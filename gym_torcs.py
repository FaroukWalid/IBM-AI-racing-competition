import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time


class TorcsEnv:
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 120

    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        else:
            os.system('torcs  -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Automatic Throttle Control by Snakeoil
        if self.throttle is False:
            sd = client.S.d
            rd = client.R.d

            # Blended curvature: 70% wide (±12°) + 30% near (±1.7°)
            far_left    = sd['track'][snakeoil3.TRACK_FAR_LEFT]
            far_right   = sd['track'][snakeoil3.TRACK_FAR_RIGHT]
            near_left   = sd['track'][snakeoil3.TRACK_LEFT]
            near_right  = sd['track'][snakeoil3.TRACK_RIGHT]
            track_front = sd['track'][snakeoil3.TRACK_FRONT]
            # (left - right): right-hand corner → far_right closer → positive curvature
            wide_curv = (far_left  - far_right)  / (far_left  + far_right  + 1e-6)
            near_curv = (near_left - near_right) / (near_left + near_right + 1e-6)
            curvature = snakeoil3.clip(0.7 * wide_curv + 0.3 * near_curv, -1, 1)

            # Curvature-based corner speed + wall-proximity backstop.
            # MIN of both signals: whichever demands more braking wins.
            corner_speed = snakeoil3.MAX_SPEED * (
                1.0 - abs(curvature) * snakeoil3.CORNER_SPEED_FACTOR)
            wall_speed   = 20 + (track_front ** 0.6) * 18
            target_speed = min(corner_speed, wall_speed)

            # Past-apex detection → aggressive throttle on exit
            curvature_sign = 1 if curvature >= 0 else -1
            on_inside  = (sd['trackPos'] * curvature_sign) < 0
            past_apex  = abs(curvature) < snakeoil3.PAST_APEX_CURV and not on_inside
            accel_ramp = snakeoil3.ACCEL_RAMP_EXIT if past_apex else snakeoil3.ACCEL_RAMP_CORNER

            speed_error = target_speed - sd['speedX']
            if speed_error > 0:
                rd['accel'] = min(rd['accel'] + accel_ramp, 1.0)
                rd['brake'] = 0
            else:
                rd['brake'] = snakeoil3.clip(-speed_error / snakeoil3.BRAKE_DIVISOR, 0, 1)
                trail_t = snakeoil3.clip(1.0 - rd['brake'], 0, 1)
                rd['accel'] = trail_t * snakeoil3.TRAIL_THROTTLE_MAX

            if sd['speedX'] < 10:
                rd['accel'] += 1 / (sd['speedX'] + .1)
                rd['brake'] = 0

            # Traction Control System
            rear_spin  = sd['wheelSpinVel'][2] + sd['wheelSpinVel'][3]
            front_spin = sd['wheelSpinVel'][0] + sd['wheelSpinVel'][1]
            slip = rear_spin - front_spin
            if slip > 5:
                rd['accel'] = max(
                    0, rd['accel'] - snakeoil3.clip(slip * 0.01, 0, 0.3))

            # ABS: release brake pressure if front wheels lock
            if rd['brake'] > 0 and front_spin < 1.0 and sd['speedX'] > 5:
                rd['brake'] *= 0.5
        else:
            action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            # RPM-based automatic transmission
            rpm  = client.S.d['rpm']
            gear = int(client.S.d['gear'])
            if gear < 1:
                gear = 1
            if rpm > 8500 and gear < 6:
                action_torcs['gear'] = gear + 1
            elif rpm < 3500 and gear > 1:
                action_torcs['gear'] = gear - 1
            else:
                action_torcs['gear'] = gear

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        track = np.array(obs['track'])
        sp    = np.array(obs['speedX'])

        # Forward progress: speed projected along heading.
        # Using speedX directly (longitudinal) rather than cos(angle) weighting
        # so the agent is rewarded for raw speed even mid-corner.
        progress = sp * np.cos(obs['angle'])

        # Racing-line deviation penalty:
        # Compare actual trackPos to the ideal position for this corner.
        # Being AT the apex is rewarded, not penalised.
        far_l     = obs['track'][snakeoil3.TRACK_FAR_LEFT]
        far_r     = obs['track'][snakeoil3.TRACK_FAR_RIGHT]
        near_l    = obs['track'][snakeoil3.TRACK_LEFT]
        near_r    = obs['track'][snakeoil3.TRACK_RIGHT]
        wide_c    = (far_l  - far_r)  / (far_l  + far_r  + 1e-6)
        near_c    = (near_l - near_r) / (near_l + near_r + 1e-6)
        curv      = snakeoil3.clip(0.7 * wide_c + 0.3 * near_c, -1, 1)
        ideal_tp  = snakeoil3.clip(-curv * snakeoil3.RACING_LINE_FACTOR, -0.9, 0.9)
        line_err  = abs(obs['trackPos'] - ideal_tp) / 2.0  # normalised 0–1

        # Speed bonus: reward carrying high speed through corners.
        # Normalise by default_speed so the bonus scales to ~0-1.
        speed_bonus = sp / self.default_speed

        # Composite reward:
        #  - Forward progress is the primary signal (weight 1.0)
        #  - Speed bonus adds incentive for corner speed (weight 0.3)
        #  - Racing-line deviation subtracts up to 30% of total (was 40%)
        #    → reduced penalty so agent is not over-trained to crawl safely
        reward = (progress + 0.3 * speed_bonus) * (1.0 - 0.3 * line_err)

        # Collision detection — damage override.
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement #########################
        episode_terminate = False
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = - 1
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': u[2]})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12288,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
