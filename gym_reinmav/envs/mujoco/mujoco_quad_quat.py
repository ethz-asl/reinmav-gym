import numpy as np
from numpy import linalg
import os

from gym_reinmav.envs.mujoco import MujocoQuadEnv


class MujocoQuadQuaternionEnv(MujocoQuadEnv):
    def __init__(self, xml_name="quadrotor_quat.xml"):
        super(MujocoQuadQuaternionEnv, self).__init__(xml_name=xml_name)

        self.goal_pos = np.array([0.0, 0.0, 1.0])
        self.alive_bonus = 100



    def step(self, a):
        self.do_simulation(self.clip_action(a), self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:3]
        quat = ob[3:7] 
        lin_vel = ob[7:10]
        ang_vel= ob[10:13]
        lin_acc = ob[13:16]
        ang_acc = ob[16:19]
        

        reward_position = - linalg.norm(pos-self.goal_pos) * 10 
        reward_linear_velocity = - linalg.norm(lin_vel) * 0.1 
        reward_angular_velocity = - linalg.norm(ang_vel) * 0.1
        reward_action = - linalg.norm(a)
        reward_alive = self.alive_bonus

        reward = reward_position \
                 + reward_linear_velocity \
                 + reward_angular_velocity \
                 + reward_action \
                 + reward_alive 

        notdone = np.isfinite(ob).all() \
                  and pos[2] > 0.3 \
                  and abs(pos[0]) < 2.0 \
                  and abs(pos[1]) < 2.0

        info = {
          'rp': reward_position,
          'rlv': reward_linear_velocity,
          'rav': reward_angular_velocity,
          'ra': reward_action,
          'ra2': reward_alive,
        }

        #if done=True indicates the episode has terminated and it's time to reset the environment. (For example, perhaps the pole tipped too far, or you lost your last life.) https://gym.openai.com/docs/
        done = not notdone

        #If the episode is done, then add some variations to the initial state that will be exploited for the next ep. The low and high bounds empirically set.
        
        return ob, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=3, low=-0.05, high=0.05)
        qvel = self.init_qvel + self.np_random.uniform(size=3, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()