import numpy as np
import os

from gym_reinmav.envs.mujoco import MujocoQuadEnv

class MujocoQuadForceEnv(MujocoQuadEnv):
    def __init__(self):
        super(MujocoQuadForceEnv, self).__init__()

    def step(self, a):
        reward = 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        return ob, reward, done, {}