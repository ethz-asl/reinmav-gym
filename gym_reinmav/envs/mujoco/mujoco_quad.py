import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env


class MujocoQuadEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        xml_path = os.path.join(os.path.dirname(__file__), "./assets", "quadrotor_force.xml")

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        reward = 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
