import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env


class MujocoQuadEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_name="quadrotor_ground.xml"):

        xml_path = os.path.join(os.path.dirname(__file__), "./assets", xml_name)

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        reward = 0
        self.do_simulation(self.clip_action(a), self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        return ob, reward, done, {}

    def clip_action(self, action):
        """
        clip action to [0, inf]
        :param action:
        :return: clipped action
        """
        action = np.clip(action, a_min=0, a_max=np.inf)
        return action

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel, self.sim.data.qacc]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 10

    @property
    def mass(self):
        return self.model.body_mass[1]

    @property
    def gravity(self):
        return self.model.opt.gravity