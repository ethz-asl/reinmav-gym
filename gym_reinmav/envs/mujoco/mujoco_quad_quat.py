import numpy as np
from numpy import linalg
import os

from gym_reinmav.envs.mujoco import MujocoQuadEnv


class MujocoQuadQuaternionEnv(MujocoQuadEnv):
    def __init__(self, xml_name="quadrotor_quat.xml"):
    #def __init__(self, xml_name="quadrotor_hovering.xml"):
        super(MujocoQuadQuaternionEnv, self).__init__(xml_name=xml_name)

    def step(self, a):
        reward = 0 #reward = -cost
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        #print("ob=",ob)
        # notdone = np.isfinite(ob).all()
        # done = not notdone
        terminated_penalty=-2
        alive_bonus = 5
        goal_pos = np.array([0.0, 0.0, 1.0])
        pos = ob[0:3]
        quat = ob[3:7] 
        lin_vel = ob[7:10]
        ang_vel= ob[10:13]
        lin_acc = ob[13:16]
        ang_acc = ob[16:]

        # dist_goal=linalg.norm(goal_pos,2)
        # cost  = ( linalg.norm(pos-goal_pos)* 4*1e-3 \
        # 		 +linalg.norm(ang_vel)* 3*1e-4 \
        # 		 +linalg.norm(lin_vel)* 5*1e-4 \
        # 		 +linalg.norm(lin_acc)* 2*1e-4 \
        # 		 +linalg.norm(ang_acc)* 4*1e-5 )
        # reward = -cost

        # done = not (np.isfinite(ob).all() and \
        # 	   abs(pos[2]) > 0.3 and \
        # 	   abs(pos[0]) < 2.0 and \
        # 	   abs(pos[1]) < 2.0)
        # done = bool(done)


        #=========================
        # reward from quad hover
        #=========================

        alive_bonus = 100
        reward = - np.sum(np.square(ob[0:3] - np.array([0.0, 0, 1.0]))) * 10 \
                 - np.sum(np.square(ob[7:13] - np.zeros(6))) * 0.1 \
                 - np.sum(np.square(ob[13:] - np.zeros(6))) * 0.01\
                 - np.sum(np.square(a)) \
                 + np.sum(a) * 0.1 \
                 + alive_bonus

        notdone = np.isfinite(ob).all() \
                  and ob[2] > 0.3 \
                  and abs(ob[0]) < 2.0 \
                  and abs(ob[1]) < 2.0
        done = not notdone
        #if done=True indicates the episode has terminated and it's time to reset the environment. (For example, perhaps the pole tipped too far, or you lost your last life.) https://gym.openai.com/docs/
        return ob, reward, done, {}
