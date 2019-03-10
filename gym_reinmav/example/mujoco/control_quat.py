# **********************************************************************
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Dongho Kang <eastsky.kang@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************
import gym
import numpy as np
from numpy import linalg
from pyquaternion import Quaternion

from gym_reinmav.envs.mujoco import MujocoQuadQuaternionEnv


class Quadrotor:
    def __init__(self, ref_pos, ref_vel):
        self._state = None
        self._ref_pos = ref_pos
        self._ref_vel = ref_vel
        self._g = np.array([0.0, 0.0, -9.81])

    @property
    def state(self):
        return self._state

    @property
    def ref_pos(self):
        return self._ref_pos

    @property
    def ref_vel(self):
        return self._ref_vel

    @property
    def g(self):
        return self._g

    @ref_pos.setter
    def ref_pos(self, ref_pos):
        self._ref_pos = ref_pos

    @state.setter
    def state(self, state):
        self._state = state


def control(quat):
    def acc2quat(desired_acc, yaw):  # TODO: Yaw rotation
        """
        acceleration -> quaternion
        """
        zb_des = desired_acc / linalg.norm(desired_acc)
        yc = np.array([0.0, 1.0, 0.0])
        xb_des = np.cross(yc, zb_des)
        xb_des = xb_des / linalg.norm(xb_des)
        yb_des = np.cross(zb_des, xb_des)
        zb_des = zb_des / linalg.norm(zb_des)

        rotmat = np.array([[xb_des[0], yb_des[0], zb_des[0]],
                           [xb_des[1], yb_des[1], zb_des[1]],
                           [xb_des[2], yb_des[2], zb_des[2]]])

        desired_att = Quaternion(matrix=rotmat)

        return desired_att

    Kp = np.array([-5.0, -5.0, -5.0])
    Kv = np.array([-4.0, -4.0, -4.0])
    tau = 0.3

    state = quat.state
    ref_pos = quat.ref_pos
    ref_vel = quat.ref_vel

    pos = np.array([state[0], state[1], state[2]]).flatten()
    att = np.array([state[3], state[4], state[5], state[6]]).flatten()
    vel = np.array([state[7], state[8], state[9]]).flatten()

    error_pos = pos - ref_pos
    error_vel = vel - ref_vel

    # %% Calculate desired acceleration
    reference_acc = np.array([0.0, 0.0, 0.0])
    feedback_acc = Kp * error_pos + Kv * error_vel

    desired_acc = reference_acc + feedback_acc - quat.g

    desired_att = acc2quat(desired_acc, 0.0)

    desired_quat = Quaternion(desired_att)
    current_quat = Quaternion(att)

    error_att = current_quat.conjugate * desired_quat
    qe = error_att.elements

    w = (2 / tau) * np.sign(qe[0]) * qe[1:4]

    thrust = desired_acc.dot(current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])))

    action = np.array([thrust, w[0], w[1], w[2]])

    return action


def main():

    # TODO no hardcoding!
    dt = 0.01
    R = 0.5  # trajectory radius
    w = 1.0  # trajectory angular speed (rad/s)

    ref_z = 1.0

    ref_pos = np.array([0.0, 0.0, ref_z])
    ref_vel = np.array([0.0, 0.0, 0.0])

    env = gym.make('MujocoQuadQuat-v0')
    quat = Quadrotor(ref_pos=ref_pos, ref_vel=ref_vel)

    # [x, y, z, q0, q1, q2, q3]
    observation = env.reset()
    quat.state = observation
    for t in range(10000):
        env.render()

        quat.ref_pos = np.array([
            R * np.cos(w * dt * t),
            R * np.sin(w * dt * t),
            ref_z
        ])

        action = control(quat=quat)
        observation, reward, done, info = env.step(action)
        quat.state = observation
        # if done:
        #     break


if __name__ == "__main__":
    main()
