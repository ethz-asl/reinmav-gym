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

from gym_reinmav.envs.mujoco import MujocoQuadEnv
from gym_reinmav.controller import RpyController


class Trajectory:
    R = 0.5     # trajectory radius
    w = 1.0     # trajectory angular speed (rad/s)

def main():

    env = gym.make('MujocoQuadForce-v0')

    dt = env.dt
    mass = env.mass
    gravity = env.gravity[2]

    # controller
    ctrl = RpyController(dt, mass, gravity=gravity)

    # [x, y, z, q0, q1, q2, q3]
    observation = env.reset()
    for t in range(1000):

        env.render()

        # current position and quaternion
        s = np.array([observation[0], observation[1], observation[2]])
        q = observation[3:7]

        # desired trajectory (position and yaw)
        pos_d = np.array([
            Trajectory.R * np.cos(Trajectory.w * dt * t),
            Trajectory.R * np.sin(Trajectory.w * dt * t),
            1.0
        ])
        yaw_d = (Trajectory.w * dt * t + np.pi) % (2 * np.pi) - np.pi

        # control
        action = ctrl.control(s, q, pos_d, yaw_d)
        observation, reward, done, info = env.step(action)

        if done:
            break

if __name__ == "__main__":
    main()
