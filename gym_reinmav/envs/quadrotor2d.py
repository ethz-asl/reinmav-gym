#Copyright (C) 2018, by Jaeyoung Lim, jaeyoung@auterion.com
# 2D quadrotor environment using rate control inputs (continuous control)

#This is free software: you can redistribute it and/or modify
#it under the terms of the GNU Lesser General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
 
#This software package is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Lesser General Public License for more details.

#You should have received a copy of the GNU Leser General Public License.
#If not, see <http://www.gnu.org/licenses/>.


import gym
from gym import error, spaces, utils

class Quadrotor2D(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.mass = 1.0
		self.dt = 0.01

		self.att = 0.0
		self.pos = [0.0, 0.0]
		self.vel = [0.0, 0.0]


	def step(self, action):
		thrust = action[0] # Thrust command
		w = action[1] # Angular velocity command

        acc = thrust/self.mass * [cos(self.att + pi()/2), sin(self.att + pi()/2)] + [0.0, -9.8];
        self.vel = self.vel + acc * self.dt;
        self.pos = self.pos + self.vel * self.dt + 0.5*acc*self.dt*self.dt;
        self.att = self.att + w * dt;


	def reset(self):
		print("reset")
		#self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
		return np.array(self.state)

	def render(self, mode='human', close=False):
		print("render() called")
