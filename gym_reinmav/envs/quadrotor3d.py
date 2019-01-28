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
from math import cos, sin, pi, atan2
import numpy as np
from numpy import linalg

class Quadrotor3D(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.mass = 1.0
		self.dt = 0.01
		self.g = np.array([0.0, -9.8])

		self.att = np.array([0.0])
		self.pos = np.array([0.3, 0.0])
		self.vel = np.array([3.0, 0.0])

		self.ref_pos = np.array([0.0, 0.0])
		self.ref_vel = np.array([0.0, 0.0])

		self.viewer = None
		self.render_quad = None
		self.reftrans = None
		self.x_range = 1.0


	def step(self, action):
		thrust = action[0] # Thrust command
		w = action[1] # Angular velocity command
		# acc = thrust/self.mass * np.array([cos(self.att + pi()/2), sin(self.att + pi()/2)]) + self.g
		acc = thrust/self.mass * np.array([cos(self.att + pi/2), sin(self.att + pi/2)]) + self.g
		self.vel = self.vel + acc * self.dt
		self.pos = self.pos + self.vel * self.dt + 0.5*acc*self.dt*self.dt
		self.att = self.att + w * self.dt

	def control(self):
		Kp = -5.0
		Kv = -4.0
		tau = 0.1;

		error_pos = self.pos - self.ref_pos
		error_vel = self.vel - self.ref_vel
		# %% Calculate desired acceleration
		desired_acc = Kp * error_pos + Kv * error_vel + [0.0, 9.8];
		desired_att = atan2(desired_acc[1], desired_acc[0]) - pi/2;
		error_att = self.att - desired_att;
		w = (-1/tau) * error_att;
		thrust = self.mass * linalg.norm(desired_acc, 2);

		action = np.array([thrust, w])

		return action 

	def reset(self):
		print("reset")
		#self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
		return np.array(self.state)

	def render(self, mode='human', close=False):
		if self.viewer is None:
			from vpython import box, sphere, color, vector, rate, canvas
			rate(100)
			
			self.render_quad = sphere (canvas = self.viewer, pos=vector(self.pos[0],self.pos[1],0), radius=1, color=color.red, make_trail=False)

		if self.pos is None: return None
		
		self.render_quad.pos.x = self.pos[0]
		self.render_quad.pos.y = self.pos[1]
		self.render_quad.pos.z = 0
		
		return True