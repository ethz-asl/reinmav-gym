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
		self.quadtrans = None
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
		screen_width = 600
		screen_height = 400

		world_width = self.x_range*2
		scale = screen_width/world_width
		quadwidth = 80.0
		quadheight = 10.0
		ref_size = 5.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			# Draw drone
			l,r,t,b = -quadwidth/2, quadwidth/2, quadheight/2, -quadheight/2
			quad = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.quadtrans = rendering.Transform()
			quad.add_attr(self.quadtrans)
			self.viewer.add_geom(quad)
			# Draw refereence
			ref = rendering.make_circle(ref_size)
			self.reftrans = rendering.Transform()
			ref.add_attr(self.reftrans)
			ref.set_color(1,0,0)
			self.viewer.add_geom(ref)

		if self.pos is None: return None

		x = self.pos
		theta = self.att
		quad_x = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		quad_y = x[1]*scale+screen_height/2.0 # MIDDLE OF CART
		self.quadtrans.set_translation(quad_x, quad_y)
		self.quadtrans.set_rotation(theta)

		y = self.ref_pos
		ref_x = y[0]*scale+screen_width/2.0
		ref_y = y[1]*scale+screen_height/2.0
		self.reftrans.set_translation(ref_x, ref_y)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
