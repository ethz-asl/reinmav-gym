#Copyright (C) 2018, by Jaeyoung Lim, jaeyoung@auterion.com
# 3D quadrotor environment using rate control inputs (continuous control)

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
from pyquaternion import Quaternion

class Quadrotor3D(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.mass = 1.0
		self.dt = 0.01
		self.g = np.array([0.0, 0.0, -9.8])

		self.att = np.array([1.0, 0.0, 0.0, 0.0])
		self.pos = np.array([0.3, 0.0, 1.0])
		self.vel = np.array([3.0, 0.0, 0.0])

		self.ref_pos = np.array([0.0, 0.0, 1.0])
		self.ref_vel = np.array([0.0, 0.0, 0.0])

		self.viewer = None
		self.render_quad = None
		self.render_ref = None
		self.x_range = 1.0


	def step(self, action):
		thrust = action[0] # Thrust command
		w = action[1:4] # Angular velocity command
		att_quaternion = Quaternion(self.att)
		att_rotmat = att_quaternion.rotation_matrix

		acc = thrust/self.mass * np.dot(att_rotmat, np.array([0.0, 0.0, 1.0])) + self.g
		
		self.vel = self.vel + acc * self.dt
		self.pos = self.pos + self.vel * self.dt + 0.5*acc*self.dt*self.dt
		
		q_dot = att_quaternion.derivative(w)
		self.att = self.att + q_dot * self.dt


	def control(self):
		def acc2quat(desired_acc, yaw): # TODO: Yaw rotation
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
		tau = 0.1;

		error_pos = self.pos - self.ref_pos
		error_vel = self.vel - self.ref_vel

		# %% Calculate desired acceleration
		reference_acc = np.array([0.0, 0.0, 0.0])
		feedback_acc = Kp * error_pos + Kv * error_vel 
		desired_acc = reference_acc + feedback_acc + self.g

		desired_att = acc2quat(desired_acc, 0.0)
		desired_quat = Quaternion(desired_att)
		current_quat = Quaternion(self.att)
		error_att = desired_quat / current_quat
		qe = error_att.elements
		
		w = (-2/tau) * np.sign(qe[0])*qe[1:4]
		
		thrust = desired_acc.dot(current_quat.rotation_matrix*np.array([0.0, 0.0, 1.0]))

		action = np.array([thrust, w[0], w[1], w[2]])

		return action 

	def reset(self):
		print("reset")
		#self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
		return np.array(self.state)

	def render(self, mode='human', close=False):
		from vpython import box, sphere, color, vector, rate, canvas

		if self.viewer is None:
			self.viewer = canvas(title='Quadrotor 3D', width=640, height=480, center=vector(1, 0,1), background=color.white)
			self.render_quad = box(canvas = self.viewer, pos=vector(self.pos[0],self.pos[1],0), axis=vector(self.att[1],self.att[2],self.att[3]), length=0.2, height=0.05, width=0.05)
			self.render_ref = sphere(canvas = self.viewer, pos=vector(self.ref_pos[0], self.ref_pos[1], self.ref_pos[2]), radius=0.02, color=color.blue, make_trail = 0)

		if self.pos is None: return None

		self.render_quad.pos.x = self.pos[0]
		self.render_quad.pos.y = self.pos[1]
		self.render_quad.pos.z = self.pos[2]
		# self.render_quad.axis.x = self.att[1]
		# self.render_quad.axis.y = self.att[2]	
		# self.render_quad.axis.z = self.att[3]
		self.render_ref.pos.x = self.ref_pos[0]
		self.render_ref.pos.y = self.ref_pos[1]
		self.render_ref.pos.z = self.ref_pos[2]
		print(self.pos[0], self.pos[1], self.pos[2])


		rate(100)

		return True