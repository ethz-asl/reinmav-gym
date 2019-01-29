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

class Quadrotor3DSlungload(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.mass = 1.0
		self.dt = 0.01
		self.g = np.array([0.0, 0.0, -9.8])

		self.att = np.array([1.0, 0.0, 0.0, 0.0])
		self.pos = np.array([0.3, 0.1, 1.0])
		self.vel = np.array([3.0, 0.0, 0.0])

		self.ref_pos = np.array([0.0, 0.0, 1.0])
		self.ref_vel = np.array([0.0, 0.0, 0.0])

		self.viewer = None
		self.render_quad1 = None
		self.render_quad2 = None
		self.render_rotor1 = None
		self.render_rotor2 = None
		self.render_rotor3 = None
		self.render_rotor4 = None
		self.render_velocity = None
		self.render_ref = None
		self.x_range = 1.0


	def step(self, action):
		thrust = action[0] # Thrust command
		w = action[1:4] # Angular velocity command
		att_quaternion = Quaternion(self.att)

		acc = thrust/self.mass * att_quaternion.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])) + self.g
		
		vel = self.vel
		self.vel = vel + acc * self.dt
		self.pos = self.pos + vel * self.dt + 0.5*acc*self.dt*self.dt
		
		q_dot = att_quaternion.derivative(w)
		self.att = self.att + q_dot.elements * self.dt

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
		tau = 0.3;

		error_pos = self.pos - self.ref_pos
		error_vel = self.vel - self.ref_vel

		# %% Calculate desired acceleration
		reference_acc = np.array([0.0, 0.0, 0.0])
		feedback_acc = Kp * error_pos + Kv * error_vel 

		desired_acc = reference_acc + feedback_acc - self.g

		desired_att = acc2quat(desired_acc, 0.0)

		desired_quat = Quaternion(desired_att)
		current_quat = Quaternion(self.att)

		error_att = current_quat.conjugate * desired_quat
		qe = error_att.elements

		
		w = (2/tau) * np.sign(qe[0])*qe[1:4]

		
		thrust = desired_acc.dot(current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])))
		
		action = np.array([thrust, w[0], w[1], w[2]])

		return action 

	def reset(self):
		print("reset")
		#self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
		return np.array(self.state)

	def render(self, mode='human', close=False):
		from vpython import box, sphere, color, vector, rate, canvas, cylinder, arrow
		current_quat = Quaternion(self.att)
		x_axis = current_quat.rotation_matrix.dot(np.array([1.0, 0.0, 0.0]))
		y_axis = current_quat.rotation_matrix.dot(np.array([0.0, 1.0, 0.0]))
		z_axis = current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0]))

		if self.viewer is None:
			self.viewer = canvas(title='Quadrotor 3D', width=640, height=480, center=vector(0, 0, 0), forward=vector(1, 1, -1), up=vector(0, 0, 1), background=color.white)
			self.render_quad1 = box(canvas = self.viewer, pos=vector(self.pos[0],self.pos[1],0), axis=vector(x_axis[0],x_axis[1],x_axis[2]), length=0.2, height=0.05, width=0.05)
			self.render_quad2 = box(canvas = self.viewer, pos=vector(self.pos[0],self.pos[1],0), axis=vector(y_axis[0],y_axis[1],y_axis[2]), length=0.2, height=0.05, width=0.05)
			self.render_rotor1 = cylinder(canvas = self.viewer, pos=vector(self.pos[0],self.pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor2 = cylinder(canvas = self.viewer, pos=vector(self.pos[0],self.pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor3 = cylinder(canvas = self.viewer, pos=vector(self.pos[0],self.pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor4 = cylinder(canvas = self.viewer, pos=vector(self.pos[0],self.pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_velocity = pointer = arrow(pos=vector(self.pos[0],self.pos[1],0), axis=vector(self.vel[0],self.vel[1],self.vel[2]), shaftwidth=0.05, color=color.green)
			self.render_ref = sphere(canvas = self.viewer, pos=vector(self.ref_pos[0], self.ref_pos[1], self.ref_pos[2]), radius=0.02, color=color.blue, make_trail = True)

		if self.pos is None: return None

		self.render_quad1.pos.x = self.pos[0]
		self.render_quad1.pos.y = self.pos[1]
		self.render_quad1.pos.z = self.pos[2]
		self.render_quad2.pos.x = self.pos[0]
		self.render_quad2.pos.y = self.pos[1]
		self.render_quad2.pos.z = self.pos[2]
		rotor_pos = 0.5*x_axis
		self.render_rotor1.pos.x = self.pos[0] + rotor_pos[0]
		self.render_rotor1.pos.y = self.pos[1] + rotor_pos[1]
		self.render_rotor1.pos.z = self.pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*x_axis
		self.render_rotor2.pos.x = self.pos[0] + rotor_pos[0]
		self.render_rotor2.pos.y = self.pos[1] + rotor_pos[1]
		self.render_rotor2.pos.z = self.pos[2] + rotor_pos[2]
		rotor_pos = 0.5*y_axis
		self.render_rotor3.pos.x = self.pos[0] + rotor_pos[0]
		self.render_rotor3.pos.y = self.pos[1] + rotor_pos[1]
		self.render_rotor3.pos.z = self.pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*y_axis
		self.render_rotor4.pos.x = self.pos[0] + rotor_pos[0]
		self.render_rotor4.pos.y = self.pos[1] + rotor_pos[1]
		self.render_rotor4.pos.z = self.pos[2] + rotor_pos[2]
		self.render_velocity.pos.x = self.pos[0]
		self.render_velocity.pos.y = self.pos[1]
		self.render_velocity.pos.z = self.pos[2]

		self.render_quad1.axis.x = x_axis[0]
		self.render_quad1.axis.y = x_axis[1]	
		self.render_quad1.axis.z = x_axis[2]
		self.render_quad2.axis.x = y_axis[0]
		self.render_quad2.axis.y = y_axis[1]
		self.render_quad2.axis.z = y_axis[2]
		self.render_rotor1.axis.x = 0.01*z_axis[0]
		self.render_rotor1.axis.y = 0.01*z_axis[1]
		self.render_rotor1.axis.z = 0.01*z_axis[2]
		self.render_rotor2.axis.x = 0.01*z_axis[0]
		self.render_rotor2.axis.y = 0.01*z_axis[1]
		self.render_rotor2.axis.z = 0.01*z_axis[2]
		self.render_rotor3.axis.x = 0.01*z_axis[0]
		self.render_rotor3.axis.y = 0.01*z_axis[1]
		self.render_rotor3.axis.z = 0.01*z_axis[2]
		self.render_rotor4.axis.x = 0.01*z_axis[0]
		self.render_rotor4.axis.y = 0.01*z_axis[1]
		self.render_rotor4.axis.z = 0.01*z_axis[2]
		self.render_velocity.axis.x = 0.5 * self.vel[0]
		self.render_velocity.axis.y = 0.5 * self.vel[1]
		self.render_velocity.axis.z = 0.5 * self.vel[2]


		self.render_quad1.up.x = z_axis[0]
		self.render_quad1.up.y = z_axis[1]
		self.render_quad1.up.z = z_axis[2]
		self.render_quad2.up.x = z_axis[0]
		self.render_quad2.up.y = z_axis[1]
		self.render_quad2.up.z = z_axis[2]


		self.render_ref.pos.x = self.ref_pos[0]
		self.render_ref.pos.y = self.ref_pos[1]
		self.render_ref.pos.z = self.ref_pos[2]

		rate(100)

		return True