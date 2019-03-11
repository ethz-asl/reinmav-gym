# **********************************************************************
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Jaeyoung Lim <jalim@student.ethz.ch>
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
from gym import error, spaces, utils, logger
from math import cos, sin, pi, atan2
import numpy as np
from numpy import linalg
from gym.utils import seeding
from pyquaternion import Quaternion

class Quadrotor3DSlungload(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.mass = 1.0
		self.load_mass = 0.1
		self.dt = 0.01
		self.g = np.array([0.0, 0.0, -9.8])

		self.state = None

		self.ref_pos = np.array([0.0, 0.0, 1.0])
		self.ref_vel = np.array([0.0, 0.0, 0.0])

		# Conditions to fail the episode
		self.pos_threshold = 3.0
		self.vel_threshold = 10.0

		self.tether_length = 1.5

		self.viewer = None
		self.render_quad1 = None
		self.render_quad2 = None
		self.render_rotor1 = None
		self.render_rotor2 = None
		self.render_rotor3 = None
		self.render_rotor4 = None
		self.render_velocity = None
		self.render_ref = None
		self.render_load = None
		self.render_tether = None
		self.x_range = 1.0
		self.steps_beyond_done = None

		self.action_space = spaces.Box(low=-10.0, high=10.0,
                                       shape=(4,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                        shape=(16,), dtype=np.float32)
		self.seed()
		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


	def step(self, action):
		thrust = action[0] # Thrust command
		w = action[1:4] # Angular velocity command

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos = np.array([state[0], state[1], state[2]]).flatten()
		att = np.array([state[3], state[4], state[5], state[6]]).flatten()
		vel = np.array([state[7], state[8], state[9]]).flatten()
		load_pos = np.array([state[10], state[11], state[12]]).flatten()
		load_vel = np.array([state[13], state[14], state[15]]).flatten()

		tether_vec = load_pos - pos
		unit_tether_vec = tether_vec / linalg.norm(tether_vec)

		if linalg.norm(tether_vec) >= self.tether_length:

			#Quadrotor Dynamics
			att_quaternion = Quaternion(att)

			thrust_vec = thrust*att_quaternion.rotation_matrix.dot(np.array([0.0, 0.0, 1.0]))
			load_acceleration = np.inner(unit_tether_vec, thrust_vec - self.mass * self.tether_length * np.inner(load_vel, load_vel)) * unit_tether_vec
			load_acceleration = (1/(self.mass + self.load_mass)) * load_acceleration + self.g
			load_pos = load_pos + load_vel * self.dt + 0.5 * load_acceleration * self.dt * self.dt
			load_vel = load_vel + load_acceleration * self.dt

			T = self.load_mass * linalg.norm(-self.g + load_acceleration) * unit_tether_vec

			#Quadrotor Dynamics
			acc = thrust/self.mass * att_quaternion.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])) + self.g + T/self.mass
			pos = pos + vel * self.dt + 0.5*acc*self.dt*self.dt
			vel = vel + acc * self.dt
			
			q_dot = att_quaternion.derivative(w)
			att = att + q_dot.elements * self.dt

			## Enforce kinematic constraints
			load_direction = (load_pos - pos) / linalg.norm(load_pos - pos)
			load_pos = pos + load_direction * self.tether_length
			load_vel = load_vel - np.inner(load_vel - vel, load_direction) * load_direction


		else:
			att_quaternion = Quaternion(att)

			# Load dynamics
			load_acceleration = self.g
			load_pos = load_pos + load_vel * self.dt + 0.5 * load_acceleration * self.dt * self.dt
			load_vel = load_vel + load_acceleration * self.dt

			# Quadrotor Dynamics
			acc = thrust/self.mass * att_quaternion.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])) + self.g			
			pos = pos + vel * self.dt + 0.5*acc*self.dt*self.dt
			vel = vel + acc * self.dt
			
			q_dot = att_quaternion.derivative(w)
			att = att + q_dot.elements * self.dt

		self.state = (pos[0], pos[1], pos[2], att[0], att[1], att[2], att[3], vel[0], vel[1], vel[2], load_pos[0], load_pos[1], load_pos[2], load_vel[0], load_vel[1], load_vel[2])

		done =  linalg.norm(load_pos, 2) < -self.pos_threshold \
			or  linalg.norm(load_pos, 2) > self.pos_threshold \
			or linalg.norm(vel, 2) < -self.vel_threshold \
			or linalg.norm(vel, 2) > self.vel_threshold
		done = bool(done)

		if not done:
		    reward = (-linalg.norm(load_pos, 2))
		elif self.steps_beyond_done is None:
		    # Pole just fell!
		    self.steps_beyond_done = 0
		    reward = 1.0
		else:
		    if self.steps_beyond_done == 0:
		        logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
		    self.steps_beyond_done += 1
		    reward = 0.0

		return np.array(self.state), reward, done, {}

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

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos = np.array([state[0], state[1], state[2]]).flatten()
		att = np.array([state[3], state[4], state[5], state[6]]).flatten()
		vel = np.array([state[7], state[8], state[9]]).flatten()
		load_pos = np.array([state[10], state[11], state[12]]).flatten()
		load_vel = np.array([state[13], state[14], state[15]]).flatten()


		error_pos = pos - ref_pos
		error_vel = vel - ref_vel

		# %% Calculate desired acceleration
		reference_acc = np.array([0.0, 0.0, 0.0])
		feedback_acc = Kp * error_pos + Kv * error_vel 

		desired_acc = reference_acc + feedback_acc - self.g

		desired_att = acc2quat(desired_acc, 0.0)

		desired_quat = Quaternion(desired_att)
		current_quat = Quaternion(att)

		error_att = current_quat.conjugate * desired_quat
		qe = error_att.elements

		
		w = (2/tau) * np.sign(qe[0])*qe[1:4]

		
		thrust = desired_acc.dot(current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0])))
		
		action = np.array([thrust, w[0], w[1], w[2]])

		return action 

	def reset(self):
		print("reset")
		self.state = np.array(self.np_random.uniform(low=-1.0, high=1.0, size=(16,)))
		return np.array(self.state)

	def render(self, mode='human', close=False):
		from vpython import box, sphere, color, vector, rate, canvas, cylinder, arrow

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos = np.array([state[0], state[1], state[2]]).flatten()
		att = np.array([state[3], state[4], state[5], state[6]]).flatten()
		vel = np.array([state[7], state[8], state[9]]).flatten()
		load_pos = np.array([state[10], state[11], state[12]]).flatten()
		load_vel = np.array([state[13], state[14], state[15]]).flatten()

		current_quat = Quaternion(att)
		x_axis = current_quat.rotation_matrix.dot(np.array([1.0, 0.0, 0.0]))
		y_axis = current_quat.rotation_matrix.dot(np.array([0.0, 1.0, 0.0]))
		z_axis = current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0]))
		tether_vec = load_pos - pos

		if self.viewer is None:
			self.viewer = canvas(title='Quadrotor 3D Slungload', width=640, height=480, center=vector(0, 0, 0), forward=vector(1, 1, -1), up=vector(0, 0, 1), background=color.white)
			self.render_quad1 = box(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(x_axis[0],x_axis[1],x_axis[2]), length=0.2, height=0.05, width=0.05)
			self.render_quad2 = box(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(y_axis[0],y_axis[1],y_axis[2]), length=0.2, height=0.05, width=0.05)
			self.render_rotor1 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor2 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor3 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor4 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_velocity = pointer = arrow(pos=vector(pos[0],pos[1],0), axis=vector(vel[0],vel[1],vel[2]), shaftwidth=0.05, color=color.green)
			self.render_ref = sphere(canvas = self.viewer, pos=vector(ref_pos[0], ref_pos[1], ref_pos[2]), radius=0.02, color=color.blue, make_trail = False)
			self.render_tether = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(tether_vec[0],tether_vec[1],tether_vec[2]), radius=0.01, color=color.black)
			self.render_load = sphere(canvas = self.viewer, pos=vector(load_pos[0], load_pos[1], load_pos[2]), radius=0.1, color=color.red, make_trail = False)
		if self.state is None: return None

		self.render_quad1.pos.x = pos[0]
		self.render_quad1.pos.y = pos[1]
		self.render_quad1.pos.z = pos[2]
		self.render_quad2.pos.x = pos[0]
		self.render_quad2.pos.y = pos[1]
		self.render_quad2.pos.z = pos[2]
		rotor_pos = 0.5*x_axis
		self.render_rotor1.pos.x = pos[0] + rotor_pos[0]
		self.render_rotor1.pos.y = pos[1] + rotor_pos[1]
		self.render_rotor1.pos.z = pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*x_axis
		self.render_rotor2.pos.x = pos[0] + rotor_pos[0]
		self.render_rotor2.pos.y = pos[1] + rotor_pos[1]
		self.render_rotor2.pos.z = pos[2] + rotor_pos[2]
		rotor_pos = 0.5*y_axis
		self.render_rotor3.pos.x = pos[0] + rotor_pos[0]
		self.render_rotor3.pos.y = pos[1] + rotor_pos[1]
		self.render_rotor3.pos.z = pos[2] + rotor_pos[2]
		rotor_pos = (-0.5)*y_axis
		self.render_rotor4.pos.x = pos[0] + rotor_pos[0]
		self.render_rotor4.pos.y = pos[1] + rotor_pos[1]
		self.render_rotor4.pos.z = pos[2] + rotor_pos[2]
		self.render_velocity.pos.x = pos[0]
		self.render_velocity.pos.y = pos[1]
		self.render_velocity.pos.z = pos[2]
		self.render_load.pos.x = load_pos[0]
		self.render_load.pos.y = load_pos[1]
		self.render_load.pos.z = load_pos[2]
		self.render_tether.pos.x = pos[0]
		self.render_tether.pos.y = pos[1]
		self.render_tether.pos.z = pos[2]

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
		self.render_tether.axis.x = tether_vec[0]
		self.render_tether.axis.y = tether_vec[1]
		self.render_tether.axis.z = tether_vec[2]
		self.render_velocity.axis.x = 0.5 * vel[0]
		self.render_velocity.axis.y = 0.5 * vel[1]
		self.render_velocity.axis.z = 0.5 * vel[2]


		self.render_quad1.up.x = z_axis[0]
		self.render_quad1.up.y = z_axis[1]
		self.render_quad1.up.z = z_axis[2]
		self.render_quad2.up.x = z_axis[0]
		self.render_quad2.up.y = z_axis[1]
		self.render_quad2.up.z = z_axis[2]
		self.render_rotor1.up.x = y_axis[0]
		self.render_rotor1.up.y = y_axis[1]
		self.render_rotor1.up.z = y_axis[2]
		self.render_rotor2.up.x = y_axis[0]
		self.render_rotor2.up.y = y_axis[1]
		self.render_rotor2.up.z = y_axis[2]
		self.render_rotor3.up.x = y_axis[0]
		self.render_rotor3.up.y = y_axis[1]
		self.render_rotor3.up.z = y_axis[2]
		self.render_rotor4.up.x = y_axis[0]
		self.render_rotor4.up.y = y_axis[1]
		self.render_rotor4.up.z = y_axis[2]

		self.render_ref.pos.x = ref_pos[0]
		self.render_ref.pos.y = ref_pos[1]
		self.render_ref.pos.z = ref_pos[2]

		rate(100)

		return True

	def close(self):
		if self.viewer:
			self.viewer = None