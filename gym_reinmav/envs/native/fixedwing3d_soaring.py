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
from gym.utils import seeding
from pyquaternion import Quaternion

class Fixedwing3DSoaring(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.mass = 1.0
		self.dt = 0.01
		self.g = np.array([0.0, 0.0, -9.8])

		self.state = None

		self.ref_pos = np.array([0.0, 0.0, 2.0])
		self.ref_vel = np.array([0.0, 0.0, 0.0])

		# Conditions to fail the episode
		self.pos_threshold = 0.1
		self.vel_threshold = 0.1

		self.seed()
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

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		thrust = action[0] # Thrust command
		w = action[1:4] # Angular velocity command

		dphi_dt = w[0] # Roll rate
		dgamma_a_dt = w[1]  # Pitch rate  

		Jw = wind_gradient;

		state = self.state

		gamma_a = np.array([state[0]]).flatten()
		psi_a = np.array([state[1]]).flatten()
		phi = np.array([state[2]]).flatten()
		pos = np.array([state[3], state[4], state[5]]).flatten()
		vel = np.array([state[6], state[7], state[8]]).flatten()
		v_a = np.array([state[9]]).flatten()

		L = m/cos(phi) * (v_a * dgamma_a_dt + g * cos(gamma_a) - [sin(gamma_a)*cos(psi_a), sin(gamma_a) * sin(psi_a), cos(gamma_a)]* Jw * vel')			
		if liftisfeasible(state, L): #  Check if lift is feasible
		    L =  0.5 * Cl_max * (rho * v_a^2 * S );
		    dgamma_a_dt = (1/v_a) * ( L / m * cos(phi) - g * cos(gamma_a) + [sin(gamma_a)*cos(psi_a), sin(gamma_a) * sin(psi_a), cos(gamma_a)] * Jw * vel')

		D = 0.5 * rho * v_a^2 * S * Cd_0 + L^2 / (0.5 * rho * v_a^2 * S * pi() * AR * e)
		dva_dt = (-D) / m - g * sin(gamma_a) - [cos(gamma_a)*cos(psi_a), cos(gamma_a) * sin(psi_a), -sin(gamma_a)] * Jw * vel'

		dpsi_a_dt = (1 / (v_a * cos(gamma_a))) * (L / m * sin(phi) + [sin(psi_a), -cos(psi_a), 0] * Jw * vel')
		Cia = rpy2rotmat(phi, gamma_a, psi_a) 		# Calculate intertial forces
		acc = (1 / m) * (Cia * [ -D, 0, -L]' + m * [0, 0, 9.8]')'
		vel = vel + acc * dt;
		pos = pos + vel * dt + 0.5 * acc * dt^2;
		v_a = v_a + dva_dt * dt ;
		psi_a = psi_a + dpsi_a_dt * dt;
		phi = phi + dphi_dt * dt;
		gamma_a = gamma_a + dgamma_a_dt * dt;

		state = setstate(gamma_a, psi_a, phi, pos, vel, v_a);


		self.state = (gamma_a, psi_a, phi, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], v_a)

		done =  linalg.norm(pos, 2) < -self.pos_threshold \
			and  linalg.norm(pos, 2) > self.pos_threshold \
			and linalg.norm(vel, 2) < -self.vel_threshold \
			and linalg.norm(vel, 2) > self.vel_threshold
		done = bool(done)

		if not done:
		    reward = (-linalg.norm(pos, 2))
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


	def reset(self):
		print("reset")
		self.state = np.array(self.np_random.uniform(low=-1.0, high=1.0, size=(10,1)))
		return np.array(self.state)

	def render(self, mode='human', close=False):
		from vpython import box, sphere, color, vector, rate, canvas, cylinder, arrow, curve

		def make_grid(unit, n):
			nunit = unit * n
			for i in range(n+1):
				if i%5==0:
					lcolor = vector(0.5,0.5,0.5)
				else:
					lcolor = vector(0.5, 0.5, 0.5)
				curve(pos=[(0,i*unit,0), (nunit, i*unit, 0)],color=lcolor)
				curve(pos=[(i*unit,0,0), (i*unit, nunit, 0)],color=lcolor)
				curve(pos=[(0,-i*unit,0), (-nunit, -i*unit, 0)],color=lcolor)
				curve(pos=[(-i*unit,0,0), (-i*unit, -nunit, 0)],color=lcolor)
				curve(pos=[(0,i*unit,0), (-nunit, -i*unit, 0)],color=lcolor)
				curve(pos=[(i*unit,0,0), (-i*unit, -nunit, 0)],color=lcolor)				

		state = self.state
		ref_pos = self.ref_pos
		ref_vel = self.ref_vel

		pos = np.array([state[0], state[1], state[2]]).flatten()
		att = np.array([state[3], state[4], state[5], state[6]]).flatten()
		vel = np.array([state[7], state[8], state[9]]).flatten()


		current_quat = Quaternion(att)
		x_axis = current_quat.rotation_matrix.dot(np.array([1.0, 0.0, 0.0]))
		y_axis = current_quat.rotation_matrix.dot(np.array([0.0, 1.0, 0.0]))
		z_axis = current_quat.rotation_matrix.dot(np.array([0.0, 0.0, 1.0]))

		if self.viewer is None:
			self.viewer = canvas(title='Quadrotor 3D', width=640, height=480, center=vector(0, 0, 2), forward=vector(1, 1, -0.5), up=vector(0, 0, 1), background=color.white, range=4.0, autoscale = False)
			self.render_quad1 = box(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(x_axis[0],x_axis[1],x_axis[2]), length=0.2, height=0.05, width=0.05)
			self.render_quad2 = box(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(y_axis[0],y_axis[1],y_axis[2]), length=0.2, height=0.05, width=0.05)
			self.render_rotor1 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor2 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor3 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_rotor4 = cylinder(canvas = self.viewer, pos=vector(pos[0],pos[1],0), axis=vector(0.01*z_axis[0],0.01*z_axis[1],0.01*z_axis[2]), radius=0.2, color=color.cyan, opacity=0.5)
			self.render_velocity = pointer = arrow(pos=vector(pos[0],pos[1],0), axis=vector(vel[0],vel[1],vel[2]), shaftwidth=0.05, color=color.green)
			self.render_ref = sphere(canvas = self.viewer, pos=vector(ref_pos[0], ref_pos[1], ref_pos[2]), radius=0.02, color=color.blue, make_trail = True)
			grid_xy = make_grid(5, 100)
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