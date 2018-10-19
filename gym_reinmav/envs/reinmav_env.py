#Copyright (C) 2018, by Inkyu Sa, enddl22@gmail.com
# Adaptation of the MountainCar Environment (continuous control)

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
from gym.utils import seeding
import math
import numpy as np
from timeit import default_timer as timer
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class ReinmavEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		self.min_action = -100.0
		self.max_action = 100.0 #in N/m, Force
		self.min_position = -100 #in meter
		self.max_position = 100
		self.max_speed = 100 #in m/s
		self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
		
		self.low_state = np.array([self.min_position, -self.max_speed])
		self.high_state = np.array([self.max_position, self.max_speed])

		self.viewer = None

		self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,),dtype=np.float32)
		self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

		self.m=10 #kg
		self.k=50 #N/m
		self.c=1/2*np.sqrt(self.k*self.m) #Critical damping
		self.c1=self.c/self.m;
		self.c2=self.k/self.m;
		self.init_state=[10.0,10.0] #init value for position and velocity.
		self.dt=1.0/100.0 #10ms
		self.action=0
		self.state=self.init_state

		self.seed()
		self.cum_state=self.state
		#self.reset()
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		self.action=action
		start = timer()
		state = odeint(self.mass_spring_damping, self.state, [0,self.dt])
		self.state = state[-1]
		self.cum_state = np.vstack([self.cum_state,self.state])
		position = self.state[0] # We only care about the state at the ''final timestep'', self.dt
		velocity = self.state[1]
		done = bool(position >= self.goal_position)
		reward = 0
		if done:
			reward = 100.0
		reward-= math.pow(action,2)*0.1
		self.state = np.array([position, velocity])
		end = timer()
		#print("step odeint duration={:0.4f}ms".format((end - start)*1e3)) #in us
		return self.state, reward, done, {}

	def plot_state(self):
		t=np.arange(0.0,len(self.cum_state)*self.dt,self.dt)
		plt.plot(t, self.cum_state)
		plt.title("title")
		plt.xlabel("Time(s)")
		plt.ylabel("y-label")
		plt.legend(["position","velocity"])
		plt.show()
		
	def mass_spring_damping(self,state,t):
		# 1st order ODE
		ret0 = state[1] #vel
		ret1 = self.action/self.m-self.c1*state[1]-self.c2*state[0] #acc, input is force not accelation
		# return the two state derivatives
		return [ret0, ret1]

	def reset(self):
		self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
		return np.array(self.state)

	def render(self, mode='human', close=False):
		print("render() called")
