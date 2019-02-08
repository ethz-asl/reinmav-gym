#Copyright (C) 2018, by Jaeyoung Lim, jalim@student.ethz.ch
# 2D quadrotor environment using rate control inputs  (continuous control)

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

# core modules
import unittest

# 3rd party modules
import gym

# internal modules
import gym_reinmav
from timeit import default_timer as timer

class Environments(unittest.TestCase):
	def test_env(self):
		env = gym.make('quadrotor3d-v0')
		env.reset()
		start_t=timer()
		for i,_ in enumerate(range(400)): #dt=0.01, 400*0.01=4s

			action = env.control()
			_, reward, done, _ = env.step(action) # take a random action
			env.render() # take a random action
			if(done):
				env.reset()
		end_t=timer()
		print("simulation time=",end_t-start_t)
		# env.plot_state()
if __name__ == "__main__":
	env=Environments()
	env.test_env()