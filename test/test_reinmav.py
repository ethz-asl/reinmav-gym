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

# core modules
import unittest

# 3rd party modules
import gym

# internal modules
import gym_reinmav


class Environments(unittest.TestCase):
	def test_env(self):
		env = gym.make('reinmav-v0')
		#env.reset()
		for _ in range(1000):
			#env.render()
			env.step(env.action_space.sample()) # take a random action
		env.plot_state()
if __name__ == "__main__":
	env=Environments()
	env.test_env()