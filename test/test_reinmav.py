# core modules
import unittest

# 3rd party modules
import gym

# internal modules
import gym_reinmav
from timeit import default_timer as timer

class Environments(unittest.TestCase):
	def test_env(self):
		env = gym.make('reinmav-v0')
		#env.reset()
		#for i, _ in enumerate(dict_of_list[key]):
		start_t=timer()
		for i,_ in enumerate(range(400)): #dt=0.01, 400*0.01=4s
		# 	#env.render()
			#print("============step {}============".format(i))
			#print("step=",i)
		 	#env.step(env.action_space.sample()) # take a random action
			env.step() # take a random action
		end_t=timer()
		print("simulation time=",end_t-start_t)
		env.plot_state()
if __name__ == "__main__":
	env=Environments()
	env.test_env()