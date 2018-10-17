# core modules
import unittest

# 3rd party modules
import gym

# internal modules
import gym_reinmav


class Environments(unittest.TestCase):
    def test_env(self):
        env = gym.make('reinmav-v0')
        env.reset()
        env.step(0)
        env.render()

if __name__ == "__main__":
	env=Environments()
	env.test_env()

