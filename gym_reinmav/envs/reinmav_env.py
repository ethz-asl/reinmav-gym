import gym
from gym import error, spaces, utils
from gym.utils import seeding

class ReinmavEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    print("__init__ called")
  def step(self, action):
    print("step() called")
  def reset(self):
    print("reset() called")
  def render(self, mode='human', close=False):
    print("render() called")
