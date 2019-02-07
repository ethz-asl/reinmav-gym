#Copyright (C) 2019, by Jaeyoung Lim, jaeyoung@auterion.com and Inkyu Sa, enddl22@gmail.com
# A test code for 2D quadrotor controlling using DDPG

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

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# internal modules
import gym_reinmav
from timeit import default_timer as timer

ENV_NAME = 'quadrotor2d-v0'
#gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 2
nb_actions = env.action_space.shape[0]
print("nb_actions=",nb_actions)
WINDOW_LENGTH = 1

print("env.action_space.shape=",env.action_space.shape)
print("env.observation_space.shape=",env.observation_space.shape)

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(WINDOW_LENGTH,) + env.observation_space.shape, name='observation_input')

flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
#agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
#                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
#                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-0)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


agent.fit(env, nb_steps=50000, visualize=True, verbose=1, nb_max_episode_steps=200)