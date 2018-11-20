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
from timeit import default_timer as timer

from gym import spaces
import numpy as np
import random
import pickle
import sys
import datetime
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Reinforce(object):
    def init_params(self):
        #Hyperparameter definition
        self.total_tr_epi = 100000
        self.total_test_epi=10000
        self.max_steps_tr=10000
        self.max_steps_test=10000
        self.batch_size=32
        
        self.lr=0.001#1e-1*0.7
        self.min_lr=0.2
        self.gamma=0.99 #discouting rate
        #The larger the gamma, the smaller the discount. This means the learning agent cares more about the long term reward. On the other hand, the smaller the gamma, the bigger the discount. This means our agent cares more about the short term reward (the nearest cheese).

        #Exploitation params
        self.epsilon = 1.0 #Exploration rate, https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe
        self.max_epsilon=1.0
        self.min_epsilon = 0.01
        self.epsilon_decay= 0.995 
        self.dr=1e-2 #decay rate
        self.save_model=False

        # Angle at which to fail the episode
        self.num_actions = self.env.action_space.shape[0]
        print("self.num_actions=",self.num_actions)
        #self.buckets = (3,6,12,24) # (x, x', theta, theta') for discretization
        self.num_state = self.env.observation_space.shape[0]
        print("self.num_state=",self.num_state)
        self.memory = deque(maxlen=1000)
        #self.init_state=[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #init value for [x, y, z, dx, dy, dz, qw, qx, qy, qz, p, q, r]
		#self.state=self.init_state


    def __init__(self):
        self.env = gym.make('reinmav-v0') #e.g. "Taxi-v2", "CartPole-v0"
        self.init_params()
        #self.Q = np.zeros(self.buckets + (self.num_actions,)) # Creating a Q-Table for each state-action pair
        self.build_model()

    def build_model(self):
    	#MLP
        model = Sequential()
        model.add(Dense(24, input_dim=self.num_state, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.lr))
        self.model=model

    def history(self,state,action,reward,done,next_state):
        self.memory.append((state,action,reward,done,next_state))
        #print("history")

    def train_with_history(self):
        #print("self.memory.cout=",self.memory.count)
        miniBatch = random.sample(list(self.memory),self.batch_size)
        y_stack=np.zeros((self.batch_size,self.num_actions))
        x_stack=np.zeros((self.batch_size,self.num_state))
        cnt=0
        for state,action,reward,done,next_state in miniBatch:
            if done:
                y=action#reward
            else:
                y_pred=self.model.predict(next_state)[0]
                y= reward + self.gamma * np.amax(y_pred)
            y_f = self.model.predict(state)
            #print("action=",action)
            #print("y_f=",y_f)
            #y_f[0][action] = y
            y_stack[cnt,:]=np.reshape(y_f,[1,self.num_actions])
            x_stack[cnt,:]=np.reshape(state,[1,self.num_state])
            cnt+=1
        self.model.fit(x_stack, y_stack, batch_size=self.batch_size,epochs=100, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def train(self):
        #For training
        scores = []
        print("Start training")
        self.lr=self.get_learning_rate(0)
        self.epsilon=self.get_explore_rate(0)

        #==================
        #  Starting episode
        #==================
        for episode in range(self.total_tr_epi):
            state = np.reshape(self.env.reset(),[1,self.num_state])
            done=False
            tick_cnt=0
            #=========================================================================
            #  Steps until either terminal conditions meet or max steps for training
            #=========================================================================
            for step in range(self.max_steps_tr):
                #To choose action
                action = self.get_action(state)
                #print("action in mav_ctrl=",action)
                #Once action selected, iterating
                next_state,reward,done,_ = self.env.step(action)
                reward = reward if not done else -10
                #print("new_state.shape=",new_state.shape)
                next_state=np.reshape(next_state,[1,self.num_state])
                self.history(state,action,reward,done,next_state)
                state=next_state
                if done: #in case an episode finished with trj_err > self.tracking_error
                    #print("episode: {}/{}, score: {}".format(episode, self.total_tr_epi, step))
                    break
                if len(self.memory) > self.batch_size:
                    self.train_with_history()
            
            scores.append(tick_cnt)
            mean_score = np.mean(scores)
            if episode%1000==0:
                print("Current episode={} and mean score = {}".format(episode,mean_score))
            if episode % self.max_steps_tr == 0:
                print('[Episode {}] - Mean survival time over last {} episodes was {} ticks.'.format(episode, self.max_steps_tr,mean_score))
            scores.clear()

        if self.save_model:
            model_name='Q_iter'+str(self.total_tr_epi)+"_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())+'.h5'
            #self.save_Q(model_name)
            self.save(model_name)
            print("model saved as ",model_name)

    def get_explore_rate(self,t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t+1)/25)))

    def get_learning_rate(self,t):
        return max(self.min_lr, min(0.5, 1.0 - math.log10((t+1)/25)))
    
    def get_action(self,state):
        #print("get_action called")
        # Select a random action in case of exploration
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
        # using the trained model to predict best action (exploitation)
            predic_reward = self.model.predict(state)
            #action = np.argmax(predic_reward[0])
            action = predic_reward#np.reshape(predic_reward,[1,self.num_actions])
        return np.squeeze(action)

    def test(self):
        #For testing
        print("Start testing")
        self.env.reset()
        rewards=[] #store a reward for every single episod

        for episode in range(self.total_test_epi):
            state = self.env.reset()
            state = np.reshape(state,[1,self.num_state])
            #state=self.discretize(state)
            cumu_rewards=0
            done=False
            print("=======Episode======")
            for step in range(self.max_steps_test):
                predic_reward = self.model.predict(state)
                #print("predic_reward=",predic_reward)
                action=np.argmax(predic_reward[0])
                self.env.render()
                new_state,reward,done,info = self.env.step(action)
                cumu_rewards +=reward
                state = new_state
                state = np.reshape(state,[1,self.num_state])
                if done == True:
                    rewards.append(cumu_rewards)
                    print ("Score :{}".format(cumu_rewards))
                    #break
        self.env.close()
        print ("Score over time: {}".format(sum(rewards)/self.total_test_epi))

    def load(self, name):
        self.model.load_weights(name)
        print("load mode")

    def save(self, name):
        self.model.save_weights(name)
        print("save model")


# class Environments(unittest.TestCase):
# 	def test_env(self):
# 		env = gym.make('reinmav-v0')
		#env.reset()
		#for i, _ in enumerate(dict_of_list[key]):
		# start_t=timer()
		# for i,_ in enumerate(range(400)): #dt=0.01, 400*0.01=4s
		# # 	#env.render()
		# 	#print("============step {}============".format(i))
		# 	#print("step=",i)
		#  	#env.step(env.action_space.sample()) # take a random action
		# 	env.step() # take a random action
		# end_t=timer()
		# print("simulation time=",end_t-start_t)
		# env.plot_state()
		# env.train()
if __name__ == "__main__":
	myReinforce=Reinforce()
	myReinforce.train()
	# arg_length = len(sys.argv)
    # print ("Python template with # args={}".format(arg_length))    
    # if arg_length != 2:
    #     print("usage: python ./cart_ctrl.py train")
    # else:
    #     myReinforce=Reinforce();
    #     if sys.argv[1]=="train":
    #         myReinforce.train()
    #     else:
    #         myReinforce.load('Q_iter1000_20181011T194259.h5')
    #         myReinforce.test()
	#env=Environments()
	#env.test_env()