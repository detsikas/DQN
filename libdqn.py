#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 11:46:00 2017

@author: me
"""

import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import StandardScaler

class DQN:
    def __init__(self, env, gamma, 
                 experience_buffer_length, batch_size, 
                 activation_function, layers, bias_term, 
                 optimizer, optimizer_learning_rate, training_epochs):
        self.experience_buffer_length = experience_buffer_length
        self.action_space_size = env.action_space.n
        self.state_space_size = env.observation_space.shape[0]
        self.total_experience = 0
        self.batch_size = batch_size
        self.env = env
        self.gamma = gamma
        self.training_epochs = training_epochs

        # Experience buffer holes SARS'
        # Its size is state_space_size for s and s' respectively, and 1 position for A and R respectively
        # Experience format: 
        # | state vector | action | reward | state vector | done
        self.experience_size = self.state_space_size+1+1+self.state_space_size+1
        self.experience_buffer = np.zeros((self.experience_buffer_length, self.experience_size))
        
        self.regressor = self.build_regressor(activation_function, layers, bias_term, 
                                              optimizer, optimizer_learning_rate)
        self.target_regressor = self.build_regressor(activation_function, layers, bias_term,
                                                     optimizer, optimizer_learning_rate)
        
        self.sc = self.build_scaler()
        
    def build_scaler(self):
        sc = StandardScaler()
        '''
        observation_samples = np.array([]).reshape(0, self.state_space_size)
        for i in range(1000):
            s = self.env.reset()
            done = False
            max_moves = 20
            
            while not done and max_moves>0:
                s, r, done, info = self.env.step(self.env.action_space.sample())
                observation_samples = np.concatenate((observation_samples, s.reshape((1, self.state_space_size))))
                max_moves-=1
        '''
        observation_samples = np.array([self.env.observation_space.sample() for x in range(40000)])
        sc.fit(observation_samples)
        return sc
        
    def build_regressor(self, activation_function, layers, bias_term, optimizer, optimizer_learning_rate):
        # Build regressor
        regressor = Sequential()
        # Adding the input layer and the first hidden layer with dropout
        #classifier.add(Dense(6, input_shape=(1,11), kernel_initializer='glorot_uniform', activation='relu'))
        regressor.add(Dense(layers[0], input_shape=(self.state_space_size,), 
                            activation=activation_function, use_bias=bias_term))
        
        for l in layers[1:]:
            regressor.add(Dense(l, activation=activation_function, use_bias=bias_term))
    
        # Adding the output layer
        regressor.add(Dense(self.action_space_size, use_bias=bias_term))
        
        # Compiling the ANN
        if optimizer=="adam":
            regressor.compile(optimizer=Adam(lr=0.001), loss = 'mse', metrics=['mse'])
        else:
            regressor.compile(optimizer=RMSprop(lr=0.001), loss = 'mse', metrics=['mse'])
            
        return regressor
    
    def transform_state(self, s):
        return self.sc.transform(np.atleast_2d(s))
        
    def add_experience(self, s, a, r, s_prime, done):
        self.experience_buffer = np.roll(self.experience_buffer, 1, 0)
        experience_vector = self.transform_state(s)
        experience_vector = np.append(experience_vector, a)
        experience_vector = np.append(experience_vector, r)
        experience_vector = np.append(experience_vector, self.transform_state(s_prime))
        experience_vector = np.append(experience_vector, 1 if done else 0)
        
        self.experience_buffer[0,:] = experience_vector
        self.total_experience += 1
        
    def pick_action(self, env, s, epsilon):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.regressor.predict(self.transform_state(s))[0])
        
    def train(self):
        if self.total_experience<self.experience_buffer_length:
            return 0
        
        # Select a random batch from the experience
        rows = np.random.choice(self.experience_buffer.shape[0], size = self.batch_size, replace = False)
        random_experience_batch = self.experience_buffer[rows]
        
        # Slice the input data (state)
        X = random_experience_batch[:, :self.state_space_size]
        
        # Slice the pieces of information that will be used to calculate the output data
        actions = random_experience_batch[:, self.state_space_size].astype(int)
        rewards = random_experience_batch[:, self.state_space_size+1]
        s_primes = random_experience_batch[:, self.state_space_size+2:-1]
        dones = random_experience_batch[:, -1].astype(int)
        
        # Predict the Q values at each state for all actions ...
        Y = self.target_regressor.predict(X)

        # ... but for the actions taken the values will be updatad from the Bellman equations,
        # if the game did not end with this action
        # Do we really need to do that? Normally yes, as we do not want the value function of a terminal state
        target_rewards = rewards+self.gamma*np.amax(self.target_regressor.predict(s_primes), axis=1)
        
        # if it did we just use the earned reward
        target_rewards[np.nonzero(dones)] = rewards[np.nonzero(dones)]
        
        # Update the output values for the actions taken
        Y[np.arange(self.batch_size), actions] = target_rewards
        
        # Fit the model
        history = self.regressor.fit(X, Y, batch_size = self.batch_size, epochs = self.training_epochs, verbose = 0)
        loss = history.history['loss'][0]
        
        return loss
    
    def copy_regressor(self):
        self.target_regressor.set_weights(self.regressor.get_weights())
