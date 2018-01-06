#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 16:07:55 2017

@author: me
"""

import argparse

def params_to_filename_specifier(args):
    filename_specifier = "__activ_func_"+args.activation_function+"__"+\
                        "layers_"+str(args.layers)+"__"+\
                        "gamma_"+str(args.gamma)+"__"+\
                        "ebl_"+str(args.experience_buffer_length)+"__"+\
                        "ebbs_"+str(args.experience_buffer_batch_size)+"__"+\
                        "bias_"+str(not args.without_bias_term)+"__"+\
                        "optimizer_"+args.optimizer+"__"+\
                        "olr_"+str(args.optimizer_learning_rate)+"__"+\
                        "cpy_prd_"+str(args.optimizer)+"__"+\
                        "trn_epchs_"+str(args.training_epochs)+"__"+\
                        "min_eps_"+str(args.min_epsilon)+"__"+\
                        "scaler_"+args.scaler
    return filename_specifier

parser = argparse.ArgumentParser('DQN parameters')
parser.add_argument("--activation-function", "--activation_function", type=str, 
                    help="activation function used in NN layers (default: tanh)", 
                    choices=["relu","tanh"], default="tanh")
parser.add_argument("--layers", type=int, nargs="+", 
                    help="list of hidden layers in NN with their node counts (default: 200, 200)", default=[200,200])
parser.add_argument("--gamma", type=float, help="gamma value", default=0.99)
parser.add_argument("--experience-buffer-length", "--experience_buffer_length", type=int, 
                    help="the length of the experience buffer (default: 200)", default=200)
parser.add_argument("--experience-buffer-batch-size", "--experience_buffer_batch_size", type=int, 
                    help="the number of random samples taken from the experience buffer for training (default: 32)",
                    default=32)
parser.add_argument("--without-bias-term", "--without_bias_term", action="store_true", 
                    help="if a bias term should be added to the NN layers (default: false)", default=False)
parser.add_argument("--optimizer", type=str, help="optimizer used for the NN training (default: adam)",
                    choices=["adam", "rmsprop"], default="adam")
parser.add_argument("--optimizer-learning-rate", "--optimizer_learning_rate", type=float,
                    help="the learning rate of the optimizer (default: 0.001)", default=0.001)
parser.add_argument("--copy-period", "--copy_period", type=int, help="how often the NN weights are copied (default: 50)",
                    default=50)
parser.add_argument("--training-epochs", "--training_epochs", type=int, 
                    help="number of training epochs for each training step (default: 1)", default=1)
parser.add_argument("--min-epsilon", "--min_epsilon", type=float, help="minimum epsilon value (default: 0.01)", default=0.01)
parser.add_argument("--scaler", type=str, help="scaler samples source (default: play)", choices=["play", "random"], default="play")

args = parser.parse_args()

print "Learning CartPole-v0 with the following parameters"
print "--------------------------------------------------"
print "Activation function: "+args.activation_function
print "Neural network layers: "+str(args.layers)
print "Gamma value: "+str(args.gamma)
print "Experience buffer length: "+str(args.experience_buffer_length)
print "Experience buffer batch size: "+str(args.experience_buffer_batch_size)
print "With bias term: "+str(not args.without_bias_term)
print "Optimizer: "+args.optimizer
print "Optimizer learning rate: "+str(args.optimizer_learning_rate)
print "Copy period: "+str(args.copy_period)
print "Training epochs: "+str(args.training_epochs)
print "Minimum epsilon: "+str(args.min_epsilon)
print "Scaler: "+args.scaler
print

    
filename_specifier = params_to_filename_specifier(args)
    
import numpy as np
import gym

from libdqn import DQN

# Experiences buffer length
env = gym.make('CartPole-v0')
max_number_of_episodes = 10000
# MDP variables
epsilonDecay = 1.0
dqn = DQN(env, args.gamma, args.experience_buffer_length, args.experience_buffer_batch_size, 
          args.activation_function, args.layers, not args.without_bias_term, args.optimizer,
          args.optimizer_learning_rate, args.training_epochs, args.scaler)

last_100_total_rewards = np.zeros(100)
loss_averages = []
loss_variances = []
rewards_100 = []
i = 0
solved = False
while i<max_number_of_episodes and not solved: 
    s = env.reset()
    
    epsilonDecay+=0.01
    epsilon = 1.0/epsilonDecay
    if epsilon<args.min_epsilon:
        epsilon = args.min_epsilon
    #epsilon = 1.0/np.sqrt(i+1)
    done = False
    total_reward = 0
    episode_losses = []

    while not done:
        action = dqn.pick_action(env, s, epsilon)
        s_prime, r, done, info = env.step(action)
        total_reward += r
        #if done:
        #    r = -200
        dqn.add_experience(s, action, r, s_prime, done)
        loss = dqn.train()
        episode_losses.append(loss)
        s = s_prime
        
        if i%args.copy_period==0:
            dqn.copy_regressor()
            
    episode_losses = np.asarray(episode_losses)
    loss_variance = np.var(episode_losses)
    loss_mean = np.average(episode_losses)
            
    last_100_total_rewards = np.roll(last_100_total_rewards, 1)
    last_100_total_rewards[0] = total_reward
    
    average = np.average(last_100_total_rewards)
    
    print "Episode: "+str(i)+"\teps: "+str('%0.4f' % epsilon)+\
        "\ttotal reward: "+str(total_reward)+\
        "\tloss variance: "+str('%0.2f' % loss_variance)+\
        "\taverage loss: "+str('%0.2f' % loss_mean)+\
        "\taverage(last 100): " +str(average)
    if average>=195:
        print "Solved after "+str(i)+" episodes, with last 100 episodes average total reward "+\
            str(average)
        solved = True
    i+=1
    
    loss_averages.append(loss_mean)
    loss_variances.append(loss_variance)
    rewards_100.append(average)
    
losses_filename = "losses_"+filename_specifier
variances_filename = "loss_var_"+filename_specifier
rewards_100_filename = "rewards_100"+filename_specifier

np.save(losses_filename, loss_averages)
np.save(variances_filename, loss_averages)
np.save(rewards_100_filename, rewards_100)

print "-------------- END ----------- "
print 
print
