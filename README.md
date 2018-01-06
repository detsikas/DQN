# DQN
Deep Q Network

Highly customizable Deep Q Network for Reinforcement Learning.

The project solves the [CartPole](https://github.com/openai/gym/wiki/CartPole-v0) environemnt using a Deep Q Network. 

Multiple algorithm parameters can be customized so that the code can be tailored to solve other environments or Reinforcement Learning problems.

## Parameters
Here is a list of the customizable parameters of the algorithm:
* Activation function: The activation function used in the Neural Network. Options are tanh, relu and softmax (default: tanh)
* Neural network layers: A list of the number of nodes in each layer of the Neural Network (default: [200,200])
* Gamma value: The gamma value used in the update of the Bellman equations Q values (default: 0.99)
* Experience buffer length: The length of the experience buffer (default: 200)
* Experience buffer batch size: The size of each training batch, picked randomly from the experience buffer (default: 32)
* With bias term: Whether a bias term will be included in the Neural Network layers (default: true)
* Optimizer: The optimizer used in the training of the Neural Network. Options are adam and rmsprop (default: adam)
* Optimizer learning rate: The learning rate of the optimizer (default: 0.001)
* Copy period: After how many episodes the target Neural Network value will be copied over to the Q value approximation Neural Network (default: 50)
* Training epochs: The number of training epochs. Each training epoch uses a different random barch from the experience buffer (default: 1)
* Minimum epsilon: The minimum value of epsilon. Epsilon is decaying in time. (default: 0.01)
* Scaler: The algorithm used for scaling the observation values. Options are play and random. Play plays games and samples the observations values. Random creates random observation samples (default: play)

## Usage
To get help simply type
./dqn_parameterized --help or
./dqn_parameterized -h

To use the default values simply type
./dqn_parameterized

## Sample output
$ ./dqn_parametrized.py 
Learning CartPole-v0 with the following parameters
--------------------------------------------------
Activation function: tanh
Neural network layers: [200, 200]
Gamma value: 0.99
Experience buffer length: 200
Experience buffer batch size: 32
With bias term: True
Optimizer: adam
Optimizer learning rate: 0.001
Copy period: 50
Training epochs: 1
Minimum epsilon: 0.01
Scaler: play

Using TensorFlow backend.
[2018-01-06 17:28:17,151] Making new env: CartPole-v0
2018-01-06 17:28:19.079289: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1
Episode: 0	eps: 0.9901	total reward: 15.0	loss variance: 0.00	average loss: 0.00	average(last 100): 0.15
Episode: 1	eps: 0.9804	total reward: 31.0	loss variance: 0.00	average loss: 0.00	average(last 100): 0.46
Episode: 2	eps: 0.9709	total reward: 15.0	loss variance: 0.00	average loss: 0.00	average(last 100): 0.61
Episode: 3	eps: 0.9615	total reward: 15.0	loss variance: 0.00	average loss: 0.00	average(last 100): 0.76
Episode: 4	eps: 0.9524	total reward: 25.0	loss variance: 0.00	average loss: 0.00	average(last 100): 1.01
Episode: 5	eps: 0.9434	total reward: 9.0	loss variance: 0.00	average loss: 0.00	average(last 100): 1.1
Episode: 6	eps: 0.9346	total reward: 14.0	loss variance: 0.00	average loss: 0.00	average(last 100): 1.24
Episode: 7	eps: 0.9259	total reward: 14.0	loss variance: 0.00	average loss: 0.00	average(last 100): 1.38
Episode: 8	eps: 0.9174	total reward: 16.0	loss variance: 0.00	average loss: 0.00	average(last 100): 1.54
Episode: 9	eps: 0.9091	total reward: 24.0	loss variance: 0.00	average loss: 0.00	average(last 100): 1.78
Episode: 10	eps: 0.9009	total reward: 41.0	loss variance: 0.06	average loss: 0.23	average(last 100): 2.19
Episode: 11	eps: 0.8929	total reward: 9.0	loss variance: 0.00	average loss: 0.34	average(last 100): 2.28
Episode: 12	eps: 0.8850	total reward: 19.0	loss variance: 0.00	average loss: 0.33	average(last 100): 2.47
Episode: 13	eps: 0.8772	total reward: 32.0	loss variance: 0.00	average loss: 0.32	average(last 100): 2.79
Episode: 14	eps: 0.8696	total reward: 11.0	loss variance: 0.00	average loss: 0.32	average(last 100): 2.9
Episode: 15	eps: 0.8621	total reward: 56.0	loss variance: 0.00	average loss: 0.32	average(last 100): 3.46
Episode: 16	eps: 0.8547	total reward: 37.0	loss variance: 0.00	average loss: 0.31	average(last 100): 3.83
Episode: 17	eps: 0.8475	total reward: 20.0	loss variance: 0.00	average loss: 0.30	average(last 100): 4.03
Episode: 18	eps: 0.8403	total reward: 11.0	loss variance: 0.00	average loss: 0.29	average(last 100): 4.14
Episode: 19	eps: 0.8333	total reward: 12.0	loss variance: 0.00	average loss: 0.28	average(last 100): 4.26
Episode: 20	eps: 0.8264	total reward: 19.0	loss variance: 0.00	average loss: 0.28	average(last 100): 4.45
Episode: 21	eps: 0.8197	total reward: 12.0	loss variance: 0.00	average loss: 0.29	average(last 100): 4.57
Episode: 22	eps: 0.8130	total reward: 30.0	loss variance: 0.00	average loss: 0.30	average(last 100): 4.87
...
...
Episode: 772	eps: 0.1145	total reward: 200.0	loss variance: 0.25	average loss: 1.16	average(last 100): 194.48
Episode: 773	eps: 0.1144	total reward: 200.0	loss variance: 3.78	average loss: 2.00	average(last 100): 194.48
Episode: 774	eps: 0.1143	total reward: 200.0	loss variance: 0.69	average loss: 1.85	average(last 100): 194.48
Episode: 775	eps: 0.1142	total reward: 200.0	loss variance: 1934.40	average loss: 19.99	average(last 100): 194.61
Episode: 776	eps: 0.1140	total reward: 200.0	loss variance: 1848.05	average loss: 19.80	average(last 100): 194.61
Episode: 777	eps: 0.1139	total reward: 188.0	loss variance: 1611.46	average loss: 17.17	average(last 100): 194.49
Episode: 778	eps: 0.1138	total reward: 200.0	loss variance: 61.27	average loss: 2.24	average(last 100): 194.53
Episode: 779	eps: 0.1136	total reward: 200.0	loss variance: 0.23	average loss: 1.33	average(last 100): 194.53
Episode: 780	eps: 0.1135	total reward: 200.0	loss variance: 7.96	average loss: 3.22	average(last 100): 194.53
Episode: 781	eps: 0.1134	total reward: 200.0	loss variance: 2429.36	average loss: 22.90	average(last 100): 194.53
Episode: 782	eps: 0.1133	total reward: 200.0	loss variance: 1.29	average loss: 2.29	average(last 100): 194.53
Episode: 783	eps: 0.1131	total reward: 200.0	loss variance: 0.41	average loss: 1.45	average(last 100): 194.53
Episode: 784	eps: 0.1130	total reward: 200.0	loss variance: 4.89	average loss: 2.37	average(last 100): 195.36
Solved after 784 episodes, with last 100 episodes average total reward 195.36
