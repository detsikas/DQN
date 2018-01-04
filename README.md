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
