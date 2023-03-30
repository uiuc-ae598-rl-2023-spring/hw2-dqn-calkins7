import numpy as np
import math
from collections import namedtuple, deque
from itertools import count
import random
from epsilon_greedy import epsilon_greedy, random_action

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN_NN(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dim):
        """
        Initializes neural network
        :param num_obs: dimension of input
        :param output_dim: dimension of output
        :param hidden_dim: dimension of hidden layers
        :param num_hidden_layers: number of hidden layers
        """
        super(DQN_NN, self).__init__()
        self.layer1 = nn.Linear(num_obs, hidden_dim)  # num_obsx64
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)  # 64x64
        self.layer3 = nn.Linear(hidden_dim, num_actions)  # 64xnum_actions

    def forward(self, x):
        """
        Performs forward pass. Called with either one element to determine next action, or a batch during optimization.
        :param x: input
        :return: output
        """
        y1 = nn.Tanh(self.layer1(x))
        y2 = nn.Tanh(self.layer2(y1))
        y3 = self.layer3(y2)
        return y3

class DQN_NN(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dim):
        """
        Initializes neural network
        :param num_obs: dimension of input
        :param output_dim: dimension of output
        :param hidden_dim: dimension of hidden layers
        :param num_hidden_layers: number of hidden layers
        """
        super(DQN_NN, self).__init__()
        self.layer1 = nn.Linear(num_obs, hidden_dim)  # num_obsx64
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)  # 64x64
        self.layer3 = nn.Linear(hidden_dim, num_actions)  # 64xnum_actions

    def forward(self, x):
        """
        Performs forward pass. Called with either one element to determine next action, or a batch during optimization.
        :param x: input
        :return: output
        """
        activation = nn.Tanh()
        y1 = activation(self.layer1(x))
        y2 = activation(self.layer2(y1))
        y3 = self.layer3(y2)
        return y3


def update_Q_weights(Q, replay_buffer, gamma, Q_target, optimizer, loss_fn, batch_size, target_Q_on):
    """
    Updates the weights of Q at each time step
    :param replay_buffer: ('state', 'action', 'next_state', 'reward')
    :param gamma: discount factor
    :param Q_target: target Q network
    :param optimizer: optimizer object to perform SGD
    :param loss_fn: loss function to compute loss
    :param batch_size: number of experiences to sample from episode replay buffer
    :param target_Q_on: boolean to determine whether to use target Q network
    :return: updates weights of Q NN
    """
    # Sample a batch of transitions from replay buffer
    batch = replay_buffer.sample(batch_size)
    # Convert batch to tensors
    batch = Transition(*zip(*batch))
    # Create tensors
    state_batch = batch.state
    action_batch = batch.action
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = batch.next_state
    # Compute policy Q(s_t, a_t) of the main Q network
    state_action_values = torch.empty(batch_size)
    for ii in range(len(state_batch)):
        state_action_values[ii] = Q(state_batch[ii])[action_batch[ii]]

    # Compute max_a Q_target(s_t+1, a)
    next_state_values = torch.empty(batch_size)
    for ii in range(len(state_batch)):
        if target_Q_on: # use target Q network
            next_state_values[ii] = Q_target(next_state_batch[ii]).detach().max()
        else:  # use main Q network
            next_state_values[ii] = Q(next_state_batch[ii]).max()

    # Compute target of the Bellman equation
    expected_state_action_values = reward_batch + gamma * next_state_values
    # Compute loss
    loss = loss_fn(state_action_values, expected_state_action_values)
    # Optimize the model (update the weights based on SGD)
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients
    torch.nn.utils.clip_grad_value_(Q.parameters(), 1)
    optimizer.step()


def fill_replay_buffer(env, replay_buffer, replay_size):
    """
    Fills replay buffer with random actions before Q learning
    :param env:
    :param replay_buffer:
    :param replay_size:
    :return:
    """
    # Fill up replay buffer
    print('...filling replay buffer...')
    # Initialize state
    state = env.reset()
    # convert to tensor
    state = torch.Tensor.float(torch.from_numpy(state))
    while len(replay_buffer) < replay_size:
        a = random_action(env.num_actions)
        (next_state, reward, done) = env.step(a)
        # Convert to tensors
        next_state = torch.Tensor.float(torch.from_numpy(next_state))
        reward = torch.tensor(reward)
        # Store transition in replay ('state', 'action', 'next_state', 'reward')
        replay_buffer.push(state, a, next_state, reward)
        # Update state
        if not done:
            state = next_state
        else:
            state = env.reset()
            state = torch.Tensor.float(torch.from_numpy(state))
    print('...replay buffer filled with %i values...' % len(replay_buffer))
    return replay_buffer


def DQN(replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate, NN_dim, num_episodes,
        env, batch_size, target_Q_on, init_replay_size):
    """
    Performs deep Q Learning

    :param replay_size: size of experience replay buffer
    :param learning_rate: learning rate
    :param gamma: discount factor
    :param eps: epsilon for epsilon-greedy policy
    :param target_update_rate: target Q network update rate
    :param NN_dim: dimension of hidden layers
    :param NN_num_layers: number of hidden layers
    :param num_episodes: number of episodes to run
    :param env: environment
    :param batch_size: size of minibatch for SGD
    :param target_Q_on: boolean to turn on target Q network
    :param init_replay_size: number of random actions to take before Q learning
    :return log: dictionary of episode returns
    :return Q: trained Q network
    """
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_size)
    # Fill replay buffer
    replay_buffer = fill_replay_buffer(env, replay_buffer, init_replay_size)

    # Initialize Q and target Q with random weights
    Q = DQN_NN(env.num_states, env.num_actions, NN_dim)
    if target_Q_on:
        target_Q = DQN_NN(env.num_states, env.num_actions, NN_dim)
    else:
        target_Q = []
    # Initialize optimizer
    optimizer = optim.RMSprop(Q.parameters(), lr=learning_rate)
    # Initialize loss function
    loss_fn = nn.MSELoss()
    # Number of timesteps for each sim
    num_steps = 0
    # Initialize log
    log = {'return': []}

    # Intialized steps done
    steps_done = 0

    # Run episodes
    for i_episode in range(num_episodes):
        # Initialize episode return
        episode_return = 0
        # Initialize state
        state = env.reset()
        # convert to tensor
        state = torch.Tensor.float(torch.from_numpy(state))
        # Run steps
        done = False
        step_counter = 1
        while not done:
            # Select action
            action = epsilon_greedy(eps_start, eps_end, eps_decay, steps_done, env.num_actions, state, Q)
            steps_done += 1
            # Take action
            (next_state, reward, done) = env.step(action)
            # Convert to tensors
            next_state = torch.Tensor.float(torch.from_numpy(next_state))
            reward = torch.tensor(reward)
            # Store transition in replay ('state', 'action', 'next_state', 'reward')
            replay_buffer.push(state, action, next_state, reward)
            # Update state
            state = next_state
            # Update episode return
            episode_return += gamma**step_counter * reward.numpy()
            # Update number of steps
            num_steps += 1
            # Perform SGD step with respect to network parameters
            # if not (len(replay_buffer) < replay_size):
            update_Q_weights(Q, replay_buffer, gamma, target_Q, optimizer, loss_fn, batch_size, target_Q_on)

            # Update target Q with weights from Q every target_update_rate steps
            if (i_episode % target_update_rate == 0) and (step_counter == 1) and target_Q_on:
                target_Q.load_state_dict(Q.state_dict())
                print('...Updating target Q network')

            # Print episode return
            # if step_counter % 5 == 0:
            #     print('Episode: %d, Step: %d, Return: %f' % (i_episode, step_counter, episode_return))
            step_counter += 1

            # Break if done
            if done:
                break

        # if i_episode % 100 == 0:
        print('### Episode: %d / %d, Return: %f' % (i_episode, num_episodes, episode_return))

        # Update log
        log['return'].append(episode_return)

    return log, Q



