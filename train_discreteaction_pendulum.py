import numpy as np
from dqn import DQN
from discreteaction_pendulum import Pendulum
from plotters import *
import pickle
import os
from get_state_value_function import *


def main():
    """Apply DQN to the discrete action pendulum"""

    runTag = 'test_rahil_'
    tag_train_Q = False
    tag_plot_results = True
    tag_ablation_study = False

    ##  PARAMETERS
    replay_size = 100000
    learning_rate = 0.00025
    gamma = 0.95
    eps_start = 1
    eps_end = 0.1
    NN_dim = 64
    num_episodes = 150
    target_update_rate = 10  # Number of episodes to update target weights after
    eps_decay = num_episodes*100  # decay over the whole simulation
    batch_size = 32
    init_replay_size = 500  # number of steps to fill replay buffer before training
    target_Q_on = True

    if tag_train_Q:
        print("training Q...")
        # Initialize environment
        env = Pendulum()
        # Run DQN
        log, trained_Q = DQN(replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
                             NN_dim, num_episodes, env, batch_size, target_Q_on, init_replay_size)
        print('...Training complete.')

        # Save trained DQN, log, and env
        # Store data (serialize)
        with open(os.path.join('data', runTag + 'dqn.pickle'), 'wb') as handle:
            pickle.dump(trained_Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join('data', runTag + 'log.pickle'), 'wb') as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join('data', runTag + 'env.pickle'), 'wb') as handle:
            pickle.dump(env, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('data saved.')

    if tag_plot_results:
        print("plotting results...")
        # Load data (deserialize)
        with open(os.path.join('data', runTag + 'dqn.pickle'), 'rb') as handle:
            trained_Q = pickle.load(handle)
        with open(os.path.join('data', runTag + 'log.pickle'), 'rb') as handle:
            log = pickle.load(handle)
        with open(os.path.join('data', runTag + 'env.pickle'), 'rb') as handle:
            env = pickle.load(handle)
        print('data loaded.')

        # Plot learning curve (return versus number of episodes)
        # plot_learning_curve(log, num_episodes, filename='my_figures/'+runTag+'_learning_curve.png')
        # print('learning curve saved. (close figure to continue)')

        # Plot example trajectory for one trained agent
        # plot_trajectory(env, trained_Q, filename='my_figures/'+runTag+'_trained_trajectory.png')
        # print('trajectory saved. (close figure to continue)')

        # plot example trajectory gif for one trained agent
        # env.video_NN(trained_Q, filename='my_figures/'+runTag+'_trained_pendulum.gif')
        # print('animation saved.')

        # plot trained policy
        plot_policy(env, trained_Q, num_theta=100, num_theta_dot=100, filename='my_figures/'+runTag+'_trained_policy.png')
        print('policy saved. (close figure to continue)')

        # plot state value function for one trained agent
        num_theta = 100
        num_theta_dot = 100
        # V = get_state_value_function_TD0(env, trained_Q, num_theta, num_theta_dot, alpha=0.1, gamma=0.95, num_episodes=1000)
        V = get_state_value_function(env, trained_Q, num_theta, num_theta_dot)
        plot_state_value_function(V, num_theta, num_theta_dot, env, filename='my_figures/'+runTag+'_value_function.png')
        print('value function saved. (close figure to continue)')

    # Ablation study
    if tag_ablation_study:
        print('running ablation study... (this may take a while)')
        # Load data (deserialize)
        with open(os.path.join('data', runTag + 'dqn.pickle'), 'rb') as handle:
            trained_Q = pickle.load(handle)
        with open(os.path.join('data', runTag + 'log.pickle'), 'rb') as handle:
            log = pickle.load(handle)
        with open(os.path.join('data', runTag + 'env.pickle'), 'rb') as handle:
            env = pickle.load(handle)
        print('data loaded.')

        # With replay, with target Q (standard DQN)
        case1 = max(log['return'])
        # With replay, without target Q (DQN with fixed Q)
        log, trained_Q = DQN(replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
                             NN_dim, num_episodes, env, batch_size, target_Q_on=False)
        case2 = max(log['return'])
        # Without replay, with target Q (DQN with fixed replay buffer)
        replay_size = 1
        batch_size = 1
        log, trained_Q = DQN(replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
                             NN_dim, num_episodes, env, batch_size, target_Q_on=True)
        case3 = max(log['return'])
        # set replay_size to 1 to turn off replay
        # Without replay, without target Q (DQN with fixed replay buffer and fixed Q)
        replay_size = 1
        batch_size = 1
        log, trained_Q = DQN(replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
                             NN_dim, num_episodes, env, batch_size, target_Q_on=False)
        case4 = max(log['return'])
        plot_ablation_study(case1, case2, case3, case4, filename='my_figures/'+runTag+'_ablation_study.png')


if __name__ == '__main__':
    main()