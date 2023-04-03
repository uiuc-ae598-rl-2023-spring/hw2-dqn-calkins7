import numpy as np
from dqn import DQN, DQN_avg
from discreteaction_pendulum import Pendulum
from plotters import *
import pickle
import os
from helpers import *


def main():
    """Apply DQN to the discrete action pendulum"""

    runTag = 'hw_final_'
    tag_train_Q = False
    tag_plot_results = False
    tag_ablation_study_run = True
    tag_ablation_study_plot = True

    ##  PARAMETERS
    replay_size = 100000
    learning_rate = 0.00025
    gamma = 0.95
    eps_start = 1
    eps_end = 0.1
    NN_dim = 64
    num_episodes = 100
    target_update_rate = 10  # Number of episodes to update target weights after
    eps_decay = num_episodes*100  # decay over the whole simulation
    batch_size = 32
    init_replay_size = 500  # number of steps to fill replay buffer before training
    target_Q_on = True
    numRuns = 5
    if numRuns > 1:
        avgTag = True
    else:
        avgTag = False

    if tag_train_Q:
        print("training Q...")
        # Initialize environment
        env = Pendulum()
        # Run lse
        log, trained_Q = DQN_avg(numRuns, replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
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
        plot_learning_curve(log, num_episodes, filename='my_figures/'+runTag+'_learning_curve.png', avgTag=avgTag)
        print('learning curve saved. (close figure to continue)')

        # Plot example trajectory for one trained agent
        plot_trajectory(env, trained_Q, filename='my_figures/'+runTag+'_trained_trajectory.png', avgTag=avgTag)
        print('trajectory saved. (close figure to continue)')

        # plot example trajectory gif for one trained agent
        env.video_NN(trained_Q, filename='my_figures/'+runTag+'_trained_pendulum.gif', avgTag=avgTag)
        print('animation saved.')

        # plot trained policy
        plot_policy(env, trained_Q, num_theta=100, num_theta_dot=100,
                    filename='my_figures/'+runTag+'_trained_policy.png', avgTag=avgTag)
        print('policy saved. (close figure to continue)')

        # plot state value function for one trained agent
        num_theta = 100
        num_theta_dot = 100
        # V = get_state_value_function_TD0(env, trained_Q, num_theta, num_theta_dot, alpha=0.1, gamma=0.95, num_episodes=1000)
        V = get_state_value_function(env, trained_Q, num_theta, num_theta_dot, avgTag=avgTag)
        plot_state_value_function(V, num_theta, num_theta_dot, env, filename='my_figures/'+runTag+'_value_function.png')
        print('value function saved. (close figure to continue)')

    # Ablation study
    if tag_ablation_study_run:
        numRuns = 4
        # Initialize environment
        env = Pendulum()

        # With replay, with target Q (DQN)
        wrwt = DQN_avg(numRuns, replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
                             NN_dim, num_episodes, env, batch_size, target_Q_on=True, init_replay_size=init_replay_size)
        with open(os.path.join('data', runTag + 'wrwt.pickle'), 'wb') as handle:
            pickle.dump(wrwt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # With replay, without target Q (DQN with fixed Q)
        wrnt = DQN_avg(numRuns, replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
                             NN_dim, num_episodes, env, batch_size, target_Q_on=False, init_replay_size=init_replay_size)
        with open(os.path.join('data', runTag + 'wrnt.pickle'), 'wb') as handle:
            pickle.dump(wrnt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Without replay, with target Q (DQN with fixed replay buffer)
        replay_size = 1
        init_replay_size = 1
        batch_size = 1
        nrwt = DQN_avg(numRuns, replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
                             NN_dim, num_episodes, env, batch_size, target_Q_on=True, init_replay_size=init_replay_size)
        with open(os.path.join('data', runTag + 'nrwt.pickle'), 'wb') as handle:
            pickle.dump(nrwt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Without replay, without target Q (DQN with fixed replay buffer and fixed Q)
        replay_size = 1
        init_replay_size = 1
        batch_size = 1
        nrnt= DQN_avg(numRuns, replay_size, learning_rate, gamma, eps_start, eps_end, eps_decay, target_update_rate,
                             NN_dim, num_episodes, env, batch_size, target_Q_on=False, init_replay_size=init_replay_size)
        with open(os.path.join('data', runTag + 'nrnt.pickle'), 'wb') as handle:
            pickle.dump(nrnt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if tag_ablation_study_plot:
        # Load data (deserialize)
        with open(os.path.join('data', runTag + 'wrwt' + '.pickle'), 'rb') as handle:
            wrwt = pickle.load(handle)
        with open(os.path.join('data', runTag + 'wrnt' + '.pickle'), 'rb') as handle:
            wrnt = pickle.load(handle)
        with open(os.path.join('data', runTag + 'nrwt' + '.pickle'), 'rb') as handle:
            nrwt = pickle.load(handle)
        with open(os.path.join('data', runTag + 'nrnt' + '.pickle'), 'rb') as handle:
            nrnt = pickle.load(handle)
        # Plot learning curves
        plot_ablation_study(wrwt, wrnt, nrwt, nrnt, filename='my_figures/'+runTag+'_ablation_study.png')


if __name__ == '__main__':
    main()