import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from helpers import *


def plot_learning_curve(log, num_episodes, filename, avgTag):
    """Plot learning curve (return versus number of episodes)"""
    if not avgTag:
        # Extract data from log
        returns = log['return']
        episodes = range(num_episodes)

        # get rolling average
        ret = pd.DataFrame(returns)
        ret_avg = ret.rolling(window=10).mean()
        ret_std = ret.rolling(window=10).std()
    else:
        # Extract data from log
        rets = []
        ret_std = []
        returns = []
        episodes = range(num_episodes)

        for ii, case in enumerate(log):
            rets.append(np.array(case))
            returns.append(case)
        ret_avg = pd.DataFrame(np.mean(rets, axis=0))
        ret_std = pd.DataFrame(np.std(rets, axis=0))

    # Plot learning curve
    plt.figure()
    for ret in returns:
        plt.scatter(episodes, ret, s=5, label='_nolegend_')
    plt.plot(ret_avg.index, ret_avg, 'r-', label='Running average')
    std = 1
    plt.fill_between(ret_avg.index, (ret_avg - std*ret_std).to_numpy()[:,0], (ret_avg + std*ret_std).to_numpy()[:,0], color='r', alpha=0.2, label=str(std)+' std')
    plt.xlabel('Episode Number')
    plt.ylabel('Return')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    # plt.show()

    return plt


def plot_trajectory(env, trained_Q, filename, avgTag):
    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [],
        'thetadot': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        s = torch.Tensor.float(torch.from_numpy(s))
        if not avgTag:
            a = trained_Q(s).argmax().numpy()
        else:
            a = avgDQNsa(trained_Q, s)
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(s[0])
        log['thetadot'].append(s[1])

    # Clip theta
    thetas = []
    for theta in log['theta']:
        thetas.append(((theta + np.pi) % (2 * np.pi)) - np.pi)

    # Plot data and save to png file
    fig = plt.figure(figsize=(10, 10))

    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    ax1.plot(log['t'], log['s'])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('State')
    ax2.plot(log['t'][:-1], log['a'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Action')
    ax3.plot(log['t'][:-1], log['r'])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Reward')

    ax4.plot(log['t'][1:], thetas)
    ax4.plot(log['t'][1:], log['thetadot'])
    ax4.plot(log['t'], np.ones(np.shape(log['t'])) * 0.1 * np.pi, 'g--')
    ax4.plot(log['t'], np.ones(np.shape(log['t'])) * -0.1 * np.pi, 'g--')
    ax4.legend(['θ', '$\dot{θ}$', 'Good Theta Zone'])
    ax4.set_xlabel('Time')
    ax4.set_ylabel('θ \ $\dot{θ}$ [rad]')

    plt.subplots_adjust(wspace=0.3, left=0.07, right=0.97, top=0.99)
    plt.savefig(filename)
    # plt.show()
    return


def plot_state_value_function(V, num_theta, num_theta_dot, env, filename='my_figures/value_function.png'):
    """
    Plot state-value function
    :param discrete_states:
    :param V:
    :param filename:
    :return:
    """
    thetas = np.linspace(-np.pi, np.pi, num_theta)
    thetadots = np.linspace(-env.max_thetadot, env.max_thetadot, num_theta_dot)

    [X, Y] = np.meshgrid(thetadots, thetas)
    Z = V.reshape(X.shape)

    fig = plt.figure()
    ax = plt.gca()
    p = ax.contourf(Y, X, Z, cmap='jet')
    ax.set_ylabel('θ')
    ax.set_xlabel('$\dot{θ}$')
    ax.set_title('State-Value Function')
    fig.colorbar(p, shrink=0.5, aspect=5)
    plt.savefig(filename)
    # plt.show()
    return


def plot_policy(env, trained_Q, num_theta, num_theta_dot, filename, avgTag):
    """
    Plot policy from trained Q network
    :param trained_Q:
    :param num_theta:
    :param num_theta_dot:
    :param filename:
    :param avgTag:
    :return:
    """
    thetas = np.linspace(-np.pi, np.pi, num_theta)
    thetadots = np.linspace(-env.max_thetadot, env.max_thetadot, num_theta_dot)

    [X, Y] = np.meshgrid(thetadots, thetas)
    Z = np.zeros(X.shape)
    for ii, t in enumerate(thetas):
        for jj, tdot in enumerate(thetadots):
            s = np.array([t, tdot])
            if not avgTag:
                a = trained_Q(torch.Tensor.float(torch.from_numpy(s))).mean(0).argmax().numpy()
            else:
                a = avgDQNsa(trained_Q, torch.Tensor.float(torch.from_numpy(s)))
            Z[ii, jj] = a

    fig = plt.figure()
    ax = plt.gca()
    # make a countour plot of the Z
    p = ax.contourf(Y, X, Z, cmap='jet')
    ax.set_ylabel('θ')
    ax.set_xlabel('$\dot{θ}$')
    ax.set_title('Optimal Policy')
    fig.colorbar(p, shrink=0.5, aspect=5)
    plt.savefig(filename)
    # plt.show()
    return


def plot_ablation_study(wrwt, wrnt, nrwt, nrnt,filename):
    """
    Plot ablation study results
    :param wrwt: With Replay, With Target
    :param wrnt: With Replay, No Target
    :param nrwt: No Replay, With Target
    :param nrnt: No Replay, No Target
    :param filename: filename to save plot
    :return:
    """

    legend = ["DQN", "No Target", "No Replay", "No Target, No Replay"]

    # Get mean of the datas passed in
    mean_data = []
    std_data = []
    for ii, case in enumerate([wrwt, wrnt, nrwt, nrnt]):
        mean_data.append(np.mean(np.array(case), axis=0))
        std_data.append(np.std(np.array(case), axis=0))

    fig = plt.figure()
    ax = plt.gca()
    for ii in range(len(mean_data)):
        ax.plot(mean_data[ii], label=legend[ii])
        ax.fill_between(np.arange(len(mean_data[ii])), mean_data[ii] - std_data[ii], mean_data[ii] + std_data[ii], alpha=0.2)
    ax.legend()
    ax.set_ylabel('Average Episode Score')
    ax.set_title('Ablation Study')
    plt.savefig(filename)
    plt.show()
    return
