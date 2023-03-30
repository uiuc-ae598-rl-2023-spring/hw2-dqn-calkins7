import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch


def plot_learning_curve(log, num_episodes, filename=None):
    """Plot learning curve (return versus number of episodes)"""
    # Extract data from log
    returns = log['return']
    episodes = range(num_episodes)

    # get rolling average
    ret = pd.DataFrame(returns)
    ret_avg = ret.rolling(window=10).mean()
    ret_std = ret.rolling(window=10).std()

    # Plot learning curve
    plt.figure()
    plt.scatter(episodes, returns, s=3, alpha=0.5, label='Return')
    plt.plot(ret_avg.index, ret_avg, 'r-', label='Running average')
    std = 1
    plt.fill_between(range(num_episodes), (ret_avg - std*ret_std).to_numpy()[:,0], (ret_avg + std*ret_std).to_numpy()[:,0], color='r', alpha=0.2, label=str(std)+' std')
    plt.xlabel('Episode Number')
    plt.ylabel('Return')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

    return plt


def plot_trajectory(env, trained_Q, filename=None):
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
        a = trained_Q(s).argmax().numpy()
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
    plt.show()
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

    [X, Y] = np.meshgrid(thetas, thetadots)
    Z = V.reshape(X.shape)

    fig = plt.figure()
    ax = plt.gca()
    p = ax.contourf(Y, X, Z, cmap='jet')
    ax.set_ylabel('θ')
    ax.set_xlabel('$\dot{θ}$')
    ax.set_title('State-Value Function')
    fig.colorbar(p, shrink=0.5, aspect=5)
    plt.savefig(filename)
    plt.show()

    return


def plot_policy(env, trained_Q, num_theta, num_theta_dot, filename):
    """
    Plot policy from trained Q network
    :param trained_Q:
    :param num_theta:
    :param num_theta_dot:
    :param filename:
    :return:
    """
    thetas = np.linspace(-np.pi, np.pi, num_theta)
    thetadots = np.linspace(-env.max_thetadot, env.max_thetadot, num_theta_dot)

    [X, Y] = np.meshgrid(thetas, thetadots)
    Z = np.zeros(X.shape)
    for ii, t in enumerate(thetas):
        for jj, tdot in enumerate(thetadots):
            s = np.array([t, tdot])
            a = trained_Q(torch.Tensor.float(torch.from_numpy(s))).argmax().numpy()
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
    plt.show()
    return


def plot_ablation_study(case1, case2, case3, case4, filename):
    """
    Plot barchart of best episode score from each ablation study
    :param case1:
    :param case2:
    :param case3:
    :param case4:
    :param filename:
    :return:
    """

    xlabels = ["With replay,\nwith target Q", "With replay,\nwithout target Q", "Without replay,\nwith target Q", "Without replay,\nwithout target Q"]

    fig = plt.figure()
    ax = plt.gca()
    ax.bar(xlabels, [case1, case2, case3, case4])
    ax.set_ylabel('Best Episode Score')
    ax.set_title('Ablation Study')
    plt.savefig(filename)
    plt.show()
    return
