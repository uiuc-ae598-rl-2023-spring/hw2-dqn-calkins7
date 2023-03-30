import numpy as np
import torch


def get_state_value_function_TD0(env, trained_Q, num_theta, num_theta_dot, alpha, gamma=0.95, num_episodes=1000):
    """
    Gets state value function using TD(0)
    :param env:
    :param trained_Q:
    :return:
    """
    # Get discrete state space representation for plotting
    thetas = np.linspace(-np.pi, np.pi, num_theta)
    thetadots = np.linspace(-env.max_thetadot, env.max_thetadot, num_theta_dot)

    if not (0 < alpha <= 1):
        print("alpha value input to TD(0) invalid")
        return

    # Make a V matrix for given theta, theta_dot
    V = np.zeros((num_theta, num_theta_dot))

    # For each episode loop
    for ii in range(num_episodes):
        # Initialize simulation
        s = env.reset()

        # Simulate until episode is done
        done = False
        while not done:
            # choose action according to pi
            a = trained_Q(torch.Tensor.float(torch.from_numpy(s))).argmax().numpy()
            (s, r, done) = env.step(a)
            # take action, observe sprime and r
            (s1, r, done) = env.step(a)
            # Figure out where s and s1 are within the discrete state space
            [t, tdot] = s
            [t1, tdot1] = s1
            t_ind = next(x for x, val in enumerate(thetas) if val > t)
            t1_ind = next(x for x, val in enumerate(thetas) if val > t1)
            tdot_ind = next(x for x, val in enumerate(thetadots) if val > tdot)
            tdot1_ind = next(x for x, val in enumerate(thetadots) if val > tdot1)

            V[t_ind][tdot_ind] = V[t_ind][tdot_ind] + alpha * (r + gamma * V[t1_ind][tdot1_ind] - V[t_ind][tdot_ind])
            s = s1

        # Print if ii is a factor of 100
        if (ii % 100) == 0:
            print(f'TD(0) Episode {ii} / {num_episodes}')

    return V

def get_state_value_function(env, trained_Q, num_theta, num_theta_dot):
    """
    Gets state value function using max
    :param env:
    :param trained_Q:
    :param num_theta:
    :param num_theta_dot:
    :param alpha:
    :param gamma:
    :param num_episodes:
    :return:
    """
    # Get discrete state space representation for plotting
    thetas = np.linspace(-np.pi, np.pi, num_theta)
    thetadots = np.linspace(-env.max_thetadot, env.max_thetadot, num_theta_dot)

    # Make a V matrix for given theta, theta_dot
    V = np.zeros((num_theta, num_theta_dot))

    # For each theta, theta_dot pair
    for ii in range(num_theta):
        for jj in range(num_theta_dot):
            # Get state
            s = np.array([thetas[ii], thetadots[jj]])
            # Get action values
            action_values = trained_Q(torch.Tensor.float(torch.from_numpy(s)))
            # Get max action value
            V[ii][jj] = action_values.max().detach().numpy()

    return V



