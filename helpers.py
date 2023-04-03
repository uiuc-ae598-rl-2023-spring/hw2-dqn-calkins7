import numpy as np
import torch

def get_state_value_function(env, trained_Q, num_theta, num_theta_dot, avgTag):
    """
    Gets state value function using max
    :param env:
    :param trained_Q:
    :param num_theta:
    :param num_theta_dot:
    :param avgTag:
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
            if avgTag:
                action_values = avgDQNsV(trained_Q, torch.Tensor.float(torch.from_numpy(s)))
                # Get max action value
                V[ii][jj] = action_values.max()
            else:
                action_values = trained_Q(torch.Tensor.float(torch.from_numpy(s)))
                # Get max action value
                V[ii][jj] = action_values.max().detach().numpy()

    return V


def avgDQNsa(trained_Qs, s):
    """
    Returns the average action from a list of DQN's
    :param trained_Qs: List of DQN's
    :return:
    """
    Q_vecs = []
    for trained_Q in trained_Qs:
        Q_vecs.append(trained_Q(s).detach().numpy())

    best_action = np.mean(Q_vecs, axis=0).argmax()
    return best_action


def avgDQNsV(trained_Qs, s):
    """
    Returns the average action from a list of DQN's
    :param trained_Qs: List of DQN's
    :return:
    """
    Q_vecs = []
    for trained_Q in trained_Qs:
        Q_vecs.append(trained_Q(s).detach().numpy())

    V_max = np.mean(Q_vecs, axis=0).max()
    return V_max



