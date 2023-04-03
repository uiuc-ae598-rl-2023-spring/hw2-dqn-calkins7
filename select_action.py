import random
import numpy as np
import torch

def epsilon_greedy(eps_start, eps_end, eps_decay, steps_done, num_actions, s, Q_pi):
    """
    Samples an action for a given state following e-greedy policy based on the given Q(s,a)

    :param epsilon: e small
    :param num_actions: number of actions in environment
    :param s: state
    :param Q_pi: NN for Q(s,a)
    :return a_greedy: the greedy action
    """
    # Identify greedy action
    a_greedy = int(Q_pi(s).argmax())

    # Compute linear epsilon decay until the specified number of steps
    if steps_done < eps_decay:
        epsilon = eps_start - (eps_start - eps_end) / eps_decay * steps_done
    else:
        epsilon = eps_end

    # Compute epsilon greedy probabilities
    weights = np.ones(num_actions) * epsilon / num_actions
    weights[a_greedy] = 1 - epsilon + epsilon / num_actions

    # Choose action with epsilon greedy weights
    a_greedy = random.choices(range(num_actions), weights=weights, k=1)[0]
    # Convert to tensor
    a_greedy = torch.Tensor.long(torch.from_numpy(np.array(a_greedy)))
    return a_greedy


def random_action(num_actions):
    """
    Samples a random action

    :param num_actions: number of actions in environment
    :return a: the random action
    """
    a = random.randrange(num_actions)
    # Convert to tensor
    a = torch.Tensor.long(torch.from_numpy(np.array(a)))
    return a
