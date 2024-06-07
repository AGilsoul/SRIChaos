import numpy as np
from typing import Union, Callable
import matplotlib.pyplot as plt
from helper_code.maps import logistic_map


def LCE(sys: callable, x_0: float, r: float, n: int, verbose=False) -> float:
    """
    Computes the Lyapunov exponent of a system.

    Parameters
    ----------
    sys : callable
        The system to be analyzed, computes an orbit with the given params

    x_0 : float
        Initial condition

    r   : float
        Map parameter

    n   : int
        iteration

    Returns
    -------
    float
        The computed Lyapunov exponent
    """
    if verbose:
        print('-------------------------------------')
        print('Computing Lyapunov exponent ...')
    x_vals = sys(r, x_0, n)  # computes an orbit using the provided system
    if verbose:
        print(f'Computed orbit : {x_vals}')
    total = 0  # set sum to 0
    # use provided equation for LCE
    for i in range(1, n + 1):
        total += np.emath.logn(2, np.abs(r * (1 - 2 * x_vals[i])))
    lce = total / n
    if verbose:
        print(f'LCE: {lce}')
        print('-------------------------------------\n')
    return lce


def encode_orbit(orbit: Union[np.array, list], partition: Union[np.array, list], verbose=False) -> str:
    if verbose:
        print('-------------------------------------')
        print(f'Encoding orbit ...')
    if len(partition) > 25:
        raise ValueError('Max of 26 partitions')
    if verbose:
        print(f'Using paritition {partition}')
    encoded_string = ''
    for x in orbit:
        part_found = False
        for p in range(len(partition)):
            if x < partition[p]:
                encoded_string += chr(p + 65)
                part_found = True
                break
        if not part_found:
            encoded_string += chr(65 + len(partition))
    if verbose:
        print(f'Encoded orbit[:L]: {encoded_string}')
        print('-------------------------------------\n')
    return encoded_string


def rolling_dist(system: Callable[[float, float, int], float], r: float, x_0: float, n: int, L: int,
                 partition: Union[np.array, list], verbose=False) -> float:
    i = L
    joint_dist = {}
    total_str = ''
    orbit = system(r, x_0, L - 1).tolist()
    while i < n:
        if i == L:
            string = encode_orbit(orbit, partition, verbose)
        else:
            string = encode_orbit(orbit, partition, False)
        total_str += string
        x_0 = orbit[-1]

        if string in joint_dist:
            joint_dist[string] += 1
        else:
            joint_dist[string] = 1
        i += 1
        orbit.pop(0)
        orbit.append(system(r, x_0, 1))

    joint_count = sum([x for x in joint_dist.values()])
    joint_dist = {x: y / joint_count for x, y in joint_dist.items()}
    return joint_dist


def H(distr: dict, verbose=False) -> float:
    """
    Computes the Shannon Entropy of a provided distribution.

    Parameters
    ----------
    distr : dict
        Probability distribution

    Returns
    -------
    float
        The computed Shannon Entropy
    """
    if verbose:
        print('-------------------------------------')
        print(f'Computing Shannon Entropy ...')
    shannon_sum = 0  # sets the sum to 0
    for x in distr.values():  # for every event
        if x != 0:
            log_2 = np.emath.logn(2, x)  # take the log base 2 of the event
            shannon_sum -= x * log_2  # multiply log base 2 by the probability of the event, subtract from sum

    if verbose:
        print(f'Shannon Entropy: {shannon_sum}')
        print('-------------------------------------\n')
    return shannon_sum


def hu(joint_dist: dict, verbose=False) -> float:
    """
    Computes the Shannon Entropy rate of a provided joint distribution.

    Parameters
    ----------
    joint_dist : dict
        Joint probability distribution

    Returns
    -------
    float
        The computed Shannon Entropy rate
    """
    if verbose:
        print('_____________________________________')
        print(f'Computing Shannon Entropy rate ...')
    marginal_dist = {}  # Generate marginal distribution
    for word in joint_dist.keys():  # for every word in the joint distribution
        sub_word = word[:-1]  # Remove the last letter from the current word to get the sub word
        if sub_word in marginal_dist:  # if the sub word is already in the marginal distribution
            marginal_dist[sub_word] += joint_dist[word]  # increment the sub word count
        else:  # if sub word not in the marginal, distribution, add it
            marginal_dist[sub_word] = joint_dist[word]
    H_joint = H(joint_dist, verbose)  # Compute Shannon Entropy of the joint distribution
    H_marginal = H(marginal_dist, verbose)  # Compute Shannon Entropy of the marginal distribution
    h_u = H_joint - H_marginal
    if verbose:
        print(f'H[joint] = {H_joint}')
        print(f'H[marginal] = {H_marginal}')
        print(f'h_u: {h_u}')
        print('_____________________________________\n')
    return h_u  # return differences in Shannon Entropy (rate)


def approx_log_hu(x_0, r, n, L, partition, verbose=False) -> float:
    distribution = rolling_dist(logistic_map, r, x_0, n, L, partition, verbose)
    h_u = hu(distribution, verbose)
    return h_u


def plot_hu_conv(x_0, r, n, L_range, partition, ax):
    hu_range = [approx_log_hu(x_0, r, n, L, partition) for L in L_range]
    print(f'hu_range: {hu_range}')
    ax.plot(L_range, hu_range, label=f'p={len(partition)}')
    return hu_range


def partition_test(x_0, r, n):
    L_range = range(3, 20)
    lce = LCE(logistic_map, x_0, r, 10000, verbose=True)
    partitions = [np.arange(0, 1, 1 / (num * 2))[1:] for num in range(1, 10)]
    fig, ax = plt.subplots(1, 1)
    for partition in partitions:
        plot_hu_conv(x_0, r, n, L_range, partition, ax)
    ax.plot(L_range, [lce for _ in L_range], label='approx LCE')
    ax.set_ylim(0, 1)
    ax.set_xlabel('L')
    ax.set_ylabel('h_u')
    plt.legend()
    plt.show()


x_0 = 0.2
r = 3.9
n = 10000
partition_test(x_0, r, n)

