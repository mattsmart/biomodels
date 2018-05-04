import numpy as np
from random import random

from noneq_settings import BETA


def hamming(s1, s2):
    """Calculate the Hamming distance between two bit lists"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def hamiltonian(state_vec, intxn_matrix):
    return -0.5 * reduce(np.dot, [state_vec.T, intxn_matrix, state_vec])  # plus some other field terms... do we care for these? ie. "-sum h_i*s_i"


def internal_field(state, spin_idx, t, intxn_matrix):
    internal_field = np.dot(intxn_matrix[spin_idx,:], state[:,t])
    return internal_field


def glauber_dynamics_update(state, spin_idx, t, intxn_matrix, app_field=None):
    r1 = random()
    total_field = internal_field(state, spin_idx, t, intxn_matrix)
    if app_field is not None:
        total_field += app_field[spin_idx]
    prob_on_after_timestep = 1 / (1 + np.exp(-2*BETA*total_field))  # probability that site i will be "up" after the timestep
    #prob_on_after_timestep = 1 / (1 + np.exp(-BETA*total_field))  # (note remove factor of 2 because h = 0.5*J*s) probability that site i will be "up" after the timestep
    if prob_on_after_timestep > r1:
        #state[spin_idx, t + 1] = 1.0
        state[spin_idx, t] = 1.0
    else:
        #state[spin_idx, t + 1] = -1.0
        state[spin_idx, t] = -1.0
    return state


def state_to_label(state):
    # Idea: assign integer label (0 to 2^N - 1) to the state
    # state acts like binary representation of integers
    # "0" corresponds to all -1
    # 2^N - 1 corresponds to all +1
    label = 0
    bitlist = (1+np.array(state, dtype=int))/2
    for bit in bitlist:
        label = (label << 1) | bit
    return label


def label_to_state(label, N, use_neg=True):
    # n is the integer label of a set of spins
    bitlist = [1 if digit=='1' else 0 for digit in bin(label)[2:]]
    if len(bitlist) < N:
        tmp = bitlist
        bitlist = np.zeros(N, dtype=int)
        bitlist[-len(tmp):] = tmp[:]
    if use_neg:
        state = np.array(bitlist)*2 - 1
    else:
        state = np.array(bitlist)
    return state


def get_adjacent_labels(state):
    # TODO slow, how to speedup with permutation?
    N = len(state)
    labels = [0] * N
    tmp = np.zeros(N, dtype=int)
    for i in xrange(N):
        tmp[:] = state[:]
        tmp[i] = -1 * state[i]
        labels[i] = state_to_label(tmp)
    return labels
