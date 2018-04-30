import numpy as np
from random import random

from noneq_constants import BETA


def hamiltonian(state_vec):
    return -0.5*reduce(np.dot, [state_vec.T, J, state_vec])  # plus some other field terms... do we care for these? ie. "-sum h_i*s_i"


def internal_field(state, spin_idx, t, intxn_matrix):
    internal_field = np.dot(intxn_matrix[spin_idx,:], state[:,t])
    return internal_field


def glauber_dynamics_update(state, spin_idx, t, intxn_matrix, app_field=None):
    r1 = random()
    total_field = internal_field(state, spin_idx, t, intxn_matrix)
    if app_field is not None:
        total_field += app_field[spin_idx]
    prob_on_after_timestep = 1 / (1 + np.exp(-2*BETA*total_field))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > r1:
        state[spin_idx, t + 1] = 1.0
    else:
        state[spin_idx, t + 1] = -1.0
    return state


def state_to_label(state):
    # Idea: assign integer label (0 to 2^N - 1) to the state
    # state acts like binary representation of integers
    # "0" corresponds to all -1
    # 2^N - 1 corresponds to all +1
    label = 0
    bitlist = (1+state)/2
    for bit in bitlist:
        label = (label << 1) | bit
    return label


def label_to_state(label, N):
    # n is the integer label of a set of spins
    bitlist = [1 if digit=='1' else 0 for digit in bin(label)[2:]]
    print bitlist
    if len(bitlist) < N:
        tmp = bitlist
        bitlist = np.zeros(N, dtype=int)
        bitlist[-len(tmp):] = tmp[:]
    print bitlist
    state = np.array(bitlist)*2 - 1
    return state
