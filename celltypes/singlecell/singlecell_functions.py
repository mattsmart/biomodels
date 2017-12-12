import numpy as np
from random import random

from singlecell_constants import BETA
from singlecell_simsetup import N, XI, A_INV, J

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def hamiltonian(state, t):
    return -0.5*reduce(np.dot, [state[:,t].T, J, state[:,t]])  # plus some other field terms... do we care for these?


def local_field(state, gene_idx, t):
    h_i = 0
    intxn_list = range(0, gene_idx) + range(gene_idx+1, N)
    for j in intxn_list:
        h_i += J[gene_idx,j] * state[j,t]  # plus some other field terms... do we care for these?
    return h_i


def glauber_dynamics_update(state, gene_idx, t):
    r1 = random()
    beta_h = BETA * local_field(state, gene_idx, t)
    prob_on_after_timestep = 1 / (1 + np.exp(-2*beta_h))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > r1:
        state[gene_idx, t+1] = 1
    else:
        state[gene_idx, t + 1] = -1
    return state


def state_memory_overlap(state_arr, time):
    return np.dot(XI.T, state_arr[:, time]) / N


def state_memory_projection(state_arr, time):
    return np.dot(A_INV, state_memory_overlap(state_arr, time))
