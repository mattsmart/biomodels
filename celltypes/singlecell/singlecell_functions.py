import numpy as np
from random import random

from singlecell_constants import BETA, PARAM_EXOSOME
from singlecell_simsetup import N, XI, A_INV, J, CELLTYPE_LABELS

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def hamiltonian(state_vec):
    return -0.5*reduce(np.dot, [state_vec.T, J, state_vec])  # plus some other field terms... do we care for these?


def local_field(state, gene_idx, t):
    h_i = 0
    intxn_list = range(0, gene_idx) + range(gene_idx+1, N)
    for j in intxn_list:
        h_i += J[gene_idx,j] * state[j,t]  # plus some other field terms... do we care for these?
    return h_i


def glauber_dynamics_update(state, gene_idx, t, field=None):
    r1 = random()
    if field is None:
        beta_h = BETA * local_field(state, gene_idx, t)
    else:
        beta_h = BETA * (local_field(state, gene_idx, t) + PARAM_EXOSOME * field[gene_idx])
    prob_on_after_timestep = 1 / (1 + np.exp(-2*beta_h))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > r1:
        state[gene_idx, t + 1] = 1
    else:
        state[gene_idx, t + 1] = -1
    return state


def state_memory_overlap(state_arr, time):
    return np.dot(XI.T, state_arr[:, time]) / N


def state_memory_projection(state_arr, time):
    return np.dot(A_INV, state_memory_overlap(state_arr, time))


def check_memory_energies():
    # in projection method, expect all to have value -N/2, global minimum value (Mehta 2014)
    # TODO: what is expectation in hopfield method?
    for idx, label in enumerate(CELLTYPE_LABELS):
        mem = XI[:,idx]
        h = hamiltonian(mem)
        print idx, label, h
    return
