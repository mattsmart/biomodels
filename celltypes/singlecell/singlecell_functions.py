import numpy as np
from random import random

from singlecell_constants import BETA, PARAM_EXOSOME
from singlecell_simsetup import N, XI, A_INV, J, CELLTYPE_LABELS

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def hamiltonian(state_vec):
    return -0.5*reduce(np.dot, [state_vec.T, J, state_vec])  # plus some other field terms... do we care for these? ie. "-sum h_i*s_i"


def internal_field(state, gene_idx, t):
    """
    Original slow summation:
    h_1 = 0
    intxn_list = range(0, gene_idx) + range(gene_idx+1, N)
    for j in intxn_list:
        h_1 += J[gene_idx,j] * state[j,t]  # plus some other field terms... do we care for these?
    """
    internal_field = np.dot(J[gene_idx,:], state[:,t])  # note diagonals assumed to be zero (enforce in J definition)
    return internal_field


def glauber_dynamics_update(state, gene_idx, t, external_field=None):
    """
    See page 107-111 Amit for discussion on functional form
    """
    r1 = random()
    if external_field is None:
        total_field = internal_field(state, gene_idx, t)
    else:
        total_field = internal_field(state, gene_idx, t) + PARAM_EXOSOME * external_field[gene_idx]
    prob_on_after_timestep = 1 / (1 + np.exp(-2*BETA*total_field))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > r1:
        state[gene_idx, t + 1] = 1.0
    else:
        state[gene_idx, t + 1] = -1.0
    return state


def state_subsample(state_vec, ratio_to_remove=0.5):
    state_subsample = np.zeros(len(state_vec))
    state_subsample[:] = state_vec[:]
    vals_to_remove = np.random.choice(range(len(state_vec)), int(np.round(ratio_to_remove*len(state_vec))), replace=False)
    for val in vals_to_remove:
        state_subsample[val] = 0.0
    return state_subsample


def state_only_on(state_vec):
    state_only_on = np.zeros(len(state_vec))
    for idx, val in enumerate(state_vec):
        if val < 0.0:
            state_only_on[idx] = 0.0
        else:
            state_only_on[idx] = val
    return state_only_on


def state_only_off(state_vec):
    state_only_off = np.zeros(len(state_vec))
    for idx, val in enumerate(state_vec):
        if val > 0.0:
            state_only_off[idx] = 0.0
        else:
            state_only_off[idx] = val
    return state_only_off


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
