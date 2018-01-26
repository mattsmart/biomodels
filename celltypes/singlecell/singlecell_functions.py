import numpy as np
from random import random

from singlecell_constants import BETA, EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH
from singlecell_simsetup import N, XI, A_INV, J, CELLTYPE_LABELS, GENE_ID, ETA

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


def glauber_dynamics_update(state, gene_idx, t, ext_field=None, ext_field_strength=EXT_FIELD_STRENGTH, app_field=None, app_field_strength=APP_FIELD_STRENGTH):
    """
    See page 107-111 Amit for discussion on functional form
    ext_field - N x 1 - field external to the cell in a signalling sense; exosome field in multicell sym
    ext_field_strength  - scaling factor for ext_field
    app_field - N x 1 - unnatural external field (e.g. force TF on for some time period experimentally)
    app_field_strength - scaling factor for appt_field
    """
    r1 = random()
    total_field = internal_field(state, gene_idx, t)
    if ext_field is not None:
        total_field += ext_field_strength * ext_field[gene_idx]
    if app_field is not None:
        total_field += app_field_strength * app_field[gene_idx]
    prob_on_after_timestep = 1 / (1 + np.exp(-2*BETA*total_field))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > r1:
        state[gene_idx, t + 1] = 1.0
    else:
        state[gene_idx, t + 1] = -1.0
    return state


def state_subsample(state_vec, ratio_to_remove=0.5):
    state_subsample = np.zeros(len(state_vec))
    state_subsample[:] = state_vec[:]
    idx_to_remove = np.random.choice(range(len(state_vec)), int(np.round(ratio_to_remove*len(state_vec))), replace=False)
    for idx in idx_to_remove:
        state_subsample[idx] = 0.0
    return state_subsample


def state_burst_errors(state_vec, ratio_to_flip=0.02):
    state_burst_errors = np.zeros(len(state_vec))
    state_burst_errors[:] = state_vec[:]
    idx_to_flip = np.random.choice(range(len(state_vec)), int(np.round(ratio_to_flip*len(state_vec))), replace=False)
    for idx in idx_to_flip:
        state_burst_errors[idx] = -state_vec[idx]
    return state_burst_errors


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


def state_memory_projection_single(state_arr, time, memory_idx):
    #a = np.dot(ETA[memory_idx,:], state_arr[:,time])
    #b = state_memory_projection(state_arr, time)[memory_idx]
    return np.dot(ETA[memory_idx,:], state_arr[:,time])


def check_memory_energies():
    # in projection method, expect all to have value -N/2, global minimum value (Mehta 2014)
    # TODO: what is expectation in hopfield method?
    for idx, label in enumerate(CELLTYPE_LABELS):
        mem = XI[:,idx]
        h = hamiltonian(mem)
        print idx, label, h
    return


def construct_app_field_from_genes(gene_list, num_steps):
    app_field = np.zeros((N, num_steps))
    for label in gene_list:
        app_field[GENE_ID[label], :] += 1
        #print app_field[GENE_ID[label]-1:GENE_ID[label]+2,0:5]
    return app_field
