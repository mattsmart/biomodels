import numpy as np
from random import random

from singlecell_constants import BETA, EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH, MEMS_MEHTA, FIELD_PROTOCOL
from singlecell_simsetup import singlecell_simsetup, unpack_simsetup

"""
Conventions follow from Lang & Mehta 2014, PLOS Comp. Bio
- note the memory matrix is transposed throughout here (dim N x p instead of dim p x N)
"""


def hamming(s1, s2):
    """Calculate the Hamming distance between two bit lists"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def hamiltonian(state_vec, intxn_matrix):
    # TODO add applied field
    return -0.5*reduce(np.dot, [state_vec.T, intxn_matrix, state_vec])  # plus some other field terms... do we care for these? ie. "-sum h_i*s_i"


def internal_field(state, gene_idx, t, intxn_matrix):
    """
    Original slow summation:
    h_1 = 0
    intxn_list = range(0, gene_idx) + range(gene_idx+1, N)
    for j in intxn_list:
        h_1 += J[gene_idx,j] * state[j,t]  # plus some other field terms... do we care for these?
    """
    internal_field = np.dot(intxn_matrix[gene_idx,:], state[:,t])  # note diagonals assumed to be zero (enforce in J definition)
    return internal_field


def glauber_dynamics_update(state, gene_idx, t, intxn_matrix, unirand, beta=BETA, ext_field=None, app_field=None,
                            ext_field_strength=EXT_FIELD_STRENGTH, app_field_strength=APP_FIELD_STRENGTH):
    """
    unirand: pass a uniform 0,1 random number
        - note previously unirand = random() OR unirand = np.random_intel.random() from intel python distribution
    See page 107-111 Amit for discussion on functional form
    ext_field - N x 1 - field external to the cell in a signalling sense; exosome field in multicell sym
    ext_field_strength  - scaling factor for ext_field
    app_field - N x 1 - unnatural external field (e.g. force TF on for some time period experimentally)
    app_field_strength - scaling factor for appt_field
    """
    total_field = internal_field(state, gene_idx, t, intxn_matrix=intxn_matrix)
    if ext_field is not None:
        total_field += ext_field_strength * ext_field[gene_idx]
    if app_field is not None:
        total_field += app_field_strength * app_field[gene_idx]
    prob_on_after_timestep = 1 / (1 + np.exp(-2*beta*total_field))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > unirand:
        state[gene_idx, t] = 1.0
    else:
        state[gene_idx, t] = -1.0
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


def state_memory_overlap(state_arr, time, N, xi):
    return np.dot(xi.T, state_arr[:, time]) / N


def state_memory_projection(state_arr, time, a_inv, N, xi):
    return np.dot(a_inv, state_memory_overlap(state_arr, time, N, xi))


def single_memory_projection(state_arr, time, memory_idx, eta):
    """
    Given state_array (N genes x T timesteps) and time t, return projection onto single memory (memory_idx) at t
    - this should be faster than performing the full matrix multiplication (its just a row.T * col dot product)
    - this should be faster if we want many single memories, say less than half of num memories
    """
    return np.dot(eta[memory_idx,:], state_arr[:,time])


def single_memory_projection_timeseries(state_array, memory_idx, eta):
    """
    Given state_array (N genes x T timesteps), return projection (T x 1) onto single memory specified by memory_idx
    """
    num_steps = np.shape(state_array)[1]
    timeseries = np.zeros(num_steps)
    for time_idx in xrange(num_steps):
        timeseries[time_idx] = single_memory_projection(state_array, time_idx, memory_idx, eta)
    return timeseries


def check_memory_energies(xi, celltype_labels, intxn_matrix):
    # in projection method, expect all to have value -N/2, global minimum value (Mehta 2014)
    # TODO: what is expectation in hopfield method?
    for idx, label in enumerate(celltype_labels):
        mem = xi[:,idx]
        h = hamiltonian(mem, intxn_matrix)
        print idx, label, h
    return


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


if __name__ == '__main__':
    label = 511
    N = 9
    print label_to_state(label, N, use_neg=True)
