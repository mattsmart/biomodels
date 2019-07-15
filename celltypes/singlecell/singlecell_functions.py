import numpy as np
from random import random, shuffle

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


def hamiltonian(state_vec, intxn_matrix, field=None, fs=0.0):
    """
    fs is applied_field_strength
    """
    if field is None:
        H = -0.5 * reduce(np.dot, [state_vec.T, intxn_matrix, state_vec])
    else:
        H = -0.5 * reduce(np.dot, [state_vec.T, intxn_matrix, state_vec]) - fs * np.dot(state_vec, field)
    return H


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


def sorted_energies(simsetup, field=None, fs=0.0):
    N = simsetup['N']
    num_states = 2 ** N
    energies = np.zeros(num_states)
    for label in xrange(num_states):
        state = label_to_state(label, N, use_neg=True)
        energies[label] = hamiltonian(state, simsetup['J'], field=field, fs=fs)
    energies_ranked = np.argsort(energies)
    sorted_data = {}
    last_rank = 0
    last_energy = 0
    for rank, idx in enumerate(energies_ranked):
        energy = energies[idx]
        if np.abs(energy - last_energy) < 1e-4:
            sorted_data[last_rank]['labels'].append(idx)
            sorted_data[last_rank]['ranks'].append(idx)
        else:
            sorted_data[rank] = {'energy': energy, 'labels': [idx], 'ranks': [rank]}
            last_rank = rank
            last_energy = energy
    return sorted_data, energies


def get_all_fp(simsetup, field=None, fs=0.0):
    # TODO 1 - is it possible to partition all 2^N into basins? are many of the points ambiguous where they wont roll into one basin but multiple?
    N = simsetup['N']
    num_states = 2 ** N
    energies = np.zeros(num_states)
    X = np.array([label_to_state(label, N) for label in xrange(num_states)])

    for label in xrange(num_states):
        energies[label] = hamiltonian(X[label,:], simsetup['J'], field=field, fs=fs)

    minima = []
    maxima = []
    fp_annotation = {}
    for label in xrange(num_states):
        is_fp, is_min = check_min_or_max(simsetup, X[label,:], energy=energies[label], field=field, fs=fs)
        if is_fp:
            if is_min:
                minima.append(label)
            else:
                maxima.append(label)
            fp_info = [0 for _ in xrange(N)]
            for idx in xrange(N):
                nbr_state = np.copy(X[label, :])
                nbr_state[idx] = -1 * nbr_state[idx]
                nbr_label = state_to_label(nbr_state)
                fp_info[idx] = energies[label] <= energies[nbr_label]  # higher or equal energy after flip -> True, else False (nbr is lower)
            fp_annotation[label] = fp_info
    return fp_annotation, minima, maxima


def calc_state_dist_to_local_min(simsetup, minima, X=None):
    N = simsetup['N']
    num_states = 2 ** N
    if X is None:
        X = np.array([label_to_state(label, N) for label in xrange(num_states)])
    minima_states = np.array([label_to_state(a, N) for a in minima])
    overlaps = np.dot(X, minima_states.T)
    hamming_dist = 0.5 * (N - overlaps)
    return hamming_dist


def check_min_or_max(simsetup, state, energy=None, field=None, fs=0.0):
    # 1) is it a fixed point of the deterministic dynamics?
    is_fp = False
    field_term = 0
    if field is not None:
        field_term = field * fs
    total_field = np.dot(simsetup['J'], state) + field_term
    # TODO speedup
    if np.array_equal(np.sign(total_field), np.sign(state)):
        is_fp = True

    # 2) is it a min or a max?
    is_min = None
    if is_fp:
        state_perturb = np.zeros(state.shape)
        state_perturb[:] = state[:]
        state_perturb[0] = -1 * state[0]
        energy_perturb = hamiltonian(state_perturb, simsetup['J'], field, fs)
        if energy is None:
            energy = hamiltonian(state, simsetup['J'], field, fs)
        if energy > energy_perturb:
            is_min = False
        else:
            is_min = True
    return is_fp, is_min


def fp_of_state(simsetup, state_start, app_field=0, dynamics='sync', zero_override=True):
    """
    Given a state e.g. (1,1,1,1, ... 1) i.e. hypercube vertex, return the corresponding FP of specified dynamics
    """
    # TODO how to handle the flickering/oscillation in sync mode? store extra state, catch 2 cycle, and impute FP?
    assert dynamics in ['sync', 'async_fixed', 'async_batch']
    i = 0
    state_next = np.copy(state_start)
    state_current = np.zeros(state_start.shape)
    if zero_override:
        override_sign = 1
        app_field = app_field + np.ones(app_field.shape) * 1e-8 * zero_override
    if dynamics == 'sync':
        while not np.array_equal(state_next, state_current):
            state_current = state_next
            total_field = np.dot(simsetup['J'], state_next) + app_field
            state_next = np.sign(total_field)
            i += 1
    elif dynamics == 'async_fixed':
        while not np.array_equal(state_next, state_current):
            state_current = state_next
            for i in xrange(simsetup['N']):
                total_field_on_i = np.dot(simsetup['J'][i, :], state_next) + app_field[i]
                state_next[i] = np.sign(total_field_on_i)
                i += 1
    else:
        assert dynamics == 'asynch_batch'
        sites = shuffle(range(simsetup['N']))  # randomize site ordering each timestep updates
        while not np.array_equal(state_next, state_current):
            state_current = state_next
            for i in sites:
                total_field_on_i = np.dot(simsetup['J'][i, :], state_next) + app_field[i]
                state_next[i] = np.sign(total_field_on_i)
                i += 1
    fp = state_next
    return fp


def partition_basins(simsetup, X=None, minima=None, field=None, fs=0.0, dynamics='sync'):
    assert dynamics in ['sync', 'async_fixed', 'async_batch']
    if minima is None:
        _, minima, _ = get_all_fp(simsetup, field=field, fs=fs)
    N = simsetup['N']
    num_states = 2 ** N
    if field is not None:
        app_field = field * fs
    else:
        app_field = np.zeros(N)
    basins_dict = {label: [] for label in minima}
    if X is None:
        X = np.array([label_to_state(label, N) for label in xrange(num_states)])
    for label in xrange(num_states):
        state = X[label, :]
        fp = fp_of_state(simsetup, state, app_field=app_field, dynamics=dynamics)
        fp_label = state_to_label(fp)
        if fp_label in minima:
            basins_dict[fp_label].append(label)
        else:
            print "WARNING -- fp_label not in minima;", label, 'went to', fp_label, minima
            basins_dict[fp_label] = [label]
    return basins_dict


if __name__ == '__main__':
    label = 511
    N = 9
    print label_to_state(label, N, use_neg=True)
