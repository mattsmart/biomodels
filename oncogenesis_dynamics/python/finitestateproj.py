import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp_sparse
from os import sep

from constants import OUTPUT_DIR
from data_io import write_matrix_data_and_idx_vals, read_matrix_data_and_idx_vals
from expv import expv
from formulae import map_init_name_to_init_cond, reaction_propensities_lowmem
from presets import presets

# main stuff to finish
# TODO 2 - build fsp_matrix


def fsp_statespace(params, fpt_flag=False):
    # TODO may be waste to store the state_to_int dict in memory
    assert params.N <= 100.0  # only works for small pop size bc memory issues
    assert params.numstates <= 3
    pop_buffer = int(params.N * 0.1 + 1)  # TODO tune this, need buffer at least +1 for pop indexing
    statespace_length = int(params.N + pop_buffer)
    statespace_volume = statespace_length ** params.numstates
    if fpt_flag:  # Note: could alternatively add extra state index to the tuple (more expensive though)
        statespace_volume += 1
    return statespace_volume, statespace_length


def fsp_statespace_map(params, fpt_flag=False):
    statespace_volume, statespace_length = fsp_statespace(params, fpt_flag=fpt_flag)
    state_to_int = {}
    count = 0

    def recloop(state_to_int, state_list, level, counter):
        # loop to encode state as L-ary representation of integer (i.e. L is side length of hypercube)
        if level == 0:
            state_to_int[tuple(state_list)] = counter  # could do vol - counter?
            return
        else:
            for idx in xrange(statespace_length):
                state_list_new = state_list + [idx]  # may be faster to preallocate and assign idx to level slot
                recloop(state_to_int, state_list_new, level - 1, counter + idx*statespace_length**(level-1))  # TODO fix broken (change buffer)

    recloop(state_to_int, [], params.numstates, count)  # TODO tricky function, maybe unit test ith binary case

    if fpt_flag:  # Note: could alternatively add extra state index to the tuple (more expensive though)
        state_to_int["firstpassage"] = statespace_volume - 1  # i.e. last socket is for fpt absorb state

    """
    for i in xrange(statespace_length):
        for j in xrange(statespace_length):
            for k in xrange(statespace_length):
                print i,j,k, "to", state_to_int[(i,j,k)]
    print "firstpassage idx", state_to_int["firstpassage"]
    """

    int_to_state = {v: k for k, v in state_to_int.items()}

    return state_to_int, int_to_state


def fsp_matrix(params, fpt_flag=False):
    """
    sparse library url: https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    # set FSP state space
    statespace_volume, statespace_length = fsp_statespace(params, fpt_flag=fpt_flag)
    print "FSP CME matrix dimensions %d x %d" % (statespace_volume, statespace_volume)
    state_to_int, int_to_state = fsp_statespace_map(params, fpt_flag=fpt_flag)

    # collect FSP (truncated CME) matrix information
    # TODO use make sure diags negative sum of col at end.. do via sparse operations or do before create coo
    # TODO issue of absorbing remainder "g" last index from the 2006 paper...
    update_dict = params.update_dict
    count = 0
    fpt_rxn_idx = np.max(update_dict.keys())
    sparse_info = dict()
    for state_start_idx in xrange(statespace_volume-1):
        state_start = int_to_state[state_start_idx]
        rxn_prop = reaction_propensities_lowmem(state_start, params, fpt_flag=fpt_flag)
        for rxn_idx, prop in enumerate(rxn_prop):
            if prop != 0.0:
                if rxn_idx == fpt_rxn_idx:
                    state_end_idx = state_to_int["firstpassage"]  # i.e. fpt absorbing state is the last one
                    sparse_info[count] = [prop, state_end_idx, state_start_idx]  # note A index as [end, start]
                    count += 1
                else:
                    state_end = [state_start[k] + update_dict[rxn_idx][k] for k in xrange(params.numstates)]
                    if not any(np.array(state_end) >= statespace_length):  # this is the truncation step
                        state_end_idx = state_to_int[tuple(state_end)]
                        sparse_info[count] = [prop, state_end_idx, state_start_idx]  # note A index as [end, start]
                        count += 1

    print "done collecting FSP generator elements, loading sparse matrix"
    # build sparse FSP matrix
    data = np.zeros(count)
    row = np.zeros(count)
    col = np.zeros(count)
    for idx in xrange(count):
        data[idx] = sparse_info[idx][0]
        row[idx] = sparse_info[idx][1]
        col[idx] = sparse_info[idx][2]
    fsp = sp_sparse.coo_matrix((data, (row, col)), shape=(statespace_volume, statespace_volume))

    print "done building FSP step generator, converting coo format to csc"
    fsp = fsp.tocsc()

    print "filling in diagonals..."
    column_sum = -1 * np.asarray(fsp.sum(axis=0))
    column_sum = column_sum.reshape(statespace_volume,)
    diag = sp_sparse.diags(column_sum, offsets=0, shape=(statespace_volume, statespace_volume))
    fsp = fsp + diag

    print "done setup: %d nonzero elems (and %d diag)" % (count, statespace_volume)
    return fsp


def fsp_dtmc_step_explicit(fsp, step_dt):
    # take matrix exp
    print "Start: matrix exp"
    dtmc_step = sp_sparse.linalg.expm(fsp*step_dt)
    print "Done: matrix exp"
    return dtmc_step


def prob_at_t_oneshot(fsp, init_prob, t, explicit=False):
    if explicit:
        dtmc_step = fsp_dtmc_step_explicit(fsp, t)
        prob_at_t = dtmc_step.dot(init_prob)
    else:
        prob_at_t = expv(t, fsp, init_prob)[0]
    return prob_at_t


def prob_at_t_timeseries(params, init_prob, t0=0.0, t1=1000.0, dt=1.0, fpt_flag=False, explicit=False, save=True):
    trange = np.arange(t0,t1,dt)
    fsp = fsp_matrix(params, fpt_flag=fpt_flag)
    p_of_t = np.zeros( (len(init_prob), len(trange)) )
    p_of_t[:,0] = init_prob
    if explicit:
        print "here", type(init_prob), type(trange), init_prob.shape, trange.shape
        print len(init_prob), len(trange)
        dtmc_step = fsp_dtmc_step_explicit(fsp, dt)
        for idx, t in enumerate(trange[:-1]):
            p_of_t[:,idx+1] = dtmc_step.dot(p_of_t[:, idx])  # TODO is this right and faster then re-exponent
    else:
        for idx, t in enumerate(trange[:-1]):
            print "Start: expv", t
            p_of_t[:, idx + 1] = expv(dt, fsp, p_of_t[:, idx])[0]
    if save:
        write_matrix_data_and_idx_vals(p_of_t, [], trange, 'p_of_t', 'idx', 'times', binary=True)
    return p_of_t, trange


def fsp_fpt_cdf(p_of_t, fpt_idx=-1):
    #p_of_t, trange = prob_at_t_timeseries(params, init_prob, t1=t1, dt=dt, fpt_flag=True)
    fpt_cdf = p_of_t[fpt_idx,:]  # TODO what is generic location of the FPT index? last? second last? use state_id["firstpassage"]
    return fpt_cdf


def conv_cdf_to_pdf(cdf, domain):
    dt = domain[1] - domain[0]
    pdf = np.zeros(len(cdf))
    pdf[0] = cdf[0]                              # TODO adjust
    for idx in xrange(1,len(domain)):
        pdf[idx] = (cdf[idx] - cdf[idx-1]) / dt  # TODO check form
    return pdf


def plot_distr(distr, domain, title):
    plt.plot(domain, distr)
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('prob')
    plt.show()
    return


if __name__ == "__main__":
    # SCRIPT PARAMETERS
    switch_generate = True
    default_path_p_of_t = OUTPUT_DIR + sep + 'p_of_t.npy'
    default_path_p_of_t_idx = OUTPUT_DIR + sep + 'p_of_t_idx.npy'
    default_path_p_of_t_times = OUTPUT_DIR + sep + 'p_of_t_times.npy'

    # DYNAMICS PARAMETERS
    params = presets('preset_xyz_constant')  # preset_xyz_constant, preset_xyz_constant_fast, valley_2hit
    params = params.mod_copy({'N': 20})  # TODO had memory error with N = 50 once it got to expm call, had delayed memory error for N = 20
    t1 = 0.4 * 1e5
    dt = 10 * 1e2

    # INITIAL PROBABILITY VECTOR
    statespace_vol, statespace_length = fsp_statespace(params, fpt_flag=True)
    state_to_int, int_to_state = fsp_statespace_map(params, fpt_flag=True)
    init_prob = np.zeros(statespace_vol)
    init_state = tuple(map_init_name_to_init_cond(params, "x_all"))
    init_prob[state_to_int[init_state]] = 1.0
    assert np.sum(init_prob) == 1.0

    # get p_of_t data
    if switch_generate:
        p_of_t, trange = prob_at_t_timeseries(params, init_prob, t1=t1, dt=dt, fpt_flag=True)
    else:
        path_p_of_t = default_path_p_of_t
        path_p_of_t_idx = default_path_p_of_t_idx
        path_p_of_t_times = default_path_p_of_t_times
        p_of_t, _, trange = read_matrix_data_and_idx_vals(path_p_of_t, path_p_of_t_idx, path_p_of_t_times, binary=True)

    # get fpt distribution
    fpt_cdf = fsp_fpt_cdf(p_of_t, fpt_idx=-1)
    fpt_pdf = conv_cdf_to_pdf(fpt_cdf, trange)

    # plot
    plot_distr(fpt_cdf, trange, 'FPT cdf')
    plot_distr(fpt_pdf, trange, 'FPT pdf')

