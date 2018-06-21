import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp_sparse

from formulae import map_init_name_to_init_cond
from presets import presets

# main stuff to finish
# TODO 1 - build fsp_statespace_map
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

    for i in xrange(statespace_length):
        for j in xrange(statespace_length):
            for k in xrange(statespace_length):
                print i,j,k, "to", state_to_int[(i,j,k)]
    print "firstpassage idx", state_to_int["firstpassage"]

    return state_to_int


def fsp_matrix(params, fpt_flag=False):
    """
    sparse library url: https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    # set FSP state space
    statespace_size = fsp_statespace(params, fpt_flag=fpt_flag)
    print "FSP CME matrix dimensions %d x %d" % (statespace_size, statespace_size)

    # build FSP (truncated CME) matrix
    fsp = None  # TODO todo the hard part, use csc format?
    # given state i j k, what are transitions?
    # in general all can increment by one so we must populate accordingly
    # care for edge cases
    # care for not re assigning same elements as iterating cross matrix
    for state in states:
        fsp = fsp + sparse(......) # fsp add states from stoich dict sparsely
    # TODO use stoich matrix from params somehow, only add nonzero elements to A, dont re-add elements?

    print "done building FSP step generator"
    return fsp


def fsp_dtmc_step(fsp, step_dt):
    # take matrix exp
    dtmc_step = sp_sparse.linalg.expm(fsp*step_dt)
    return dtmc_step


def prob_at_t_oneshot(fsp, init_prob, t):
    dtmc_step = fsp_dtmc_step(fsp, t)
    prob_at_t = dtmc_step.dot(init_prob)
    return prob_at_t


def prob_at_t_timeseries(params, init_prob, t0=0.0, t1=1000.0, dt=1.0, fpt_flag=False):
    trange = np.arange(t0,t1,dt)
    fsp = fsp_matrix(params, fpt_flag=fpt_flag)
    dtmc_step = fsp_dtmc_step(fsp, dt)
    p_of_t = np.zeros(len(init_prob), len(trange))
    p_of_t[:,0] = init_prob
    for idx, t in enumerate(trange[:-1]):
        p_of_t[:,idx+1] = dtmc_step.dot(p_of_t[:, idx])  # TODO is this right and faster then re-exponent
    return p_of_t, trange


def fsp_fpt_cdf(params, init_prob, fpt_idx=-1):
    p_of_t, trange = prob_at_t_timeseries(params, init_prob, fpt_flag=True)
    fpt_cdf = p_of_t[fpt_idx,:]  # TODO what is genric location of the FPT index? last? second last? use state_id["firstpassage"]
    return fpt_cdf, trange       # TODO check that it should be cdf, been using pdf though


def conv_cdf_to_pdf(cdf, domain):
    dt = domain[1] - domain[0]
    pdf = np.zeros(len(cdf))
    pdf[0] = cdf[0]                              # TODO adjust
    for idx in xrange(1,len(domain)):
        pdf[idx] = (cdf[idx] - cdf[idx-1]) / dt  # TODO check form
    return pdf


def plot_distr(distr, domain):
    plt.plot(domain, distr)
    plt.xlabel('t')
    plt.ylabel('prob')
    plt.show()
    return


if __name__ == "__main__":
    # DYNAMICS PARAMETERS
    params = presets('preset_xyz_constant')  # preset_xyz_constant, preset_xyz_constant_fast, valley_2hit
    params.N = 3
    # INITIAL PROBABILITY VECTOR
    statespace_vol, statespace_length = fsp_statespace(params, fpt_flag=True)
    state_to_int = fsp_statespace_map(params, fpt_flag=True)
    init_prob = np.zeros(statespace_vol)
    init_state = tuple(map_init_name_to_init_cond(params, "x_all"))
    init_prob[state_to_int[init_state]] = 1.0
    assert np.sum(init_prob) == 1.0

    """
    # get fpt distribution
    fpt_cdf, trange = fsp_fpt_cdf(params, init_prob, fpt_idx=-1)
    fpt_pdf = conv_cdf_to_pdf(fpt_cdf, trange)

    # plot
    plot_distr(fpt_cdf, trange)
    plot_distr(fpt_pdf, trange)
    """
