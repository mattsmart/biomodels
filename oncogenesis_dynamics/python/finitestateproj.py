import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp_sparse

from formulae import map_init_name_to_init_cond
from presets import presets


def fsp_statespace(params):
    # TODO may be waste to store the state_to_int dict in memory
    assert params.N <= 100.0  # only works for small pop size bc memory issues
    assert params.numstates <= 3
    pop_buffer = params.N * 0.1 + 10  # TOO tune this
    statespace = (params.N + pop_buffer) * params.numstates

    state_to_idx = {}
    count = 0
    for idx in xrange(params.numstates):
        state_to_idx[state] = idx
        count += 1

    return statespace, state_to_int


def fsp_matrix(params):
    """
    sparse library url: https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    # set FSP state space
    statespace_size = fsp_statespace(params)
    print "FSP CME matrix dimensions %d x %d" % (statespace_size, statespace_size)
    # build FSP (truncated CME) matrix
    fsp = ...  # TODO todo the hard part, use csc format?
    return fsp


def fsp_dtmc_step(fsp, step_dt):
    # take matrix exp
    dtmc_step = sp_sparse.linalg.expm(fsp*step_dt)
    return dtmc_step


def prob_at_t_oneshot(fsp, init_prob, t):
    dtmc_step = fsp_dtmc_step(fsp, t)
    prob_at_t = dtmc_step.dot(init_prob)  # TODO check works
    return prob_at_t


def prob_at_t_timeseries(params, init_prob, t0=0.0, t1=1000.0, dt=1.0):
    trange = np.linspace(t0,t1,dt)
    fsp = fsp_matrix(params)
    dtmc_step = fsp_dtmc_step(fsp, dt)
    p_of_t = np.zeros(len(init_prob), len(trange))
    p_of_t[:,0] = init_prob
    for idx, t in enumerate(trange[:-1]):
        p_of_t[:,idx+1] = dtmc_step.dot(p_of_t[:, idx])  # TODO is this right and faster then re-exponent
    return p_of_t, trange


def fsp_fpt_cdf(params, init_prob, fpt_idx=-1):
    p_of_t, trange = prob_at_t_timeseries(params, init_prob)
    fpt_cdf = p_of_t[fpt_idx,:]  # TODO what is genric location of the FPT index? last? second last?
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

    # INITIAL PROBABILITY VECTOR
    statepace, state_to_int = fsp_statespace(params)
    init_prob = np.zeros(statepace)
    init_state = tuple(map_init_name_to_init_cond(params, "x_all"))
    init_prob[state_to_int[init_state]]
    assert np.sum(init_prob) == 1.0  # TODO should assert sum of init cond is 1.0

    # get fpt distribution
    fpt_cdf, trange = fsp_fpt_cdf(params, init_prob, fpt_idx=-1)
    fpt_pdf = conv_cdf_to_pdf(fpt_cdf, trange)

    # plot
    plot_distr(fpt_cdf, trange)
    plot_distr(fpt_pdf, trange)
