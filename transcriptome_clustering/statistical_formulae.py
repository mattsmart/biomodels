import numpy as np

from settings import DEFAULT_PARAMS


def build_diffusion(multitraj, params):
    """
    multitraj: of the form NUM_STEPS x STATE_DIM x NUM_TRAJ
    Note: assume last step is roughly at steady state, without testing
    """
    # setup
    p = params
    D = np.zeros((p.dim, p.dim))
    # identify steadystate time slice
    steadystate_samples = multitraj[-1, :, :]
    # fill in matrix
    for idx in xrange(params.dim):
        state_mean = np.mean(steadystate_samples[idx,:])
        if idx < params.dim_master:
            D[idx, idx] = 2 * state_mean / params.tau
        else:
            slave_idx = idx - params.dim_master
            D[idx, idx] = 2 * state_mean / params.taus[slave_idx]
    return D


def build_covariance(multitraj, params):
    """
    multitraj: of the form NUM_STEPS x STATE_DIM x NUM_TRAJ
    Note: assume last step is roughly at steady state, without testing
    """
    p = params
    cov = np.zeros((p.dim, p.dim))
    # TODO
    return cov


def infer_interactions(multitraj, params):
    """
    multitraj: of the form NUM_STEPS x STATE_DIM x NUM_TRAJ
    Note: assume last step is roughly at steady state, without testing
    """
    p = params
    J = np.zeros((p.dim, p.dim))
    # TODO
    return J
