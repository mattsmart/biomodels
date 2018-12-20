import numpy as np

from inference import build_linear_problem, solve_regularized_linear_problem, matrixify_vector


def build_diffusion(params, state_means):
    """
    steadystate_samples: of the form STATE_DIM x NUM_TRAJ
    """
    # setup
    D = np.zeros((params.dim, params.dim))
    # fill in matrix
    for idx in xrange(params.dim):
        if idx < params.dim_master:
            D[idx, idx] = 2 * state_means[idx] / params.tau
        else:
            slave_idx = idx - params.dim_master
            D[idx, idx] = 2 * state_means[idx] / params.taus[slave_idx]
    return D


def build_covariance(params, steadystate_samples, use_numpy=True, state_means=None):
    """
    steadystate_samples: of the form STATE_DIM x NUM_TRAJ
    """
    if use_numpy:
        cov = np.cov(steadystate_samples)
    else:
        p = params
        sample = steadystate_samples
        cov = np.zeros((p.dim, p.dim))
        num_traj = steadystate_samples.shape[-1]
        denom_correction = num_traj - 1
        if state_means is None:
            state_means = np.mean(steadystate_samples, axis=1)
        # fill in
        for i in xrange(p.dim):
            for j in xrange(p.dim):
                for k in xrange(num_traj):
                    cov[i, j] += (sample[i, k] - state_means[i]) * (sample[j, k] - state_means[j])
                cov[i, j] = cov[i, j] / denom_correction
    return cov


def infer_interactions(C, D):
    """
    Method to solve for J in JC + (JC)^T = -D
    - convert problem to linear one: underdetermined Ax=b
    - use lasso (lagrange multiplier with L1-norm on J) to find candidate J
    """
    # TODO why is result so poor
    A, b = build_linear_problem(C, D, order='C')
    x = solve_regularized_linear_problem(A, b)
    J = matrixify_vector(x, order='C')
    return J


def collect_multitraj_info(multitraj, params):
    """
    steadystate_samples: of the form STATE_DIM x NUM_TRAJ
    Note: assume last step is roughly at steady state, without testing
    """
    # identify steadystate time slice and compute sample means
    steadystate_samples = multitraj[-1, :, :]
    # compute means to pass to array functions
    state_means = np.mean(steadystate_samples, axis=1)
    assert len(state_means) == params.dim
    # obtain three matrices in fluctuation-dissipation relation
    D = build_diffusion(params, state_means)
    C = build_covariance(params, steadystate_samples, use_numpy=True)
    J = infer_interactions(C, D)
    return D, C, J
