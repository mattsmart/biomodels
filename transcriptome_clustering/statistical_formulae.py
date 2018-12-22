import numpy as np

from inference import infer_interactions


def build_diffusion_from_expt(params, state_means):
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


def build_diffusion_from_langevin(params, noise):
    # TODO fix
    D = (noise ** 2) * np.eye(params.dim)                # if N noise sampled independently for each gene
    #D = noise ** 2 * np.ones((params.dim, params.dim))  # if same noise sample applied to all genes
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


def build_covariance_at_step(multitraj, params, covstep=-1):
    samples = multitraj[covstep, :, :]
    return build_covariance(params, samples, use_numpy=True)


def collect_multitraj_info(multitraj, params, noise, alpha=0.1, tol=1e-4, skip_infer=False, covstep=-1):
    """
    steadystate_samples: of the form STATE_DIM x NUM_TRAJ
    Note: assume last step is roughly at steady state, without testing (call this "covstep")
    """
    # obtain three matrices in fluctuation-dissipation relation
    D = build_diffusion_from_langevin(params, noise)
    C = build_covariance_at_step(multitraj, params, covstep=covstep)
    if skip_infer:
        J = None
    else:
        J = infer_interactions(C, D, alpha=alpha, tol=tol)
    return D, C, J
