import numpy as np
import matplotlib.pyplot as plt

from pitchfork_langevin import jacobian_pitchfork, steadystate_pitchfork, langevin_dynamics
from settings import DEFAULT_PARAMS, FOLDER_OUTPUT, TIMESTEP, INIT_COND, NUM_TRAJ, NUM_STEPS
from statistical_formulae import collect_multitraj_info


def many_traj_varying_params(params_list, num_steps=NUM_STEPS, dt=TIMESTEP, init_cond=INIT_COND, num_traj=NUM_TRAJ):
    """
    Computes num_traj langevin trajectories, for num_steps, for each params in params_list
    Returns:
        (1) multitraj_varying: NUM_STEPS x NUM_STATES x NUM_TRAJ x PARAM_IDX
    """
    # TODO decide if dict would work better
    multitraj_varying = np.zeros((num_steps, p.dim, num_traj, len(params_list)))
    for idx, p in enumerate(params_list):
        for traj in xrange(num_traj):
            langevin_states, _ = langevin_dynamics(init_cond=init_cond, dt=dt, num_steps=num_steps, params=p)
            multitraj_varying[:, :, traj, idx] = langevin_states
    return multitraj_varying


def gen_params_list(params=DEFAULT_PARAMS):
    """
    Creates a list of params based off DEFAULT_PARAMS
    Default behaviour is to vary tau across the bifurcation which occurs (originally at tau=2.0)
    """
    # TODO
    """
    tau_range = ...
    params_list = [0] * len(tau_range)
    for idx, tau in enumerate(tau_range):...
    """
    return params_list


if __name__ == '__main__':
    params_list = gen_params_list()
    multitraj_varying = many_traj_varying_params(params_list)
