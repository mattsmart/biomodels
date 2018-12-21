import numpy as np
import matplotlib.pyplot as plt
import time

from pitchfork_langevin import jacobian_pitchfork, steadystate_pitchfork, langevin_dynamics
from settings import DEFAULT_PARAMS, PARAMS_ID, FOLDER_OUTPUT, TIMESTEP, INIT_COND, NUM_TRAJ, NUM_STEPS, NOISE
from spectrums import get_spectrums, plot_spectrum_hists
from statistical_formulae import collect_multitraj_info


def many_traj_varying_params(params_list, num_steps=NUM_STEPS, dt=TIMESTEP, init_cond=INIT_COND,
                             num_traj=NUM_TRAJ, noise=NOISE):
    """
    Computes num_traj langevin trajectories, for num_steps, for each params in params_list
    Returns:
        (1) multitraj_varying: NUM_STEPS x NUM_STATES x NUM_TRAJ x PARAM_IDX
    """
    # TODO decide if dict would work better
    base_params = params_list[0]
    print "Generating: num_steps x base_params.dim x num_traj x len(params_list)"
    print num_steps, base_params.dim, num_traj, len(params_list)
    multitraj_varying = np.zeros((num_steps, base_params.dim, num_traj, len(params_list)))
    t0 = time.time()
    for idx, p in enumerate(params_list):
        for traj in xrange(num_traj):
            langevin_states, _ = langevin_dynamics(init_cond=init_cond, dt=dt, num_steps=num_steps, params=p,
                                                   noise=noise)
            multitraj_varying[:, :, traj, idx] = langevin_states
    print "Done, timer:", time.time() - t0
    return multitraj_varying


def gen_params_list(pv_name, pv_low, pv_high, pv_num=10, params=DEFAULT_PARAMS):
    """
    Creates a list of params based off DEFAULT_PARAMS
    Default behaviour is to vary tau across the bifurcation which occurs (expect tau=2.0)
    """
    assert pv_name in PARAMS_ID.values()
    pv_range = np.linspace(pv_low, pv_high, pv_num)
    params_list = [0] * len(pv_range)
    for idx, pv in enumerate(pv_range):
        params_with_pv = params.mod_copy({pv_name: pv})
        params_list[idx] = params_with_pv
    return params_list, pv_range


if __name__ == '__main__':
    noise = 0.1
    pv_name = 'tau'
    params_list, pv_range = gen_params_list(pv_name, 0.5, 3.0)
    multitraj_varying = many_traj_varying_params(params_list, noise=noise)
    for idx, pv in enumerate(pv_range):
        title_mod = '$s_%.3f' % (pv_name, pv)
        print "idx, pv:", idx, title_mod
        params = params_list[idx]
        C, D, _ = collect_multitraj_info(multitraj_varying[:, :, :, idx], params, noise, skip_infer=True)
        fp_mid = steadystate_pitchfork(params)[0, :]
        J_true = jacobian_pitchfork(params, fp_mid, print_eig=False)
        # get U spectrums
        specs, labels = get_spectrums(C, D, method='U')
        plot_spectrum_hists(specs, labels, method='U', hist='default', title_mod=title_mod)
        # get infer spectrums
        specs, labels = get_spectrums(C, D, method='infer')
        plot_spectrum_hists(specs, labels, method='U', hist='default', title_mod=title_mod)
