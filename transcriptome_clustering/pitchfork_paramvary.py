import matplotlib.pyplot as plt
import numpy as np
import time

from pitchfork_langevin import jacobian_pitchfork, steadystate_pitchfork, langevin_dynamics
from settings import DEFAULT_PARAMS, PARAMS_ID, FOLDER_OUTPUT, TIMESTEP, INIT_COND, NUM_TRAJ, NUM_STEPS, NOISE
from spectrums import get_spectrums, plot_spectrum_hists, get_spectrum_from_J, plot_rank_order_spectrum, scan_J_truncations, plot_spectrum_extremes
from statistical_formulae import collect_multitraj_info


# TODO store output and params in OUTPUT dir


def many_traj_varying_params(params_list, num_steps=NUM_STEPS, dt=TIMESTEP, init_cond=INIT_COND,
                             num_traj=NUM_TRAJ, noise=NOISE):
    """
    Computes num_traj langevin trajectories, for num_steps, for each params in params_list
    Returns:
        (1) multitraj_varying: NUM_STEPS x NUM_STATES x NUM_TRAJ x PARAM_IDX
    """
    # TODO decide if dict would work better
    base_params = params_list[0]
    print "Generating: num_steps x base_params.dim x num_traj x len(params_list) --", \
        num_steps, base_params.dim, num_traj, len(params_list)
    multitraj_varying = np.zeros((num_steps, base_params.dim, num_traj, len(params_list)))
    t0 = time.time()
    for idx, p in enumerate(params_list):
        print "on param_list %d of %d" % (idx, len(params_list))
        for traj in xrange(num_traj):
            langevin_states, _ = langevin_dynamics(init_cond=init_cond, dt=dt, num_steps=num_steps, params=p,
                                                   noise=noise)
            multitraj_varying[:, :, traj, idx] = langevin_states
    print "done, timer:", time.time() - t0
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
    skip_inference = True
    plot_hists_all = False
    plot_rank_order_selection = False
    scan_truncations = False
    verbosity = False
    spectrum_extremes = True

    noise = 0.1
    pv_name = 'tau'
    #params_list, pv_range = gen_params_list(pv_name, 0.1, 5.0, pv_num=5)
    params_list, pv_range = gen_params_list(pv_name, 1.2, 2.2, pv_num=5)
    multitraj_varying = many_traj_varying_params(params_list, noise=noise)
    for idx, pv in enumerate(pv_range):
        title_mod = '(%s_%.3f)' % (pv_name, pv)
        print "idx, pv:", idx, title_mod
        params = params_list[idx]
        C, D, _ = collect_multitraj_info(multitraj_varying[:, :, :, idx], params, noise, skip_infer=True)
        fp_mid = steadystate_pitchfork(params)[:, 0]
        J_true = jacobian_pitchfork(params, fp_mid, print_eig=False)
        # get U spectrums
        list_of_J_u, specs_u, labels_u = get_spectrums(C, D, method='U')
        # get infer spectrums
        if not skip_inference:
            list_of_J_infer, specs_infer, labels_infer = get_spectrums(C, D, method='infer')
        # get J_true spectrum
        spectrum_true = np.zeros((1, D.shape[0]))
        spectrum_true[0, :] = get_spectrum_from_J(J_true, real=True)
        label_true = 'J_true'

        if plot_hists_all:
            plot_spectrum_hists(specs_u, labels_u, method='U', hist='default', title_mod=title_mod)
            plot_spectrum_hists(specs_u, labels_u, method='U', hist='violin', title_mod=title_mod)
            if not skip_inference:
                plot_spectrum_hists(specs_infer, labels_infer, method='infer', hist='default', title_mod=title_mod)
                plot_spectrum_hists(specs_infer, labels_infer, method='infer', hist='violin', title_mod=title_mod)
            plot_spectrum_hists(spectrum_true, [label_true], method='true', hist='default', title_mod=title_mod)
            plot_spectrum_hists(spectrum_true, [label_true], method='true', hist='violin', title_mod=title_mod)
        if plot_rank_order_selection:
            plot_rank_order_spectrum(specs_u[0, :], labels_u[0], method='U0', title_mod=title_mod)
            if not skip_inference:
                plot_rank_order_spectrum(specs_infer[0, :], labels_infer[0], method='infer_%s' % (labels_infer[0]), title_mod=title_mod)
                plot_rank_order_spectrum(specs_infer[1, :], labels_infer[1], method='infer_%s' % (labels_infer[1]), title_mod=title_mod)
                plot_rank_order_spectrum(specs_infer[4, :], labels_infer[4], method='infer_%s' % (labels_infer[4]), title_mod=title_mod)
            plot_rank_order_spectrum(spectrum_true[0, :], label_true, method='true', title_mod=title_mod)
            plt.close('all')
        if spectrum_extremes:
            print "Scanning truncations for J U method (U=0 choice)"
            spec, spec_perturb = scan_J_truncations(list_of_J_u[0], verbose=verbosity, spectrum_unperturbed=specs_u[0, :])
            plot_spectrum_extremes(spec, spec_perturb, method='U_%s' % labels_u[0], title_mod=title_mod, max=True)
            plot_spectrum_extremes(spec, spec_perturb, method='U_%s' % labels_u[0], title_mod=title_mod, max=False)
            print "Scanning truncations for J U method (U=uni[0,scale] antisym choice)"
            spec, spec_perturb = scan_J_truncations(list_of_J_u[1], verbose=verbosity, spectrum_unperturbed=specs_u[1, :])
            plot_spectrum_extremes(spec, spec_perturb, method='U_%s' % labels_u[1], title_mod=title_mod, max=True)
            plot_spectrum_extremes(spec, spec_perturb, method='U_%s' % labels_u[1], title_mod=title_mod, max=False)
            if not skip_inference:
                print "Scanning truncations for J inferred %s" % labels_infer[1]
                spec, spec_perturb = scan_J_truncations(list_of_J_infer[1], verbose=verbosity, spectrum_unperturbed=specs_infer[1, :])
                plot_spectrum_extremes(spec, spec_perturb, method='infer_%s' % (labels_infer[1]), title_mod=title_mod, max=True)
                plot_spectrum_extremes(spec, spec_perturb, method='infer_%s' % (labels_infer[1]), title_mod=title_mod, max=False)
                print "Scanning truncations for J inferred %s" % labels_infer[3]
                spec, spec_perturb = scan_J_truncations(list_of_J_infer[3], verbose=verbosity, spectrum_unperturbed=specs_infer[3, :])
                plot_spectrum_extremes(spec, spec_perturb, method='infer_%s' % (labels_infer[3]), title_mod=title_mod, max=True)
                plot_spectrum_extremes(spec, spec_perturb, method='infer_%s' % (labels_infer[3]), title_mod=title_mod, max=False)
            print "Scanning truncations for J_true"
            spec, spec_perturb = scan_J_truncations(J_true, verbose=verbosity, spectrum_unperturbed=spectrum_true[0, :])
            plot_spectrum_extremes(spec, spec_perturb, method='true', title_mod=title_mod, max=True)
            plot_spectrum_extremes(spec, spec_perturb, method='true', title_mod=title_mod, max=False)
