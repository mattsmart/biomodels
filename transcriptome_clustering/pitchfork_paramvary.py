import matplotlib.pyplot as plt
import numpy as np
import time

from inference import solve_true_covariance_from_true_J
from pitchfork_langevin import jacobian_pitchfork, steadystate_pitchfork, langevin_dynamics
from settings import DEFAULT_PARAMS, PARAMS_ID, FOLDER_OUTPUT, TIMESTEP, INIT_COND, NUM_TRAJ, NUM_STEPS, NOISE
from spectrums import get_spectrums, plot_spectrum_hists, get_spectrum_from_arr, plot_rank_order_spectrum, \
    scan_J_truncations, plot_spectrum_extremes, plot_sliding_tau_scores, gene_control_scores
from statistical_formulae import collect_multitraj_info, build_diffusion_from_langevin
from visualize_matrix import plot_matrix


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
    avoid_traj = False
    skip_inference = False
    plot_hists_all = False
    plot_rank_order_selection = True
    verbosity = False
    spectrum_extremes = False
    sliding_tau_cg_plot = True

    noise = 0.1
    pv_name = 'tau'
    #params_list, pv_range = gen_params_list(pv_name, 0.1, 5.0, pv_num=5)
    #params_list, pv_range = gen_params_list(pv_name, 1.2, 2.2, pv_num=5)
    params_list, pv_range = gen_params_list(pv_name, 1.2, 2.2, pv_num=10)
    num_genes = params_list[0].dim

    # prepare main scoring object TODO consider convert to class and data import/export method maybe pickle
    score_labels = ['J_true']
    int_U_to_use = [0, 1]
    int_infer_to_use = [0, 1, 4]
    # prep remainder of score labels depending on run flags
    C_list = ['_lyap']
    if not avoid_traj:
        C_list.append('_data')
    for mod in C_list:
        score_labels.append('C%s' % mod)
        for elem in int_U_to_use:
            score_labels.append('J_U%d%s' % (elem, mod))
        if not skip_inference:
            for elem in int_infer_to_use:
                score_labels.append('J_infer%d%s' % (elem, mod))
    # now fill in rest, dependent on whether inference and C_data are being used
    print "List of score labels that will be analyzed:\n", score_labels
    score_dict = {label: {'skip': False,
                          'method_list': [0]*len(pv_range),
                          'matrix_list': [0]*len(pv_range),
                          'spectrums_unperturbed': np.zeros((num_genes, len(pv_range))),
                          'spectrums_perturbed': np.zeros((num_genes, num_genes - 1, len(pv_range))),
                          'cg_min': np.zeros((num_genes, len(pv_range))),
                          'cg_max': np.zeros((num_genes, len(pv_range)))} for label in score_labels}

    # optionally skip generating trajectories and use theoretical covariance
    if not avoid_traj:
        multitraj_varying = many_traj_varying_params(params_list, noise=noise)

    for idx, pv in enumerate(pv_range):
        title_mod = '(%s_%.3f)' % (pv_name, pv)
        print "idx, pv:", idx, title_mod
        params = params_list[idx]
        fp_mid = steadystate_pitchfork(params)[:, 0]
        J_true = jacobian_pitchfork(params, fp_mid, print_eig=False)
        D_true = build_diffusion_from_langevin(params, noise)
        # build data covariance or solve asymptotic covariance
        C_lyap = solve_true_covariance_from_true_J(J_true, D_true)
        if not avoid_traj:
            _, C_data, _ = collect_multitraj_info(multitraj_varying[:, :, :, idx], params, noise, skip_infer=True)

        # score spectrum fill in begin
        score_dict['J_true']['method_list'][idx] = 'J_true'
        score_dict['J_true']['matrix_list'][idx] = J_true
        score_dict['J_true']['spectrums_unperturbed'][:, idx] = get_spectrum_from_arr(J_true, real=True)

        # now fill in rest, dependent on whether inference and C_data are being used
        C_list = [(C_lyap, '_lyap')]
        if not avoid_traj:
            C_list.append((C_data, '_data'))
        for C, mod in C_list:
            # fill in C info
            label = 'C%s' % mod
            score_dict[label]['method_list'][idx] = 'covariance%s' % mod
            score_dict[label]['matrix_list'][idx] = C
            score_dict[label]['spectrums_unperturbed'][:, idx] = get_spectrum_from_arr(C, real=True)
            # do J(U) method
            list_of_J_u, specs_u, labels_u = get_spectrums(C, D_true, method='U%s' % mod)
            if plot_hists_all:
                plot_spectrum_hists(specs_u, labels_u, method='U%s' % mod, hist='violin', title_mod=title_mod)
            # fill in J(U) info
            for elem in int_U_to_use:
                label = 'J_U%d%s' % (elem, mod)
                score_dict[label]['method_list'][idx] = labels_u[elem]
                score_dict[label]['matrix_list'][idx] = list_of_J_u[elem]
                score_dict[label]['spectrums_unperturbed'][:, idx] = specs_u[elem, :]
            # do inference method
            if not skip_inference:
                list_of_J_infer, specs_infer, labels_infer = get_spectrums(C, D_true, method='infer%s' % mod)
                if plot_hists_all:
                    plot_spectrum_hists(specs_infer, labels_infer, method='infer%s' % mod, hist='violin', title_mod=title_mod)
                # fill in inference info
                for elem in int_infer_to_use:
                    label = 'J_infer%d%s' % (elem, mod)
                    score_dict[label]['method_list'][idx] = labels_infer[elem]
                    score_dict[label]['matrix_list'][idx] = list_of_J_infer[elem]
                    score_dict[label]['spectrums_unperturbed'][:, idx] = specs_infer[elem, :]

        # plot sorted rank order distributions of each spectrum (for each tau)
        if plot_rank_order_selection:
            for label in score_dict.keys():
                if not score_dict[label]['skip']:
                    spec = score_dict[label]['spectrums_unperturbed'][:, idx]
                    method = label + '_' + score_dict[label]['method_list'][idx]
                    plot_rank_order_spectrum(spec, method=method, title_mod=title_mod)
                    plt.close('all')

        # perform spectrum perturbation scanning (slow step)
        for label in score_dict.keys():
            if not score_dict[label]['skip']:
                print "Scanning truncations for matrix %s" % label
                spec, spec_perturb = scan_J_truncations(score_dict[label]['matrix_list'][idx],
                                                        spectrum_unperturbed=score_dict[label]['spectrums_unperturbed'][:, idx])
                score_dict[label]['spectrums_perturbed'][:, :, idx] = spec_perturb
                score_dict[label]['cg_min'][:, idx] = gene_control_scores(spec, spec_perturb, use_min=True)
                score_dict[label]['cg_max'][:, idx] = gene_control_scores(spec, spec_perturb, use_min=False)
                if spectrum_extremes:
                    method = label + '_' + score_dict[label]['method_list'][idx]
                    plot_spectrum_extremes(spec, spec_perturb, method=method, title_mod=title_mod, max=True)
                    plot_spectrum_extremes(spec, spec_perturb, method=method, title_mod=title_mod, max=False)

    if sliding_tau_cg_plot:
        for label in score_dict.keys():
            if score_dict[label]['skip']:
                print 'Skipping label %s because skip flag is true' % label
            else:
                print "Generating sliding tau plot for label %s" % label
                # TODO have full label in title somehow... e.g. alpha float U float
                plot_sliding_tau_scores(pv_range, score_dict[label]['cg_min'].T, label, 'cg_min')
                plot_sliding_tau_scores(pv_range, score_dict[label]['cg_max'].T, label, 'cg_max')
