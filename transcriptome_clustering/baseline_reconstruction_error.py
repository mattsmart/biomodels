import matplotlib.pyplot as plt
import numpy as np
import os

from inference import error_fn, infer_interactions, choose_J_from_general_form, solve_true_covariance_from_true_J
from pitchfork_langevin import jacobian_pitchfork, gen_multitraj, steadystate_pitchfork
from settings import DEFAULT_PARAMS, FOLDER_OUTPUT, TAU
from statistical_formulae import collect_multitraj_info, build_diffusion_from_langevin, build_covariance_at_step
from visualize_matrix import plot_matrix


"""
Assess error in JC + (JC)^T + D = 0 as num_traj varies, since C computed from num_traj
"""
# TODO plot heatmaps fn for each step in get_errors_from_one_traj


def get_errors_for_replicates(num_traj=500, num_steps=500, replicates=10, params=DEFAULT_PARAMS, noise=1.0):
    true_errors = np.zeros(replicates)
    infer_errors = np.zeros(replicates)
    # get true J
    fp_mid = steadystate_pitchfork(params)[:, 0]
    J_true = jacobian_pitchfork(params, fp_mid, print_eig=False)
    for k in xrange(replicates):
        trials_states, _ = gen_multitraj(num_traj, init_cond=fp_mid, num_steps=num_steps, params=params, noise=noise)
        D, C_est, J_infer = collect_multitraj_info(trials_states, params, noise, alpha=0.01, tol=1e-6)
        true_errors[k] = error_fn(C_est, D, J_true)
        infer_errors[k] = error_fn(C_est, D, J_infer)
    return true_errors, infer_errors


def get_errors_from_one_traj(covperiod=5, num_traj=500, num_steps=5000, params=DEFAULT_PARAMS, noise=0.1, infer=True, alpha=0.01):
    # get points to measure at
    num_pts = int(num_steps/covperiod)
    covsteps = [a*covperiod for a in xrange(num_pts)]
    plotperiod = covperiod * 100
    # prep error vectors
    true_errors = np.zeros(num_pts)
    infer_errors = None
    J_infer_errors = None
    if infer:
        infer_errors = np.zeros(num_pts)
        J_infer_errors = np.zeros(num_pts)
    J_U0choice_errors = np.zeros(num_pts)
    cov_lyap_errors = np.zeros(num_pts)
    # get true J and D
    fp_mid = steadystate_pitchfork(params)[:, 0]
    J_true = jacobian_pitchfork(params, fp_mid, print_eig=False)
    D = build_diffusion_from_langevin(params, noise)
    C_lyap = solve_true_covariance_from_true_J(J_true, D)
    print 'norm of C_lyap', np.linalg.norm(C_lyap)
    plot_matrix(C_lyap, method='C_lyap', title_mod='static', plotdir=FOLDER_OUTPUT)
    # compute long traj
    multitraj, _ = gen_multitraj(num_traj, init_cond=fp_mid, num_steps=num_steps, params=params, noise=noise)
    # get error for all covsteps
    for idx, step in enumerate(covsteps):
        C_est = build_covariance_at_step(multitraj, params, covstep=step)
        J_U0choice = choose_J_from_general_form(C_est, D, scale=0.0)
        true_errors[idx] = error_fn(C_est, D, J_true)
        J_U0choice_errors[idx] = np.linalg.norm(J_true - J_U0choice)
        print step, covperiod*100, step % covperiod*100
        if step % plotperiod == 0:
            plot_matrix(C_est, method='C_data', title_mod='step%d' % step, plotdir=FOLDER_OUTPUT)
        if infer:
            print "inferring..."
            J_infer = infer_interactions(C_est, D, alpha=alpha, tol=1e-6)
            print "done"
            infer_errors[idx] = error_fn(C_est, D, J_infer)
            J_infer_errors[idx] = np.linalg.norm(J_true - J_infer)
        cov_lyap_errors[idx] = np.linalg.norm(C_lyap - C_est)
        print idx, step, np.linalg.norm(C_est), cov_lyap_errors[idx]
    return covsteps, true_errors, infer_errors, J_infer_errors, J_U0choice_errors, cov_lyap_errors


if __name__ == '__main__':
    # run settings
    many_reps_endpt = False
    one_rep_long = True

    if many_reps_endpt:
        reps = 10
        mod = 'num_steps'
        assert mod in ['num_traj', 'num_steps']

        num_traj_set = [int(a) for a in np.linspace(10, 600, 6)]
        num_steps_set = [int(a) for a in np.linspace(10, 2000, 5)]
        param_vary_set = {'num_traj': num_traj_set, 'num_steps': num_steps_set}[mod]

        true_errors_mid = np.zeros(len(param_vary_set))
        true_errors_sd = np.zeros(len(param_vary_set))
        infer_errors_mid = np.zeros(len(param_vary_set))
        infer_errors_sd = np.zeros(len(param_vary_set))
        # compute errors and do inference
        for i, elem in enumerate(param_vary_set):
            print "point %d (%s %d)" % (i, mod, elem)
            if mod == 'num_traj':
                true_errors, infer_errors = get_errors_for_replicates(num_traj=elem, replicates=reps, noise=0.1)
            else:
                true_errors, infer_errors = get_errors_for_replicates(num_steps=elem, replicates=reps, noise=0.1)
            true_errors_mid[i] = np.mean(true_errors)
            true_errors_sd[i] = np.std(true_errors)
            infer_errors_mid[i] = np.mean(infer_errors)
            infer_errors_sd[i] = np.std(infer_errors)
        # plot
        plt.errorbar(param_vary_set, true_errors_mid, yerr=true_errors_sd, label='true J errors', fmt='o')
        plt.errorbar(param_vary_set, infer_errors_mid, yerr=infer_errors_sd, label='infer J errors', fmt='o')
        plt.title('Reconstruction error (true J vs inferred) for varying %s' % mod)
        plt.xlabel('%s' % mod)
        plt.ylabel('F-norm of JC + (JC)^T + D')
        plt.legend()
        plt.show()

    # alternate: errors for one long multi-traj at increasing timepoints points
    infer = False
    if one_rep_long:
        alpha = 1e-8
        num_steps = 5000
        num_traj = 500 #5000
        covsteps, true_errors, infer_errors, J_infer_errors, J_U0choice_errors, cov_errors = \
            get_errors_from_one_traj(alpha=alpha, num_steps=num_steps, num_traj=num_traj, infer=infer)
        # plotting
        f = plt.figure(figsize=(16, 8))
        plt.plot(covsteps, true_errors, '--k', label='true error')
        if infer:
            plt.plot(covsteps, infer_errors, '--b', label='inference error')
        plt.title('Reconstruction error (true J vs inference alpha=%.1e) for 1 multiraj (num_steps %s, num_traj %d)' % (alpha, num_steps, num_traj))
        plt.xlabel('step')
        plt.ylabel('F-norm of JC + (JC)^T + D')
        plt.legend()
        plt.savefig(FOLDER_OUTPUT + os.sep + 'fnorm_reconstruct_flucdiss_a%.1e_traj%d_steps%d_tau%.2f.png' % (alpha, num_traj, num_steps, TAU))
        # J error
        f2 = plt.figure(figsize=(16, 8))
        if infer:
            plt.plot(covsteps, J_infer_errors, '--b', label='inference error')
        plt.plot(covsteps, J_U0choice_errors, '--r', label='general form + choose U=0 error')
        plt.title('Reconstruction error of J (U=0 choice vs inference alpha=%.1e) for 1 multiraj (num_steps %s, num_traj %d)' % (alpha, num_steps, num_traj))
        plt.xlabel('step')
        plt.ylabel('F-norm of J_true - J_method')
        plt.legend()
        plt.savefig(FOLDER_OUTPUT + os.sep + 'fnorm_reconstruct_J_a%.1e_traj%d_steps%d_tau%.2f.png' % (alpha, num_traj, num_steps, TAU))
        plt.close()
        # C_lyap vs C_data error
        f3 = plt.figure(figsize=(16, 8))
        plt.plot(covsteps, cov_errors, '--b', label='cov error')
        plt.title(
            'Reconstruction error of C_lyap from asymptotic C_data for 1 multiraj (num_steps %s, num_traj %d)' %
            (num_steps, num_traj))
        plt.xlabel('step')
        plt.ylabel('F-norm of C_lyap - C_data')
        plt.legend()
        plt.savefig(FOLDER_OUTPUT + os.sep + 'fnorm_reconstruct_C_lyap_traj%d_steps%d_tau%.2f.png' % (num_traj, num_steps, TAU))