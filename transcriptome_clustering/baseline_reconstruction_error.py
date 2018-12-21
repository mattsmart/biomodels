import numpy as np
import matplotlib.pyplot as plt

from inference import error_fn
from pitchfork_langevin import jacobian_pitchfork, gen_multitraj, langevin_dynamics, steadystate_info, steadystate_pitchfork
from settings import DEFAULT_PARAMS, PARAMS_ID, FOLDER_OUTPUT, TIMESTEP, INIT_COND, NUM_TRAJ, NUM_STEPS
from statistical_formulae import collect_multitraj_info

"""
Assess error in JC + (JC)^T + D = 0 as num_traj varies, since C computed from num_traj
"""


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


if __name__ == '__main__':
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
