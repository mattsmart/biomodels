import numpy as np
import matplotlib.pyplot as plt

from constants import PARAMS_ID_INV
from formulae import fp_location_general, jacobian_3d
from trajectory import trajectory_simulate


ROOT_PARAMS_VALID = ["mu", "c"]


def a0_eval(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params
    Delta_b = b - a
    Delta_c = c - a
    return (Delta_c + alpha_plus) * (Delta_c - Delta_b + alpha_minus + mu) - alpha_minus*alpha_plus


def a1_eval(params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params
    Delta_b = b - a
    Delta_c = c - a
    return 2*Delta_c - Delta_b + alpha_plus + alpha_minus + mu


def a0_roots(alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base):
    # TODO: currently only works for 'c' param, extend to all
    delta = a-b
    rhat = alpha_plus + alpha_minus + mu
    p = np.array([1, delta + rhat, alpha_plus*(delta+mu)])
    roots = np.roots(p)
    print "a0 roots for s*:", roots
    rootparams_0 = [alpha_plus, alpha_minus, mu, a, b, roots[0] + a, N, v_x, v_y, v_z, mu_base]
    print "a0 val at root 0:", a0_eval(rootparams_0)
    rootparams_1 = [alpha_plus, alpha_minus, mu, a, b, roots[1] + a, N, v_x, v_y, v_z, mu_base]
    print "a0 val at root 1:", a0_eval(rootparams_1)
    return roots


def a1_roots(alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base, root_param="mu"):
    # TODO: currently only works for 'c' param, extend to all
    delta = a-b
    rhat = alpha_plus + alpha_minus + mu
    root = 0.5*(-delta-rhat)
    print "a1 root for s*:", root
    rootparams = [alpha_plus, alpha_minus, mu, a, b, root + a, N, v_x, v_y, v_z, mu_base]
    print "a1 vals at root:", a1_eval(rootparams)
    return root


def a0_crit_param(params, root_param="mu"):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params
    if root_param == "mu":
        Delta_b = b - a
        Delta_c = c - a
        slope = (Delta_c + alpha_plus)
        crit = (alpha_minus*alpha_plus - slope * (Delta_c - Delta_b + alpha_minus) ) / slope

        params_to_check = params
        params_to_check[PARAMS_ID_INV[root_param]] = crit
        print "CHECK IF 0:", a0_eval(params_to_check)
    elif root_param == "c":
        Delta_b = b - a
        param_sum = alpha_minus + mu - Delta_b
        p = np.array([1, param_sum + alpha_plus, alpha_plus * (param_sum - alpha_minus)])
        crit = np.roots(p)  # list of 2 in general

        for elem in crit:
            params_to_check = params
            params_to_check[PARAMS_ID_INV[root_param]] = elem
            print "CHECK if a0 eval to 0:", a0_eval(params_to_check)
    else:
        assert root_param in ROOT_PARAMS_VALID
        crit = None
    return crit


def a1_crit_param(params, root_param="mu"):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params
    if root_param == "mu":
        Delta_b = b - a
        Delta_c = c - a
        crit = Delta_b - 2*Delta_c - alpha_minus - alpha_plus
    elif root_param == "c":
        Delta_b = b - a
        crit = 0.5 * (a + b - alpha_minus - alpha_plus - mu)
    else:
        assert root_param in ROOT_PARAMS_VALID
        crit = None

    params_to_check = params
    params_to_check[PARAMS_ID_INV[root_param]] = crit
    print "CHECK if a1 eval to 0:", a1_eval(params_to_check)
    return crit


def plot_x0_stab_conditional_test(params, root_param="mu", plot_sweep=True):
    # TODO: currently only works for 'c' param, extend to all
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params

    # ===========================================================
    print "a1, a0 crit param for %s" % root_param
    a1_crit_val = a1_crit_param(params, root_param=root_param)
    a0_crit_val = a0_crit_param(params, root_param=root_param)
    print "a1 crit at: %.2f" % a1_crit_val
    print "a0 crit at: %.2f" % a0_crit_val
    # ===========================================================

    if plot_sweep:
        crit_param_vals = np.linspace(-50.0, 10.0, 1000)
        a0_array = np.zeros(len(crit_param_vals))
        a1_array = np.zeros(len(crit_param_vals))
        stab_array = np.zeros(len(crit_param_vals))
        for idx, crit_to_check in enumerate(crit_param_vals):
            params_to_check = params
            params_to_check[PARAMS_ID_INV[root_param]] = crit_to_check
            a0_at_check = a0_eval(params_to_check)
            a1_at_check = a1_eval(params_to_check)
            a0_array[idx] = a0_at_check
            a1_array[idx] = a1_at_check
            stab_array[idx] = 1000 * int(a0_at_check > 0 and a1_at_check > 0)

        line_a0, = plt.plot(crit_param_vals, a0_array, label="a0 at params")
        line_a1, = plt.plot(crit_param_vals, a1_array, label="a1 at params")
        line_stab, = plt.plot(crit_param_vals, stab_array, label="All-z aka x0 FP is stable?")
        plt.legend(handles=[line_a0,line_a1,line_stab])
        plt.xlabel('param %s val' % root_param)
        plt.ylabel('a0 and a1 val, stability step line')
        plt.gca().grid(True)
        plt.show()

    roots_from_a0 = a0_roots(*params)
    root_from_a1 = a1_roots(*params)
    print "mid", 0.5*(roots_from_a0[0] + roots_from_a0[1])

    return


def get_predicted_fp_locations_with_traj():
    alpha_plus = 0.004  # 0.05 #0.4
    alpha_minus = 0.05  # 4.95 #0.5
    mu = 0.000001  # 0.77 #0.01
    a = 1.0
    b = 0.1
    c = 1.01  # 2.6 #1.2
    N = 100.0  # 100
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0
    params_test = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]
    SYSTEM = "default"
    sol_list = fp_location_general(params_test, SYSTEM, solver_fsolve=False, solver_fast=False)
    print "Predicted FP locations:"
    for fp in sol_list:
        J = jacobian_3d(params_test, fp)
        eigenvalues, V = np.linalg.eig(J)
        print fp, " |  Eigenvalues:", eigenvalues

    # print N*((alpha_plus + alpha_minus + mu)/2 - np.sqrt((alpha_plus - alpha_minus - mu)**2 + 4*alpha_plus*alpha_minus)/2)
    print "\nNow simulating trajectory..."
    r, times, ax_traj, ax_mono = trajectory_simulate(params_test, SYSTEM, init_cond=[99.9, 0.1, 0.0], t0=0, t1=20000.0,
                                                     num_steps=2000, sim_method="libcall",
                                                     flag_showplt=False, flag_saveplt=False)
    print "traj endpoint", r[-1]
    return


if __name__ == '__main__':
    alpha_plus = 0.05
    alpha_minus = 0.09 #4.95
    mu = 0.77
    a = 1.0
    b = 1.1  #1.376666
    c = 0.8
    N = 100.0
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0
    delta = 1 - b
    s = c - 1
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]
    eps = 0.1
    print "main params:", params, '\n'

    root_param = "mu"

    plot_x0_stab_conditional_test(params, root_param=root_param)
    get_predicted_fp_locations_with_traj()
