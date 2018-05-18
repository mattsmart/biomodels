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


def a0_crit_param(params, root_param="mu"):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params
    if root_param == "mu":
        Delta_b = b - a
        Delta_c = c - a
        slope = (Delta_c + alpha_plus)
        crit = (alpha_minus*alpha_plus - slope * (Delta_c - Delta_b + alpha_minus) ) / slope
    elif root_param == "c":
        Delta_b = b - a
        param_sum = alpha_minus + mu - Delta_b
        p = np.array([1, param_sum + alpha_plus, alpha_plus * (param_sum - alpha_minus)])
        crit = np.roots(p)  # list of 2 in general
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
    return crit


def x0_stab_conditional_region_printer(params, root_param="mu"):
    assert root_param == "mu"  #TODO implement b, c params
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params
    if root_param == "mu":
        Delta_b = b - a
        Delta_c = c - a
        slope = (Delta_c + alpha_plus)

        mu_crit_a1 = a1_crit_param(params, root_param=root_param)  # a1 always has slope +1 for mu
        mu_crit_a0 = a0_crit_param(params, root_param=root_param)  # a0 has slope (Delta_c + alpha_plus) for mu
        slope_a1 = np.sign(1.0)
        slope_a0 = np.sign((Delta_c + alpha_plus))

        if slope_a1 > 0 and slope_a0 > 0:
            mu_crit = max(mu_crit_a1, mu_crit_a0)
            print "All z FP stable after mu_crit =", mu_crit

        elif slope_a1 < 0 and slope_a0 < 0:
            mu_crit = min(mu_crit_a1, mu_crit_a0)
            print "All z FP stable before mu_crit =", mu_crit

        else:  # they have opposite sign of slope (or one is zero)
            if slope_a1 > 0 and mu_crit_a1 < mu_crit_a0:
                print "All z FP stable in region (A,B):", mu_crit_a1, mu_crit_a0
            elif slope_a0 > 0 and mu_crit_a0 < mu_crit_a1:
                print "All z FP stable in region (A,B):", mu_crit_a0, mu_crit_a1
            else:
                print "All z FP is not stable for any mu"
    return


def plot_x0_stab_conditional_test(params, root_param="mu", plot_sweep=True):
    # TODO: currently only works for 'c' param, extend to all
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params

    # ===========================================================
    print "a1, a0 crit param for %s" % root_param
    a1_crit_val = a1_crit_param(params, root_param=root_param)
    a0_crit_val = a0_crit_param(params, root_param=root_param)
    params_pre = params[:]
    params_post = params[:]
    eps = 0.01
    params_pre[PARAMS_ID_INV[root_param]] = a1_crit_val - eps
    params_post[PARAMS_ID_INV[root_param]] = a1_crit_val + eps
    print "a1 crit at: %.4f" % a1_crit_val, "with a1 sign before:", a1_eval(params_pre), "after:",  a1_eval(params_post)
    params_pre[PARAMS_ID_INV[root_param]] = a0_crit_val - eps
    params_post[PARAMS_ID_INV[root_param]] = a0_crit_val + eps
    print "a0 crit at: %.4f" % a0_crit_val, "with a0 sign before:", a0_eval(params_pre), "after:",  a0_eval(params_post)
    # ===========================================================

    if plot_sweep:
        crit_param_vals = np.linspace(-50.0, 10.0, 1000)
        a0_array = np.zeros(len(crit_param_vals))
        a1_array = np.zeros(len(crit_param_vals))
        stab_array = np.zeros(len(crit_param_vals))
        for idx, crit_to_check in enumerate(crit_param_vals):
            params_to_check = params[:]
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

    flag_sweep = True

    alpha_plus = 0.02
    alpha_minus = 0.05 #4.95
    mu = 0.77
    a = 1.0
    b = 0.95 #1.376666
    c = 0.99 #1 - alpha_plus*0.9
    N = 100.0
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]
    eps = 0.1
    print "main params:", params, '\n'

    root_param = "mu"

    plot_x0_stab_conditional_test(params, root_param=root_param)
    get_predicted_fp_locations_with_traj()

    if flag_sweep:
        params_sweep = params[:]
        for b_sweep in np.linspace(0.9,1.0,20):
            for c_sweep in np.linspace(0.9,1.0,20):
                params_sweep[PARAMS_ID_INV["b"]] = b_sweep
                params_sweep[PARAMS_ID_INV["c"]] = c_sweep
                print "For b, c =", b_sweep, c_sweep
                x0_stab_conditional_region_printer(params_sweep, root_param="mu")
