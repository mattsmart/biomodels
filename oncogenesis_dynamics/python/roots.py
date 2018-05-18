import numpy as np
import matplotlib.pyplot as plt

from formulae import fp_location_general, jacobian_3d
from trajectory import trajectory_simulate


def a0(alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base):
    delta = 1 - b
    s = c - 1
    return (s+alpha_plus)*(s+delta+alpha_minus+mu) - alpha_minus*alpha_plus


def a1(alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base):
    delta = 1 - b
    s = c - 1
    #print 2*s + delta + alpha_plus + alpha_minus + mu
    return 2*s + delta + alpha_plus + alpha_minus + mu


def a0_roots(alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base):
    # TODO: currently only works for 'c' param, extend to all
    delta = 1-b
    rhat = alpha_plus + alpha_minus + mu
    p = np.array([1, delta + rhat, alpha_plus*(delta+mu)])
    roots = np.roots(p)
    print "a0 roots for s*:", roots
    rootparams_0 = [alpha_plus, alpha_minus, mu, a, b, roots[0] + 1, N, v_x, v_y, v_z, mu_base]
    print "a0 val at root 0:", a0(*rootparams_0)
    rootparams_1 = [alpha_plus, alpha_minus, mu, a, b, roots[1] + 1, N, v_x, v_y, v_z, mu_base]
    print "a0 val at root 1:", a0(*rootparams_1)
    return roots


def a1_roots(alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base):
    # TODO: currently only works for 'c' param, extend to all
    delta = 1-b
    rhat = alpha_plus + alpha_minus + mu
    root = 0.5*(-delta-rhat)
    print "a1 root for s*:", root
    rootparams = [alpha_plus, alpha_minus, mu, a, b, root + 1, N, v_x, v_y, v_z, mu_base]
    print "a1 vals at root:", a1(*rootparams)
    return root


def plot_x0_stab_conditional_test():
    # TODO: currently only works for 'c' param, extend to all

    alpha_plus = 0.05
    alpha_minus = 0.09 #4.95
    mu = 0.77

    """
    alpha_plus = 0.2
    alpha_minus = 0.5 #4.95
    mu = 0.5
    """

    a = 1.0
    b = 1.1  #1.376666
    c = 2.6
    N = 100.0
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0
    delta = 1 - b
    s = c - 1
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]
    eps = 0.1
    print "params:", params, '\n'

    s_vals = np.linspace(-50.0, 10.0, 1000)
    a0_array = np.zeros(len(s_vals))
    a1_array = np.zeros(len(s_vals))
    stab_array = np.zeros(len(s_vals))
    for idx, s_to_check in enumerate(s_vals):
        c_to_check = s_to_check + 1
        params_to_check = [alpha_plus, alpha_minus, mu, a, b, c_to_check, N, v_x, v_y, v_z, mu_base]
        a0_at_s = a0(*params_to_check)
        a1_at_s = a1(*params_to_check)
        a0_array[idx] = a0_at_s
        a1_array[idx] = a1_at_s
        stab_array[idx] = 1000 * int(a0_at_s > 0 and a1_at_s > 0)

    line_a0, = plt.plot(s_vals, a0_array, label="a0")
    line_a1, = plt.plot(s_vals, a1_array, label="a1")
    line_stab, = plt.plot(s_vals, stab_array, label="x0 fp stab?")
    plt.legend(handles=[line_a0,line_a1,line_stab])
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
    print "traj", r[-1]
    return


if __name__ == '__main__':
    plot_x0_stab_conditional_test()
    get_predicted_fp_locations_with_traj()
