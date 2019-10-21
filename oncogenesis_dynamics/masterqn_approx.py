import matplotlib.pyplot as plt
import numpy as np

from params import Params
from trajectory import trajectory_simulate, get_centermanifold_traj

"""
def get_s_arr(params, Nval):
    pmc = params.mod_copy({'N': Nval})

    time_end = 200.0  # 20.0
    num_steps = 200  # number of timesteps in each trajectory

    assert params.b == 0.8
    assert params.mult_inc == 100.0  # saddle point hardcoded to this rn
    saddle = np.array([40.61475564788107, 40.401927055159106, 18.983317296959825]) / 100.0
    saddle_below = np.array([40.62, 40.41, 18.97]) / 100.0 * Nval
    saddle_above = np.array([40.6, 40.4, 19.0]) / 100.0 * Nval

    num_pts = 200 * 3
    mid_a = 200
    mid_b = 400
    z_arr = np.zeros(num_pts)
    y_arr = np.zeros(num_pts)
    s_xyz_arr = np.zeros(num_pts)
    f_xyz_arr = np.zeros(num_pts)
    s_xy_arr = np.zeros(num_pts)
    f_xy_arr = np.zeros(num_pts)

    r_a_fwd, times_a_fwd = trajectory_simulate(pmc, init_cond=[Nval, 0, 0], t0=0.0, t1=time_end,
                                               num_steps=num_steps, sim_method='libcall')
    r_b_bwd, times_b_bwd = trajectory_simulate(pmc, init_cond=saddle_below, t0=0.0, t1=time_end,
                                               num_steps=num_steps, sim_method='libcall')
    r_c_fwd, times_c_fwd = trajectory_simulate(pmc, init_cond=saddle_above, t0=0.0, t1=time_end,
                                               num_steps=num_steps, sim_method='libcall')

    for idx in xrange(num_pts):
        if idx < mid_a:
            traj_idx = idx
            r = r_a_fwd
        elif idx < mid_b:
            traj_idx = mid_b - idx
            r = r_b_bwd
        else:
            traj_idx = idx - mid_b
            r = r_c_fwd
        x, y, z = r[traj_idx, :]
        f_xyz_arr[idx] = (pmc.a * x + pmc.b * y + pmc.c * z) / Nval
        f_xy_arr[idx] = (pmc.a * x + pmc.b * y) / (Nval - z)
        s_xyz_arr[idx] = pmc.c / f_xyz_arr[idx] - 1
        s_xy_arr[idx] = pmc.c / f_xy_arr[idx] - 1
        z_arr[idx] = z
        y_arr[idx] = y
    return z_arr, s_xyz_arr, f_xyz_arr, y_arr
"""

def map_n_to_sf_idx(params, z_arr, s_xyz_arr, f_xyz_arr, y_arr):
    Nval = int(params.N)
    z_of_n = np.zeros(Nval+1)
    s_of_n = np.zeros(Nval+1)
    f_of_n = np.zeros(Nval+1)
    y_of_n = np.zeros(Nval+1)

    for n in range(Nval + 1):
        for idx_z, z in enumerate(z_arr):
            if idx_z == len(z_arr)-1 or n == Nval:
                print 'warning map_n_to_sf_idx edge case', n, 'zofn is', z_arr[-1]
                z_of_n[n] = z_arr[-1]
                s_of_n[n] = s_xyz_arr[-1]
                f_of_n[n] = f_xyz_arr[-1]
                y_of_n[n] = y_arr[-1]
                break
            elif z > n:
                z_of_n[n] = z_arr[idx_z]
                s_of_n[n] = s_xyz_arr[idx_z]
                f_of_n[n] = f_xyz_arr[idx_z]
                y_of_n[n] = y_arr[idx_z]
                break
    return z_of_n, s_of_n, f_of_n, y_of_n


def make_mastereqn_matrix(params, flag_zhat=True):
    n = params.N
    if n > 1e5:
        print 'Warning large N', n

    # BL g100 supported currently
    assert params.b in [0.8, 1.2]
    if params.b == 0.8:
        if params.mult_inc == 1.0 or params.feedback == 'constant':
            y_0_frac = 14.585869420527702 / 100.0
        elif params.mult_inc == 4.0 and params.feedback != 'constant':
            y_0_frac = 14.89695736662967 / 100.0
        else:
            assert params.mult_inc == 100.0 and params.feedback != 'constant'
            y_0_frac = 22.471588735222426 / 100.0
    else:
        if params.mult_inc == 1.0 or params.feedback == 'constant':
            y_0_frac = 19.262827700935464 / 100.0
        elif params.mult_inc == 4.0 and params.feedback != 'constant':
            y_0_frac = 19.652759713453463 / 100.0
        else:
            assert params.mult_inc == 100.0 and params.feedback != 'constant'
            y_0_frac = 28.094892239342407 / 100.0

    f_arr, s_arr, z_arr, y_arr = get_centermanifold_traj(params, norm=False)
    z_of_n, s_of_n, f_of_n, y_of_n = map_n_to_sf_idx(params, z_arr, s_arr, f_arr, y_arr)

    statespace = int(n + 1)
    if flag_zhat:
        statespace += 1

    W = np.zeros((statespace, statespace))
    if flag_zhat:
        for i in xrange(statespace):
            for j in xrange(statespace-1):
                if i == n + 1:
                    W[i, j] = params.mu * z_of_n[j]
                elif j == 0 and i == 1:
                    W[i, j] = params.mu * y_0_frac * n
                elif j == n and i == n-1:
                    W[i, j] = 0
                elif j == n - 1 and i == n:
                    W[i, j] = (1 + s_of_n[j]) * j + 1 * params.mu * y_of_n[j]
                else:
                    if i == j + 1:
                        W[i, j] = (1 + s_of_n[j]) * j + 1 * params.mu * y_of_n[j]
                    elif i == j - 1:
                        W[i, j] = 1 * j
                    else:
                        continue
    else:
        for i in xrange(statespace):
            for j in xrange(statespace):
                if j == 0 and i == 1:
                    W[i, j] = params.mu * y_0_frac * n
                elif j == n and i == n - 1:
                    W[i, j] = 0
                elif j == n - 1 and i == n:
                    W[i, j] = (1 + s_of_n[j]) * j + 1 * params.mu * y_of_n[j]
                    print n, i, j, W[i,j], (1 + s_of_n[j]) * j, params.mu * y_of_n[j]
                else:
                    if i == j + 1:
                        W[i, j] = (1 + s_of_n[j]) * j + 1 * params.mu * y_of_n[j]
                    elif i == j - 1:
                        W[i, j] = 1 * j
                    else:
                        continue
    for d in xrange(statespace):
        W[d, d] = - np.sum(W[:,d]) + W[d,d]  # add diagonal back in case it was not zero after for loops
    print W[-4:, -4:]
    return W


def linalg_mfpt(W=None, params=None, flag_zhat=False):
    if W is None:
        W = make_mastereqn_matrix(params, flag_zhat=flag_zhat)
    W_tilde = W[:-1, :-1]
    inv_W_tilde = np.linalg.inv(W_tilde.T)
    tau_vec = -1 * np.dot(inv_W_tilde, np.ones(len(W[0,:]) - 1))
    return tau_vec


def sort_D_V(A):
    D, V = np.linalg.eig(A)
    D_ranks = np.argsort(D)[::-1]
    D_sorted = D[D_ranks]
    V_sorted = V[:, D_ranks]
    return D_sorted, V_sorted


if __name__ == '__main__':
    param_varying_name = "N"
    assert param_varying_name == "N"

    # DYNAMICS PARAMETERS
    system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "tanh"  # "constant", "hill", "step", "pwlinear"
    params_dict = {
        'alpha_plus': 0.2,
        'alpha_minus': 1.0,  # 0.5
        'mu': 1e-4,  # 0.01
        'a': 1.0,
        'b': 0.8,
        'c': 0.9,  # 1.2
        'N': 100.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0,
        'mult_inc': 1.0,
        'mult_dec': 1.0,
    }
    params = Params(params_dict, system, feedback=feedback)

    N_range = [int(a) for a in np.logspace(1.50515, 4.13159, num=11)] + [int(a) for a in np.logspace(4.8, 7, num=4)]

    tau_guess_n0 = np.zeros(len(N_range))
    tau_guess_n1 = np.zeros(len(N_range))
    tau_guess_eval = np.zeros(len(N_range))
    for idx, n in enumerate(N_range[0:10]):
        print idx, n
        pmc = params.mod_copy({'N': n})

        W = make_mastereqn_matrix(pmc)
        D, V = sort_D_V(W)
        plt.plot(V[:, 0], label=0)
        plt.plot(V[:, 1], label=1)
        plt.legend()
        plt.show()

        tau_vec = linalg_mfpt(W=W)
        plt.plot(tau_vec[1:])
        plt.show()
        print "tau guess n=0", tau_vec[0], np.log10(tau_vec[0])
        print "tau guess n=1", tau_vec[1], np.log10(tau_vec[1])
        print "tau guess n=2", tau_vec[2], np.log10(tau_vec[2])
        print "compare eval 1", 1/(D[1]), np.log10(-1/(D[1]))

        tau_guess_n0[idx] = tau_vec[0]
        tau_guess_n1[idx] = tau_vec[1]
        tau_guess_eval[idx] = -1/(D[1])
    plt.plot(N_range, tau_guess_n0, '--x', label='mfpt_n0')
    plt.plot(N_range, tau_guess_n1, '--x', label='mfpt_n1')
    plt.plot(N_range, tau_guess_eval, '--o', label='mfpt_eval')
    plt.ylim(0.5, 2*1e6)
    plt.gca().set_yscale("log")
    plt.gca().set_xscale("log")
    plt.xlabel(r'$N$')
    plt.legend()
    plt.show()
