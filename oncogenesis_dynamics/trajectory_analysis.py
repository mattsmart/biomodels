import matplotlib.pyplot as plt
import numpy as np
import os

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from data_io import read_varying_mean_sd_fpt_and_params
from params import Params
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_table_params
from trajectory import trajectory_simulate



def corner_to_flux(corner, params):
    Z_FRACTIONS = {'BL': 0.000212,
                   'BR': 1.0,
                   'TL': 0.000141,
                   'TR': 0.507754,
                   'BL100g': 0.0004085,  # note saddle is at 18.98%
                   'TR100g': 0.16514,
                   'BL1g': 0.00020599,
                   'TR1g': 1.0}
    z_fp = Z_FRACTIONS[corner] * params.N  # number entering zhat state per unit time
    MU_1 = 0.0001
    avg_flux = 1/(z_fp * MU_1)
    print "todo fix manual corner_to_flux mu_1", MU_1
    return avg_flux


def compute_heuristic_mfpt(params):

    # SCRIPT PARAMS
    sim_method = "libcall"  # see constants.py -- sim_methods_valid
    time_start = 0.0
    time_end = 16000.0  #20.0
    num_steps = 200000  # number of timesteps in each trajectory

    init_cond = [params.N, 0, 0]
    r, times = trajectory_simulate(params, init_cond=init_cond, t0=time_start, t1=time_end, num_steps=num_steps,
                                   sim_method=sim_method)
    plt.plot(times, r[:,2])
    plt.show()
    # compute integral numerically
    z_of_t = r[:, 2]
    "I = INT_0_inf t N mu z e^(-INT_0_TAU N mu z(t') dt') dt     <- compare vs 1/(mu z_fp)"

    dt = times[1] - times[0]
    expweights = np.zeros(len(times))
    last_expweight = 0
    for idx in xrange(times.shape[0]):
        expweights[idx] = last_expweight
        expweights[idx] += params.mu * dt * z_of_t[idx]
        last_expweight = expweights[idx]
        #weight[i] = params.N * params.mu * dt * np.dot(times[0:idx], z_of_t[0:idx])  big oneline repeat dot

    normalization = 0
    for idx in xrange(times.shape[0]):
        normalization += dt * np.exp(-expweights[idx]) * params.mu * z_of_t[idx]

    mfpt = 0
    for idx in xrange(times.shape[0]):
        mfpt += dt * np.exp(-expweights[idx]) * params.mu * z_of_t[idx] * times[idx]
    mfpt = mfpt / normalization
    print mfpt, normalization
    return mfpt


def plot_heuristic_mfpt(N_range, curve_heuristic, param_vary_name, param_set, params,
                        show_flag=False, figname_mod="", outdir=OUTPUT_DIR, fs=20, ax=None):
    # load data to compare against
    dataid = 'TR100g'
    mfpt_data_dir = 'data' + os.sep + 'mfpt' + os.sep + 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g100'
    mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
        read_varying_mean_sd_fpt_and_params(mfpt_data_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N.txt',
                                            mfpt_data_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N_params.csv')

    curve_fpflux = [corner_to_flux(dataid, params.mod_copy({'N':n})) for n in N_range]
    print 'using flux TR for plot_heuristic_mfpt'

    plt.plot(N_range, curve_fpflux, '--k', label='curve_fpflux')
    plt.plot(N_range, curve_heuristic, '-or', label='curve_heuristic')
    plt.plot(N_range, mean_fpt_varying, '-ok', label='data')


    ax = plt.gca()
    ax.set_xlabel(r'$%s$' % param_vary_name, fontsize=fs)
    ax.set_ylabel(r'$\tau$', fontsize=fs)
    #ax.set_ylabel(r'$\langle\tau\rangle$', fontsize=fs)
    #ax.set_ylabel(r'$\delta\tau$', fontsize=fs)

    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)

    # log options
    flag_xlog10 = True
    flag_ylog10 = True
    if flag_xlog10:
        #ax.set_xscale("log", nonposx='clip')
        ax.set_xscale("log")
        #ax_dual.set_xscale("log", nonposx='clip')
        ax.set_xlim([np.min(param_set)*0.9, 1.5*1e4])
    if flag_ylog10:
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")
        #ax_dual.set_yscale("log", nonposx='clip')
        ax.set_ylim([6*1e-1, 3*1e6])

    ax.legend(fontsize=fs-6, ncol=2, loc='upper right')
    plt_save = "mean_fpt_varying_heuristic_" + dataid
    plt.savefig(outdir + os.sep + plt_save + '.pdf', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


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
        'b': 1.2,
        'c': 1.1,  # 1.2
        'N': 100.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0,
        'mult_inc': 100.0,
        'mult_dec': 100.0,
    }
    params = Params(params_dict, system, feedback=feedback)

    N_range = [int(a) for a in np.logspace(1.50515, 4.13159, num=11)]
    # TODO more fine grained N?

    # OTHER PARAMETERS
    #init_cond = np.zeros(params.numstates, dtype=int)
    #init_cond[0] = int(params.N)

    curve_heuristic = [0]*len(N_range)
    for idx, N in enumerate(N_range):
        pv = params.mod_copy({'N': N})
        curve_heuristic[idx] = compute_heuristic_mfpt(pv)
        print N, curve_heuristic[idx]

    plot_heuristic_mfpt(N_range, curve_heuristic, 'N', N_range, params,
                        show_flag=False, figname_mod="", outdir=OUTPUT_DIR, fs=20)
