import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from formulae import bifurc_value, fp_from_timeseries
from params import Params
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_table_params
from trajectory import trajectory_simulate

if __name__ == '__main__':
    # PARAM TO VARY
    param_varying_name = "mu"
    if param_varying_name == "c":
        flag_log = False
        assert param_varying_name in PARAMS_ID_INV.keys()
        search_start = 0.7
        search_end = 1.02
        search_amount = 100  #20
    elif param_varying_name == "b":
        flag_log = False
        assert param_varying_name in PARAMS_ID_INV.keys()
        search_start = 0.8
        search_end = 1.2
        search_amount = 100  #20
    else:  # assume "mu" in general
        flag_log = True
        assert param_varying_name in PARAMS_ID_INV.keys()
        search_start = 1e-5  #0.55
        search_end = 1e-1   #1.45
        search_amount = 80  #20

    if flag_log:
        param_varying_values = np.logspace(np.log10(search_start), np.log10(search_end), num=search_amount)  # log axis
    else:
        param_varying_values = np.linspace(search_start, search_end, search_amount)

    # SCRIPT PARAMS
    sim_method = "libcall"  # see constants.py -- sim_methods_valid
    time_start = 0.0
    time_end = 10*16000.0  #20.0
    num_steps = 2000  # number of timesteps in each trajectory
    flag_table = True

    # DYNAMICS PARAMETERS
    system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "hill"  # "constant", "hill", "step", "pwlinear"
    params_dict = {
        'alpha_plus': 0.02,
        'alpha_minus': 0.1,  # 0.5
        'mu': 1e-4,  # 0.01
        'a': 1.0,
        'b': 0.92,
        'c': 0.99,  # 1.2
        'N': 100.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0
    }
    params = Params(params_dict, system, feedback=feedback)

    # OTHER PARAMETERS
    #init_cond = np.zeros(params.numstates, dtype=int)
    #init_cond[0] = int(params.N)
    init_cond = [98.0, 1.0, 1.0] #[99.9, 0.1, 0.0]


    # VARYING PARAM SPECIFICATION
    """
    idx_varying = PARAMS_ID_INV[param_varying_name]
    if param_varying_name in ["b", "c"]:
        param_varying_bifurcname = BIFURC_DICT[idx_varying]
        param_center = bifurc_value(params, param_varying_bifurcname)
        print "Bifurcation in %s possibly at %.5f" % (param_varying_bifurcname, param_center)
    else:
        param_center = params[idx_varying]
    print "Searching in window: %.8f to %.8f with %d points" \
          % (SEARCH_START * param_center, SEARCH_END * param_center, SEARCH_AMOUNT)
    """

    # GET TRAJECTORIES
    ax_comp = None
    r_inf_list = np.zeros((len(param_varying_values), 3))

    for idx, pv in enumerate(param_varying_values):
        params_step = params.mod_copy({param_varying_name: pv})
        r, times, ax_traj, ax_mono = trajectory_simulate(params_step, init_cond=init_cond, t0=time_start, t1=time_end, num_steps=num_steps,
                                                         sim_method=sim_method, flag_showplt=False, flag_saveplt=False)
        ax_mono = plot_trajectory_mono(r, times, params, False, False, ax_mono=ax_comp, mono="z")
        ax_comp = ax_mono
        #assert np.abs(np.sum(r[-1, :]) - N) <= 0.001
        if idx % 10 == 0:
            ax_comp.text(times[-1], r[-1, 2], '%.3f' % param_varying_values[idx])
        r_inf_list[idx] = fp_from_timeseries(r, sim_method)
    if flag_table:
        plot_table_params(ax_mono, params)
    plt.savefig(OUTPUT_DIR + sep + "trajectory_mono_z_composite" + ".png")
    plt.show()

    ax_endpts = plot_endpoint_mono(r_inf_list, param_varying_values, param_varying_name, params, True, True,
                                   all_axis=True, conv_to_fraction=True, flag_log=flag_log,
                                   plt_save="endpoint_varying_%s" % param_varying_name,
                                   flag_table=flag_table)
