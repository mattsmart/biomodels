import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from os import sep

import formulae
from constants import OUTPUT_DIR, PARAMS_ID_INV, INIT_COND, TIME_START, TIME_END, NUM_STEPS, SIM_METHOD
from params import Params
from plotting import plot_simplex, plot_trajectory_mono, plot_trajectory


# MATPLOTLIB GLOBAL SETTINGS
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(mpl_params)


def trajectory_infoprint(init_cond, t0, t1, num_steps, params):
    # params is class
    times = np.linspace(t0, t1, num_steps + 1)
    print "ODE Setup: t0, t1:", t0, t1, "| num_steps, dt:", num_steps, times[1] - times[0]
    print "Init Cond:", init_cond
    params.printer()


def trajectory_simulate(params, init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                        sim_method=SIM_METHOD, flag_showplt=False, flag_saveplt=True, flag_info=False, plt_save="trajectory"):
    # params is "Params" class object
    # SIMULATE SETUP
    display_spacing = int(num_steps / 10)
    times = np.linspace(t0, t1, num_steps + 1)
    if flag_info:
        trajectory_infoprint(init_cond, t0, t1, num_steps, params)

    # SIMULATE
    r, times = formulae.simulate_dynamics_general(init_cond, times, params, method=sim_method)
    if flag_info:
        print 'Done trajectory\n'

    # FP COMPARISON
    """
    if v_x == 0 and v_y == 0 and v_z == 0:
        solver_numeric = False
    else:
        solver_numeric = True
    predicted_fps = fp_location_general(params, ODE_SYSTEM, solver_numeric=solver_numeric, solver_fast=False)
    print "Predicted FPs:"
    for i in xrange(3):
        print "FP", i, predicted_fps[i], "Stable:", is_stable(params, predicted_fps[i])
    """

    # PLOTTING
    if flag_showplt or flag_saveplt:
        ax_traj = plot_trajectory(r, times, params.N, flag_show=flag_showplt, flag_save=flag_saveplt, plt_save=plt_save)
        ax_mono_z = plot_trajectory_mono(r, times, flag_showplt, flag_saveplt, plt_save=plt_save + "_mono")
    else:
        ax_traj = None
        ax_mono_z = None
    return r, times, ax_traj, ax_mono_z


if __name__ == "__main__":
    # SCRIPT PARAMETERS
    system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "hill"      # "constant", "hill", "step", "pwlinear"

    # DYNAMICS PARAMETERS
    params_dict = {
        'alpha_plus': 0.2,
        'alpha_minus': 0.5,  # 0.5
        'mu': 0.001,  # 0.01
        'a': 1.0,
        'b': 0.8,
        'c': 0.95,  # 1.2
        'N': 10000.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0
    }
    params = Params(params_dict, system, feedback=feedback)

    init_cond = np.zeros(params.numstates, dtype=int)
    init_cond[0] = int(params.N)

    """
    trajectory_simulate(params, system)
    """

    ic_mixed = [0.8*params.N, 0.1*params.N, 0.1*params.N]  #TODO generalize with init cond builder fn
    param_vary = "c"
    for pv in [0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03]:
        params_step = params.mod_copy({param_vary: pv})
        fmname = "trajectory_main_%s=%.3f" % (param_vary, pv)
        trajectory_simulate(params_step, init_cond=ic_mixed, t1=2000, plt_save=fmname, flag_showplt=True)
