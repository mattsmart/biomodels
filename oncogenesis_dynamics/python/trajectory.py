import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from os import sep

#import formulae
#from formulae import fp_location_fsolve, jacobian_numerical_2d, get_physical_and_stable_fp, simulate_dynamics_general
from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, NUM_TRAJ, INIT_COND, TIME_START, TIME_END, NUM_STEPS, SIM_METHOD
from params import Params
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_simplex, plot_trajectory, plot_table_params

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
                        sim_method=SIM_METHOD, flag_showplt=False, flag_saveplt=True, flag_info=False, flag_table=False,
                        plt_save="trajectory"):
    from formulae import simulate_dynamics_general
    # params is "Params" class object
    # SIMULATE SETUP
    display_spacing = int(num_steps / 10)
    times = np.linspace(t0, t1, num_steps + 1)
    if flag_info:
        trajectory_infoprint(init_cond, t0, t1, num_steps, params)

    # SIMULATE
    r, times = simulate_dynamics_general(init_cond, times, params, method=sim_method)
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
        ax_traj = plot_trajectory(r, times, params, flag_show=flag_showplt, flag_save=flag_saveplt, plt_save=plt_save, flag_table=flag_table)
        ax_mono_z = plot_trajectory_mono(r, times, params, flag_showplt, flag_saveplt, plt_save=plt_save + "_mono", flag_table=flag_table)
    else:
        ax_traj = None
        ax_mono_z = None
    return r, times, ax_traj, ax_mono_z


def phase_portrait(params, num_traj=NUM_TRAJ, sim_method=SIM_METHOD, output_dir=OUTPUT_DIR, figname_mod="",
                   show_flag=False, basins_flag=False, flag_table=False):
    # IMPORTS
    from formulae import fp_location_fsolve, jacobian_numerical_2d, get_physical_and_stable_fp
    # GET TRAJECTORIES
    init_conds = np.zeros((num_traj, 3))
    N = params.N
    for k in xrange(num_traj):
        ak = N*np.random.random_sample()
        bk = (N-ak)*np.random.random_sample()
        ck = N - ak - bk
        init_cond = [ak, bk, ck]
        init_conds[k,:] = np.array(init_cond)

    all_fps = fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=35, tol=10e-1)
    sorted_fps = sorted(get_physical_and_stable_fp(params), key=itemgetter(2))
    print "STABLE", sorted_fps
    for fp in all_fps:
        J = jacobian_numerical_2d(params, fp[0:2])
        eigenvalues, V = np.linalg.eig(J)
        print fp, eigenvalues

    plt_title = "Phase portrait (%d traj) System: %s" % (num_traj, params.system)
    plt_save = output_dir + sep + "trajectory_simplex_multi%s.png" % figname_mod
    if basins_flag:
        plt_title = "Basins of attraction (%d traj) System: %s" % (num_traj, params.system)
        plt_save = output_dir + sep + "trajectory_simplex_basins%s.png" % figname_mod

    fig_traj = plot_simplex(N)
    ax_traj = fig_traj.gca()
    ax_traj.view_init(5, 35)  # ax.view_init(-45, -15)
    for idx, init_cond in enumerate(init_conds):
        r, times, _, _ = trajectory_simulate(params, init_cond=init_cond, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                                             sim_method=sim_method, flag_showplt=False, flag_saveplt=False)
        if basins_flag:
            endpt = r[-1,:]
            for idx, fp in enumerate(sorted_fps):
                if np.linalg.norm(endpt - fp) <= 10-2:  # check if trajectory went to that fp
                    ax_traj.plot(r[:, 0], r[:, 1], r[:, 2], label='trajectory', color=BASIN_COLOUR_DICT[idx])
        else:
            ax_traj.plot(r[:, 0], r[:, 1], r[:, 2], label='trajectory')
        #assert np.abs(np.sum(r[-1, :]) - N) <= 0.001

    # plot the fixed points
    for fp in sorted_fps:
        plt.plot([fp[0]], [fp[1]], [fp[2]], marker='o', markersize=5, markeredgecolor='black', linewidth='3', color='yellow')

    plt.title(plt_title)

    if flag_table:
        plot_table_params(ax_traj, params, loc='best', bbox=(1.1, 0.2, 0.1, 0.75))
    plt.savefig(plt_save, bbox_inches='tight')
    if show_flag:
        plt.show()
    return plt.gca()


if __name__ == "__main__":
    # MAIN RUN OPTIONS
    run_singletraj = True
    run_multitraj = False
    run_phaseportrait = True

    # PLOTTING OPTIONS
    BASIN_COLOUR_DICT = {0: 'blue', 1: 'red', 2: 'green'}
    flag_showplt = True
    flag_table = True

    # DYNAMICS PARAMETERS
    system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "hill"      # "constant", "hill", "step", "pwlinear"
    params_dict = {
        'alpha_plus': 0.2,
        'alpha_minus': 0.5,  # 0.5
        'mu': 0.001,  # 0.01
        'a': 1.0,
        'b': 0.8,
        'c': 0.85,  # 1.2
        'N': 10000.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0
    }
    params = Params(params_dict, system, feedback=feedback)

    ic_allx = np.zeros(params.numstates, dtype=int)
    ic_allx[0] = int(params.N)
    ic_mixed = [0.8*params.N, 0.1*params.N, 0.1*params.N]  #TODO generalize with init cond builder fn

    if run_singletraj:
        trajectory_simulate(params, init_cond=ic_mixed, t1=2000, plt_save='singletraj', flag_showplt=flag_showplt,
                            flag_table=flag_table)

    if run_multitraj:
        param_vary = "c"
        for pv in [0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03]:
            params_step = params.mod_copy({param_vary: pv})
            fmname = "trajectory_main_%s=%.3f" % (param_vary, pv)
            trajectory_simulate(params_step, init_cond=ic_mixed, t1=2000, plt_save=fmname, flag_showplt=flag_showplt,
                                flag_table=flag_table)

    if run_phaseportrait:
        phase_portrait(params, num_traj=280, show_flag=True, basins_flag=True, flag_table=flag_table)
