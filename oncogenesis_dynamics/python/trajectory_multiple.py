import matplotlib.pyplot as plt
import numpy as np
import random
from operator import itemgetter
from os import sep


from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, NUM_TRAJ, TIME_START, TIME_END, NUM_STEPS, SIM_METHOD
from formulae import bifurc_value, fp_from_timeseries, get_physical_and_stable_fp
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_simplex, plot_trajectory
from trajectory import trajectory_simulate


BASIN_COLOUR_DICT = {0: 'blue', 1: 'red', 2:'green'}


def phase_portrait(params, system, num_traj=NUM_TRAJ, sim_method=SIM_METHOD, output_dir=OUTPUT_DIR, figname_mod="", show_flag=False, basins_flag=False):
    # GET TRAJECTORIES
    init_conds = np.zeros((num_traj, 3))
    for k in xrange(num_traj):
        ak = N*np.random.random_sample()
        bk = (N-ak)*np.random.random_sample()
        ck = N - ak - bk
        init_cond = [ak, bk, ck]
        init_conds[k,:] = np.array(init_cond)

    sorted_fps = sorted(get_physical_and_stable_fp(params, system), key=itemgetter(2))

    plt_title = "Phase portrait (%d traj) System: %s" % (num_traj, system)
    plt_save = output_dir + sep + "trajectory_simplex_multi%s.png" % figname_mod
    if basins_flag:
        plt_title = "Basins of attraction (%d traj) System: %s" % (num_traj, system)
        plt_save = output_dir + sep + "trajectory_simplex_basins%s.png" % figname_mod

    fig_traj = plot_simplex(N)
    ax_traj = fig_traj.gca()
    ax_traj.view_init(5, 35)  # ax.view_init(-45, -15)
    for idx, init_cond in enumerate(init_conds):
        r, times, _, _ = trajectory_simulate(params, system, init_cond=init_cond, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
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
    # CREATE TABLE OF PARAMS
    # bbox is x0, y0, height, width
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals, colWidths=[0.1]*3, rowLabels=row_labels, loc='best',
                            bbox=(1.1, 0.2, 0.1, 0.75))
    plt.savefig(plt_save, bbox_inches='tight')
    if show_flag:
        plt.show()
    return plt.gca()


if __name__ == "__main__":
    # SCRIPT PARAMETERS
    system = "feedback_z"  # "default" or "feedback_z" etc

    # DYNAMICS PARAMETERS
    alpha_plus = 0.2  # 0.05 #0.4
    alpha_minus = 0.5  # 4.95 #0.5
    mu = 0.001  # 0.01
    a = 1.0
    b = 0.6
    c = 0.6  # 2.6 #1.2
    N = 100.0  # 100
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

    """
    param_vary = 'c'
    pv_vals = np.linspace(0.60,1.0,10)
    print pv_vals
    output_dir = OUTPUT_DIR + sep + "simplex_during_bifurcation_b06_cvary"
    for pv in pv_vals:
        params_step = params
        params_step[PARAMS_ID_INV[param_vary]] = pv
        fmname = "_%.2fb_%.2fc" % (b,pv)
        phase_portrait(params_step, system, num_traj=800, output_dir=output_dir, figname_mod=fmname, basins_flag=True)
        phase_portrait(params_step, system, num_traj=20, output_dir=output_dir, figname_mod=fmname, basins_flag=False)
    """
    phase_portrait(params, system, num_traj=8, show_flag=True, basins_flag=False)
