import matplotlib.pyplot as plt
import numpy as np
import random
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, NUM_TRAJ, TIME_START, TIME_END, NUM_STEPS, SIM_METHOD
from formulae import bifurc_value, fp_from_timeseries
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_simplex, plot_trajectory
from trajectory import trajectory_simulate


def phase_portrait(params, system, num_traj=NUM_TRAJ, sim_method=SIM_METHOD, figname_mod=""):
    # GET TRAJECTORIES
    init_conds = np.zeros((num_traj, 3))
    for k in xrange(num_traj):
        ak = N*np.random.random_sample()
        bk = (N-ak)*np.random.random_sample()
        ck = N - ak - bk
        init_cond = [ak, bk, ck]
        init_conds[k,:] = np.array(init_cond)

    fig_traj = plot_simplex(N)
    ax_traj = fig_traj.gca()
    ax_traj.view_init(5, 35)  # ax.view_init(-45, -15)
    for idx, init_cond in enumerate(init_conds):
        r, times, _, _ = trajectory_simulate(params, system, init_cond=init_cond, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                                             sim_method=sim_method, flag_showplt=False, flag_saveplt=False)
        ax_traj.plot(r[:, 0], r[:, 1], r[:, 2], label='trajectory')
        #assert np.abs(np.sum(r[-1, :]) - N) <= 0.001
    plt.title('Phase portrait (%d traj) System: %s' % (num_traj, system))
    # CREATE TABLE OF PARAMS
    # bbox is x0, y0, height, width
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals, colWidths=[0.1]*3, rowLabels=row_labels, loc='best',
                            bbox=(1.1, 0.2, 0.1, 0.75))
    plt.savefig(OUTPUT_DIR + sep + "trajectory_simplex_multi%s.png" % figname_mod, bbox_inches='tight')
    return plt.gca()


if __name__ == "__main__":
    # SCRIPT PARAMETERS
    system = "feedback_z"  # "default" or "feedback_z" etc

    # DYNAMICS PARAMETERS
    alpha_plus = 0.2  # 0.05 #0.4
    alpha_minus = 0.5  # 4.95 #0.5
    mu = 0.001  # 0.01
    a = 1.0
    b = 0.8
    c = 0.81  # 2.6 #1.2
    N = 100.0  # 100
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

    """
    phase_portrait(params, system, figname_mod="_main")
    """

    param_vary = "c"
    for pv in [0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99, 1.01, 1.03]:
        params_step = params
        params_step[PARAMS_ID_INV[param_vary]] = pv
        fmname = "_main_%s=%.3f" % (param_vary, pv)
        phase_portrait(params, system, figname_mod=fmname)
