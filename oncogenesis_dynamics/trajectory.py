import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

from formulae import fp_location_fsolve, jacobian_numerical_2d, get_physical_fp_stable_and_not, simulate_dynamics_general, \
                     fp_location_general, is_stable, map_init_name_to_init_cond
from constants import BASIN_COLOUR_DICT, NUM_TRAJ, INIT_COND, TIME_START, TIME_END, NUM_STEPS, SIM_METHOD
from params import Params
from plotting import plot_handler, plot_options_build, plot_trajectory_mono, plot_simplex, plot_trajectory


def trajectory_infoprint(init_cond, t0, t1, num_steps, params):
    times = np.linspace(t0, t1, num_steps + 1)
    print "ODE Setup: t0, t1:", t0, t1, "| num_steps, dt:", num_steps, times[1] - times[0]
    print "Init Cond:", init_cond
    params.printer()


def trajectory_simulate(params, init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                        sim_method=SIM_METHOD, flag_info=False, fp_comparison=False):

    times = np.linspace(t0, t1, num_steps + 1)
    if flag_info:
        trajectory_infoprint(init_cond, t0, t1, num_steps, params)

    r, times = simulate_dynamics_general(init_cond, times, params, method=sim_method)
    if flag_info:
        print 'Done trajectory\n'

    if fp_comparison:
        predicted_fps = fp_location_general(params, solver_fsolve=True)
        print "Predicted FPs:"
        for i in xrange(3):
            print "FP", i, predicted_fps[i], "Stable:", is_stable(params, predicted_fps[i])
    return r, times


def phase_portrait(params, num_traj=NUM_TRAJ, sim_method=SIM_METHOD, figname_mod="", basins_flag=False, **plot_options):

    init_conds = np.zeros((num_traj, 3))
    N = params.N
    for k in xrange(num_traj):
        init_conds[k, :] = np.array(map_init_name_to_init_cond(params, "random"))

    stable_fps = []
    unstable_fps = []
    all_fps = fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=35, tol=10e-1, buffer=True)
    for fp in all_fps:
        J = jacobian_numerical_2d(params, fp[0:2])
        eigenvalues, V = np.linalg.eig(J)
        if eigenvalues[0] < 0 and eigenvalues[1] < 0:
            print "Stable FP:", fp, "Evals:", eigenvalues
            stable_fps.append(fp)
        else:
            print "Unstable FP:", fp, "Evals:", eigenvalues
            unstable_fps.append(fp)

    plt_title = "Phase portrait (%d traj) System: %s" % (num_traj, params.system)
    plt_save = "trajectory_simplex_multi%s" % figname_mod
    if basins_flag:
        plt_title = "Basins of attraction (%d traj) System: %s" % (num_traj, params.system)
        plt_save = "trajectory_simplex_basins%s" % figname_mod

    fig_traj = plot_simplex(N)
    ax_traj = fig_traj.gca()
    ax_traj.view_init(5, 35)  # ax.view_init(-45, -15)
    plt.title(plt_title)
    for idx, init_cond in enumerate(init_conds):
        r, times, = trajectory_simulate(params, init_cond=init_cond, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS, sim_method=sim_method)
        if basins_flag:
            endpt = r[-1,:]
            for idx, fp in enumerate(all_fps):
                if np.linalg.norm(endpt - fp) <= 10-2:  # check if trajectory went to that fp
                    ax_traj.plot(r[:, 0], r[:, 1], r[:, 2], label='trajectory', color=BASIN_COLOUR_DICT[idx])
        else:
            ax_traj.plot(r[:, 0], r[:, 1], r[:, 2], label='trajectory')
        #assert np.abs(np.sum(r[-1, :]) - N) <= 0.001

    # plot the fixed points
    for fp in stable_fps:
        plt.plot([fp[0]], [fp[1]], [fp[2]], marker='o', markersize=5, markeredgecolor='black', linewidth='3', color='yellow')
    for fp in unstable_fps:
        plt.plot([fp[0]], [fp[1]], [fp[2]], marker='o', markersize=5, markeredgecolor='black', linewidth='3', color='red')

    plot_options['plt_save'] = plt_save
    plot_options['bbox_inches'] = 'tight'
    plot_options['loc_table'] = 'best'
    fig_traj, ax_traj = plot_handler(fig_traj, ax_traj, params, plot_options=plot_options)

    return ax_traj


def conserved_quantity(state, params):
    # TODO doesn't seem to be working
    # did dy/dx = g/f then solve as exact eqn Psi_y = -f, Psi_x = g so Psi dot F = 0
    x, y, z = state
    p = params
    assert p.system == 'default'
    q1 = (p.c - p.a) / p.N
    q2 = (p.c - p.b) / p.N
    xy_factor = p.alpha_plus + p.N * q1
    val = - (q1 * x**2 * y) - 0.5 * (q2 * x * y ** 2) + x * y * xy_factor + 0.5 * (p.alpha_plus * x ** 2 - p.alpha_minus * y ** 2)
    return val


if __name__ == "__main__":
    # MAIN RUN OPTIONS
    run_singletraj = False
    run_conserved = False
    plot_options_traj = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajectory")
    run_multitraj = False
    plot_options_multitraj = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajmulti")
    run_phaseportrait = False
    plot_options_trajportrait = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajportrait")
    run_multiphaseportrait = True
    plot_options_mulyitrajportrait = plot_options_build(flag_table=True, flag_show=True, flag_save=True,
                                                   plt_save="trajportrait")

    # PLOTTING OPTIONS
    sim_method = "libcall"
    num_steps = NUM_STEPS
    basins_flag = False

    # DYNAMICS PARAMETERS
    system = "default"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "constant"      # "constant", "hill", "step", "pwlinear"
    params_dict = {
        'alpha_plus': 0.2,
        'alpha_minus': 0.5,  # 0.5
        'mu': 0.001,  # 0.01
        'a': 1.0,
        'b': 1.075,
        'c': 0.8,  # 1.2
        'N': 100.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0
    }
    params = Params(params_dict, system, feedback=feedback)
    init_cond = map_init_name_to_init_cond(params, "mixed")

    if run_singletraj:

        r, times = trajectory_simulate(params, init_cond=init_cond, t1=2000, num_steps=num_steps,
                                       sim_method=sim_method)
        ax_traj = plot_trajectory(r, times, params, fig_traj=None, **plot_options_traj)
        ax_mono_x = plot_trajectory_mono(r, times, params, mono="x", **plot_options_traj)
        ax_mono_y = plot_trajectory_mono(r, times, params, mono="y", **plot_options_traj)
        ax_mono_z = plot_trajectory_mono(r, times, params, mono="z", **plot_options_traj)
        if run_conserved:
            for idx in xrange(r.shape[0]):
                state = r[idx, :]
                print times[idx], "state:", state, "and quantity:",  conserved_quantity(state, params)

    if run_multitraj:
        param_vary = "c"
        for pv in [0.975, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.08]:
            params_step = params.mod_copy({param_vary: pv})
            fmname = "trajectory_main_%s=%.3f" % (param_vary, pv)
            plot_options_multitraj['plt_save'] = fmname
            trajectory_simulate(params_step, init_cond=init_cond, t1=2000, **plot_options_multitraj)

    if run_phaseportrait:
        phase_portrait(params, num_traj=70, show_flag=True, basins_flag=False, **plot_options_trajportrait)

    if run_multiphaseportrait:
        param_vary_name = 'c'
        param_vary_range = np.linspace(1.022,1.025, 10)
        # fp_dict = {pv: [] for pv in param_vary_range}
        for i, pv in enumerate(param_vary_range):
            print "paramset %d, %s=%.5f" % (i, param_vary_name, pv)
            params_step = params.mod_copy({param_vary_name: pv})
            phase_portrait(params_step, num_traj=30, show_flag=True, basins_flag=False, **plot_options_mulyitrajportrait)
