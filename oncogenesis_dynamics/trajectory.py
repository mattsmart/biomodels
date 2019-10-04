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


def get_centermanifold_traj(params, norm=False):
    sim_method = "libcall"  # see constants.py -- sim_methods_valid
    time_start = 0.0
    time_end = 200.0  # 20.0
    num_steps = 200  # number of timesteps in each trajectory

    if (params.b == 1.2 and (params.mult_inc == 1.0 or params.feedback == 'constant')):

        num_pts = 400
        z_arr = np.zeros(num_pts)
        y_arr = np.zeros(num_pts)
        s_xyz_arr = np.zeros(num_pts)
        s_xy_arr = np.zeros(num_pts)
        f_xyz_arr = np.zeros(num_pts)
        f_xy_arr = np.zeros(num_pts)

        r_fwd, times_fwd = trajectory_simulate(params, init_cond=[params.N, 0, 0], t0=time_start, t1=time_end,
                                               num_steps=num_pts, sim_method=sim_method)
        for idx in xrange(num_pts):
            traj_idx = idx
            r = r_fwd
            x, y, z = r[traj_idx, :]
            f_xyz_arr[idx] = (params.a * x + params.b * y + params.c * z) / params.N
            f_xy_arr[idx] = (params.a * x + params.b * y) / (params.N - z)
            s_xyz_arr[idx] = params.c / f_xyz_arr[idx] - 1
            s_xy_arr[idx] = params.c / f_xy_arr[idx] - 1
            z_arr[idx] = z
            y_arr[idx] = y

    elif (params.b == 1.2 and params.feedback != 'constant') or \
            (params.b == 0.8 and (params.mult_inc == 1.0 or params.feedback == 'constant')):

        num_pts = 400
        mid = 200
        z_arr = np.zeros(num_pts)
        y_arr = np.zeros(num_pts)
        s_xyz_arr = np.zeros(num_pts)
        s_xy_arr = np.zeros(num_pts)
        f_xyz_arr = np.zeros(num_pts)
        f_xy_arr = np.zeros(num_pts)

        r_fwd, times_fwd = trajectory_simulate(params, init_cond=[params.N, 0, 0], t0=time_start, t1=time_end,
                                               num_steps=num_steps, sim_method=sim_method)
        r_bwd, times_bwd = trajectory_simulate(params, init_cond=[0, 1e-1, params.N - 1e-1], t0=time_start,
                                               t1=time_end,
                                               num_steps=num_steps, sim_method=sim_method)
        for idx in xrange(num_pts):
            if idx > mid:
                traj_idx = num_pts - idx
                r = r_bwd
            else:
                traj_idx = idx
                r = r_fwd
            x, y, z = r[traj_idx, :]
            f_xyz_arr[idx] = (params.a * x + params.b * y + params.c * z) / params.N
            f_xy_arr[idx] = (params.a * x + params.b * y) / (params.N - z)
            s_xyz_arr[idx] = params.c / f_xyz_arr[idx] - 1
            s_xy_arr[idx] = params.c / f_xy_arr[idx] - 1
            z_arr[idx] = z
            y_arr[idx] = y
    else:
        assert params.b == 0.8
        if params.mult_inc == 100.0 and params.feedback != 'constant':
            saddle_below = np.array([40.62, 40.41, 18.97]) / 100.0 * params.N
            saddle_above = np.array([40.6, 40.4, 19.0]) / 100.0 * params.N
        else:
            assert params.mult_inc == 4.0 and params.feedback != 'constant'
            saddle_below = np.array([21.57844087406341, 21.54060213939143, 56.880956986545154]) / 100.0
            saddle_above = np.array([21.55844087406341, 21.52060213939143, 56.920956986545154]) / 100.0

        num_pts = 200 * 3
        mid_a = 200
        mid_b = 400
        z_arr = np.zeros(num_pts)
        y_arr = np.zeros(num_pts)
        s_xyz_arr = np.zeros(num_pts)
        s_xy_arr = np.zeros(num_pts)
        f_xyz_arr = np.zeros(num_pts)
        f_xy_arr = np.zeros(num_pts)

        r_a_fwd, times_a_fwd = trajectory_simulate(params, init_cond=[params.N, 0, 0], t0=time_start,
                                                   t1=time_end,
                                                   num_steps=num_steps, sim_method=sim_method)
        r_b_bwd, times_b_bwd = trajectory_simulate(params, init_cond=saddle_below, t0=time_start, t1=time_end,
                                                   num_steps=num_steps, sim_method=sim_method)
        r_c_fwd, times_c_fwd = trajectory_simulate(params, init_cond=saddle_above, t0=time_start, t1=time_end,
                                                   num_steps=num_steps, sim_method=sim_method)

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
            f_xyz_arr[idx] = (params.a * x + params.b * y + params.c * z) / params.N
            f_xy_arr[idx] = (params.a * x + params.b * y) / (params.N - z)
            s_xyz_arr[idx] = params.c / f_xyz_arr[idx] - 1
            s_xy_arr[idx] = params.c / f_xy_arr[idx] - 1
            z_arr[idx] = z
            y_arr[idx] = y
    if norm:
        z_arr = z_arr / params.N
        y_arr = y_arr / params.N
    return f_xyz_arr, s_xyz_arr, z_arr, y_arr


if __name__ == "__main__":
    # MAIN RUN OPTIONS
    run_singletraj = False
    run_conserved = False
    plot_options_traj = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajectory")
    run_multitraj = False
    plot_options_multitraj = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajmulti")
    run_phaseportrait = True
    plot_options_trajportrait = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajportrait")
    run_multiphaseportrait = False
    plot_options_mulyitrajportrait = plot_options_build(flag_table=True, flag_show=True, flag_save=True,
                                                   plt_save="trajportrait")
    get_fitness_curve = False

    # PLOTTING OPTIONS
    sim_method = "libcall"
    num_steps = NUM_STEPS
    basins_flag = False

    # DYNAMICS PARAMETERS
    system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
    feedback = "tanh"      # "constant", "hill", "step", "pwlinear"
    params_dict = {
        'alpha_plus': 0.2,
        'alpha_minus': 1.0,  # 0.5
        'mu': 0.0001,  # 0.01
        'a': 1.0,
        'b': 1.2,
        'c': 1.1,
        'N': 100.0,  # 100.0
        'v_x': 0.0,
        'v_y': 0.0,
        'v_z': 0.0,
        'mu_base': 0.0,
        'c2': 0.0,
        'v_z2': 0.0,
        'mult_inc': 1.0,
        'mult_dec': 1.0
    }
    params = Params(params_dict, system, feedback=feedback)
    init_cond = map_init_name_to_init_cond(params, "x_all")

    if run_singletraj:

        r, times = trajectory_simulate(params, init_cond=init_cond, t1=200, num_steps=num_steps,
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
        param_vary_range = np.linspace(0.999, 1.001, 5)
        # fp_dict = {pv: [] for pv in param_vary_range}
        for i, pv in enumerate(param_vary_range):
            print "\nparamset %d, %s=%.5f" % (i, param_vary_name, pv)
            params_step = params.mod_copy({param_vary_name: pv})
            phase_portrait(params_step, num_traj=30, show_flag=True, basins_flag=False, **plot_options_mulyitrajportrait)

    if get_fitness_curve:

        f_xyz_arr, s_xyz_arr, z_arr, y_arr = get_centermanifold_traj(params)

        plt.plot(s_xyz_arr, label='s')
        plt.plot(f_xyz_arr, label='f')
        plt.plot(z_arr, label='z')
        plt.plot(y_arr, label='y')
        plt.legend()
        plt.show()

        plt.plot(z_arr, s_xyz_arr, '--k', label=r'$s1 = c/f_{xyz} - 1$')
        #plt.plot(z_arr, s_xy_arr, '--b', label=r'$s2 = c/f_{xy} - 1$')
        plt.xlabel(r'$z/N$')
        plt.ylabel(r'$s$')
        plt.gca().axhline(0.0, linestyle='-', color='gray')
        plt.legend()
        plt.show()
