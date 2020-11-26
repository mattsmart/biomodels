import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

from formulae import fp_location_fsolve, jacobian_numerical_2d, get_physical_fp_stable_and_not, simulate_dynamics_general, \
                     fp_location_general, is_stable, map_init_name_to_init_cond
from constants import BASIN_COLOUR_DICT, NUM_TRAJ, INIT_COND, TIME_START, TIME_END, NUM_STEPS, SIM_METHOD
from params import Params
from presets import presets
from plotting import plot_handler, plot_options_build, plot_trajectory_mono, plot_simplex, plot_trajectory


def trajectory_infoprint(init_cond, t0, t1, num_steps, params):
    times = np.linspace(t0, t1, num_steps + 1)
    print "ODE Setup: t0, t1:", t0, t1, "| num_steps, dt:", num_steps, times[1] - times[0]
    print "Init Cond:", init_cond
    params.printer()


def trajectory_simulate(params, init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                        sim_method=SIM_METHOD, flag_info=False, fp_comparison=True):

    times = np.linspace(t0, t1, num_steps + 1)
    if flag_info:
        trajectory_infoprint(init_cond, t0, t1, num_steps, params)

    r, times = simulate_dynamics_general(init_cond, times, params, method=sim_method)
    if flag_info:
        print 'Done trajectory\n'

    if fp_comparison:
        predicted_fps = fp_location_general(params, solver_fsolve=True)
        print "Predicted FPs:"
        print predicted_fps
        for i in xrange(len(predicted_fps)):
            print "FP", i, predicted_fps[i], "Stable:", is_stable(params, predicted_fps[i][0:2])
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
        r, times, = trajectory_simulate(params, init_cond=init_cond, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                                        sim_method=sim_method, fp_comparison=False)
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


def get_centermanifold_traj(params, norm=False, force_region_1=False, force_region_2=False, prune=True):
    sim_method = "libcall"  # see constants.py -- sim_methods_valid
    time_start = 0.0
    time_end = 300.0  # 20.0
    num_steps = 400*2  # number of timesteps in each trajectory

    if (params.b == 1.2 and (params.mult_inc == 1.0 or params.feedback == 'constant')) or force_region_2:

        num_pts = 800*2
        z_arr = np.zeros(num_pts)
        y_arr = np.zeros(num_pts)
        s_xyz_arr = np.zeros(num_pts)
        s_xy_arr = np.zeros(num_pts)
        f_xyz_arr = np.zeros(num_pts)
        f_xy_arr = np.zeros(num_pts)

        r_fwd, times_fwd = trajectory_simulate(params, init_cond=[params.N, 0, 0], t0=time_start, t1=time_end,
                                               num_steps=num_pts, sim_method=sim_method, fp_comparison=False)
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
            (params.b == 0.8 and (params.mult_inc == 1.0 or params.feedback == 'constant')) or force_region_1:

        num_pts = 800*2
        mid = 400*2
        z_arr = np.zeros(num_pts)
        y_arr = np.zeros(num_pts)
        s_xyz_arr = np.zeros(num_pts)
        s_xy_arr = np.zeros(num_pts)
        f_xyz_arr = np.zeros(num_pts)
        f_xy_arr = np.zeros(num_pts)

        if params.b == 0.8:
            time_end_fwd = time_end * 0.2
            time_end_bwd = time_end * 2.0
        else:
            time_end_fwd = time_end * 0.8
            time_end_bwd = time_end
        r_fwd, times_fwd = trajectory_simulate(params, init_cond=[params.N, 0, 0], t0=time_start, t1=time_end_fwd,
                                               num_steps=num_steps, sim_method=sim_method, fp_comparison=False)
        r_bwd, times_bwd = trajectory_simulate(params, init_cond=[0, 1e-3, params.N - 1e-3], t0=time_start,
                                               t1=time_end_bwd, num_steps=num_steps, sim_method=sim_method, fp_comparison=False)
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
            saddle_below = np.array([21.57844087406341, 21.54060213939143, 56.880956986545154]) / 100.0 * params.N
            saddle_above = np.array([21.55844087406341, 21.52060213939143, 56.920956986545154]) / 100.0 * params.N

        num_pts = (400*2) * 3
        mid_a = 400*2
        mid_b = 800*2
        z_arr = np.zeros(num_pts)
        y_arr = np.zeros(num_pts)
        s_xyz_arr = np.zeros(num_pts)
        s_xy_arr = np.zeros(num_pts)
        f_xyz_arr = np.zeros(num_pts)
        f_xy_arr = np.zeros(num_pts)

        # 0.02, 0.7, 1.0

        r_a_fwd, times_a_fwd = trajectory_simulate(params, init_cond=[params.N, 0, 0], t0=time_start,
                                                   t1=time_end*0.4,
                                                   num_steps=num_steps, sim_method=sim_method, fp_comparison=False)
        r_b_bwd, times_b_bwd = trajectory_simulate(params, init_cond=saddle_below, t0=time_start, t1=time_end*1.1,
                                                   num_steps=num_steps, sim_method=sim_method, fp_comparison=False)
        r_c_fwd, times_c_fwd = trajectory_simulate(params, init_cond=saddle_above, t0=time_start, t1=time_end*1,
                                                   num_steps=num_steps, sim_method=sim_method, fp_comparison=False)

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

    if prune:
        tol = 1e-4
        # move into get SM with a flag to use (once done)
        assert norm
        f_xyz_arr_pruned = [f_xyz_arr[0]]
        s_xyz_arr_pruned = [s_xyz_arr[0]]
        z_arr_pruned = [z_arr[0]]
        y_arr_pruned = [y_arr[0]]
        orig_len = len(z_arr)

        for idx in xrange(orig_len):
            if np.abs(z_arr[idx] - z_arr_pruned[-1]) > tol or np.abs(y_arr[idx] - y_arr_pruned[-1]) > tol:
                f_xyz_arr_pruned.append(f_xyz_arr[idx])
                s_xyz_arr_pruned.append(s_xyz_arr[idx])
                z_arr_pruned.append(z_arr[idx])
                y_arr_pruned.append(y_arr[idx])
        f_xyz_arr = np.array(f_xyz_arr_pruned)
        s_xyz_arr = np.array(s_xyz_arr_pruned)
        z_arr = np.array(z_arr_pruned)
        y_arr = np.array(y_arr_pruned)

    return f_xyz_arr[10:], s_xyz_arr[10:], z_arr[10:], y_arr[10:]  # consider more automated truncation of the initial QSS


if __name__ == "__main__":
    # MAIN RUN OPTIONS
    run_singletraj = True
    run_conserved = False
    plot_options_traj = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajectory")
    run_multitraj = False
    plot_options_multitraj = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajmulti")
    run_phaseportrait = False
    plot_options_trajportrait = plot_options_build(flag_table=True, flag_show=True, flag_save=True, plt_save="trajportrait")
    run_multiphaseportrait = False
    plot_options_mulyitrajportrait = plot_options_build(flag_table=True, flag_show=True, flag_save=True,
                                                   plt_save="trajportrait")
    get_slowmanifold_curves = False
    N_vary_stochplots = False

    # PLOTTING OPTIONS
    sim_method = "libcall"  # libcall, rk4, euler, gillespie
    num_steps = NUM_STEPS
    T_END = 200
    basins_flag = False

    # DYNAMICS PARAMETERS
    preset = None  #'BL1g'

    if preset is None:
        system = "feedback_z"  # "default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"
        feedback = "step"      # "constant", "hill", "step", "pwlinear", "tanh"
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
            'mult_inc': 4.0,
            'mult_dec': 4.0
        }
        params = Params(params_dict, system, feedback=feedback)
    else:
        params = presets(preset)
    fregion1 = False  # set if b=1.0, c=0.9
    fregion2 = False  # set if b=1.0, c=1.1
    if params.b==1.0 and params.c==0.9:
        fregion1 = True
    if params.b==1.0 and params.c==1.1:
        fregion2 = True

    init_cond = map_init_name_to_init_cond(params, "x_all")

    if run_singletraj:

        r, times = trajectory_simulate(params, init_cond=init_cond, t1=T_END, num_steps=num_steps,
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
        phase_portrait(params, num_traj=70, show_flag=True, basins_flag=True, **plot_options_trajportrait)

    if run_multiphaseportrait:
        param_vary_name = 'c'
        param_vary_range = np.linspace(0.999, 1.001, 5)
        # fp_dict = {pv: [] for pv in param_vary_range}
        for i, pv in enumerate(param_vary_range):
            print "\nparamset %d, %s=%.5f" % (i, param_vary_name, pv)
            params_step = params.mod_copy({param_vary_name: pv})
            phase_portrait(params_step, num_traj=30, show_flag=True, basins_flag=False, **plot_options_mulyitrajportrait)

    if get_slowmanifold_curves:

        f_xyz_arr, s_xyz_arr, z_arr, y_arr = \
            get_centermanifold_traj(params, norm=True, force_region_1=False, force_region_2=False, prune=False)

        # x, y, z vs points
        plt.plot(z_arr, label='z')
        plt.plot(y_arr, label='y')
        plt.plot(1.0 - z_arr - y_arr, label='x')
        plt.legend()
        #plt.ylim(-0.1,0.1)
        plt.show()
        plt.close()

        # fancier z,y,z curves
        plt.plot(z_arr, 1.0 - y_arr - z_arr, '--b', label=r'$x(z)$')
        plt.plot(z_arr, y_arr, '--g', label=r'$y(z)$')
        plt.plot(z_arr, z_arr, '--k', label=r'$z$')
        # plt.plot(z_arr, s_xy_arr, '--b', label=r'$s2 = c/f_{xy} - 1$')
        plt.xlabel(r'$z$')
        plt.ylabel('pop')
        plt.gca().axhline(0.0, linestyle='-', color='gray')
        plt.legend()
        plt.show()
        plt.close()

        # s, f vs points
        plt.plot(s_xyz_arr, label='s')
        plt.plot(f_xyz_arr, label='f')
        plt.legend()
        plt.ylim(-0.1, 0.1)
        plt.show()
        plt.close()

        # fancier s, f curves
        plt.plot(z_arr[0:2000], s_xyz_arr[0:2000], '--k', label=r'$s(z) = c/f_{xyz} - 1$')
        #plt.plot(z_arr, f_xyz_arr, '--b', label=r'$f(z)$')
        plt.xlabel(r'$z/N$')
        plt.ylabel(r'$s$')
        plt.gca().axhline(0.0, linestyle='-', color='gray')
        plt.legend()
        plt.show()
        plt.close()
        assert 1==2

        # check vector field values along the SM (confirm that xdot = 0 along it)
        xdot_arr = np.array([params.ode_system_vector( (1.0 - y_arr[idx] - z_arr[idx], y_arr[idx], z_arr[idx]), None)
                             for idx in xrange(len(z_arr))])
        plt.plot(z_arr, xdot_arr[:,0], '--b', label=r'$\dot x(z)$')
        plt.plot(z_arr, xdot_arr[:,1], '--g', label=r'$\dot y(z)$')
        plt.plot(z_arr, xdot_arr[:,2], '--k', label=r'$\dot z(z)$')
        # plt.plot(z_arr, s_xy_arr, '--b', label=r'$s2 = c/f_{xy} - 1$')
        plt.xlabel(r'$z$')
        plt.ylabel('Rate of change')
        plt.gca().axhline(0.0, linestyle='-', color='gray')
        plt.legend()
        plt.show()
        plt.close()

        # A(z), B(z), psi(z) curves
        A_arr = ((params.c - f_xyz_arr) * z_arr + params.mu * y_arr)# / params.N
        B_arr = ((params.c + f_xyz_arr) * z_arr + params.mu * y_arr)# / params.N

        A_noMu_arr = ((params.c - f_xyz_arr) * z_arr)# / params.N
        B_noMu_arr = ((params.c + f_xyz_arr) * z_arr)# / params.N

        def psi(zmid, Nval, int_lower=0.0):
            # make sure n and z arr are equivalently normalized or not
            intval = 0.0

            for i, z in enumerate(z_arr[:-1]):
                # TODO integral bounds low high and dz weight
                if z > zmid:
                    if 0.1 < zmid < 0.11:
                        print 'breaking at', z, zmid, 'intval', intval
                        break
                if z > int_lower:
                    dz = z_arr[i + 1] - z_arr[i]
                    #intval += Nval * A_arr[i] / B_arr[i] * dz  # note factor 1/2 already in B
                    intval += s_xyz_arr[i] / 2.0 * dz  # note factor 1/2 already in B

            #from scipy.integrate import quad
            #def A_over_B(z):
            #    s0 = -0.07
            #    s_of_z = -s0 * (1 - z)
            #    return s_of_z / 2.0
            #return np.exp(2 * Nval * quad(A_over_B, 0, zmid)[0])
            #intval = quad(A_over_B, 0, zmid)[0]
           
            return np.exp(2 * Nval * intval)

        def psi_via_quad(zmid, Nval, int_lower=0.0):
            from scipy.integrate import quad
            def A_over_B(z):
                s0 = -0.07
                s_of_z = -s0 * (1 - z)
                return s_of_z / 2.0
            intval = quad(A_over_B, 0, zmid)[0]

            if 0.1 < zmid < 0.11:
                print 'psi_via_quad zmid', zmid, 'intval', intval

            return np.exp(2 * Nval * intval)

        def int_psi_orig(low, high):
            intval = 0.0
            for i, z in enumerate(z_arr[:-1]):
                if z > high:
                    break
                if z >= low:
                    dz = z_arr[i + 1] - z_arr[i]
                    intval += 1 / (psi_table[i]) * dz
            return intval

        def int_psi(low, high):
            intval = 0.0
            for i, z in enumerate(z_arr[:-1]):
                if z > high:
                    break
                if z >= low:
                    dz = z_arr[i + 1] - z_arr[i]
                    intval += psi_table[i] * dz
            return intval

        psi_table = np.zeros(len(z_arr))
        psi_via_quad_table = np.zeros(len(z_arr))
        for i, z in enumerate(z_arr[:-1]):
            zmid = (z_arr[i + 1] + z_arr[i]) / 2
            psi_table[i] = psi(zmid, params.N)
            psi_via_quad_table[i] = psi_via_quad(zmid, params.N)

        plt.plot(z_arr, A_arr, '--b', label=r'$A(z)$')
        plt.plot(z_arr, B_arr, '--r', label=r'$B(z)$')

        #plt.plot(z_arr/params.N, psi_arr, '--k', label=r'$s\psi(z)$')
        #plt.plot(z_arr, s_xy_arr, '--b', label=r'$s2 = c/f_{xy} - 1$')
        plt.xlabel(r'$z/N$')
        #plt.ylabel(r'$$')
        plt.gca().axhline(0.0, linestyle='-', color='gray')
        plt.legend()
        plt.show()

        def A_over_B(z):
            s0 = -0.07
            s_of_z = -s0 * (1 - z)
            return s_of_z / 2.0

        plt.plot(z_arr, A_arr/B_arr, '--k', label=r'$A(z)/B(z)$')
        plt.plot(z_arr, s_xyz_arr / 2.0, '--r', label=r'$s(z)/2$')
        plt.plot(z_arr, [A_over_B(z) for z in z_arr], '--b', label='as fn')

        plt.xlabel(r'$z/N$')
        plt.gca().axhline(0.0, linestyle='-', color='gray')
        plt.legend()
        plt.show()

        plt.plot(z_arr, psi_table, '--k', label=r'$\psi (z)$')
        plt.plot(z_arr, psi_via_quad_table, '--b', label=r'$\psi (z)$ quad')
        plt.xlabel(r'$z/N$')
        plt.gca().axhline(0.0, linestyle='-', color='gray')
        plt.legend()
        plt.show()

    if N_vary_stochplots:

        # make original prob to hit N curve
        N_range = [int(a) for a in np.logspace(1.50515, 4.13159, num=11)][0:6]  # + [int(a) for a in np.logspace(4.8, 7, num=4)]

        presets_to_y0 = {
            'BL1g': 0.14585869420527702,
            'BL100g': 0.22471588735222426,
            'TR1g': 0.19262827700935464,
            'TR100g': 0.28095}

        #f_xyz_arr, s_xyz_arr, z_arr, y_arr = \
        #    get_centermanifold_traj(params, norm=True, force_region_1=False, force_region_2=False)

        # A(z), B(z), psi(z) curves
        #A_noMu_arrA_arr_normed = ((params.c - f_xyz_arr) * z_arr)
        #B_noMu_arrA_arr_normed = ((params.c + f_xyz_arr) * z_arr)

        def prob_hit_N_orig(params):

            f_xyz_arr, s_xyz_arr, z_arr, y_arr = \
                get_centermanifold_traj(params, norm=True, force_region_1=False, force_region_2=False)

            nn=10
            f_xyz_arr = f_xyz_arr[::nn]
            s_xyz_arr = s_xyz_arr[::nn]
            z_arr = z_arr[::nn]
            y_arr = y_arr[::nn]

            # Choice 1: z range from 0 to 1 (not 0 to N)
            # Choice 2: use old versions of A(z), B(z)
            # Choice 3: divide B(z) locally by 2N; integrand is A/B
            # Choice 4: possible typo -- int ( psi ) dz is written as int ( 1/psi ) dz
            # this is used in the paper and matches linalg MFPT to z=N

            def A_local(n, n_idx):
                sval = s_xyz_arr[n_idx]
                yval = y_arr[n_idx]
                return sval * n + params.mu * yval

            def B_local(n, n_idx):
                sval = s_xyz_arr[n_idx]
                return (2 + sval) * n / (2 * params.N)

            def psi(z0, int_lower=0.0):
                intval = 0.0
                for i, z in enumerate(z_arr[:-1]):
                    if z > z0:
                        break
                    if z > int_lower:
                        zmid = (z_arr[i + 1] + z_arr[i]) / 2
                        dz = z_arr[i + 1] - z_arr[i]
                        intval += A_local(zmid, i) / B_local(zmid, i) * dz  # note factor 1/2N already in B
                return np.exp(intval)

            psi_table = np.zeros(len(z_arr))
            for i, z in enumerate(z_arr[:-1]):
                zmid = (z_arr[i + 1] + z_arr[i]) / 2
                psi_table[i] = psi(zmid)

            plt.plot(z_arr, psi_table, '--ok')
            plt.title('f1')
            plt.savefig('f1.png')
            plt.close()

            def int_psi(low, high):
                intval = 0.0
                for i, z in enumerate(z_arr[:-1]):
                    if z > high:
                        break
                    if z >= low:
                        dz = z_arr[i + 1] - z_arr[i]
                        intval += 1 / (psi_table[i]) * dz
                return intval

            num = int_psi(0.0, 1.0/params.N)
            den = int_psi(0.0, 1.0)
            print "prob_hit_N_orig", params.N, "num", num
            print "prob_hit_N_orig", params.N, "den", den
            prob_exit = num / den
            return prob_exit

        def prob_hit_N_revised_A(params):

            f_xyz_arr, s_xyz_arr, z_arr, y_arr = \
                get_centermanifold_traj(params, norm=True, force_region_1=False, force_region_2=False)

            nn = 10
            f_xyz_arr = f_xyz_arr[::nn]
            s_xyz_arr = s_xyz_arr[::nn]
            z_arr = z_arr[::nn]
            y_arr = y_arr[::nn]

            """
            plt.plot(z_arr, z_arr,label='z')
            plt.plot(z_arr, y_arr, label='z')
            plt.plot(z_arr, s_xyz_arr, label='s')
            plt.plot(z_arr, f_xyz_arr, label='f')
            plt.legend()
            plt.show()
            """

            #A_arr_normed = ((params.c - f_xyz_arr) * z_arr + params.mu * y_arr)
            #B_arr_normed = ((params.c + f_xyz_arr) * z_arr + params.mu * y_arr)
            #z_arr = z_arr * params.N

            def psi(zmid, int_lower=0.0):
                # make sure n and z arr are equivalently normalized or not
                intval = 0.0
                for i, z in enumerate(z_arr[:-1]):
                    # TODO integral bounds low high and dz weight
                    if z > zmid:
                        break
                    if z > int_lower:
                        dz = z_arr[i + 1] - z_arr[i]
                        #intval_add = A_arr_normed[i] / B_arr_normed[i]
                        intval_add = s_xyz_arr[i]/2.0
                        intval += intval_add * dz  # note factor 1/2 already in B
                        #print 'prob_hit_N_revised_A',  np.sign(intval_add), intval_add, zmid, z, dz

                return np.exp(2 * params.N * intval)

            psi_table = np.zeros(len(z_arr))
            for i, z in enumerate(z_arr[:-1]):
                zmid = (z_arr[i + 1] + z_arr[i]) / 2
                psi_table[i] = psi(zmid)

            plt.plot(z_arr, psi_table, '--ok')
            plt.title('f3')
            plt.savefig('f3.png')
            plt.close()

            def int_psi(low, high):
                intval = 0.0
                for i, z in enumerate(z_arr[:-1]):
                    if z > high:
                        break
                    if z >= low:
                        dz = z_arr[i + 1] - z_arr[i]
                        intval += psi_table[i] * dz
                return intval

            num = int_psi(0.0, 1.0 / params.N)
            den = int_psi(0.0, 1.0)
            print "prob_hit_N_revised", params.N, "num", num
            print "prob_hit_N_revised", params.N, "den", den
            prob_exit = num / den
            return prob_exit

        def prob_hit_N_revised_B(params):

            f_xyz_arr, s_xyz_arr, z_arr, y_arr = \
                get_centermanifold_traj(params, norm=True, force_region_1=False, force_region_2=False)

            nn = 10
            f_xyz_arr = f_xyz_arr[::nn]
            s_xyz_arr = s_xyz_arr[::nn]
            z_arr = z_arr[::nn]
            y_arr = y_arr[::nn]

            A_arr_normed = ((params.c - f_xyz_arr) * z_arr + params.mu * y_arr)
            B_arr_normed = ((params.c + f_xyz_arr) * z_arr + params.mu * y_arr)
            #z_arr = z_arr * params.N

            def psi(zmid, int_lower=0.0):
                # make sure n and z arr are equivalently normalized or not
                intval = 0.0
                for i, z in enumerate(z_arr[:-1]):
                    # TODO integral bounds low high and dz weight
                    if z > zmid:
                        break
                    if z > int_lower:
                        dz = z_arr[i + 1] - z_arr[i]
                        #intval += A_arr_normed[i] / B_arr_normed[i] * dz  # note factor 1/2 already in B
                        intval += s_xyz_arr[i]/2.0 * dz
                return np.exp(2 * params.N * intval)

            psi_table = np.zeros(len(z_arr))
            for i, z in enumerate(z_arr[:-1]):
                zmid = (z_arr[i + 1] + z_arr[i]) / 2
                psi_table[i] = psi(zmid)

            plt.plot(z_arr, psi_table, '--ok')
            plt.title('f3')
            plt.savefig('f3.png')
            plt.close()

            def int_psi(low, high):
                intval = 0.0
                for i, z in enumerate(z_arr[:-1]):
                    if z > high:
                        break
                    if z >= low:
                        dz = z_arr[i + 1] - z_arr[i]
                        intval += (1 / psi_table[i]) * dz
                return intval

            num = int_psi(0.0, 1.0 / params.N)
            den = int_psi(0.0, 1.0)
            print "prob_hit_N_revised", params.N, "num", num
            print "prob_hit_N_revised", params.N, "den", den
            prob_exit = num / den
            return prob_exit

        curve_orig = np.zeros(len(N_range))
        curve_A = np.zeros(len(N_range))
        curve_B = np.zeros(len(N_range))
        #curve_scaling = np.zeros(len(N_range))
        for idx, Nval in enumerate(N_range):
            params_mod = params.mod_copy({'N': Nval})
            curve_orig[idx] = 1/(params.mu * Nval * presets_to_y0[preset] * prob_hit_N_orig(params_mod))
            curve_A[idx] = 1/(params.mu * Nval * presets_to_y0[preset] * prob_hit_N_revised_A(params_mod))
            curve_B[idx] = 1/(params.mu * Nval * presets_to_y0[preset] * prob_hit_N_revised_B(params_mod))
            #curve_scaling[idx] = 1 / (params.mu * Nval) * np.exp(Nval)
            print "main data loop:", idx, Nval, curve_orig[idx], curve_A[idx], curve_B[idx], '\n'

        print "curve_orig"
        print curve_orig, '\n'
        print "curve_A"
        print curve_A, '\n'
        print "curve_B"
        print curve_B, '\n'

        # plot data and heuristics on one plot
        ax = None
        fs = 12
        #colours = [X_DARK, '#ffd966', Z_DARK, BLUE, 'pink', 'brown']  # ['black', 'red', 'green', 'blue']
        # colours = [X_DARK, BLUE, Z_DARK, 'pink', 'brown']  # ['black', 'red', 'green', 'blue'] NOV 7 ALT
        plt.figure(figsize=(4, 3))
        ax = plt.gca()

        ax.plot(N_range, curve_orig, ':', marker='p', markeredgecolor='k', color='black',  # colours[idx],
                label=r'%s: $\langle\tau\rangle_{\mathrm{B}}$ prob 3 Orig' % preset, zorder=3)
        ax.plot(N_range, curve_A, ':', marker='s', markeredgecolor='k', color='green',  # colours[idx],
                label=r'%s: $\langle\tau\rangle_{\mathrm{B}}$ prob 3 explicit, $\psi$' % preset, zorder=3)
        ax.plot(N_range, curve_B, ':', marker='^', markeredgecolor=None, color='blue',  # colours[idx],
                label=r'%s: $\langle\tau\rangle_{\mathrm{B}}$ prob 3 explicit, $\psi^{-1}$' % preset, zorder=3)
        #ax.plot(N_range, curve_scaling, ':', marker='*', markeredgecolor='k', color='blue',  # colours[idx],
        #        label=r'%s: Fit' % preset, zorder=3)

        ax.set_xlabel(r'$N$', fontsize=fs)
        ax.set_ylabel(r'$\langle\tau\rangle$', fontsize=fs)
        plt.xticks(fontsize=fs - 2)
        plt.yticks(fontsize=fs - 2)
        # plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize=fs-4)
        plt.legend(fontsize=fs - 4, ncol=1)
        # log options
        ax.set_xscale("log")
        # ax.set_xlim([np.min(N_range) * 0.9, 1.5 * 1e6])
        ax.set_xlim([np.min(N_range) * 0.9, 1.5 * 1e4])
        ax.set_yscale("log")
        # ax.set_ylim([0.8 * 1e1, 2 * 1e5])
        ax.set_ylim([0.8 * 1e2, 2 * 1e6])
        plt.show()
