import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from os import sep

from constants import X0_COL, X1_COL, X2_COL, OUTPUT_DIR, STATES_ID_INV, PARAMS_ID, \
                      DEFAULT_X_COLOUR, DEFAULT_Y_COLOUR, DEFAULT_Z_COLOUR
from formulae import fp_location_fsolve, jacobian_numerical_2d
from presets import presets


# MATPLOTLIB GLOBAL SETTINGS
"""
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
mpl.rcParams.update(mpl_params)
"""

# PLOTTING CONSTANTS
PLT_SAVE = 'default'
FLAG_SAVE = True
BBOX_INCHES = None
FLAG_SHOW = True
FLAG_TABLE = True
LOC_TABLE = 'center right'
BBOX_TABLE = None


def plot_options_build(alloff=False, **plot_options):
    main_keys = ['flag_show', 'flag_save', 'flag_table']
    plot_options = {'flag_table': plot_options.get('flag_table', FLAG_TABLE),
                    'loc_table': plot_options.get('loc_table', LOC_TABLE),
                    'bbox_table': plot_options.get('bbox_table', BBOX_TABLE),
                    'plt_save': plot_options.get('plt_save', PLT_SAVE),
                    'flag_save': plot_options.get('flag_save', FLAG_SAVE),
                    'bbox_inches': plot_options.get('bbox_inches', BBOX_INCHES),
                    'flag_show': plot_options.get('flag_show', FLAG_SHOW)}
    if alloff:
        plot_options = {k: False for k in main_keys}
    return plot_options


def plot_handler(fig, ax, params, plot_options=None):
    if plot_options is None:
        plot_options = plot_options_build()

    if plot_options.get('flag_table', FLAG_TABLE):
        loc = plot_options.get('loc_table', LOC_TABLE)
        bbox = plot_options.get('bbox_table', BBOX_TABLE)
        plot_table_params(ax, params, loc=loc, bbox=bbox)

    if plot_options.get('flag_save', FLAG_SAVE):
        savename = OUTPUT_DIR + sep + plot_options.get('plt_save') + '.pdf'
        bbox_inches = plot_options.get('bbox_inches', BBOX_INCHES)
        fig.savefig(savename, bbox_inches=bbox_inches)

    if plot_options.get('flag_show', FLAG_SHOW):
        plt.show()
    return fig, ax


def plot_table_params(ax, params, loc=LOC_TABLE, bbox=None):
    """
    params is Params object
    loc options 'center right', 'best'
    bbox is x0, y0, height, width e.g. (1.1, 0.2, 0.1, 0.75)
    """
    # create table of params
    row_labels = ['system', 'feedback']
    row_labels += [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params.system], [params.feedback]]
    table_vals += [[val] for val in params.params_list]  # note weird format
    # plot table
    param_table = ax.table(cellText=table_vals,
                           colWidths=[0.1]*3,
                           rowLabels=row_labels,
                           loc=loc, bbox=bbox)
    #ax.text(12, 3.4, 'Params', size=8)
    return ax


def plot_simplex(N):
    normal = [1, 1, 1]
    intercepts = [(N, 0, 0), (0, N, 0), (0, 0, N)]

    # create surface
    x1range = np.linspace(0.0, N, 100)
    x2range = np.linspace(0.0, N, 100)
    xx, yy = np.meshgrid(x1range, x2range)
    z = (N - normal[0] * xx - normal[1] * yy) * 1. / normal[2]

    # plot surface
    cmap = colors.ListedColormap(['white', 'red'])
    bounds = [0, 5, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z, alpha=0.4, cmap=cmap, color='blue')
    ax.scatter(intercepts[0], intercepts[1], intercepts[2], color=['red', 'green', 'blue'])
    ax.view_init(5, 35)  # ax.view_init(-45, -15)

    ax.set_zlim(0.0, intercepts[2][2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # note = 'A1 = %.2f \ngamma = %.2f \nh1 = %.2f, h2 = %.2f, h3 = %.2f' % (A[0,0], W[0,0], H[0,0], H[0,1], H[0,2])
    # ax.text(intercepts[0][0]*0.55, intercepts[1][1]*0.6, intercepts[2][2]*0.6, note, fontsize=7)
    return fig


def plot_simplex2D(params, streamlines=True, fp=True, cbar=False, smallfig=False):
    N = params.N

    if smallfig:
        figsize = (2.0, 1.6)
        text_fs = 20
        ms = 10
        stlw = 0.5
        nn = 20
        ylim_mod = 0.08
    else:
        figsize=(4, 3)
        text_fs = 20
        ms = 10
        stlw = 0.5
        nn = 100
        ylim_mod = 0.04

    fig = plt.figure(figsize=figsize)

    X = np.array([[0.0, 0.0], [N, 0.0], [N / 2.0, N]])

    if smallfig:
        t1 = plt.Polygon(X[:3, :], color=(0.902, 0.902, 0.902), alpha=1.0, ec=(0.14, 0.14, 0.14), lw=1)
        plt.gca().add_patch(t1)
        plt.text(-params.N*0.12, -params.N*0.08, r'$x$', fontsize=text_fs)
        plt.text(params.N*1.045, -params.N*0.08, r'$y$', fontsize=text_fs)
        plt.text(params.N/2.0*0.93, params.N*1.115, r'$z$', fontsize=text_fs)
    else:
        t1 = plt.Polygon(X[:3, :], color='k', alpha=0.1)
        plt.gca().add_patch(t1)
        plt.text(-params.N * 0.07, -params.N * 0.05, r'$x$', fontsize=text_fs)
        plt.text(params.N * 1.03, -params.N * 0.05, r'$y$', fontsize=text_fs)
        plt.text(params.N / 2.0 * 0.96, params.N * 1.07, r'$z$', fontsize=text_fs)

    if streamlines:
        B, A = np.mgrid[0:N:nn*1j, 0:N:nn*1j]
        # need to mask outside of simplex
        ADOT = np.zeros(np.shape(A))
        BDOT = np.zeros(np.shape(A))
        SPEEDS = np.zeros(np.shape(A))
        for i in xrange(nn):
            for j in xrange(nn):
                a = A[i, j]
                b = B[i, j]
                z = b
                x = N - a - b/2.0  # TODO check
                y = N - x - z
                if b > 2.0*a or b > 2.0*(N-a) or b == 0:  # check if outside simplex
                    ADOT[i, j] = np.nan
                    BDOT[i, j] = np.nan
                else:
                    dxvecdt = params.ode_system_vector([x,y,z], None)
                    SPEEDS[i, j] = np.sqrt(dxvecdt[0]**2 + dxvecdt[1]**2 + dxvecdt[2]**2)
                    ADOT[i, j] = (-dxvecdt[0] + dxvecdt[1])/2.0  # (- xdot + ydot) / 2
                    BDOT[i, j] = dxvecdt[2]                      # zdot
        if smallfig:
            strm = plt.streamplot(A, B, ADOT, BDOT, color=(0.34, 0.34, 0.34), linewidth=stlw)
        else:
            strm = plt.streamplot(A, B, ADOT, BDOT, color=SPEEDS, linewidth=stlw, cmap=plt.cm.coolwarm)
            if cbar:
                plt.colorbar(strm.lines)

    if fp:
        stable_fps = []
        unstable_fps = []
        all_fps = fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=35, tol=10e-1, buffer=True)
        for fp in all_fps:
            J = jacobian_numerical_2d(params, fp[0:2])
            eigenvalues, V = np.linalg.eig(J)
            if eigenvalues[0] < 0 and eigenvalues[1] < 0:
                stable_fps.append(fp)
            else:
                unstable_fps.append(fp)
        for fp in stable_fps:
            fp_x = (N + fp[1] - fp[0]) / 2.0
            #plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color='k')
            plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color=(0.212, 0.271, 0.31))
        for fp in unstable_fps:
            fp_x = (N + fp[1] - fp[0]) / 2.0
            #plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', markerfacecolor="None")
            plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color=(0.902, 0.902, 0.902))

    plt.ylim(-N*ylim_mod, N*(1+ylim_mod))
    plt.axis('off')
    return fig


def plot_fp_curves_simple(x0, x0_stab, x1, x1_stab, x2, x2_stab, N, plt_title, flag_show, flag_save, plt_save="bifurcation_curves", colourbinary=False):
    # this is for data from 'default' ode system which guarantees 3 fixed point arrays
    fig_simplex = plot_simplex(N)
    ax_simplex = fig_simplex.gca()
    if colourbinary:
        x0_col_array = [X1_COL[int(i)] for i in x0_stab]
        x1_col_array = [X1_COL[int(i)] for i in x1_stab]
        x2_col_array = [X1_COL[int(i)] for i in x2_stab]
    else:
        x0_col_array = [X0_COL[int(i)] for i in x0_stab]
        x1_col_array = [X1_COL[int(i)] for i in x1_stab]
        x2_col_array = [X2_COL[int(i)] for i in x2_stab]

    ax_simplex.scatter(x0[:, 0], x0[:, 1], x0[:, 2], label='x0', color=x0_col_array)
    ax_simplex.scatter(x1[:, 0], x1[:, 1], x1[:, 2], label='x1', color=x1_col_array)
    ax_simplex.scatter(x2[:, 0], x2[:, 1], x2[:, 2], label='x2', color=x2_col_array)
    # plot settings
    ax_simplex.view_init(5, 35)  # ax.view_init(-45, -15)
    axis_scale = 1
    ax_simplex.set_xlim(-N * axis_scale, N * axis_scale)  # may need to flip order
    ax_simplex.set_ylim(-N * axis_scale, N * axis_scale)
    ax_simplex.set_zlim(-N * axis_scale, N * axis_scale)
    if not colourbinary:
        ax_simplex.legend()
    else:
        stable_pt = mlines.Line2D([], [], color=X1_COL[1], marker='o',
                                      markersize=5, label='Stable FP')
        unstable_pt = mlines.Line2D([], [], color=X1_COL[0], marker='o',
                                  markersize=5, label='Unstable FP')
        plt.legend(handles=[stable_pt, unstable_pt])
    ax_simplex.set_title(plt_title)
    if flag_show:
        plt.show()
    if flag_save:
        fig_simplex.savefig(OUTPUT_DIR + sep + plt_save + '.png')
    return fig_simplex


def plot_fp_curves_general(fp_info_dict, N, flag_show=False, plt_save="bifurcation_curves"):
    fig_simplex = plot_simplex(N)
    ax_simplex = fig_simplex.gca()

    fp_info_list = [triple for value in fp_info_dict.values() for triple in value]
    fp_array = np.array([triple[0] for triple in fp_info_list])
    fp_color_array = np.array([X1_COL[int(triple[2])] for triple in fp_info_list])
    ax_simplex.scatter(fp_array[:,0], fp_array[:,1], fp_array[:,2], color=fp_color_array)

    # PRINTING
    """
    print "len", len(fp_info_list)
    print "first elem", fp_info_list[0]
    for i in xrange(len(fp_info_list)):
        if fp_info_list[i][2]:
            print "fp_info_list[i]", fp_info_list[i][2], fp_info_list[i][1], fp_info_list[i][0]
    """

    # plot settings
    ax_simplex.view_init(5, 35)  # ax.view_init(-45, -15)
    axis_scale = 1
    ax_simplex.set_xlim(-N * axis_scale, N * axis_scale)  # may need to flip order
    ax_simplex.set_ylim(-N * axis_scale, N * axis_scale)
    ax_simplex.set_zlim(-N * axis_scale, N * axis_scale)
    stable_pt = mlines.Line2D([], [], color=X1_COL[1], marker='o',
                              markersize=5, label='Stable FP')
    unstable_pt = mlines.Line2D([], [], color=X1_COL[0], marker='o',
                                markersize=5, label='Unstable FP')
    plt.legend(handles=[stable_pt, unstable_pt])
    ax_simplex.set_title("Bifurcation curves")
    if flag_show:
        plt.show()
    if plt_save is not None:
        fig_simplex.savefig(OUTPUT_DIR + sep + plt_save + '.png')
    return fig_simplex


def plot_bifurc_dist(x1_array, bifurcation_search, bifurc_id, N, dist_type, flag_show, flag_save, plt_save="bifurcation_dist_norm_"):
    assert dist_type in ["norm", "z_only"]
    distances_to_x0 = np.zeros((len(bifurcation_search), 1))
    for idx in xrange(len(bifurcation_search)):
        x1_fp = x1_array[idx, :]
        if dist_type == "norm":
            distances_to_x0[idx] = np.linalg.norm(x1_fp - np.array([0, 0, N]))
        else:
            distances_to_x0[idx] = N - x1_fp[2]
    fig_dist = plt.figure()
    ax_dist = fig_dist.gca()
    plt.plot(bifurcation_search, distances_to_x0)
    if dist_type == "norm":
        plt.axhline(y=np.sqrt(2) * N, color='r', linestyle='--')
        ax_dist.set_ylim(-0.1, N * 3)
    else:
        plt.axhline(N, color='r', linestyle='--')
        ax_dist.set_ylim(-N, N * 3)
    plt.axhline(y=0, color='k', linestyle='-')
    ax_dist.grid(True)
    ax_dist.set_title("Bifurcation Distance (%s)" % dist_type)
    ax_dist.set_xlabel(bifurc_id)
    ax_dist.set_ylabel("x1 distance to x0 (%s)" % dist_type)
    if flag_show:
        plt.show()
    if flag_save:
        fig_dist.savefig(OUTPUT_DIR + sep + plt_save + dist_type + '.png')
    return fig_dist


def plot_trajectory(r, times, params, fig_traj=None, **plot_options):
    if fig_traj is None:
        fig_traj = plot_simplex(params.N)
    ax_traj = fig_traj.gca()
    ax_traj.view_init(5, 35)  # ax.view_init(-45, -15)
    ax_traj.plot(r[:, 0], r[:, 1], r[:, 2], label='trajectory')
    #ax_traj.plot([x1[0]], [x1[1]], [x1[2]], label='x_weird')
    ax_traj.legend()
    ax_traj.set_title("Trajectory")
    fig_traj, ax_traj = plot_handler(fig_traj, ax_traj, params, plot_options=plot_options)
    return ax_traj


def plot_trajectory_mono(r, times, params, ax_mono=None, mono="z", **plot_options):
    assert mono in STATES_ID_INV.keys()
    axis_idx = STATES_ID_INV[mono]
    if ax_mono is None:
        fig_mono = plt.figure()
        ax_mono = fig_mono.gca()
        plt.plot(times, r[:, axis_idx], )
        plt.title("Trajectory: " + mono + " only")
        ax_mono.grid(True)
        ax_mono.set_xlabel("time")
        ax_mono.set_ylabel(mono)
        plot_options['plt_save'] = plot_options.get('plt_save', "traj_mono_") + "_%s" % mono
        fig_mono, ax_mono = plot_handler(fig_mono, ax_mono, params, plot_options=plot_options)
    else:
        ax_mono.plot(times, r[:, axis_idx], )
        fig_mono, ax_mono = plot_handler(plt.gcf(), ax_mono, params, plot_options=plot_options)
    return ax_mono


def plot_endpoint_mono(fp_list, param_list, param_varying_name, params, flag_show, flag_save, ax_mono=None, mono="z",
                       plt_save="endpoint_mono_", all_axis=True, conv_to_fraction=False, flag_log=True, flag_table=False):
    assert mono in STATES_ID_INV.keys()
    #rcParams.update({'font.size': 22})
    axis_idx = STATES_ID_INV[mono]
    fig_mono = plt.figure(figsize=(8, 6), dpi=80)
    ax_mono = fig_mono.gca()
    if conv_to_fraction:
        N = params.N
        fp_list = fp_list / N
    if all_axis:
        if flag_log:
            line_x, = ax_mono.semilogx(param_list, fp_list[:, 0], '-o', color=DEFAULT_X_COLOUR, markersize=10.0, markeredgecolor='black',
                                       label="x")
            line_y, = ax_mono.semilogx(param_list, fp_list[:, 1], '-o', color=DEFAULT_Y_COLOUR, markersize=10.0, markeredgecolor='black',
                                       label="y")
            line_z, = ax_mono.semilogx(param_list, fp_list[:, 2], '-o', color=DEFAULT_Z_COLOUR, markersize=10.0, markeredgecolor='black',
                                       label="z")
        else:
            line_x, = ax_mono.plot(param_list, fp_list[:, 0], '-o', color=DEFAULT_X_COLOUR, markersize=10.0, markeredgecolor='black', label="x")
            line_y, = ax_mono.plot(param_list, fp_list[:, 1], '-o', color=DEFAULT_Y_COLOUR, markersize=10.0, markeredgecolor='black', label="y")
            line_z, = ax_mono.plot(param_list, fp_list[:, 2], '-o', color=DEFAULT_Z_COLOUR, markersize=10.0, markeredgecolor='black', label="z")
        ax_mono.set_ylabel("axis_i")
        #plt.legend(handles=[line_x, line_y, line_z], bbox_to_anchor=(1.05, 0.98))
        plt.title("axis_inf vs param_val")
    else:
        ax_mono.plot(param_list, fp_list[:, axis_idx], '-o')
        ax_mono.set_ylabel(mono + "_inf")
        plt.title(mono + "_inf vs param_val")
    #ax_mono.grid(True)
    #ax_mono.tick_params(labelsize=16)
    ax_mono.set_xlabel(param_varying_name)
    if flag_table:
        plot_table_params(ax_mono, params)
    if flag_show:
        plt.show()
    if flag_save:
        fig_mono.savefig(OUTPUT_DIR + sep + plt_save + mono + '.pdf')
    return ax_mono


if __name__ == '__main__':
    #fig = plot_simplex(100)
    #plt.show()
    params = presets('preset_xyz_tanh')
    fig = plot_simplex2D(params, smallfig=True)
    plt.savefig(OUTPUT_DIR + sep + 'simplex_plot.pdf')
    plt.show()
