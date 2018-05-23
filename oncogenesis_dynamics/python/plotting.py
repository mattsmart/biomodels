import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from os import sep

from constants import PARAMS_ID_INV, X0_COL, X1_COL, X2_COL, OUTPUT_DIR, STATES_ID_INV, PARAMS_ID, \
                      DEFAULT_X_COLOUR, DEFAULT_Y_COLOUR, DEFAULT_Z_COLOUR


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

    ax.set_zlim(0.0, intercepts[2][2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # note = 'A1 = %.2f \ngamma = %.2f \nh1 = %.2f, h2 = %.2f, h3 = %.2f' % (A[0,0], W[0,0], H[0,0], H[0,1], H[0,2])
    # ax.text(intercepts[0][0]*0.55, intercepts[1][1]*0.6, intercepts[2][2]*0.6, note, fontsize=7)
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


def plot_trajectory(r, times, N, fig_traj=None, flag_show=False, flag_save=True, plt_save="trajectory"):
    if fig_traj is None:
        fig_traj = plot_simplex(N)
    ax_traj = fig_traj.gca()
    ax_traj.view_init(5, 35)  # ax.view_init(-45, -15)
    ax_traj.plot(r[:, 0], r[:, 1], r[:, 2], label='trajectory')
    #ax_traj.plot([x1[0]], [x1[1]], [x1[2]], label='x_weird')
    ax_traj.legend()
    ax_traj.set_title("Trajectory")
    if flag_show:
        plt.show()
    if flag_save:
        fig_traj.savefig(OUTPUT_DIR + sep + plt_save + '.png')
    return ax_traj

"""
def plot_trajectory_mono(r, times, flag_show, flag_save, mono="z", plt_save="trajectory_mono_"):
    assert mono in STATES_ID_INV.keys()
    fig_mono = plt.figure()
    ax_mono = fig_mono.gca()
    axis_idx = STATES_ID_INV[mono]
    plt.plot(times, r[:, axis_idx], )
    plt.title("Trajectory: " + mono + " only")
    ax_mono.grid(True)
    ax_mono.set_xlabel("time")
    ax_mono.set_ylabel(mono)
    if flag_show:
        plt.show()
    if flag_save:
        fig_mono.savefig(OUTPUT_DIR + sep + plt_save + mono + '.png')
    return fig_mono
"""

def plot_trajectory_mono(r, times, flag_show, flag_save, ax_mono=None, mono="z", plt_save="trajectory_mono_"):
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
        if flag_show:
            plt.show()
        if flag_save:
            fig_mono.savefig(OUTPUT_DIR + sep + plt_save + mono + '.png')
    else:
        ax_mono.plot(times, r[:, axis_idx], )
        if flag_show:
            plt.show()
    return ax_mono


def plot_endpoint_mono(fp_list, param_list, param_varying_name, params, flag_show, flag_save, ax_mono=None, mono="z",
                       plt_save="endpoint_mono_", all_axis=True, conv_to_fraction=False, flag_log=True):
    assert mono in STATES_ID_INV.keys()
    #rcParams.update({'font.size': 22})
    axis_idx = STATES_ID_INV[mono]
    fig_mono = plt.figure(figsize=(8, 6), dpi=80)
    ax_mono = fig_mono.gca()
    if conv_to_fraction:
        N = params[PARAMS_ID_INV["N"]]
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
    # CREATE TABLE OF PARAMS
    """
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] if PARAMS_ID[i] != param_varying_name else [None] for i in xrange(len(PARAMS_ID))]
    print len(row_labels), len(table_vals)
    param_table = plt.table(cellText=table_vals,
                            colWidths=[0.1]*3,
                            rowLabels=row_labels,
                            loc='center right')
    #plt.text(12, 3.4, 'Params', size=8)
    """
    if flag_show:
        plt.show()
    if flag_save:
        fig_mono.savefig(OUTPUT_DIR + sep + plt_save + mono + '.pdf')
    return ax_mono
