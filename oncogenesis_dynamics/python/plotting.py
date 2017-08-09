import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from os import sep

from constants import X1_COL, X2_COL, OUTPUT_DIR


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


def plot_fp_curves(x1_array, x2_array, N, plt_title, flag_show, flag_save, plt_save="bifurcation_curves"):
    fig_simplex = plot_simplex(N)
    ax_simplex = fig_simplex.gca()
    ax_simplex.scatter(x1_array[:, 0], x1_array[:, 1], x1_array[:, 2], label='q_plus', color=X1_COL)
    ax_simplex.scatter(x2_array[:, 0], x2_array[:, 1], x2_array[:, 2], label='q_minus', color=X2_COL)
    # plot settings
    ax_simplex.view_init(5, 35)  # ax.view_init(-45, -15)
    axis_scale = 1
    ax_simplex.set_xlim(-N * axis_scale, N * axis_scale)  # may need to flip order
    ax_simplex.set_ylim(-N * axis_scale, N * axis_scale)
    ax_simplex.set_zlim(-N * axis_scale, N * axis_scale)
    ax_simplex.legend()
    ax_simplex.set_title(plt_title)
    if flag_show:
        plt.show()
    if flag_save:
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