import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import numpy as np
import os
from scipy.integrate import odeint
from matplotlib import colors

from plotting_helper import plot_handler

"""
See plotting.py in xyz repo folder for related functionality
"""


# MATPLOTLIB GLOBAL SETTINGS
"""
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
mpl.rcParams.update(mpl_params)
"""


def plot_vectorfield_2D(params, streamlines=True, smallfig=False, ax=None):

    N = 100.0

    figsize=(3, 2.5)  # 4,3 orig, else 3, 2.5 for stoch fig
    text_fs = 20
    ms = 10
    stlw = 0.5
    nn = 100  # 100
    ylim_mod = 0.04

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    X = np.array([[0.0, 0.0], [N, 0.0], [N / 2.0, N]])

    t1 = plt.Polygon(X[:3, :], color='k', alpha=0.1, zorder=1)
    ax.add_patch(t1)
    ax.text(-params.N * 0.07, -params.N * 0.05, r'$x$', fontsize=text_fs)
    ax.text(params.N * 1.03, -params.N * 0.05, r'$y$', fontsize=text_fs)
    ax.text(params.N / 2.0 * 0.96, params.N * 1.07, r'$z$', fontsize=text_fs)

    if streamlines:
        B, A = np.mgrid[0:N*0.25:nn*1j, 0:N:nn*1j]
        # need to mask outside of simplex
        ADOT = np.zeros(np.shape(A))
        BDOT = np.zeros(np.shape(A))
        SPEEDS = np.zeros(np.shape(A))
        for i in range(nn):
            for j in range(nn):
                a = A[i, j]
                b = B[i, j]
                z = b
                x = N - a - b/2.0  # TODO check
                y = N - x - z
                if b > 2.1*a or b > 2.05*(N-a) or b == 0:  # check if outside simplex
                    ADOT[i, j] = np.nan
                    BDOT[i, j] = np.nan
                else:
                    dxvecdt = params.ode_system_vector([x,y,z], None)
                    SPEEDS[i, j] = np.sqrt(dxvecdt[0]**2 + dxvecdt[1]**2 + dxvecdt[2]**2)
                    ADOT[i, j] = (-dxvecdt[0] + dxvecdt[1])/2.0  # (- xdot + ydot) / 2
                    BDOT[i, j] = dxvecdt[2]                      # zdot

        # this will color lines
        """
        strm = ax.streamplot(A, B, ADOT, BDOT, color=SPEEDS, linewidth=stlw, cmap=plt.cm.coolwarm)
        if cbar:
            plt.colorbar(strm.lines)
        """
        # this will change line thickness
        stlw_low = stlw
        stlw_high = 1.0
        speeds_low = np.min(SPEEDS)
        speeds_high = np.max(SPEEDS)
        speeds_conv = 0.3 + SPEEDS / speeds_high
        strm = ax.streamplot(A, B, ADOT, BDOT, color=(0.34, 0.34, 0.34), linewidth=speeds_conv, zorder=10)

    """
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
            ax.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color=(0.212, 0.271, 0.31), zorder=11)  # #b88c8c is pastel reddish, (0.212, 0.271, 0.31) blueish
        for fp in unstable_fps:
            fp_x = (N + fp[1] - fp[0]) / 2.0
            #plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', markerfacecolor="None")
            ax.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color=(0.902, 0.902, 0.902), zorder=11)
    """

    #ax.set_ylim(-N*ylim_mod, N*(1+ylim_mod))
    ax.set_xlim(-6, 1.05 * N)
    ax.set_ylim(-0.1, 0.2 * N)

    ax.axis('off')
    plt.savefig('test_vectorfield.pdf')
    return ax


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


if __name__ == '__main__':
    #fig = plot_simplex(100)
    #plt.show()
    params = presets('preset_xyz_tanh')
    #params = presets('preset_xyz_constant')

    ax = plot_simplex2D(params, smallfig=True)
    plt.savefig(OUTPUT_DIR + os.sep + 'simplex_plot.pdf')
    plt.show()
    plot_vectorfield_2D()