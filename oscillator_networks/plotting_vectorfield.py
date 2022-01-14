import matplotlib.pyplot as plt
import numpy as np

from class_singlecell import SingleCell
from vectorfields import set_params_ode, vectorfield_Yang2013

'''
def plot_vectorfield_2D(single_cell, streamlines=True, ax=None):

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
    ax.text(-N * 0.07, -N * 0.05, r'$x$', fontsize=text_fs)
    ax.text(N * 1.03, -N * 0.05, r'$y$', fontsize=text_fs)
    ax.text(N / 2.0 * 0.96, N * 1.07, r'$z$', fontsize=text_fs)

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
                    dxvecdt = single_cell.ode_system_vector([x,y,z])
                    SPEEDS[i, j] = np.sqrt(dxvecdt[0]**2 + dxvecdt[1]**2)
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

    #ax.set_ylim(-N*ylim_mod, N*(1+ylim_mod))
    ax.set_xlim(-6, 1.05 * N)
    ax.set_ylim(-0.1, 0.2 * N)

    ax.axis('off')
    plt.savefig('test_vectorfield.pdf')
    return ax
'''


def nan_mask(x, fill=1000):
    x_mask_arr = np.isfinite(x)
    # print(x_mask_arr)
    x[~x_mask_arr] = fill
    return x


def example_vectorfield():
    ax_lims = 10
    Y, X = np.mgrid[-ax_lims:ax_lims:100j, -ax_lims:ax_lims:100j]
    U = -10 - X**2 + Y
    V = 10 + X - Y**2

    speed = np.sqrt(U**2 + V**2)
    lw = 5*speed / speed.max()

    fig = plt.figure(figsize=(7, 9))

    #  Varying density along a streamline
    plt.axhline(0, linestyle='--', color='k')
    plt.axvline(0, linestyle='--', color='k')

    ax0 = fig.gca()
    strm = ax0.streamplot(X, Y, U, V, density=[0.5, 1], color=speed, linewidth=lw)
    fig.colorbar(strm.lines)
    ax0.set_title('Varying Density, Color, Linewidth')
    ax0.set_xlabel('U')
    ax0.set_ylabel('V')

    plt.tight_layout()
    plt.show()
    return


def Yang2013_vectorfield(z=0):
    """
    Scalar z represents static Bam concentration
    """
    # Block for example code
    ax_lims = 100
    Y, X = np.mgrid[0:ax_lims:100j, 0:ax_lims:100j]

    params = set_params_ode('Yang2013')
    U, V = vectorfield_Yang2013(params, X, Y, z=z, two_dim=True)

    U = nan_mask(U)
    V = nan_mask(V)

    # Block for example code
    speed = np.sqrt(U**2 + V**2)
    lw = 5 * speed / speed.max()

    fig = plt.figure(figsize=(7, 9))

    #  Varying density along a streamline
    plt.axhline(0, linestyle='--', color='k')
    plt.axvline(0, linestyle='--', color='k')

    ax0 = fig.gca()
    strm = ax0.streamplot(X, Y, U, V, density=[0.5, 1], color=speed, linewidth=lw)
    fig.colorbar(strm.lines)
    ax0.set_title('Varying Density, Color, Linewidth')
    ax0.set_xlabel('U')
    ax0.set_ylabel('V')

    plt.tight_layout()
    plt.show()


def Yang2013_contourplot(z=0):
    """
    Scalar z represents static Bam concentration
    """
    #ax_lims = 100
    #Y, X = np.mgrid[0:ax_lims:100j, 0:ax_lims:100j]

    delta = 0.5
    x = np.arange(0, 100.0, delta)
    y = np.arange(0, 100.0, delta)
    X, Y = np.meshgrid(x, y)

    params = set_params_ode('Yang2013')
    U, V = vectorfield_Yang2013(params, X, Y, z=z, two_dim=True)

    U = nan_mask(U)
    V = nan_mask(V)

    fig, ax = plt.subplots(1, 3)
    contours_u = ax[0].contour(X, Y, U)
    ax[0].clabel(contours_u, inline=1, fontsize=10)
    ax[0].set_title('U contours')
    ax[0].set_xlabel('U')
    ax[0].set_ylabel('V')

    contours_v = ax[1].contour(X, Y, V)
    ax[1].clabel(contours_v, inline=1, fontsize=10)
    ax[1].set_title('V contours')
    ax[1].set_xlabel('U')
    ax[1].set_ylabel('V')

    contours_dbl_u = ax[2].contour(X, Y, U)
    contours_dbl_v = ax[2].contour(X, Y, V)
    ax[2].set_title('U, V contours overlaid')
    ax[2].set_xlabel('U')
    ax[2].set_ylabel('V')

    plt.show()


def Yang2013_nullclines(z=0, flip_axis=False):
    delta = 0.5
    axmax = 150.0
    x = np.arange(0, axmax, delta)
    y = np.arange(0, axmax, delta)
    X, Y = np.meshgrid(x, y)

    params = set_params_ode('Yang2013')
    U, V = vectorfield_Yang2013(params, X, Y, z=z, two_dim=True)

    U = nan_mask(U)
    V = nan_mask(V)

    x_label = 'x'
    y_label = 'y'
    if flip_axis:
        # swap X, Y
        tmp = X
        X = Y
        Y = tmp
        # swap labels
        tmp = x_label
        x_label = y_label
        y_label = tmp

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    # plot nullclines
    nullcline_u = ax.contour(X, Y, U, (0,), colors='b', linewidths=1.5)
    ax.clabel(nullcline_u, inline=1, fmt='X nc', fontsize=10)
    nullcline_v = ax.contour(X, Y, V, (0,), colors='r', linewidths=1.5)
    ax.clabel(nullcline_v, inline=1, fmt='Y nc', fontsize=10)
    # plot labels
    ax.set_title('X, Y nullclines overlaid (blue=X, red=Y)')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()


if __name__ == '__main__':
    #example_vectorfield()

    Yang2013_vectorfield()
    Yang2013_contourplot()
    Yang2013_nullclines()
    Yang2013_nullclines(flip_axis=True)

    #sinit_cond = (10.0, 0, 0)
    #single_cell = SingleCell(init_cond)
    #plot_vectorfield_2D(single_cell)
