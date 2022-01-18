import matplotlib.pyplot as plt
import numpy as np

from class_singlecell import SingleCell
from dynamics_generic import simulate_dynamics_general
from dynamics_vectorfields import ode_choose_params, ode_choose_vectorfield, ode_integration_defaults
from settings import DYNAMICS_METHOD, STYLE_ODE, PLOT_XLABEL, PLOT_YLABEL

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


def phaseplot_general(ode_dict, init_conds=None, dynamics_method=DYNAMICS_METHOD, axlow=0., axhigh=120., k=10):
    """
    ode_kwargs:
        'z': Scalar z represents static Bam concentration
        't': Scalar t represents time
    k is the number of points between arrows along the trajectory
    """
    # integration parameters
    t0, t1, num_steps, _ = ode_integration_defaults(ode_dict['style_ode'])
    times = np.linspace(t0, t1, num_steps + 1)

    plt.figure(figsize=(5, 5))
    ax = plt.gca()

    if init_conds is None:
        nn = 2
        np.random.seed(0)
        init_conds = np.random.uniform(low=axlow, high=axhigh, size=(nn, 3))

    for init_cond in init_conds:
        print(init_cond.shape, init_cond)
        single_cell = SingleCell(init_cond, style_ode=ode_dict['style_ode'], params_ode=ode_dict['params'], label='')
        r, times = simulate_dynamics_general(init_cond, times, single_cell, method=dynamics_method)
        ax.plot(r[:, 0], r[:, 1], '-.', linewidth=0.1)
        # draw arrows every k points
        # Note: see mpl quiver to do this vectorized
        for idx in range(0, num_steps, k):
            arrow_vec = r[idx+1, :] - r[idx,:]
            dx, dy, _ = arrow_vec
            x, y, _ = r[idx, :]
            print(x, y, dx, dy)
            ax.arrow(x, y, dx, dy, zorder=10, width=1e-4)

    plt.axhline(0, linestyle='--', color='k')
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel(PLOT_XLABEL)
    plt.ylabel(PLOT_YLABEL)
    plt.title('Example trajectories')
    plt.savefig('test_arrows.pdf')
    plt.show()

def vectorfield_general(ode_dict, delta=0.1, axlow=0.0, axhigh=120.0, **ode_kwargs):
    """
    ode_kwargs:
        'z': Scalar z represents static Bam concentration
        't': Scalar t represents time
    """
    x = np.arange(axlow, axhigh, delta)
    y = np.arange(axlow, axhigh, delta)
    X, Y = np.meshgrid(x, y)

    params = ode_dict['params']
    U, V = ode_choose_vectorfield(ode_dict['style_ode'], params, X, Y, two_dim=True, **ode_kwargs)
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
    ax0.set_title('%s Vector field' % ode_dict['style_ode'])
    ax0.set_xlabel(PLOT_XLABEL)
    ax0.set_ylabel(PLOT_YLABEL)

    plt.tight_layout()
    plt.show()


def contourplot_general(ode_dict, delta=0.1, axlow=0.0, axhigh=120.0, **ode_kwargs):
    """
    ode_kwargs:
        'z': Scalar z represents static Bam concentration
        't': Scalar t represents time
    """
    #ax_lims = 100
    #Y, X = np.mgrid[0:ax_lims:100j, 0:ax_lims:100j]

    x = np.arange(axlow, axhigh, delta)
    y = np.arange(axlow, axhigh, delta)
    X, Y = np.meshgrid(x, y)

    params = ode_dict['params']
    U, V = ode_choose_vectorfield(ode_dict['style_ode'], params, X, Y, two_dim=True, **ode_kwargs)
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


def nullclines_general(ode_dict, flip_axis=False, contour_labels=True,
                       delta=0.1, axlow=0.0, axhigh=120.0, **ode_kwargs):
    """
    style_dict has the form
        'style_ode': 'PWL' or 'Yang2013'
        'params': params dict
        't': optional parameter for PWL
        'z': optional parameter for Yang2013
    """
    x = np.arange(axlow, axhigh, delta)
    y = np.arange(axlow, axhigh, delta)
    X, Y = np.meshgrid(x, y)

    params = ode_dict['params']
    U, V = ode_choose_vectorfield(ode_dict['style_ode'], params, X, Y, two_dim=True, **ode_kwargs)
    U = nan_mask(U)
    V = nan_mask(V)

    if flip_axis:
        # swap X, Y
        tmp = X
        X = Y
        Y = tmp
        # swap labels
        label_x = PLOT_YLABEL
        label_y = PLOT_XLABEL
    else:
        label_x = PLOT_XLABEL
        label_y = PLOT_YLABEL

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    # plot nullclines
    nullcline_u = ax.contour(X, Y, U, (0,), colors='b', linewidths=1.5)
    nullcline_v = ax.contour(X, Y, V, (0,), colors='r', linewidths=1.5)
    if contour_labels:
        ax.clabel(nullcline_u, inline=1, fmt='X nc', fontsize=10)
        ax.clabel(nullcline_v, inline=1, fmt='Y nc', fontsize=10)
    # gridlines
    plt.axhline(0, linewidth=1, color='k', linestyle='--')
    plt.axvline(0, linewidth=1, color='k', linestyle='--')
    # plot labels
    ax.set_title('%s nullclines (blue=%s, red=%s)' % (ode_dict['style_ode'], PLOT_XLABEL, PLOT_YLABEL))
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    plt.show()


if __name__ == '__main__':
    '''
    #example_vectorfield()

    sinit_cond = (10.0, 0, 0)
    single_cell = SingleCell(init_cond)
    plot_vectorfield_2D(single_cell)'''

    flag_Yang2013 = True
    flag_PWL = True

    if flag_Yang2013:
        params_Yang2013 = ode_choose_params('Yang2013')
        ode_dict_Yang2013 = {
            'style_ode': 'Yang2013',
            'params': params_Yang2013
        }
        kwargs_Yang2013 = {
            'z': 0
        }

        phaseplot_general(ode_dict_Yang2013, axlow=0, axhigh=120)
        '''
        vectorfield_general(ode_dict_Yang2013, axlow=0, axhigh=120, **kwargs_Yang2013)
        contourplot_general(ode_dict_Yang2013, axlow=0, axhigh=120, **kwargs_Yang2013)
        nullclines_general(ode_dict_Yang2013, axlow=0, axhigh=120, contour_labels=False, flip_axis=False, **kwargs_Yang2013)
        nullclines_general(ode_dict_Yang2013, axlow=0, axhigh=120, contour_labels=False, flip_axis=True, **kwargs_Yang2013)

    if flag_PWL:
        params_PWL = ode_choose_params('PWL')
        ode_dict_PWL = {
            'style_ode': 'PWL',
            'params': params_PWL
        }
        kwargs_PWL = {
            'z': 0,
            't': 0
        }

        phaseplot_general(ode_dict_PWL, axlow=-5.0, axhigh=5.0)
        vectorfield_general(ode_dict_PWL, delta=0.01, axlow=-5.0, axhigh=5.0, **kwargs_PWL)
        contourplot_general(ode_dict_PWL, delta=0.01, axlow=-5.0, axhigh=5.0, **kwargs_PWL)
        nullclines_general(ode_dict_PWL, delta=0.01, axlow=-5.0, axhigh=5.0, contour_labels=False, flip_axis=False, **kwargs_PWL)
        nullclines_general(ode_dict_PWL, delta=0.01, axlow=-5.0, axhigh=5.0, contour_labels=False, flip_axis=True, **kwargs_PWL)'''

