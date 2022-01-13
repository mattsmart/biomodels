import matplotlib.pyplot as plt
import numpy as np

from class_singlecell import SingleCell
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


def Yang2013_manual_vectorfield(z=0):
    """
    Scalar z represents static Bam concentration
    """
    # Block for example code
    ax_lims = 10
    Y, X = np.mgrid[-ax_lims:ax_lims:100j, -ax_lims:ax_lims:100j]

    # Bock for manual Yang2013 code
    p = {
        'k_synth': 1,  # nM / min
        'a_deg': 0.01,  # min^-1
        'b_deg': 0.04,  # min^-1
        'EC50_deg': 32,  # nM
        'n_deg': 17,  # unitless
        'a_Cdc25': 0.16,  # min^-1
        'b_Cdc25': 0.80,  # min^-1
        'EC50_Cdc25': 35,  # nM
        'n_Cdc25': 11,  # unitless
        'a_Wee1': 0.08,  # min^-1
        'b_Wee1': 0.40,  # min^-1
        'EC50_Wee1': 30,  # nM
        'n_Wee1': 3.5,  # unitless
    }
    p['k_Bam'] = 1

    # setup factors
    k_synth = p['k_synth']

    # "f(x)" factor of the review
    x_d = X ** p['n_deg']
    ec50_d = p['EC50_deg'] ** p['n_deg']
    degradation = p['a_deg'] + p['b_deg'] * x_d / (ec50_d + x_d)
    degradation_scaled = degradation / (1 + z / p['k_Bam'])  # as in p7 of SmallCellCluster Review draft

    # "g(x)" factor of the review - activation by Cdc25
    x_plus = X ** p['n_Cdc25']
    ec50_plus = p['EC50_Cdc25'] ** p['n_Cdc25']
    activation = p['a_Cdc25'] + p['b_Cdc25'] * x_plus / (ec50_plus + x_plus)

    # "k_i" factor of the review - de-activation by Wee1
    x_minus = X ** p['n_Wee1']
    ec50_minus = p['EC50_Wee1'] ** p['n_Wee1']
    deactivation = p['a_Wee1'] + p['b_Wee1'] * ec50_minus / (ec50_minus + x_minus)

    U = k_synth - degradation_scaled * X + activation * (X - Y) - deactivation * X
    V = k_synth - degradation_scaled * Y

    U_mask_arr = np.isfinite(U)
    V_mask_arr = np.isfinite(V)
    U[U_mask_arr] = 1000
    V[V_mask_arr] = 1000

    # Block for example code
    print(type(U))
    print(U.shape)
    print(U)
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


if __name__ == '__main__':
    #example_vectorfield()
    Yang2013_manual_vectorfield()

    #sinit_cond = (10.0, 0, 0)
    #single_cell = SingleCell(init_cond)
    #plot_vectorfield_2D(single_cell)