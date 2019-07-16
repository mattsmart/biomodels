#import matplotlib as mpl            # Fix to allow intermediate compatibility of radar label rotation / PyCharm SciView
#mpl.use("TkAgg")                    # Fix to allow intermediate compatibility of radar label rotation / PyCharm SciView
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi

from singlecell_functions import label_to_state, state_to_label, hamiltonian, check_min_or_max, hamming, get_all_fp, calc_state_dist_to_local_min


def plot_as_bar(projection_vec, memory_labels, alpha=1.0):
    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5)
    h = plt.bar(xrange(len(memory_labels)), projection_vec, label=memory_labels, alpha=alpha)
    plt.subplots_adjust(bottom=0.3)
    xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] for patch in h]
    plt.xticks(xticks_pos, memory_labels, ha='right', rotation=45, size=7)
    return fig, plt.gca()


def plot_as_radar(projection_vec, memory_labels, color='b', rotate_labels=True, fig=None, ax=None):
    """
    # radar plots not built-in to matplotlib
    # reference code uses pandas: https://python-graph-gallery.com/390-basic-radar-chart/
    """

    p = len(memory_labels)

    # Angle of each axis in the plot
    angles = [n / float(p) * 2 * pi for n in xrange(p)]

    # Add extra element to angles and data array to close off filled area
    angles += angles[:1]
    projection_vec_ext = np.zeros(len(angles))
    projection_vec_ext[0:len(projection_vec)] = projection_vec[:]
    projection_vec_ext[-1] = projection_vec[0]

    # Initialise the spider plot
    if fig is None:
        assert ax is None
        fig, ax = plt.subplot(111, polar=True)
        fig.set_size_inches(9, 5)
    else:
        fig = plt.gcf()
        fig.set_size_inches(9, 5)

    # Draw one ax per variable + add labels
    ax.set_xticks(angles)
    ax.set_xticklabels(memory_labels)

    # Draw ylabels
    ax.set_rlabel_position(45)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])
    ax.set_ylim(-1, 1)
    ax.tick_params(axis='both', color='grey', size=12)

    # Plot data
    ax.plot(angles, projection_vec_ext, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, projection_vec_ext, color, alpha=0.1)

    # Rotate the type labels
    if rotate_labels:
        fig.canvas.draw()  # trigger label positions to extract x, y coords
        angles = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
        angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles)
        labels=[]
        for label, angle in zip(ax.get_xticklabels(), angles):
            x, y = label.get_position()
            lab = ax.text(x, y - 0.05, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va(), size=8)
            lab.set_rotation(angle)
            labels.append(lab)
        ax.set_xticklabels([])

    return fig, ax


def plot_state_prob_map(simsetup, beta=None, field=None, fs=0.0, ax=None, decorate_FP=True):
    if ax is None:
        ax = plt.figure(figsize=(8,6)).gca()

    fstring = 'None'
    if field is not None:
        fstring = '%.2f' % fs
    N = simsetup['N']
    num_states = 2 ** N
    energies = np.zeros(num_states)
    colours = ['blue' for i in xrange(num_states)]
    fpcolor = {True: 'green', False: 'red'}
    for label in xrange(num_states):
        state = label_to_state(label, N, use_neg=True)
        energies[label] = hamiltonian(state, simsetup['J'], field=field, fs=fs)
        if decorate_FP:
            is_fp, is_min = check_min_or_max(simsetup, state, energy=energies[label], field=field, fs=fs)
            if is_fp:
                colours[label] = fpcolor[is_min]
    if beta is None:
        ax.scatter(range(2 ** N), energies, c=colours)
        ax.set_title(r'$H(s), \beta=\infty$, field=%s' % (fstring))
        #ax.set_ylim((-10,10))
    else:
        ax.scatter(range(2 ** N), np.exp(-beta * energies), c=colours)
        ax.set_yscale('log')
        ax.set_title(r'$e^{-\beta H(s)}, \beta=%.2f$, field=%s' % (beta, fstring))
    plt.show()
    return


def hypercube_visualize(simsetup, X_reduced, energies, elevate3D=True, edges=True, all_edges=False,
                        minima=[], maxima=[], colours_dict=None, basin_labels=None, surf=True, ax=None):
    """
    Plot types
        A - elevate3D=True, surf=True, colours_override=None     - 3d surf, z = energy
        B - elevate3D=True, surf=False, colours_override=None    - 3d scatter, z = energy, c = energy
        C - elevate3D=True, surf=False, colours_override=list(N) - 3d scatter, z = energy, c = predefined (e.g. basins colour-coded)
        D - elevate3D=False, colours_override=None               - 2d scatter, c = energy
        E - elevate3D=False, colours_override=list(N)            - 2d scatter, c = predefined (e.g. basins colour-coded)
        F - X_reduced is dim 2**N x 3, colours_override=None     - 3d scatter, c = energy
        G - X_reduced is dim 2**N x 3, colours_override=list(N)  - 3d scatter, c = predefined (e.g. basins colour-coded)
    All plots can have partial or full edges (neighbours) plotted
    """
    # TODO annotate minima maxima
    # TODO neighbour preserving?
    # TODO think there are duplicate points in hd rep... check this bc pics look too simple

    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

    # setup data
    N = simsetup['N']
    states = np.array([label_to_state(label, N) for label in xrange(2 ** N)])

    # setup cmap
    energies_norm = (energies + np.abs(np.min(energies))) / (np.abs(np.max(energies)) + np.abs(np.min(energies)))
    if colours_dict is None:
        colours = energies_norm
    else:
        assert surf is False
        colours = colours_dict['clist']

    if X_reduced.shape[1] == 3:
        # explicit 3D plot
        sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=colours, s=20)
    else:
        assert X_reduced.shape[1] == 2
        if elevate3D:
            # implicit 3D plot, height is energy
            if surf:
                sc = ax.plot_trisurf(X_reduced[:,0], X_reduced[:,1], energies_norm, cmap=plt.cm.viridis)
            else:
                for key in colours_dict['basins_dict'].keys():
                    indices = colours_dict['basins_dict'][key]
                    sc = ax.scatter(X_reduced[indices, 0], X_reduced[indices, 1], energies_norm[indices], s=20,
                                    c=colours_dict['fp_label_to_colour'][key],
                                    label='Basin ID#%d (size %d)' % (key, len(indices)))

                sc = ax.scatter(X_reduced[:,0], X_reduced[:,1], energies_norm, c=colours, s=20)
        else:
            # 2D plot
            if colours_dict is None:
                sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colours)
            else:
                for key in colours_dict['basins_dict'].keys():
                    indices = colours_dict['basins_dict'][key]
                    sc = ax.scatter(X_reduced[indices, 0], X_reduced[indices, 1], s=20,
                                    c=colours_dict['fp_label_to_colour'][key],
                                    label='Basin ID#%d (size %d)' % (key, len(indices)))
    # legend for colours
    if colours_dict is None:
        cbar = plt.colorbar(sc)
        cbar.set_label(r'$H(s)$')
    else:
        ax.legend()

    # annotate minima
    if basin_labels is None:
        basin_labels = {a: 'ID: %d' % a for a in minima}
    for minimum in minima:
        txt = basin_labels[minimum]
        state_new = X_reduced[minimum, :]
        if elevate3D or X_reduced.shape[1] == 3:
            if elevate3D:
                z = energies_norm[minimum]
            if X_reduced.shape[1] == 3:
                z = state_new[2]
            ax.text(state_new[0], state_new[1], z, txt, fontsize=12)
        else:
            ax.annotate(txt, xy=(state_new[0], state_new[1]), fontsize=12)

    if edges:
        print 'Adding edges to plot...'
        for label in xrange(2 ** N):
            state_orig = states[label, :]
            state_new = X_reduced[label, :]
            nbrs = [0] * N
            if all_edges or label in maxima or label in minima or abs(energies_norm[label] - 1.0) < 1e-4:
                for idx in xrange(N):
                    nbr_state = np.copy(state_orig)
                    nbr_state[idx] = -1 * nbr_state[idx]
                    nbrs[idx] = state_to_label(nbr_state)
                for nbr_int in nbrs:
                    nbr_new = X_reduced[nbr_int, :]
                    x = [state_new[0], nbr_new[0]]
                    y = [state_new[1], nbr_new[1]]
                    if X_reduced.shape[1] == 3:
                        z = [state_new[2], nbr_new[2]]
                    else:
                        z = [energies_norm[label], energies_norm[nbr_int]]
                    if elevate3D or X_reduced.shape[1] == 3:
                        ax.plot(x, y, z, alpha=0.8, color='grey', lw=0.5)
                    else:
                        ax.plot(x, y, alpha=0.8, color='grey', lw=0.5)
    ax.grid('off')
    ax.axis('off')
    plt.show()
    return


def save_manual(fig, dir, fname, close=True):
    filepath = dir + os.sep + fname + ".png"
    fig.savefig(filepath, dpi=100)
    if close:
        plt.close()
    return
