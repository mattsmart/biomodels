#import matplotlib as mpl            # Fix to allow intermediate compatibility of radar label rotation / PyCharm SciView
#mpl.use("TkAgg")                    # Fix to allow intermediate compatibility of radar label rotation / PyCharm SciView

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


def hypercube_visualize(simsetup, method, dim=2, energies=None, elevate3D=True, edges=True, all_edges=False, use_hd=False, ax=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cmx
    # TODO neighbour preserving?
    # TODO think there are duplicate points in hd rep... check this bc pics look too simple
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
    # setup data
    N = simsetup['N']
    states = np.array([label_to_state(label, N) for label in xrange(2 ** N)])
    X = states
    if use_hd:
        fp_annotation, minima, maxima = get_all_fp(simsetup, field=None, fs=0.0)
        hd = calc_state_dist_to_local_min(simsetup, minima, X=X)
        X = hd
    # setup cmap
    colours = None
    if energies is not None:
        energies_norm = (energies + np.abs(np.min(energies)))/(np.abs(np.max(energies)) + np.abs(np.min(energies)))
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim)
        X_new = pca.fit_transform(X)
    elif method == 'mds':
        from sklearn.manifold import MDS
        # simple call
        """
        X_new = MDS(n_components=2, max_iter=300, verbose=1).fit_transform(X)
        """
        statespace = 2 ** N
        dists = np.zeros((statespace, statespace), dtype=int)
        for i in xrange(statespace):
            for j in xrange(i):
                d = hamming(X[i, :], X[j, :])
                dists[i, j] = d
        dists = dists + dists.T - np.diag(dists.diagonal())
        X_new = MDS(n_components=2, max_iter=300, verbose=1, dissimilarity='precomputed').fit_transform(dists)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        perplexity_def = 30.0
        tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=perplexity_def)
        X_new = tsne.fit_transform(X)
    else:
        print 'method must be in [pca, mds, tsne]'
    if elevate3D:
        sc = ax.scatter(X_new[:,0], X_new[:,1], energies_norm, c=energies, s=20)
    else:
        sc = ax.scatter(X_new[:, 0], X_new[:, 1], c=energies)
        fig.colorbar(sc)
    for idx in xrange(2 ** N):
        print idx, X[idx,:], X_new[idx,:]
    if edges:
        print 'Adding edges to plot...'
        for label in xrange(2 ** N):
            state_orig = states[label, :]
            state_new = X_new[label, :]
            nbrs = [0] * N
            if all_edges or abs(energies_norm[label]) < 1e-4 or abs(energies_norm[label] - 1.0) < 1e-4:
                for idx in xrange(N):
                    nbr_state = np.copy(state_orig)
                    nbr_state[idx] = -1 * nbr_state[idx]
                    nbrs[idx] = state_to_label(nbr_state)
                for nbr_int in nbrs:
                    nbr_new = X_new[nbr_int, :]
                    x = [state_new[0], nbr_new[0]]
                    y = [state_new[1], nbr_new[1]]
                    z = [energies_norm[label], energies_norm[nbr_int]]
                    if elevate3D:
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
