import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell.singlecell_visualize import plot_as_radar

# TODO
# 1) alternate visualization of the state itself e.g. as arrows, w indicator for housekeeping, signalling
# 2) alternate visualization as state overtime, 2 x STEPS matrix of ints or colors
# 3) other post run output is cell cell overlap timeseries, c(t)


def simple_vis(lattice, simsetup, lattice_plot_dir, title, savemod=''):
    cell_a = lattice[0][0]
    cell_b = lattice[0][1]
    proj_a = cell_a.get_memories_projection(simsetup['A_INV'], simsetup['XI'])
    proj_b = cell_b.get_memories_projection(simsetup['A_INV'], simsetup['XI'])

    fig, axarr = plt.subplots(1, 2, subplot_kw={'polar': True})
    plot_as_radar(proj_a, simsetup['CELLTYPE_LABELS'], color='blue', rotate_labels=True, fig=fig, ax=axarr[0])
    plot_as_radar(proj_b, simsetup['CELLTYPE_LABELS'], color='red', rotate_labels=True, fig=fig, ax=axarr[1])
    plt.suptitle(title)
    # save figure
    plt.savefig(os.path.join(lattice_plot_dir, 'twocell_radar_step%s.png' % savemod), dpi=120)
    plt.close()
    return


def lattice_timeseries_grid(data_dict, simsetup, plot_data_dir, savemod='', state_int=False, conj=False):
    """
    Plots lattice overtime as a 2 x T x p grid, with colour representing projection onto each cell type, horizontal time
    """
    if conj:
        num_subplots = 3  # cell A, B + cbar
        num_steps = data_dict['memory_proj_arr'][0].shape[1]
        imshow_A = np.zeros((simsetup['P'], num_steps))
        imshow_B = np.zeros((simsetup['P'], num_steps))
    else:
        num_subplots = simsetup['P']+1

    fig, axarr = plt.subplots(num_subplots, 1, figsize=(10, 8))
    # use imshow to make grids
    cmap = plt.get_cmap('PiYG')
    imshow_kw = {'vmin': -1.0, 'vmax': 1.0}
    if conj:
        for i in xrange(simsetup['P']):
            imshow_A[i, :] = data_dict['memory_proj_arr'][i][0, :]
            imshow_B[i, :] = data_dict['memory_proj_arr'][i][1, :]
        im = axarr[0].imshow(imshow_A, cmap=cmap, **imshow_kw)
        im = axarr[1].imshow(imshow_B, cmap=cmap, **imshow_kw)
        axarr[0].set_ylabel('Cell A projections')
        axarr[1].set_ylabel('Cell B projections')
    else:
        for i in xrange(simsetup['P']):
            im = axarr[i].imshow(data_dict['memory_proj_arr'][i], cmap=cmap, **imshow_kw)
            #im = axarr[i].imshow(ll[i], cmap=cmap)
            axarr[i].set_ylabel('mem %d' % i)
    # decorate
    plt.suptitle('Memory projections over time')
    axarr[num_subplots - 1].set_xlabel('Lattice timestep')
    # create colorbar
    axarr[-1].axis('off')
    cbar_kw = {'aspect': 30, 'pad': 0.01, 'orientation': 'horizontal'}   # larger aspect, thinner bar
    cbar = plt.colorbar(im, ax=axarr[-1], **cbar_kw)
    cbarlabel = 'Memory projection'
    cbar.ax.set_xlabel(cbarlabel, va="bottom", fontsize=12, labelpad=20)
    # TODO
    if state_int:
        # data_dict['grid_state_int'] 2xT as text into bars
        print 'todo state_int vis'
    # save figure
    plt.savefig(os.path.join(plot_data_dir, 'twocell_memprojvis%s.png' % savemod), dpi=120)
    plt.close()
    return


def lattice_timeseries_overlap(data_dict, simsetup, plot_data_dir, savemod='', ax=None):
    if ax is None:
        ax = plt.figure().gca()
    ax.plot(range(len(data_dict['overlap'])), data_dict['overlap'], '-ok')
    ax.set_title('Dual cell: overlap over time')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Overlap $s^a(t) \cdot s^b(t)/N(t)$')
    ax.set_ylim((-1,1))
    ax.axhline(y=0.0, linestyle='--', c='k')
    # save figure
    plt.savefig(os.path.join(plot_data_dir, 'twocell_overlap%s.png' % savemod), dpi=120)
    plt.close()
    return


def lattice_timeseries_energy(data_dict, simsetup, plot_data_dir, savemod='', ax=None):
    if ax is None:
        ax = plt.figure().gca()

    num_steps = len(data_dict['overlap'])
    ax.plot(range(num_steps), data_dict['multi_hamiltonian'], '-ok', label=r'$H_{tot}(t)$')
    ax.plot(range(num_steps), data_dict['single_hamiltonians'][0, :], '-ob', label=r'$H_0(s^a(t))$')
    ax.plot(range(num_steps), data_dict['single_hamiltonians'][1, :], '-or', label=r'$H_0(s^b(t))$')
    ax.plot(range(num_steps), data_dict['unweighted_coupling_term'][:], '-og', label='Coupling term (unweighted)')

    ax.set_title('Dual cell hamiltonian vs single cell hamiltonian')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$H(s)$')
    #ax.set_ylim((-1,1))
    ax.axhline(y=-(simsetup['N'] - simsetup['P'])/2.0, linestyle='--', c='grey', alpha=0.7, label=r'$H_0(\xi^i)$ (global minima)')
    ax.legend()
    # save figure
    plt.savefig(os.path.join(plot_data_dir, 'twocell_energy%s.png' % savemod), dpi=120)
    plt.close()
    return
