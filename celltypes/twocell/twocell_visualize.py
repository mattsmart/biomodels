import matplotlib.pyplot as plt
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
