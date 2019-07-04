import matplotlib.pyplot as plt

from singlecell.singlecell_visualize import plot_as_radar

# TODO
# 1) alternate visualization of the state itself e.g. as arrows, w indicator for housekeeping, signalling
# 2) alternate visualization as state overtime, 2 x STEPS matrix of ints or colors
# 3) other post run output is cell cell overlap timeseries, c(t)

def simple_vis(lattice, simsetup, title):

    cell_a = lattice[0][0]
    cell_b = lattice[0][1]
    proj_a = cell_a.get_memories_projection(simsetup['A_INV'], simsetup['XI'])
    proj_b = cell_b.get_memories_projection(simsetup['A_INV'], simsetup['XI'])

    fig, axarr = plt.subplots(1, 2, subplot_kw={'polar': True})
    plot_as_radar(proj_a, simsetup['CELLTYPE_LABELS'], color='blue', rotate_labels=True, fig=fig, ax=axarr[0])
    plot_as_radar(proj_b, simsetup['CELLTYPE_LABELS'], color='red', rotate_labels=True, fig=fig, ax=axarr[1])
    plt.suptitle(title)
    plt.show()
    return
