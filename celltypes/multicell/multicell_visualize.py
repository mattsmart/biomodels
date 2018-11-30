import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

from singlecell.singlecell_functions import single_memory_projection

"""
COMMENTS:
    -radius seems to extend 85% of r, to intersect middle of line seg
        -eg. radius 10 means cell takes up almost 20 x slots
INPUT:
   1) n
   2) list of lists, of size n x n, containing labels (corresponding to colour)
OUTPUT: rectangular lattice with labels coloured appropriately
"""


# Constants
# =================================================
axis_buffer = 20.0
axis_length = 100.0
axis_tick_length = int(axis_length + axis_buffer)
memory_keys = [5,24]
memory_colour_dict = {5: 'blue', 24: 'red'}
fast_flag = False  # True - fast / simple plotting
#nutrient_text_flag = False  # True - plot nutrient quantity at each grid location (slow)  TODO: plot scalar at each location?


# Functions
# =================================================

def get_lattice_uniproj(lattice, time, n, uniplot_key, simsetup):
    proj_vals = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            proj_vals[i, j] = single_memory_projection(lattice[i][j].get_state_array(), time, uniplot_key, simsetup['ETA'])
    return proj_vals


def lattice_uniplotter(lattice, time, n, lattice_plot_dir, uniplot_key, simsetup, dict_counts=None):
    # generate figure data
    proj_vals = get_lattice_uniproj(lattice, time, n, uniplot_key, simsetup)
    # plot projection
    colourmap = plt.get_cmap('PiYG')
    plt.imshow(proj_vals, cmap=colourmap, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Lattice site-wise projection onto memory %d (%s) (Step=%d)' % (uniplot_key, simsetup['CELLTYPE_LABELS'][uniplot_key], time))
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, n, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    # save figure
    plt.savefig(os.path.join(lattice_plot_dir, 'proj%d_lattice_step%d.png' % (uniplot_key, time)), dpi=max(80.0, n/2.0))
    plt.close()
    return


def lattice_projection_composite(lattice, time, n, lattice_plot_dir, simsetup):
    assert simsetup['P'] == 63  # TODO: hardcoded rows and columns should change with P
    empty_subplots = [[-1, -1]]
    ncol = 8
    nrow = 8

    # prep figure
    fig, ax = plt.subplots(nrow, ncol)
    fig.set_size_inches(16, 16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Lattice projection onto p=%d memories (Step=%d)' % (simsetup['P'], time), fontsize=20)
    colourmap = plt.get_cmap('PiYG')  # 'PiYG' or 'Spectral'

    mem_idx = 0
    for row in xrange(nrow):
        for col in xrange(ncol):
            subax = ax[row][col]
            # plot data
            proj_vals = get_lattice_uniproj(lattice, time, n, mem_idx, simsetup)
            im = subax.imshow(proj_vals, cmap=colourmap, vmin=-1, vmax=1)
            # hide axis nums
            subax.set_title('%d (%s)' % (mem_idx, simsetup['CELLTYPE_LABELS'][mem_idx][:24]), fontsize=8)
            labels = [item.get_text() for item in subax.get_xticklabels()]
            empty_string_labels = [''] * len(labels)
            subax.set_xticklabels(empty_string_labels)
            labels = [item.get_text() for item in subax.get_yticklabels()]
            empty_string_labels = [''] * len(labels)
            subax.set_yticklabels(empty_string_labels)
            # nice gridlines
            subax.set_xticks(np.arange(-.5, n, 1))
            subax.set_yticks(np.arange(-.5, n, 1))
            subax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            mem_idx += 1
            if mem_idx == simsetup['P']:
                break

    # turn off empty boxes
    for pair in empty_subplots:
        ax[pair[0], pair[1]].axis('off')
    # plot colourbar
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), ticks=[-1, 0, 1], orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    # save figure
    plt.savefig(os.path.join(lattice_plot_dir, 'composite_lattice_step%d.png' % time), dpi=max(120.0, n/2.0))
    plt.close()
    return


def site_site_overlap(lattice, loc_1, loc_2, time, N):
    cellstate_1 = lattice[loc_1[0]][loc_1[1]].get_state_array()[:, time]
    cellstate_2 = lattice[loc_2[0]][loc_2[1]].get_state_array()[:, time]
    return np.dot(cellstate_1.T, cellstate_2) / N


def reference_overlap_plotter(lattice, time, n, lattice_plot_dir, simsetup, ref_site=(0,0)):
    # get lattice size array of overlaps
    overlaps = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            #cell = lattice[i][j]
            state_overlap = site_site_overlap(lattice, [i,j], ref_site, time, simsetup['N'])
            overlaps[i,j] = state_overlap
    # plot
    colourmap = plt.get_cmap('Spectral')  # see https://matplotlib.org/examples/color/colormaps_reference.html... used 'PiYG',
    plt.imshow(overlaps, cmap=colourmap, vmin=-1,vmax=1)  # TODO: normalize? also use this for other lattice plot fn...
    plt.colorbar()
    plt.title('Lattice site-wise overlap with ref site %d,%d (Step=%d)' % (ref_site[0], ref_site[1], time))
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, n, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    # mark reference
    ax.plot(ref_site[0], ref_site[1], marker='*', c='gold')
    # save figure
    overlapname = 'overlapRef_%d_%d' % (ref_site[0], ref_site[1])
    if not os.path.exists(os.path.join(lattice_plot_dir, overlapname)):
        os.makedirs(os.path.join(lattice_plot_dir, overlapname))
    plt.savefig(os.path.join(lattice_plot_dir, overlapname, 'lattice_%s_step%d.png' % (overlapname, time)), dpi=max(80.0, n/2.0))
    plt.close()
    return
