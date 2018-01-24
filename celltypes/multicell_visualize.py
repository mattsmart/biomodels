import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

from singlecell.singlecell_simsetup import N

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
def lattice_draw(lattice, n, uniplot_key):
    # assume cell_per_row = n, cell_per_col = n
    cell_radius = axis_length / (2 * n)
    x0 = cell_radius
    y0 = axis_length - cell_radius
    dx = cell_radius * 2.0
    dy = cell_radius * 1.75

    x = x0
    y = y0
    for i in xrange(n):

        for j in xrange(n):
            cell = lattice[i][j]
            cell_proj = cell.get_memories_projection()
            cell_ij = mpatches.Rectangle((x, y), 2 * cell_radius, 2 * cell_radius, facecolor=memory_colour_dict[uniplot_key], ec="none", alpha=np.abs(cell_proj[uniplot_key]))  # might be faster solid square plotter
            plt.gca().add_patch(cell_ij)
            x += dx

        y -= dy
        x = x0

    return


def scatterplot_dict_array(lattice, n, dict_counts):
    keys = [5,24]
    dict_array = {key: np.zeros((2, dict_counts[key]), dtype=np.float32) for key in keys}
    dict_increment = {key: 0 for key in keys}

    # assume cell_per_row = n, cell_per_col = n
    cell_radius = axis_length / (2 * n)
    x0 = cell_radius
    y0 = (axis_length - cell_radius)
    dx = cell_radius * 2.0
    dy = cell_radius * 1.75  # 1.75

    x = x0
    y = y0
    for i in xrange(n):
        for j in xrange(n):
            cell_label = lattice[i][j].label
            if cell_label != '_':
                idx = dict_increment[cell_label]
                dict_array[cell_label][0, idx] = x
                dict_array[cell_label][1, idx] = y
                dict_increment[cell_label] += 1
            x += dx
        y -= dy
        x = x0
    return dict_array


def lattice_draw_fast(lattice, n, dict_counts):
    dict_array = scatterplot_dict_array(lattice, n, dict_counts)
    size_param = 40 * (axis_length / n) ** 2  # 40 = 20 (default area) * 2
    linewidth = 0  # use 2*axis_length / (5 * n)  or  0.5  or  0
    f = plt.figure()
    # plot actual data
    for key in dict_array.keys():
        plt.scatter(dict_array[key][0], dict_array[key][1], s=size_param, color=memory_colour_dict[key], marker='s', lw=linewidth, edgecolor='black')  # can use alpha=0.5 for 3d
    return f


def lattice_plotter(lattice, time, n, lattice_plot_dir, uniplot_key, dict_counts=None):
    # generate figure
    if fast_flag:
        lattice_draw_fast(lattice, n, dict_counts)
    else:
        lattice_draw(lattice, n, uniplot_key)
    # set figure size
    fig_handle = plt.gcf()
    fig_handle.set_size_inches(16, 16)
    # pad figure to hide gaps between squares
    scale_settings = {10: {'x': (-22, 122), 'y': (-20, 120)},
                      100: {'x': (-4, 104), 'y': (-4, 104)},
                      250: {'x': (-28, 128), 'y': (-22, 122)},
                      500: {'x': (-28, 128), 'y': (-22, 122)},
                      1000: {'x': (-28, 128), 'y': (-22, 122)}}
    if n in scale_settings.keys():
        ax_handle = plt.gca()
        axis_lims = scale_settings[n]
        ax_handle.set_xlim(axis_lims['x'])
        ax_handle.set_ylim(axis_lims['y'])
        ax_handle.add_patch((Rectangle((0, 12.5), 100, 87.5, facecolor="none")))
        ax_handle.text(46, 102, 'step = %.3f' % time)
    # hide axis
    axis_ticks = range(0, axis_tick_length + 1, 10)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    plt.axis('off')
    plt.title('Lattice site-wise projection onto memory %d' % uniplot_key)
    # save figure
    plt.savefig(os.path.join(lattice_plot_dir, 'lattice_step%d_proj%d.png' % (time, uniplot_key)), dpi=max(80.0, n/2.0))
    plt.close()
    return


def site_site_overlap(lattice, loc_1, loc_2, time):
    cellstate_1 = lattice[loc_1[0]][loc_1[1]].get_state_array()[:, time]
    cellstate_2 = lattice[loc_2[0]][loc_2[1]].get_state_array()[:, time]
    return np.dot(cellstate_1.T, cellstate_2) / N


def reference_overlap_plotter(lattice, time, n, lattice_plot_dir, ref_site=[0,0]):
    overlaps = np.zeros((n,n))

    for i in xrange(n):
        for j in xrange(n):
            #cell = lattice[i][j]
            state_overlap = site_site_overlap(lattice, [i,j], ref_site, time)
            overlaps[i,j] = state_overlap

    """
    # set figure size
    fig_handle = plt.gcf()
    fig_handle.set_size_inches(16, 16)
    # pad figure to hide gaps between squares
    scale_settings = {10: {'x': (-22, 122), 'y': (-20, 120)},
                      100: {'x': (-4, 104), 'y': (-4, 104)},
                      250: {'x': (-28, 128), 'y': (-22, 122)},
                      500: {'x': (-28, 128), 'y': (-22, 122)},
                      1000: {'x': (-28, 128), 'y': (-22, 122)}}
    if n in scale_settings.keys():
        ax_handle = plt.gca()
        axis_lims = scale_settings[n]
        ax_handle.set_xlim(axis_lims['x'])
        ax_handle.set_ylim(axis_lims['y'])
        ax_handle.add_patch((Rectangle((0, 12.5), 100, 87.5, facecolor="none")))
        ax_handle.text(46, 102, 'step = %.3f' % time)
    # hide axis
    axis_ticks = range(0, axis_tick_length + 1, 10)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    plt.axis('off')
    """
    # plot
    colourmap = plt.get_cmap('PiYG')  # see https://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(overlaps, cmap=colourmap)  # TODO: normalize? also use this for other lattice plot fn...
    plt.title('Lattice site-wise projection onto site %d, %d' % (ref_site[0],ref_site[1]))
    plt.colorbar()
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, n, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    # mark reference
    ax.plot(ref_site[0], ref_site[1], marker='*')
    # save figure
    overlapdir = 'overlapRef_%d_%d' % (ref_site[0], ref_site[1])
    if not os.path.exists(os.path.join(lattice_plot_dir, overlapdir)):
        os.makedirs(os.path.join(lattice_plot_dir, overlapdir))
    plt.savefig(os.path.join(lattice_plot_dir, overlapdir, 'lattice_%s_step%d.png' % (overlapdir, time)), dpi=max(80.0, n/2.0))
    plt.close()
    return
