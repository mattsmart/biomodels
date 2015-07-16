import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


"""
COMMENTS:
    -radius seems to extend 85% of r, to intersect middle of line seg
        -eg. radius 10 means cell takes up almost 20 x slots
    -JAMES: will try circles
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
label_colour_dict = {'R': "seagreen",  # red
                     'D': "crimson",  # cyan
                     '_': "none"}  # or white
fast_flag = True  # True - fast / simple plotting
nutrient_text_flag = False  # True - plot nutrient quantity at each grid location (slow)


# Functions
# =================================================
def lattice_draw(lattice, n):
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
            cell_label = lattice[i][j].label
            cell_colour = label_colour_dict[cell_label]
            """
            if fast_flag:
                if cell_label != '_':
                    cell_ij = plt.Circle((x, y), radius=cell_radius / 2, color=cell_colour, ec="k")
                    plt.gca().add_artist(cell_ij)
            else:
                if cell_label != '_':
                    cell_ij = mpatches.Rectangle((x, y), 2 * cell_radius, 2 * cell_radius, facecolor=cell_colour, ec="k")  # might be faster solid square plotter
                    plt.gca().add_patch(cell_ij)
            if nutrient_text_flag:
                cell_nutrients = lattice[i][j].nutrients
                plt.text(x + 0.2 * dx, y + 0.8 * dy, str(cell_nutrients), fontsize=12)
            """
            cell_ij = mpatches.Rectangle((x, y), 2 * cell_radius, 2 * cell_radius, facecolor=cell_colour, ec="none")  # might be faster solid square plotter
            plt.gca().add_patch(cell_ij)
            x += dx

        y -= dy
        x = x0

    return


def scatterplot_dict_array(lattice, n, dict_counts):
    keys = ['_', 'R', 'D']
    dict_array = {key: np.zeros((2, dict_counts[key]), dtype=np.float32) for key in keys}
    dict_increment = {key: 0 for key in keys}

    # assume cell_per_row = n, cell_per_col = n
    cell_radius = axis_length / (2 * n)
    x0 = cell_radius
    y0 = (axis_length - cell_radius)
    dx = cell_radius * 2.0
    dy = cell_radius * 1.65  # 1.75

    x = x0
    y = y0
    for i in xrange(n):
        for j in xrange(n):
            cell_label = lattice[i][j].label
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
    f = plt.figure()
    size_param = 40 * (axis_length / n) ** 2  # 40 = 20 (default area) * 2
    linewidth = axis_length / (10 * n)  # 0.5  # TODO FIX try cell radius / 5 or 10
    for key in dict_array.keys():
        plt.scatter(dict_array[key][0], dict_array[key][1], s=size_param, color=label_colour_dict[key], marker='s', lw=linewidth, edgecolor='black')  # can use alpha=0.5 for 3d
    return f


def lattice_plotter(lattice, time, n, dict_counts, lattice_plot_dir):
    if fast_flag:
        lattice_draw_fast(lattice, n, dict_counts)
    else:
        lattice_draw(lattice, n)
    f_handle = plt.gcf()
    f_handle.set_size_inches(16.0, 16.0)
    axis_ticks = range(0, axis_tick_length + 1, 10)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    plt.axis('off')
    plt.savefig(lattice_plot_dir + 'lattice_at_time_%f.png' % time, bbox_inches='tight', dpi=max(80.0, n/2.0))
    plt.close()
    return
