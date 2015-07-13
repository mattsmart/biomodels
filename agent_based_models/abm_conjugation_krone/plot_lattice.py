import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
COMMENTS:
    -radius seems to extend 85% of r, to intersect middle of line seg
        -eg. radius 10 means hex takes up almost 20 x slots
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
label_colour_dict = {'R': "red",
                     'D': "cyan",  # "green"
                     '_': "white"}  # or no fill
separation_flag = False  # True if you want some distance between agents


# Functions
# =================================================
def lattice_draw(lattice, n):
    # assume hex_per_row = n, hex_per_col = n
    hex_radius = axis_length / (2*n)
    x0 = hex_radius
    y0 = axis_length - hex_radius
    dx = hex_radius*2.0
    dy = hex_radius*1.75

    x = x0
    y = y0
    for i in xrange(n):

        for j in xrange(n):
            cell_label = lattice[i][j].label
            hex_colour = label_colour_dict[cell_label]
            if hex_flag:
                hex_ij = mpatches.RegularPolygon((x,y), numVertices=6, radius=lattice_radius, facecolor=lattice_colour, ec="k")
                plt.gca().add_patch(hex_ij)
            else:
                hex_ij = plt.Circle((x,y), radius=lattice_radius, color=lattice_colour, ec="k")
                plt.gca().add_artist(hex_ij)
            x = x + dx

        y = y - dy
        x = x0 + ((i+1)%2)*hex_radius # shift odd rows to the right a bit

    return


def lattice_plotter(lattice, time, n, lattice_plot_dir):
    lattice_draw(lattice, n)
    f_handle = plt.gcf()
    f_handle.set_size_inches(16.0,16.0)
    axis_ticks = range(0, axis_tick_length + 1, 10)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    plt.axis('off')
    plt.savefig(lattice_plot_dir + 'lattice_at_time_%f.png' % time, bbox_inches='tight')
    plt.clf()
    return

# MAIN

##def main():
##    n = 100
##    hex_lattice_draw(lattice, n)
##    plotter(lattice, n)
##    print "plotted"
##
##main()
