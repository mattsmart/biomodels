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
OUTPUT: hexagonal grid with labels coloured appropriately
"""

# GLOBALS
axis_buffer = 20.0
axis_length = 100.0
axis_tick_length = int(axis_length + axis_buffer)
label_colour_dict = {'R': "red",
                     'D': "cyan",#"green",
                     '_': "white"} # or nofill
hex_flag = False # True if you want actual hexagons instead of circles

# FUNCTIONS

def hex_grid_draw(grid, n):
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
            cell_label = grid[i][j].label
            hex_colour = label_colour_dict[cell_label]
            if hex_flag:
                hex_ij = mpatches.RegularPolygon((x,y), numVertices=6, radius=hex_radius, facecolor=hex_colour, ec="k")
                plt.gca().add_patch(hex_ij)
            else:
                hex_ij = plt.Circle((x,y), radius=hex_radius, color=hex_colour, ec="k")
                plt.gca().add_artist(hex_ij)
            x = x + dx
                
        y = y - dy
        x = x0 + ((i+1)%2)*hex_radius # shift odd rows to the right a bit

    return

def hex_plotter(grid, time, n, hex_plot_dir):
    hex_grid_draw(grid, n)
    f_handle = plt.gcf()
    f_handle.set_size_inches(16.0,16.0)
    axis_ticks = range(0, axis_tick_length + 1, 10)
    plt.xticks(axis_ticks)
    plt.yticks(axis_ticks)
    plt.axis('off')
    plt.savefig(hex_plot_dir + 'grid_at_time_%f.png' % time, bbox_inches='tight')
    plt.clf()
    return

# MAIN

##def main():
##    n = 100
##    hex_grid_draw(grid, n)
##    plotter(grid, n)
##    print "plotted"
##
##main()
