import matplotlib.pyplot as plt
import numpy as np
import os

from multicell.multicell_lattice import read_grid_state_int
from singlecell.singlecell_functions import label_to_state
from singlecell.singlecell_simsetup import singlecell_simsetup

turquoise = [30, 223, 214]

white = [255,255,255]
soft_grey = [225, 220, 222]
soft_grey_alt1 = [206, 199, 182]
soft_grey_alt2 = [219, 219, 219]
beige = [250, 227, 199]

soft_blue = [148, 210, 226]
soft_blue_alt1 = [58, 128, 191]

soft_red = [192, 86, 64]
soft_red_alt1 = [240, 166, 144]
soft_red_alt2 = [255, 134, 113]

soft_yellow = [237, 209, 112]

soft_orange = [250, 173, 63]
soft_orange_alt1 = [248, 200, 140]

soft_green = [120, 194, 153]
sharp_green = [142, 200, 50]

soft_purple = [177, 156, 217]

color_anchor = np.array(beige) / 255.0
color_A_pos = np.array(soft_blue_alt1) / 255.0
color_A_neg = np.array(soft_orange) / 255.0

color_B_pos = np.array(soft_red) / 255.0
color_B_neg = np.array(soft_green) / 255.0

color_C_pos = np.array(soft_green) / 255.0
color_C_neg = np.array(soft_purple) / 255.0


def state_int_to_colour(state_int, simsetup, proj=True, noanti=True):

    def linear_interpolate(val, c2, c1=color_anchor):
        assert 0.0 <= val <= 1.0
        return c1 + val * (c2 - c1)

    # gewt similarities i.e. proj or overlap with all cell types
    state = label_to_state(state_int, simsetup['N'], use_neg=True)
    similarities = np.dot(simsetup['XI'].T, state) / simsetup['N']
    if proj:
        similarities = np.dot(simsetup['A_INV'], similarities)

    # convert similarities to colours as rgb
    assert simsetup['P'] == 3
    if noanti:
        colour_a = linear_interpolate(max(0, similarities[0]), color_A_pos)
        colour_b = linear_interpolate(max(0, similarities[1]), color_B_pos)
        colour_c = linear_interpolate(max(0, similarities[2]), color_C_pos)
        idx_max = np.argmax(similarities)
    else:
        colour_a = linear_interpolate(np.abs(similarities[0]), [color_A_pos, color_A_neg][similarities[0] < 0])
        colour_b = linear_interpolate(np.abs(similarities[1]), [color_B_pos, color_B_neg][similarities[0] < 0])
        colour_c = linear_interpolate(np.abs(similarities[2]), [color_C_pos, color_C_neg][similarities[0] < 0])
        idx_max = np.argmax(np.abs(similarities))
    #rgb = color_a + colk  # TODO decide if want to avg the 3 colours in this fn or use all 3 with some alpha?
    #print colour_a, colour_b, colour_c
    #print proj, similarities, colour_a, colour_b, colour_c, idx_max

    sa = np
    colour_mix = [(max(0, similarities[0])*colour_a[i] + max(0, similarities[1])*colour_b[i] + max(0, similarities[2])*colour_c[i]) /(max(0, similarities[0])+max(0, similarities[1])+max(0, similarities[2])) for i in range(3)]
    return colour_mix
    #return colour_a, colour_b, colour_c, idx_max


def reference_overlap_plotter(lattice_ints, n, lattice_plot_dir, simsetup, ref_site=(0,0), state_int=False):
    # get lattice size array of overlaps
    celltype_overlaps = np.zeros((n, n, 3))
    lattice_colours = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            #cell = lattice[i][j]
            state_int = lattice_ints[i, j]
            lattice_colours[i, j, :] = state_int_to_colour(state_int)


def replot(filename, simsetup):
    grid_state_int = read_grid_state_int(filename)

    n = len(grid_state_int)
    imshowcolours_TOP = np.zeros((n, n, 3))
    imshowcolours_A = np.zeros((n, n, 3))
    imshowcolours_B = np.zeros((n, n, 3))
    imshowcolours_C = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            """
            c1, c2, c3, idx_max = state_int_to_colour(grid_state_int[i,j], simsetup)
            top_color = [c1,c2,c3][idx_max]
            imshowcolours_TOP[i,j] = top_color
            """

            imshowcolours_TOP[i, j] = state_int_to_colour(grid_state_int[i, j], simsetup)

            """
            imshowcolours_A[i, j] = c1
            imshowcolours_B[i, j] = c2
            imshowcolours_C[i, j] = c3
            """

    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    """
    plt.imshow(imshowcolours_A, alpha=0.65)
    plt.imshow(imshowcolours_B, alpha=0.65)
    plt.imshow(imshowcolours_C, alpha=0.65)
    """
    plt.imshow(imshowcolours_TOP)

    """
    if state_int:
        state_ints = get_graph_lattice_state_ints(lattice, n)
        for (j, i), label in np.ndenumerate(state_ints):
            plt.gca().text(i, j, label, color='black', ha='center', va='center')
    """
    #plt.title('Lattice site-wise overlap with ref site %d,%d (Step=%d)' % (ref_site[0], ref_site[1], time))
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, n, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    # save figure
    plotname = os.path.dirname(filename) + os.sep + os.path.basename(filename)[:-4] + '.jpg'
    plt.savefig(plotname)
    plt.close()
    return


def replot_overlap(filename, simsetup, ref_site=(0,0), state_int=False):
    grid_state_int = read_grid_state_int(filename)
    ref_state = label_to_state(grid_state_int[0, 0], simsetup['N'], use_neg=True)
    print(grid_state_int)

    def site_site_overlap(loc):
        cellstate = label_to_state(grid_state_int[loc[0], loc[1]], simsetup['N'], use_neg=True)
        return np.dot(ref_state.T, cellstate) / float(simsetup['N'])

    # get lattice size array of overlaps
    n = grid_state_int.shape[0]
    overlaps = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            overlaps[i, j] = site_site_overlap([i, j])
    # plot
    fig = plt.figure(figsize=(12, 12))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    colourmap = plt.get_cmap('Spectral')  # see https://matplotlib.org/examples/color/colormaps_reference.html... used 'PiYG',
    plt.imshow(overlaps, cmap=colourmap, vmin=-1,vmax=1)  # TODO: normalize? also use this for other lattice plot fn...

    plt.colorbar()
    # draw gridlines
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, n, 1))
    ax.set_yticks(np.arange(-.5, n, 1))
    # mark reference
    #ax.plot(ref_site[0], ref_site[1], marker='*', c='gold')
    # save figure
    plotname = os.path.dirname(filename) + os.sep + os.path.basename(filename)[:-4] + '_ref00.jpg'
    plt.savefig(plotname)
    plt.close()
    return


if __name__ == '__main__':

    # load simsetup
    random_mem = False
    random_W = False
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W)

    # load data
    basedir = 'runs' + os.sep + 'poster_results'
    files = [basedir + os.sep + 'initcond.txt',
             basedir + os.sep + 'radius1.txt',
             basedir + os.sep + 'radius2.txt',
             basedir + os.sep + 'radius4.txt',
             basedir + os.sep + 'W1.txt',
             basedir + os.sep + 'W2.txt',
             basedir + os.sep + 'W3.txt',
             basedir + os.sep + 'W4.txt',
             basedir + os.sep + 'W5.txt',
             basedir + os.sep + 'W6.txt',
             basedir + os.sep + 'W7.txt',
             basedir + os.sep + 'W8.txt'
             ]
    for f in files:
        #replot(f, simsetup)
        replot_overlap(f, simsetup)
