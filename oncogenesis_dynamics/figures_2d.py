import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

from constants import OUTPUT_DIR, Z_TO_COLOUR_BISTABLE_WIDE, Z_TO_COLOUR_ORIG
from data_io import read_matrix_data_and_idx_vals, read_params


def figure_2d_gapdist(gap_data_2d, params_general, param_1_name, param_1_range, param_2_name, param_2_range,
                      axis_gap="z", figname_mod="", flag_show=True, colours=Z_TO_COLOUR_BISTABLE_WIDE,
                      outdir=OUTPUT_DIR):
    fs = 16
    # custom cmap for gap diagram
    if params_general.feedback == 'constant':
        colours = Z_TO_COLOUR_ORIG
    xyz_cmap_gradient = LinearSegmentedColormap.from_list('xyz_cmap_gradient', colours, N=100)
    # normalize
    N = params_general.N
    gap_data_2d = gap_data_2d / float(N)
    # plot image
    plt.imshow(gap_data_2d, cmap=xyz_cmap_gradient, interpolation="none", origin='lower', aspect=0.2,
               extent=[param_2_range[0], param_2_range[-1], param_1_range[0], param_1_range[-1]])
    #plt.imshow(gap_data_2d, cmap=xyz_cmap_gradient, interpolation="none", origin='lower', aspect='auto',
    #           extent=[param_2_range[0], param_2_range[-1], param_1_range[0], param_1_range[-1]])
    ax = plt.gca()
    #ax.grid(which='major', axis='both', linestyle='-')
    #plt.title("Gap in %s between FPs, vary %s, %s" % (axis_gap, param_1_name, param_2_name), fontsize=fs)
    ax.set_xlabel(param_2_name, fontsize=fs)
    ax.set_ylabel(param_1_name, fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    # Now adding the colorbar
    cbar = plt.colorbar(orientation='horizontal', pad=0.18)
    #cbar = plt.colorbar(orientation='vertical', pad=0.08)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label('z/N', rotation=0.0, fontsize=fs)

    """
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    """

    plt.savefig(outdir + os.sep + 'fig_gap_data_2d_%s_%s_%s.pdf' % (param_1_name, param_2_name, figname_mod), bbox_inches='tight')
    if flag_show:
        plt.show()
    return plt.gca()


if __name__ == "__main__":
    gapdist = True
    basedir = "figures"

    # loaf data
    if gapdist:
        figdir = basedir + os.sep + "march 7 plots" + os.sep + "noFeedback" + os.sep + "mu 1"
        row_name = 'c'  # aka param 2 is row
        col_name = 'b'  # aka param 1 is col
        datapath = figdir + os.sep + "gapdist2d_full.txt"
        rowpath = figdir + os.sep + "gapdist2d_full_%s.txt" % row_name
        colpath = figdir + os.sep + "gapdist2d_full_%s.txt" % col_name
        paramsname = "gapdist2d_full_params.csv"

        gap_data_2d, param_2_range, param_1_range = read_matrix_data_and_idx_vals(datapath, rowpath, colpath)
        param_1_name = col_name
        param_2_name = row_name

        params_general = read_params(figdir, paramsname)
        print params_general

        figure_2d_gapdist(gap_data_2d, params_general, param_1_name, param_1_range, param_2_name, param_2_range,
                          axis_gap="z", figname_mod="", flag_show=True, outdir=figdir)
