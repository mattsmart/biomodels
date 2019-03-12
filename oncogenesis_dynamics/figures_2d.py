import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

from constants import OUTPUT_DIR, Z_TO_COLOUR_BISTABLE_WIDE, Z_TO_COLOUR_ORIG
from data_io import read_matrix_data_and_idx_vals, read_params


def truncate_data(arr, param_1_range, param_2_range, low_1=None, high_1=None, low_2=None, high_2=None):
    print arr.shape, (len(param_2_range), len(param_1_range))
    assert arr.shape == (len(param_1_range), len(param_2_range))

    low_1_int = 1
    low_2_int = 1
    high_1_int = len(param_1_range)
    high_2_int = len(param_2_range)

    if low_1 is not None:
        assert param_1_range[0] <= low_1 <= param_1_range[-1]
        while param_1_range[low_1_int] < low_1:
            low_1_int += 1

    if low_2 is not None:
        assert param_2_range[0] <= low_2 <= param_2_range[-1]
        while param_2_range[low_2_int] < low_2:
            low_2_int += 1

    if high_1 is not None:
        assert param_1_range[0] <= high_1 <= param_1_range[-1]
        while param_1_range[high_1_int-1] > high_1:
            high_1_int -= 1

    if high_2 is not None:
        assert param_2_range[0] <= high_2 <= param_2_range[-1]
        while param_2_range[high_2_int-1] > high_2:
            high_2_int -= 1

    trunc_param_1_range = param_1_range[low_1_int-1:high_1_int]
    trunc_param_2_range = param_2_range[low_2_int-1:high_2_int]
    trunc_arr = arr[low_1_int-1:high_1_int, low_2_int-1:high_2_int]
    assert trunc_arr.shape == (len(trunc_param_1_range), len(trunc_param_2_range))

    return trunc_arr, trunc_param_1_range, trunc_param_2_range



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
    plt.figure(figsize=(4,8))
    plt.imshow(gap_data_2d, cmap=xyz_cmap_gradient, interpolation="none", origin='lower', aspect = 0.8,
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
    cbar = plt.colorbar(orientation='horizontal', pad=0.115, aspect=15)
    #cbar = plt.colorbar(orientation='vertical', pad=0.08)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label(r'$z/N$', rotation=0.0, fontsize=fs)

    """
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    """

    plt.savefig(outdir + os.sep + 'fig_gap_data_2d_%s_%s_%s.pdf' % (param_1_name, param_2_name, figname_mod), bbox_inches='tight')
    if flag_show:
        plt.show()
    return plt.gca()


def figure_2d_fpcount(fp_data_2d, params_general, param_1_name, param_1_range, param_2_name, param_2_range,
                      axis_gap="z", figname_mod="", flag_show=True, colours=Z_TO_COLOUR_BISTABLE_WIDE,
                      outdir=OUTPUT_DIR):
    fs = 16
    # custom cmap for fp count diagram
    colours = [(0.0, 'thistle'), (0.333333, 'lightgrey'), (0.6666666, 'lightsteelblue'), (1.0, 'moccasin')]
    fpcount_cmap_gradient = LinearSegmentedColormap.from_list('fpcount_cmap_gradient', colours, N=4)
    # regularize values
    np.clip(fp_data_2d, 0, 4, fp_data_2d)
    # plot image
    plt.figure(figsize=(5,10))
    plt.imshow(fp_data_2d, cmap=fpcount_cmap_gradient, interpolation="none", origin='lower', aspect=0.6,
               extent=[param_2_range[0], param_2_range[-1], param_1_range[0], param_1_range[-1]])
    ax = plt.gca()
    #ax.grid(which='major', axis='both', linestyle='-')
    ax.set_xlabel(param_2_name, fontsize=fs)
    ax.set_ylabel(param_1_name, fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    # Now adding the colorbar
    """
    cbar = plt.colorbar(orientation='horizontal', pad=0.115, aspect=15)
    #cbar = plt.colorbar(orientation='vertical', pad=0.08)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label(r'$z/N$', rotation=0.0, fontsize=fs)
    """
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.savefig(outdir + os.sep + 'fig_fpsum_data_2d_%s_%s_%s.pdf' % (param_1_name, param_2_name, figname_mod), bbox_inches='tight')
    if flag_show:
        plt.show()
    return plt.gca()


if __name__ == "__main__":
    gapdist = False
    fpcount = True
    basedir = "figures"

    # loaf data
    if gapdist:
        figdir = basedir + os.sep + "fig4" + os.sep + "b_c_optionAdata" + os.sep + "b_vs_c_feedback_mu0pt5_0210"
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
        # truncate block
        gap_data_2d, param_1_range, param_2_range = truncate_data(gap_data_2d, param_1_range, param_2_range,
                                                                  low_1=0.2, low_2=0.2, high_2=1.4)

        figure_2d_gapdist(gap_data_2d, params_general, param_1_name, param_1_range, param_2_name, param_2_range,
                          axis_gap="z", figname_mod="", flag_show=True, outdir=figdir)

    if fpcount:
        figdir = basedir + os.sep + "fig4" + os.sep + "b_c_optionAdata" + os.sep + "b_vs_c_feedback_mu0pt5_0210"
        row_name = 'c'  # aka param 2 is row
        col_name = 'b'  # aka param 1 is col
        datapath = figdir + os.sep + "fpPhysStableSumfpcount2d_full.txt"
        rowpath = figdir + os.sep + "fpPhysStableSumfpcount2d_full_%s.txt" % row_name
        colpath = figdir + os.sep + "fpPhysStableSumfpcount2d_full_%s.txt" % col_name
        paramsname = "fpPhysStableSumfpcount2d_full_params.csv"

        fpsum_data_2d, param_2_range, param_1_range = read_matrix_data_and_idx_vals(datapath, rowpath, colpath)
        param_1_name = col_name
        param_2_name = row_name
        params_general = read_params(figdir, paramsname)
        print params_general
        # truncate block
        fpsum_data_2d, param_1_range, param_2_range = truncate_data(fpsum_data_2d, param_1_range, param_2_range,
                                                                  low_1=0.2, low_2=0.2, high_1=1.65, high_2=1.205)

        figure_2d_fpcount(fpsum_data_2d, params_general, param_1_name, param_1_range, param_2_name, param_2_range,
                          figname_mod="", flag_show=True, outdir=figdir)
