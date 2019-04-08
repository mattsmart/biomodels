import matplotlib.pyplot as plt
import numpy as np
import os

from constants import OUTPUT_DIR, COLOURS_DARK_BLUE
from data_io import read_matrix_data_and_idx_vals, read_params, read_fpt_and_params


def subsample_data():
    # TODO
    return fpt_data_subsampled


def figure_fpt_multihist(multi_fpt_list, labels, figname_mod="def", bin_linspace=80, colours=COLOURS_DARK_BLUE,
                         figsize=(8,6), ec='k', lw=0.5, flag_norm=False, flag_xlog10=False, flag_ylog10=False,
                         flag_disjoint=False, flag_show=True, outdir=OUTPUT_DIR):

    # resize fpt lists if not all same size (to the min size)
    fpt_lengths = [len(fpt) for fpt in multi_fpt_list]
    ensemble_size = np.min(fpt_lengths)

    # cleanup data to same size
    if sum(fpt_lengths - ensemble_size) > 0:
        print "Resizing multi_fpt_list elements:", fpt_lengths, "to the min size of:", ensemble_size
        # TODO should randomize to not bias data
        for idx in xrange(len(fpt_lengths)):
            multi_fpt_list[idx] = multi_fpt_list[idx][:ensemble_size]
    bins = np.linspace(np.min(multi_fpt_list), np.max(multi_fpt_list), bin_linspace)

    # normalize
    if flag_norm:
        y_label = 'Probability'
        weights = np.ones_like(multi_fpt_list) / ensemble_size
    else:
        y_label = 'Frequency'
        weights = np.ones_like(multi_fpt_list)

    # prep fig before axes mod
    fig = plt.figure(figsize=figsize, dpi=120)
    ax = plt.gca()

    # mod axes (log)
    if flag_xlog10:
        ax.set_xscale("log", nonposx='clip')
        max_log = np.ceil(np.max(np.log10(multi_fpt_list)))
        bins = np.logspace(0.1, max_log, 100)
    if flag_ylog10:
        ax.set_yscale("log", nonposx='clip')

    # plot calls
    if flag_disjoint:
        plt.hist(multi_fpt_list, bins=bins, color=colours, label=labels, weights=weights, edgecolor=ec, linewidth=lw)
    else:
        for idx, fpt_list in enumerate(multi_fpt_list):
            plt.hist(fpt_list, bins=bins, alpha=0.6, color=colours[idx], label=labels[idx],
                     weights=weights[idx,:])
            plt.hist(fpt_list, histtype='step', bins=bins, alpha=0.6, color=colours[idx],
                     label=None,weights=weights[idx,:], edgecolor=ec, linewidth=lw, fill=False)

    # labels
    plt.title('First-passage time histogram (%d runs)' % (ensemble_size), fontsize=fs)
    ax.set_xlabel('First-passage time (cell division timescale)', fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    plt.legend(loc='upper right', fontsize=fs)
    ax.tick_params(labelsize=fs)
    # plt.locator_params(axis='x', nbins=4)

    # save and show
    plt_save = 'fig_fpt_multihist_%s.pdf' % figname_mod
    plt.savefig(outdir + os.sep + plt_save, bbox_inches='tight')
    if flag_show:
        plt.show()
    return plt.gca()



if __name__ == "__main__":
    multihist = True
    basedir = "figures"
    dbdir = basedir + os.sep + "data_fpt"

    if multihist:
        # TODO
        print 'generating multihist figure'
        flag_norm = True
        #dbdir_c80 = dbdir + "fpt_feedback_z_ens1040_c0.80_params"
        c80_header = "fpt_feedback_z_ens1040_c80_N100"
        c88_header = "fpt_feedback_z_ens1040_c88_N100"
        c95_header = "fpt_feedback_z_ens1040_c95_N100"
        fp_times_xyz_c80, params_a = read_fpt_and_params(dbdir, "%s_data.txt" % c80_header, "%s_params.csv" % c80_header)
        fp_times_xyz_c88, params_b = read_fpt_and_params(dbdir, "%s_data.txt" % c88_header, "%s_params.csv" % c88_header)
        fp_times_xyz_c95, params_c = read_fpt_and_params(dbdir, "%s_data.txt" % c95_header, "%s_params.csv" % c95_header)
        fpt_histogram(fp_times_xyz_c88, params_b, flag_ylog10=False, figname_mod="_xyz_feedbackz_N10k_c88_may25")
        plt.close('all')
        fpt_histogram(fp_times_xyz_c88, params_b, flag_ylog10=True, figname_mod="_xyz_feedbackz_N10k_c88_may25_logy")
        plt.close('all')
        multi_fpt = [fp_times_xyz_c80, fp_times_xyz_c88, fp_times_xyz_c95]
        labels = ("c=0.80 (Region I)", "c=0.88 (Region IV)", "c=0.95 (Region III)")
        fpt_histogram_multi(multi_fpt, labels, flag_show=True, flag_ylog10=False, flag_norm=flag_norm, fs=FS, ec=EC, lw=LW, figsize=FIGSIZE)
        fpt_histogram_multi(multi_fpt, labels, flag_show=True, flag_ylog10=True, flag_norm=flag_norm, fs=FS, ec=EC, lw=LW, figsize=FIGSIZE)
        fpt_histogram_multi(multi_fpt, labels, flag_show=True, flag_ylog10=True, flag_norm=False, fs=FS, ec=EC, lw=LW, figsize=FIGSIZE, flag_disjoint=True)
