import matplotlib.pyplot as plt
import numpy as np
import os

from constants import OUTPUT_DIR, COLOURS_DARK_BLUE
from data_io import read_matrix_data_and_idx_vals, read_params, read_fpt_and_params
from firstpassage import fpt_histogram


def subsample_data():
    # TODO
    fpt_data_subsampled = 0
    return fpt_data_subsampled


def figure_fpt_multihist(multi_fpt_list, labels, figname_mod="def", bin_linspace=80, fs=16, colours=COLOURS_DARK_BLUE,
                         figsize=(8,6), ec='k', lw=0.5, flag_norm=False, flag_xlog10=False, flag_ylog10=False,
                         flag_disjoint=False, flag_show=True, outdir=OUTPUT_DIR, years=True):

    # resize fpt lists if not all same size (to the min size)
    fpt_lengths = [len(fpt) for fpt in multi_fpt_list]
    ensemble_size = np.min(fpt_lengths)

    if years:
        multi_fpt_list = [np.array(arr) / 365.0 for arr in multi_fpt_list]

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
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")

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
    if years:
        ax.set_xlabel('First-passage time (years)', fontsize=fs)
    else:
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
        # TODO improve presentation by having cell divison axis timescale [2,4,5...]*10^6 etc but then say mean time in years in caption
        # plot settings
        flag_norm = True
        fs = 16
        ec = 'k'
        lw = 0.5
        figsize = (8, 6)
        # data setup
        hist_headers = ("fpt_TR_ens2064", "fpt_BR_ens2064", "fpt_BL2_ens1024")
        hist_labels = ("b=0.80, c=1.10 (Region II)", "b=1.20, c=1.10 (Region III)", "b=0.80, c=0.95 (Region IV)")
        hist_data_and_params = [read_fpt_and_params(dbdir, "%s_data.txt" % header, "%s_params.csv" % header)
                                for header in hist_headers]
        hist_data = [pair[0] for pair in hist_data_and_params]
        hist_params = [pair[1] for pair in hist_data_and_params]
        num_hists = len(hist_headers)
        # plot indiv histograms
        # TODO port local custom single hist plotter function from firstpassage.py
        for i, header in enumerate(hist_headers):
            fpt_histogram(hist_data[i], hist_params[i], flag_ylog10=False, figname_mod="_%s" % header, outdir=basedir)
            plt.close('all')
            fpt_histogram(hist_data[i], hist_params[i], flag_ylog10=True, figname_mod="_%s_logy" % header, outdir=basedir)
            plt.close('all')
        # plot various multihists
        figure_fpt_multihist(hist_data, hist_labels, figname_mod="def", flag_show=True, flag_ylog10=False,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
        figure_fpt_multihist(hist_data, hist_labels, figname_mod="logy", flag_show=True, flag_ylog10=True,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
        figure_fpt_multihist(hist_data, hist_labels, figname_mod="logy_nonorm", flag_show=True, flag_ylog10=True,
                             flag_norm=False, fs=fs, ec=ec, lw=lw, figsize=figsize, flag_disjoint=True, outdir=basedir)
