import matplotlib.pyplot as plt
import numpy as np
import os

from constants import OUTPUT_DIR, COLOURS_DARK_BLUE
from data_io import read_matrix_data_and_idx_vals, read_params, read_fpt_and_params
from firstpassage import fpt_histogram, exponential_scale_estimate, sample_exponential, simplex_heatmap, fp_state_zloc_hist


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
    multihist = False
    simplices = True

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
        hist_headers = ("fpt_TL_ens1848", "fpt_TR_ens2064", "fpt_BR_ens2064", "fpt_BL_ens2048")
        hist_labels = ("b=0.80, c=0.90 (Region I)", "b=0.80, c=1.10 (Region II)", "b=1.20, c=1.10 (Region III)", "b=0.80, c=0.95 (Region IV)")
        hist_data_and_params = [read_fpt_and_params(dbdir, filename_times="%s_data.txt" % header, filename_params="%s_params.csv" % header)
                                for header in hist_headers]
        hist_times = [triple[0] for triple in hist_data_and_params]
        hist_params = [triple[2] for triple in hist_data_and_params]
        num_hists = len(hist_headers)
        # plot indiv histograms
        # TODO port local custom single hist plotter function from firstpassage.py
        for i, header in enumerate(hist_headers):
            fpt_histogram(hist_times[i], hist_params[i], flag_ylog10=False, figname_mod="_%s" % header, outdir=basedir)
            plt.close('all')
            fpt_histogram(hist_times[i], hist_params[i], flag_ylog10=True, figname_mod="_%s_logy" % header, outdir=basedir)
            plt.close('all')
            # add model comparisons
            exp_scale = exponential_scale_estimate(hist_times[i])
            model_data = sample_exponential(len(hist_times[i]), 1/exp_scale)  # note diff convention inverse
            print i, header, model_data.shape, exp_scale, 1/exp_scale
            data_vs_model = [hist_times[i], model_data]
            data_vs_model_labels = [hist_labels[i], r'$\frac{1}{\beta}e^{-t/\beta}, \beta=%.2e$ years' % (1/exp_scale / 365)]
            figure_fpt_multihist(data_vs_model, data_vs_model_labels, figname_mod="compare_model%d" % i, flag_show=True,
                                 flag_ylog10=True, flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
            plt.close()

        # plot various multihists from data
        figure_fpt_multihist(hist_times, hist_labels, figname_mod="def", flag_show=True, flag_ylog10=False,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
        plt.close()
        figure_fpt_multihist(hist_times, hist_labels, figname_mod="logy", flag_show=True, flag_ylog10=True,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
        plt.close()
        figure_fpt_multihist(hist_times, hist_labels, figname_mod="logy_nonorm", flag_show=True, flag_ylog10=True,
                             flag_norm=False, fs=fs, ec=ec, lw=lw, figsize=figsize, flag_disjoint=True, outdir=basedir)

    if simplices:

        fpt_BR, fps_BR, params_BR = read_fpt_and_params(dbdir + os.sep + 'BRstate')
        fpt_TR, fps_TR, params_TR = read_fpt_and_params(dbdir + os.sep + 'TRstate')

        hist_id = ['BRstate', 'TRstate']
        hist_times = [fpt_BR, fpt_TR]
        hist_states = [fps_BR, fps_TR]
        hist_params = [params_BR, params_TR]
        hist_labels = ("b=1.20, c=1.10 (Region III)", "b=0.80, c=0.95 (Region IV)")
        num_hists = len(hist_labels)

        # simplex/distro individual plots
        for i, header in enumerate(hist_id):
            # TODO fix normalization to avoid weird stripe neat allz (note it may be the actual data since when it hits all z there are no more x or y it just moves on the z pole)
            # TODO idea maybe to have special case for when x, y, or z are the whole pop? or just z at least?
            # TODO 2 - fix cutoff top of composite figures and the grid issue which wasn't there with smallfig OFF
            ax1 = simplex_heatmap(hist_times[i], hist_states[i], hist_params[i], smallfig=False, flag_show=False, figname_mod='_%s' % hist_id[i], outdir=basedir)
            plt.close('all')
            ax2 = fp_state_zloc_hist(hist_times[i], hist_states[i], hist_params[i], normalize=True, flag_show=False, kde=True, figname_mod='_%s' % hist_id[i], outdir=basedir)
            plt.close('all')
        # as subplots
        for i, header in enumerate(hist_id):
            f, axarr = plt.subplots(1, 2, sharey=True, figsize=(8, 2.5))
            ax1 = simplex_heatmap(hist_times[i], hist_states[i], hist_params[i], flag_show=False, smallfig=True,
                                  ax=axarr[0], cbar=False, save=False)
            ax2 = fp_state_zloc_hist(hist_times[i], hist_states[i], hist_params[i], normalize=True, flag_show=False,
                                     kde=True, ax=axarr[1], save=False)
            #plt.subplots_adjust()
            plt.savefig(basedir + os.sep + 'simplex_composite_%s.pdf' % hist_id[i], bbox_inches='tight')
