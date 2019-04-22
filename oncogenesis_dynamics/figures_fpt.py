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


def corner_to_flux(corner, params):
    df = {'BL': 0.000212,
          'BR': 1.0,
          'TL': 0.000141,
          'TR': 0.507754}
    z_fp = df[corner] * params.N  # number entering zhat state per unit time
    avg_flux = 1/(z_fp * params.mu)
    return avg_flux


def load_datadict(basedir="figures" + os.sep + "data_fpt"):
    # form is {'BL_N100_xall': {'states': X, 'times': Y, 'params': Z},
    #          'TR_N10k_icfp':  {'states': X, 'times': Y, 'params': Z}, etc. }
    datadict = {}
    # TODO maybe convert to non loop form and just explicitly name subdirs and desired plot headers?
    subdirs = [sd for sd in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, sd))]
    for subdir in subdirs:
        fpt, fps, params = read_fpt_and_params(basedir + os.sep + subdir)
        datadict[subdir] = {'times': fpt, 'states': fps, 'params': params}
    return datadict


def figure_fpt_multihist(multi_fpt_list, labels, figname_mod="def", bin_linspace=80, fs=16, colours=COLOURS_DARK_BLUE,
                         figsize=(8,6), ec='k', lw=0.5, flag_norm=False, flag_xlog10=False, flag_ylog10=False,
                         flag_disjoint=False, flag_show=True, outdir=OUTPUT_DIR, years=True, save=True, ax=None):

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
        y_label = r'$P(\tau)$'  #'Probability'
        weights = np.ones_like(multi_fpt_list) / ensemble_size
    else:
        y_label = 'Frequency'
        weights = np.ones_like(multi_fpt_list)

    # prep fig before axes mod
    if ax is None:
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
        ax.hist(multi_fpt_list, bins=bins, color=colours, label=labels, weights=weights, edgecolor=ec, linewidth=lw)
    else:
        for idx, fpt_list in enumerate(multi_fpt_list):
            ax.hist(fpt_list, bins=bins, alpha=0.6, color=colours[idx], label=labels[idx],
                     weights=weights[idx,:])
            ax.hist(fpt_list, histtype='step', bins=bins, alpha=0.6, color=colours[idx],
                     label=None,weights=weights[idx,:], edgecolor=ec, linewidth=lw, fill=False)

    # labels
    ax.set_title(r'$\tau$ histogram (%d samples)' % (ensemble_size), fontsize=fs)
    if years:
        label = r'$\tau$ (years)' # 'First-passage time (years)'
        ax.set_xlabel(label, fontsize=fs)
    else:
        ax.set_xlabel('First-passage time (cell division timescale)', fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.legend(loc='upper right', fontsize=fs)
    ax.tick_params(labelsize=fs)
    # plt.locator_params(axis='x', nbins=4)

    # save and show
    if save:
        plt_save = 'fig_fpt_multihist_%s.pdf' % figname_mod
        plt.savefig(outdir + os.sep + plt_save, bbox_inches='tight')
    if flag_show:
        plt.show()
    return plt.gca()


if __name__ == "__main__":
    multihist = False
    simplex_and_zdist = False
    composite_simplex_zdist = False
    composite_hist_simplex_zdist = False
    inspect_fpt_flux = True

    basedir = "figures"
    dbdir = basedir + os.sep + "data_fpt"
    datdict = load_datadict(basedir=dbdir)

    title = "N100_xall"
    keys = ['TL_N100_xall', 'BR_N100_xall', 'TR_N100_xall', 'BL_N100_xall']
    labels = ["b=1.20, c=0.90 (Region I)", "b=0.80, c=1.10 (Region II)", "b=1.20, c=1.10 (Region III)",
              "b=0.80, c=0.90 (Region IV)"]
    corners = ['TL', 'BR', 'TR', 'BL']
    num_hists = len(keys)

    if multihist:
        # TODO improve presentation by having cell divison axis timescale [2,4,5...]*10^6 etc but then say mean time in years in caption
        # TODO convert to new datadict io format
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

    if simplex_and_zdist:
        # simplex/distro individual plots
        for i, key in enumerate(keys):
            # TODO fix normalization to avoid weird stripe neat allz (note it may be the actual data since when it hits all z there are no more x or y it just moves on the z pole)
            # TODO idea maybe to have special case for when x, y, or z are the whole pop? or just z at least?
            # TODO 2 - fix cutoff top of composite figures and the grid issue which wasn't there with smallfig OFF
            # TODO add fpt  hist to the composite?
            fpt, fps, params = datdict[key]['times'], datdict[key]['states'], datdict[key]['params']
            label = labels[i]
            ax1 = simplex_heatmap(fpt, fps, params, smallfig=False, flag_show=False, figname_mod='_%s' % key, outdir=basedir)
            plt.close('all')
            ax2 = fp_state_zloc_hist(fpt, fps, params, normalize=True, flag_show=False, kde=True, figname_mod='_%s' % key, outdir=basedir)
            plt.close('all')
    if composite_simplex_zdist:
        # as 2 subplots
        for i, key in enumerate(keys):
            fpt, fps, params = datdict[key]['times'], datdict[key]['states'], datdict[key]['params']
            f, axarr = plt.subplots(1, 2, sharey=True, figsize=(8, 2.5))
            ax1 = simplex_heatmap(fpt, fps, params, flag_show=False, smallfig=True, ax=axarr[0], cbar=False, save=False)
            ax2 = fp_state_zloc_hist(fpt, fps, params, normalize=True, flag_show=False, kde=True, ax=axarr[1], save=False)
            #plt.subplots_adjust()
            plt.savefig(basedir + os.sep + 'simplex_composite_%s.pdf' % key, bbox_inches='tight')
    if composite_hist_simplex_zdist:
        # as 3 subplots
        for i, key in enumerate(keys):
            fpt, fps, params = datdict[key]['times'], datdict[key]['states'], datdict[key]['params']
            f, axarr = plt.subplots(1, 3, sharey=False, figsize=(8, 2.5))
            # prep exp model hist
            fs = 16
            ec = 'k'
            lw = 0.5
            exp_scale = exponential_scale_estimate(fpt)
            model_data = sample_exponential(len(fpt), 1 / exp_scale)  # note diff convention inverse
            print i, key, model_data.shape, exp_scale, 1 / exp_scale, corner_to_flux(corners[i], params)
            data_vs_model = [fpt, model_data]
            data_vs_model_labels_long = [labels[i], r'$\frac{1}{\beta}e^{-t/\beta}, \beta=%.2e$ years' % (1 / exp_scale / 365)]
            data_vs_model_labels = ['data', r'$\frac{1}{\beta}e^{-t/\beta}, \beta=%.2e$ years' % (1 / exp_scale / 365)]

            ax1 = figure_fpt_multihist(data_vs_model, data_vs_model_labels, flag_show=False, flag_ylog10=True,
                                       flag_norm=True, fs=fs, ec=ec, lw=lw, outdir=basedir, save=False, ax=axarr[0])
            ax2 = simplex_heatmap(fpt, fps, params, flag_show=False, smallfig=True, ax=axarr[1], cbar=False, save=False)
            ax3 = fp_state_zloc_hist(fpt, fps, params, normalize=True, flag_show=False, kde=True, ax=axarr[2], save=False)
            plt.subplots_adjust(wspace=0.3)  # 0.2 default width
            plt.savefig(basedir + os.sep + 'triple_composite_%s.pdf' % key, bbox_inches='tight')
            plt.close('all')

    if inspect_fpt_flux:
        print "Running inspect_fpt_flux"
        print "Note: expect xall data to overestimate the FPT timescale beta"
        print "Note: expect init cond at FP (icfp) data to correspond to the FPT timescale beta, except in Region IV"
        print "Remark: in Region IV trajectories jumping between stable FPs (for low N) causes the FPT to be faster " \
              "than one would expect for exponential flux from the lower stable FP (compare N=100 to N=10,000)"
        for flavor in ['N100_xall', 'N100_icfp', 'N10k_xall', 'N10k_icfp']:
            print
            for corner in corners:
                key = corner + '_' + flavor
                fpt, fps, params = datdict[key]['times'], datdict[key]['states'], datdict[key]['params']
                exp_scale = exponential_scale_estimate(fpt)
                print key, len(fpt), exp_scale, 1/exp_scale, corner_to_flux(corner, params)
