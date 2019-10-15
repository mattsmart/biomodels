import matplotlib.pyplot as plt
import numpy as np
import os

from constants import OUTPUT_DIR, COLOURS_DARK_BLUE, COLOURS_DARK_BLUE_YELLOW, X_DARK, Z_DARK, BLUE
from data_io import read_matrix_data_and_idx_vals, read_mfpt_heuristic, read_fpt_and_params, read_varying_mean_sd_fpt_and_params
from firstpassage import fpt_histogram, exponential_scale_estimate, sample_exponential, simplex_heatmap, \
    fp_state_zloc_hist, fp_zloc_times_joint
from plotting import plot_simplex2D
from trajectory_analysis import corner_to_flux


def subsample_data():
    # TODO
    fpt_data_subsampled = 0
    return fpt_data_subsampled


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


def figure_fpt_multihist(multi_fpt_list, labels, figname_mod="def", bin_linspace=80, fs=16, colours=COLOURS_DARK_BLUE_YELLOW,
                         figsize=(8,6), ec='k', lw=0.5, flag_norm=False, flag_xlog10=False, flag_ylog10=False,
                         flag_disjoint=False, flag_show=True, outdir=OUTPUT_DIR, years=False, save=True, ax=None, reframe=False):

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
        #ax.set_xscale("log", nonposx='clip')
        ax.set_xscale("log")
        min_log = np.floor(np.min(np.log10(multi_fpt_list)))
        max_log = np.ceil(np.max(np.log10(multi_fpt_list)))
        bins = np.logspace(min_log, max_log, bin_linspace)
    if flag_ylog10:
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")
    if reframe:
        ax.set_xlim(0.5, np.max(multi_fpt_list))
        ax.set_ylim(0.0, 0.16)

    # plot calls
    if flag_disjoint:
        ax.hist(multi_fpt_list, bins=bins, color=colours, label=labels, weights=weights.T, edgecolor=ec, linewidth=lw)
    else:
        for idx, fpt_list in enumerate(multi_fpt_list):
            ax.hist(fpt_list, bins=bins, alpha=0.6, color=colours[idx], label=labels[idx],
                     weights=weights[idx,:])
            ax.hist(fpt_list, histtype='step', bins=bins, alpha=0.6, color=colours[idx],
                     label=None,weights=weights[idx,:], edgecolor=ec, linewidth=lw, fill=False)

    # labels
    #ax.set_title(r'$\tau$ histogram (%d samples)' % (ensemble_size), fontsize=fs)
    if years:
        label = r'$\tau$ (years)' # 'First-passage time (years)'
        ax.set_xlabel(label, fontsize=fs)
    else:
        #ax.set_xlabel('First-passage time (cell division timescale)', fontsize=fs)
        ax.set_xlabel(r'$\tau$', fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.legend(fontsize=fs-2)  # loc='upper right'
    ax.tick_params(labelsize=fs)
    # plt.locator_params(axis='x', nbins=4)

    # save and show
    if save:
        plt_save = 'fig_fpt_multihist_%s.pdf' % figname_mod
        plt.savefig(outdir + os.sep + plt_save, bbox_inches='tight')
    if flag_show:
        plt.show()
    return plt.gca()


def figure_mfpt_varying(mean_fpt_varying, sd_fpt_varying, param_vary_name, param_set, params, samplesize, SEM_flag=True,
                        show_flag=False, figname_mod="", outdir=OUTPUT_DIR, fs=16):
    if SEM_flag:
        sd_fpt_varying = sd_fpt_varying / np.sqrt(samplesize)  # s.d. from CLT since sample mean is approx N(mu, sd**2/n)
    plt.errorbar(param_set, mean_fpt_varying, yerr=sd_fpt_varying, label="sim")
    plt.title("Mean FP Time, %s varying (sample=%d)" % (param_vary_name, samplesize))
    ax = plt.gca()
    ax.set_xlabel(r'$%s$' % param_vary_name, fontsize=fs)
    ax.set_ylabel(r'$\langle\tau\rangle$', fontsize=fs)

    # log options
    for i in xrange(len(mean_fpt_varying)):
        print i, param_set[i], mean_fpt_varying[i], sd_fpt_varying[i]
    flag_xlog10 = True
    flag_ylog10 = True
    if flag_xlog10:
        #ax.set_xscale("log", nonposx='clip')
        ax.set_xscale("log")
        #ax.set_xlim([0.8*1e2, 1*1e7])
    if flag_ylog10:
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")
        #ax.set_ylim([0.8*1e2, 3*1e5])

    # create table of params
    plt_save = "mean_fpt_varying" + figname_mod
    plt.savefig(outdir + os.sep + plt_save + '.pdf', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


def figure_mfpt_varying_dual(mean_fpt_varying, sd_fpt_varying, param_vary_name, param_set, params, samplesize, SEM_flag=True,
                        show_flag=False, figname_mod="", outdir=OUTPUT_DIR, fs=20, ax=None):
    if SEM_flag:
        sd_fpt_varying = sd_fpt_varying / np.sqrt(samplesize)  # s.d. from CLT since sample mean is approx N(mu, sd**2/n)
    if ax is None:
        plt.figure()
        ax = plt.gca()
    #ax_dual = ax.twinx()

    heuristic_x = np.logspace(np.log10(np.min(param_set)), np.log10(np.max(param_set)), 100)
    heuristic_y = [corner_to_flux('BL', params.mod_copy({param_vary_name: p})) for p in heuristic_x]

    ax.plot(param_set, mean_fpt_varying, '--', marker='o', color='k', label=r'$\langle\tau\rangle$ (simulation)')
    ax.plot(param_set, sd_fpt_varying, '-.', marker='^', color='r', label=r'$\delta\tau$ (simulation)')
    ax.plot(heuristic_x, heuristic_y, 'k', label=r"Flux to $\hat z$ from FP")
    #plt.title("Mean FP Time, %s varying (sample=%d)" % (param_vary_name, samplesize))
    ax.set_xlabel(r'$%s$' % param_vary_name, fontsize=fs)
    ax.set_ylabel(r'$\tau$', fontsize=fs)
    #ax.set_ylabel(r'$\langle\tau\rangle$', fontsize=fs)
    #ax.set_ylabel(r'$\delta\tau$', fontsize=fs)

    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)

    # hacky fix to sharey
    #print "HACKY"
    #print ax.get_ylim()
    #ax_dual.set_ylim(ax.get_ylim())

    # log options
    for i in xrange(len(mean_fpt_varying)):
        print i, param_set[i], mean_fpt_varying[i], sd_fpt_varying[i]
    flag_xlog10 = True
    flag_ylog10 = True
    if flag_xlog10:
        #ax.set_xscale("log", nonposx='clip')
        ax.set_xscale("log")
        #ax_dual.set_xscale("log", nonposx='clip')
        #ax.set_xlim([0.8*1e2, 1*1e7])
    if flag_ylog10:
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")
        #ax_dual.set_yscale("log", nonposx='clip')
        #ax.set_ylim([0.8*1e2, 3*1e5])

    # create table of params
    ax.legend(fontsize=fs-2)
    plt_save = "mean_fpt_varying_dual" + figname_mod
    plt.savefig(outdir + os.sep + plt_save + '.pdf', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


def figure_mfpt_varying_composite(means, sds, param_vary_name, param_set, params,
                        show_flag=False, figname_mod="", outdir=OUTPUT_DIR, fs=20, ax=None):
    if ax is None:
        plt.figure(figsize=(5,4))
        ax = plt.gca()

    num_sets = 6
    assert num_sets == len(means)
    colours = [X_DARK, Z_DARK, '#ffd966', BLUE, 'pink', 'brown']  #['black', 'red', 'green', 'blue']
    #TODO markers =
    labels = [r"$b=0.8$, $c=0.9$, $\gamma=1$", r"$b=0.8$, $c=0.9$, $\gamma=4$", r"$b=0.8$, $c=0.9$, $\gamma=100$",
              r"$b=1.2$, $c=1.1$, $\gamma=1$", r"$b=1.2$, $c=1.1$, $\gamma=4$", r"$b=1.2$, $c=1.1$, $\gamma=100$", ]
    region_labels = [r"$b=0.8$, $c=0.9$, $\gamma=1$", r"$b=0.8$, $c=0.9$, $\gamma=4$", r"$b=0.8$, $c=0.9$, $\gamma=100$",
                     r"$b=1.2$, $c=1.1$, $\gamma=1$", r"$b=1.2$, $c=1.1$, $\gamma=4$", r"$b=1.2$, $c=1.1$, $\gamma=100$"]
    corners = ['BL1g', 'BL', 'BL100g', 'TR1g', 'TR', 'TR100g']#, 'TR1g', 'TR']

    heuristic_x = np.logspace(np.log10(np.min(param_set)), np.log10(np.max(param_set)), 100)
    for idx in xrange(num_sets):
        print idx, len(param_set), len(means[idx]), len(sds[idx])
        ax.plot(param_set, means[idx], '.-', marker='o', markeredgecolor='k', color=colours[idx], label=r'%s: $\langle\tau\rangle$' % region_labels[idx], zorder=3)
        ax.plot(param_set, sds[idx], '-.', marker='^', markeredgecolor='k', color=colours[idx], label=r'%s: $\delta\tau$' % region_labels[idx], zorder=2)
        #ax.plot(param_set, sds[idx], '-.', marker='^', markeredgecolor='k', color=colours[idx], zorder=2)
        heuristic_y = [corner_to_flux(corners[idx], params.mod_copy({param_vary_name: p})) for p in heuristic_x]
        ax.plot(heuristic_x, heuristic_y, '--k', zorder=1) #, label=r"Flux to $\hat z$ from FP (Region %s)" % region_labels[idx])
        #plt.title("Mean FP Time, %s varying (sample=%d)" % (param_vary_name, samplesize))

    ax.set_xlabel(r'$%s$' % param_vary_name, fontsize=fs)
    ax.set_ylabel(r'$\tau$', fontsize=fs)
    #ax.set_ylabel(r'$\langle\tau\rangle$', fontsize=fs)
    #ax.set_ylabel(r'$\delta\tau$', fontsize=fs)

    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)

    # log options
    flag_xlog10 = True
    flag_ylog10 = True
    if flag_xlog10:
        #ax.set_xscale("log", nonposx='clip')
        ax.set_xscale("log")
        #ax_dual.set_xscale("log", nonposx='clip')
        ax.set_xlim([np.min(param_set)*0.9, 1.5*1e4])
    if flag_ylog10:
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")
        #ax_dual.set_yscale("log", nonposx='clip')
        ax.set_ylim([6*1e-1, 3*1e6])

    ax.legend(fontsize=fs-6, ncol=2, loc='upper right')
    plt_save = "mean_fpt_varying_composite" + figname_mod
    plt.savefig(outdir + os.sep + plt_save + '.pdf', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


def figure_mfpt_varying_collapsed(means, sds, param_vary_name, param_set, params,
                        show_flag=False, figname_mod="", outdir=OUTPUT_DIR, fs=20, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    num_sets = 6
    assert num_sets == len(means)
    colours = [X_DARK, Z_DARK, '#ffd966', BLUE, 'pink', 'brown']  #['black', 'red', 'green', 'blue']
    region_labels = ['BLg1', 'BLg4', 'BLg100', 'TRg1', 'TRg4', 'TRg100']#, 'TRg1', 'TRg4']
    corners = ['BL1g', 'BL', 'BL100g', 'TR1g', 'TR', 'TR100g']#, 'TR1g', 'TR']


    for idx in xrange(num_sets):
        print idx, len(param_set), len(means[idx]), len(sds[idx])
        means_scaled = [means[idx][i] / corner_to_flux(corners[idx], params.mod_copy({param_vary_name: param_set[i]})) for i in xrange(len(means[idx]))]
        ax.plot(param_set, means_scaled, '.-', marker='o', markeredgecolor='k', color=colours[idx], label=r'Region %s' % region_labels[idx], zorder=3)
        #ax.plot(param_set, sd_scaled, '-.', marker='^', markeredgecolor='k', color=colours[idx], label=r'$\delta\tau$ (Region %s)' % region_labels[idx], zorder=2)

        #ax.plot(heuristic_x, heuristic_y, '--k', zorder=1) #, label=r"Flux to $\hat z$ from FP (Region %s)" % region_labels[idx])
        #plt.title("Mean FP Time, %s varying (sample=%d)" % (param_vary_name, samplesize))
        plt.axhline(1.0, color='k', ls='--', lw=1.0)

    ax.set_xlabel(r'$%s$' % param_vary_name, fontsize=fs)
    ax.set_ylabel(r'$\mu z^{\ast} \langle\tau\rangle_{sim}$', fontsize=fs)
    #ax.set_ylabel(r'$\langle\tau\rangle$', fontsize=fs)
    #ax.set_ylabel(r'$\delta\tau$', fontsize=fs)

    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)

    # log options

    flag_xlog10 = True
    flag_ylog10 = True
    if flag_xlog10:
        #ax.set_xscale("log", nonposx='clip')
        ax.set_xscale("log")
        #ax_dual.set_xscale("log", nonposx='clip')
        ax.set_xlim([np.min(param_set)*0.9, 1.5*1e4])
    if flag_ylog10:
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")
        #ax_dual.set_yscale("log", nonposx='clip')
        #ax.set_ylim([0.0, 10])

    ax.legend(fontsize=fs-6)
    plt_save = "mean_fpt_varying_collapsed" + figname_mod
    plt.savefig(outdir + os.sep + plt_save + '.pdf', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


if __name__ == "__main__":
    multihist = False
    simplex_and_zdist = False
    only_fp_zloc_times_joint = False
    composite_simplex_zdist = False
    composite_hist_simplex_zdist = False
    inspect_fpt_flux = False
    mfpt_single = False
    mfpt_composite = False
    mfpt_details = False
    mfpt_composite_TR = True
    mfpt_composite_BL = True


    basedir = "data"
    if any([multihist, only_fp_zloc_times_joint, simplex_and_zdist, composite_simplex_zdist, composite_hist_simplex_zdist, inspect_fpt_flux]):
        dbdir = basedir + os.sep + "fpt"
        datdict = load_datadict(basedir=dbdir)

        """
        title = "N100_xall"
        keys = ['BR_%s' % title, 'TR_%s' % title, 'BL_%s' % title, 'TL_%s' % title]
        #labels = ["b=0.80, c=1.10 (Region II)", "b=1.20, c=1.10 (Region III)", "b=0.80, c=0.90 (Region IV)", "b=1.20, c=0.90 (Region I)"]
        labels = [r"(II)  $b=0.8$, $c=1.1$", r"(III) $b=1.2$, $c=1.1$", r"(IV) $b=0.8$, $c=0.9$", r"(I)   $b=1.2$, $c=0.9$"]
        corners = ['BR', 'TR', 'BL', 'TL']
        num_hists = len(keys)
        """

        title = "N100_xall"
        keys = ['BL_%s_g1' % title, 'BL_%s_g4' % title, 'BL_%s_g100' % title,
                'TR_%s_g1' % title, 'TR_%s_g4' % title, 'TR_%s_g100' % title]
        #labels = ["b=0.80, c=1.10 (Region II)", "b=1.20, c=1.10 (Region III)", "b=0.80, c=0.90 (Region IV)", "b=1.20, c=0.90 (Region I)"]
        labels = [r"$b=0.8$, $c=0.9$, $\gamma=1$", r"$b=0.8$, $c=0.9$, $\gamma=4$", r"$b=0.8$, $c=0.9$, $\gamma=100$",
                  r"$b=1.2$, $c=1.1$, $\gamma=1$", r"$b=1.2$, $c=1.1$, $\gamma=4$", r"$b=1.2$, $c=1.1$, $\gamma=100$",]
        corners = ['BL1g', 'BL4g', 'BL100g', 'TR1g', 'TR4g', 'TR100g']
        num_hists = len(keys)

    if multihist:
        # TODO improve presentation by having cell divison axis timescale [2,4,5...]*10^6 etc but then say mean time in years in caption
        # TODO convert to new datadict io format
        # plot settings
        flag_norm = True
        fs = 16
        ec = 'k'
        lw = 0.5
        figsize = (4, 4)
        # data setup
        hist_times = [0] * len(keys)
        hist_params = [0] * len(keys)
        for i, key in enumerate(keys):
            hist_times[i], hist_params[i] = datdict[key]['times'], datdict[key]['params']
        num_hists = len(hist_times)
        # plot indiv histograms
        # TODO port local custom single hist plotter function from firstpassage.py
        """
        for i, header in enumerate(keys):
            fpt_histogram(hist_times[i], hist_params[i], flag_ylog10=False, figname_mod="_%s" % header, outdir=basedir)
            plt.close('all')
            fpt_histogram(hist_times[i], hist_params[i], flag_ylog10=True, figname_mod="_%s_logy" % header, outdir=basedir)
            plt.close('all')
            # add model comparisons
            exp_scale = exponential_scale_estimate(hist_times[i])
            model_data = sample_exponential(len(hist_times[i]), 1/exp_scale)  # note diff convention inverse
            print i, header, model_data.shape, exp_scale, 1/exp_scale
            data_vs_model = [hist_times[i], model_data]
            data_vs_model_labels = [labels[i], r'$\frac{1}{\beta}e^{-t/\beta}, \beta=%.2e$ years' % (1/exp_scale / 365)]
            figure_fpt_multihist(data_vs_model, data_vs_model_labels, figname_mod="compare_model%d" % i, flag_show=False,
                                 flag_ylog10=True, flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
            plt.close()
        """
        # plot various multihists from data
        figure_fpt_multihist(hist_times, labels, figname_mod="def", flag_show=False, flag_ylog10=False,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
        plt.close()
        figure_fpt_multihist(hist_times, labels, figname_mod="logy", flag_show=False, flag_ylog10=True,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
        plt.close()
        figure_fpt_multihist(hist_times, labels, figname_mod="logx", flag_show=False, flag_xlog10=True,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
        plt.close()
        figure_fpt_multihist(hist_times, labels, figname_mod="logx_reframe", flag_show=False, flag_xlog10=True,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir, reframe=True)
        plt.close()
        figure_fpt_multihist(hist_times, labels, figname_mod="logBoth", flag_show=False, flag_ylog10=True, flag_xlog10=True,
                             flag_norm=flag_norm, fs=fs, ec=ec, lw=lw, figsize=figsize, outdir=basedir)
        plt.close()
        figure_fpt_multihist(hist_times, labels, figname_mod="logy_nonorm", flag_show=False, flag_ylog10=True,
                             flag_norm=False, fs=fs, ec=ec, lw=0.0, figsize=figsize, flag_disjoint=True, outdir=basedir)

    if simplex_and_zdist:
        # simplex/distro individual plots
        for i, key in enumerate(keys):
            # TODO fix normalization to avoid weird stripe neat allz (note it may be the actual data since when it hits all z there are no more x or y it just moves on the z pole)
            # TODO idea maybe to have special case for when x, y, or z are the whole pop? or just z at least?
            # TODO 2 - fix cutoff top of composite figures and the grid issue which wasn't there with smallfig OFF
            # TODO add fpt  hist to the composite?
            fpt, fps, params = datdict[key]['times'], datdict[key]['states'], datdict[key]['params']
            label = labels[i]
            ax1 = plot_simplex2D(params, smallfig=False)
            plt.savefig(basedir + os.sep + 'simplex_%s' % key + '.pdf')
            plt.close('all')
            ax2 = simplex_heatmap(fpt, fps, params, smallfig=False, flag_show=False, figname_mod='_%s' % key, outdir=basedir, colour=False)
            plt.close('all')
            ax3 = fp_state_zloc_hist(fpt, fps, params, normalize=True, flag_show=False, kde=True, figname_mod='_%s' % key, outdir=basedir)
            plt.close('all')
    if only_fp_zloc_times_joint:
        for i, key in enumerate(keys):
            # TODO all
            #key = 'TR_N100_icfp_extra'
            print i, key
            fpt, fps, params = datdict[key]['times'], datdict[key]['states'], datdict[key]['params']
            #fluxval = corner_to_flux('TR', params)
            fluxval = corner_to_flux(corners[i], params)
            # fluxval = None
            print key, np.min(fpt), np.max(fpt), np.mean(fpt), len(fpt)
            ax1 = fp_zloc_times_joint(fpt, fps, params, normalize=True, flag_show=False, color='RosyBrown',
                                      figname_mod='_%s_logx_color' % key, logx=True, fluxval=None, outdir=basedir)
            plt.close('all')
            ax2 = fp_zloc_times_joint(fpt, fps, params, normalize=True, flag_show=False,
                                      figname_mod='_%s_logx' % key, logx=True, fluxval=None, outdir=basedir)
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

    if mfpt_single:
        samplesize = 240
        mfpt_dir = basedir + os.sep + 'data_mfpt' + os.sep + 'mfpt_Nvary_mu1e-4_BL_ens240'
        mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
            read_varying_mean_sd_fpt_and_params(mfpt_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N.txt',
                                                mfpt_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N_params.csv')
        figure_mfpt_varying(mean_fpt_varying, sd_fpt_varying, 'N', param_set, params, samplesize,
                            SEM_flag=False, show_flag=False, figname_mod="", outdir=basedir)
        figure_mfpt_varying_dual(mean_fpt_varying, sd_fpt_varying, 'N', param_set, params, samplesize,
                                 SEM_flag=False, show_flag=False, figname_mod="", outdir=basedir)

    if mfpt_composite:
        subdirs = ['mfpt_Nvary_mu1e-4_BL_ens240_xall_g1',
                   'mfpt_Nvary_mu1e-4_BL_ens240_xall_g4',
                   'mfpt_Nvary_mu1e-4_BL_ens240_xall_g100',
                   'mfpt_Nvary_mu1e-4_TR_ens240_xall_g1',
                   'mfpt_Nvary_mu1e-4_TR_ens240_xall_g4',
                   'mfpt_Nvary_mu1e-4_TR_ens240_xall_g100']
                   #'mfpt_Nvary_mu1e-4_TR_ens240_xall_g1', 'mfpt_Nvary_mu1e-4_TR_ens240_xall']
        means = []
        sds = []
        for subdir in subdirs:
            mfpt_dir = basedir + os.sep + 'mfpt' + os.sep + subdir
            mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
                read_varying_mean_sd_fpt_and_params(mfpt_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N.txt',
                                                    mfpt_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N_params.csv')
            means.append(mean_fpt_varying)
            sds.append(sd_fpt_varying)
        figure_mfpt_varying_composite(means, sds, 'N', param_set, params, show_flag=False, figname_mod="",
                                      outdir=OUTPUT_DIR, fs=20)
        figure_mfpt_varying_collapsed(means, sds, 'N', param_set, params, show_flag=False, figname_mod="",
                                      outdir=OUTPUT_DIR, fs=20)

    if mfpt_details:
        detailsdir = 'data' + os.sep + 'mfpt_details' + os.sep + 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g100'
        basen = 'means'
        for idx in xrange(1,12):
            fpath = detailsdir + os.sep + basen + str(idx) + os.sep + "output"
            fpt, fps, params = read_fpt_and_params(fpath)
            fluxval = corner_to_flux('TRg100', params)
            # fluxval = None
            print idx, np.min(fpt), np.max(fpt), np.mean(fpt), len(fpt)
            ax2 = fp_zloc_times_joint(fpt, fps, params, normalize=True, flag_show=False, kde=False,
                                      figname_mod='_%s_logx' % idx, logx=True, fluxval=fluxval, outdir=basedir)
            plt.close('all')

    if mfpt_composite_TR:

        data_ids = ['TR1g', 'TR100g'] #'TR4g'
        data_ids_to_data = {'TR1g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g1',
                            #'TR4g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g4',
                            'TR100g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g100'}
        mfpt_dict = {key: {} for key in data_ids}
        for key in data_ids:
            mfpt_dir = basedir + os.sep + 'mfpt' + os.sep + data_ids_to_data[key]
            mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
                read_varying_mean_sd_fpt_and_params(mfpt_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N.txt',
                                                    mfpt_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N_params.csv')
            mfpt_dict[key]['data'] = {'x': param_set, 'y': mean_fpt_varying}
        # build heuristics for each data_id...
        for key in data_ids:
            heuristic_dir = basedir + os.sep + 'heuristic'
            fnames = os.listdir(heuristic_dir)
            for fname in fnames:
                fpath = heuristic_dir + os.sep + fname
                file_split_by_underscore = fname.split('_')
                if file_split_by_underscore[2] == key:
                    Narr, mfpt_heuristic = read_mfpt_heuristic(fpath)
                    mfpt_dict[key][file_split_by_underscore[3][:-4]] = {'x': Narr, 'y': mfpt_heuristic}
        # plot data and heuristics on one plot
        ax=None
        fs=12
        colours = [X_DARK, '#ffd966', Z_DARK, BLUE, 'pink', 'brown']  # ['black', 'red', 'green', 'blue']

        if ax is None:
            plt.figure(figsize=(5, 3))
            ax = plt.gca()        # TODO
        for idx, key in enumerate(data_ids):
            subdict = mfpt_dict[key]
            print key, subdict.keys()
            for datakey in subdict.keys():
                x = subdict[datakey]['x']
                y = subdict[datakey]['y']

                if key == 'TR1g':
                    gammaval = 1
                else:
                    assert key == 'TR100g'
                    gammaval = 100

                if datakey == 'data':
                    ax.plot(x, y, '-', marker='o', markeredgecolor='k', color=colours[idx],
                            label=r'$\gamma=%d$: $\langle\tau\rangle$' % gammaval, zorder=3)
                elif datakey == 'linalgALLZ':
                    if key == 'TR100g':
                        x = x[0:7]
                        y = y[0:7]
                    ax.plot(x, y, '-.', marker='*', markeredgecolor='k', color=colours[idx],
                            label=r'$\gamma=%d$: ME allz' % gammaval, zorder=3)
                elif datakey == 'linalgZHAT':
                    ax.plot(x, y, '-.', marker='^', markeredgecolor='k', color=colours[idx],
                            label=r'$\gamma=%d$: ME zhat' % gammaval, zorder=3)
                elif datakey == 'fpRouteFlux':
                    continue
                    #ax.plot(x, y, '--', marker=None, markeredgecolor='k', color=colours[idx],
                    #        label=r'$\gamma=%d$: FP route flux' % gammaval, zorder=3)
                elif datakey == 'fpFlux':
                    continue
                    #ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',#colours[idx],
                    #        label=r'$\gamma=%d$: FP flux' % gammaval, zorder=3)
                elif datakey == 'guessPfixThreeTerm' and key=='TR1g':
                    ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                            label=r'$\gamma=1$: pfix3', zorder=3)
                elif datakey == 'guessBlobtimes' and key=='TR100g':
                    continue
                    #ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='gray',  # colours[idx],
                    #        label=r'$\gamma=100$: blobtime', zorder=3)
                elif datakey == 'guessBlobtimesPosSvals' and key=='TR100g':
                    ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                            label=r'$\gamma=100$: blobtime $s>0$', zorder=3)

        ax.set_xlabel(r'$N$', fontsize=fs)
        ax.set_ylabel(r'$\tau$', fontsize=fs)
        plt.xticks(fontsize=fs - 2)
        plt.yticks(fontsize=fs - 2)
        plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize=fs-4)
        #plt.legend()
        # log options
        flag_xlog10 = True
        flag_ylog10 = True
        if flag_xlog10:
            # ax.set_xscale("log", nonposx='clip')
            ax.set_xscale("log")
            # ax_dual.set_xscale("log", nonposx='clip')
            ax.set_xlim([np.min(param_set) * 0.9, 1.5 * 1e4])
        if flag_ylog10:
            # ax.set_yscale("log", nonposx='clip')
            ax.set_yscale("log")
            # ax_dual.set_yscale("log", nonposx='clip')
            #ax.set_ylim([6 * 1e-1, 3 * 1e6])
            ax.set_ylim([1e2, 3 * 1e6])
        #plt.show()
        plt.savefig(basedir + os.sep + 'mfpt_TR_heuristics.pdf')

    if mfpt_composite_BL:

        data_ids = ['BL1g', 'BL100g'] #'TR4g'
        data_ids_to_data = {'BL1g': 'mfpt_Nvary_mu1e-4_BL_ens240_xall_g1',
                            #'TR4g': 'mfpt_Nvary_mu1e-4_TR_ens240_xall_g4',
                            'BL100g': 'mfpt_Nvary_mu1e-4_BL_ens240_xall_g100'}
        mfpt_dict = {key: {} for key in data_ids}
        for key in data_ids:
            mfpt_dir = basedir + os.sep + 'mfpt' + os.sep + data_ids_to_data[key]
            mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
                read_varying_mean_sd_fpt_and_params(mfpt_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N.txt',
                                                    mfpt_dir + os.sep + 'fpt_stats_collected_mean_sd_varying_N_params.csv')
            mfpt_dict[key]['data'] = {'x': param_set, 'y': mean_fpt_varying}
        # build heuristics for each data_id...
        for key in data_ids:
            heuristic_dir = basedir + os.sep + 'heuristic'
            fnames = os.listdir(heuristic_dir)
            for fname in fnames:
                fpath = heuristic_dir + os.sep + fname
                file_split_by_underscore = fname.split('_')
                if file_split_by_underscore[2] == key:
                    Narr, mfpt_heuristic = read_mfpt_heuristic(fpath)
                    mfpt_dict[key][file_split_by_underscore[3][:-4]] = {'x': Narr, 'y': mfpt_heuristic}
        # plot data and heuristics on one plot
        ax=None
        fs=12
        colours = [X_DARK, '#ffd966', Z_DARK, BLUE, 'pink', 'brown']  # ['black', 'red', 'green', 'blue']

        if ax is None:
            plt.figure(figsize=(5, 3))
            ax = plt.gca()        # TODO
        for idx, key in enumerate(data_ids):
            subdict = mfpt_dict[key]
            print key, subdict.keys()
            for datakey in subdict.keys():
                x = subdict[datakey]['x']
                y = subdict[datakey]['y']

                if key == 'BL1g':
                    gammaval = 1
                else:
                    assert key == 'BL100g'
                    gammaval = 100


                if datakey == 'data':
                    ax.plot(x, y, '-', marker='o', markeredgecolor='k', color=colours[idx],
                            label=r'$\gamma=%d$: $\langle\tau\rangle$' % gammaval, zorder=3)
                elif datakey == 'linalgALLZ':
                    if key == 'BL100g':
                        x = x[0:8]
                        y = y[0:8]
                    if key == 'BL1g':
                        x = x[0:4]
                        y = y[0:4]
                    ax.plot(x, y, '-.', marker='*', markeredgecolor='k', color=colours[idx],
                            label=r'$\gamma=%d$: ME allz' % gammaval, zorder=3)
                elif datakey == 'linalgZHAT':
                    ax.plot(x, y, '-.', marker='^', markeredgecolor='k', color=colours[idx],
                            label=r'$\gamma=%d$: ME zhat' % gammaval, zorder=3)
                elif datakey == 'fpRouteFlux':
                    ax.plot(x, y, '--', marker=None, markeredgecolor='k', color=colours[idx],
                            label=r'$\gamma=%d$: FP route flux' % gammaval, zorder=3)
                elif datakey == 'fpFlux':
                    #ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',#colours[idx],
                    #        label=r'$\gamma=%d$: FP flux' % gammaval, zorder=3)
                    continue
                elif datakey == 'guessBoundaryProb1' and key != 'BL1g':
                    #ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                    #        label=r'$\gamma=%d$: BP1' % gammaval, zorder=3)
                    continue
                elif datakey == 'guessBoundaryProb2':
                    #ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                    #        label=r'$\gamma=%d$: BP2' % gammaval, zorder=3)
                    continue
                elif datakey == 'guessBoundaryProb3':
                    ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                            label=r'$\gamma=%d$: BP3' % gammaval, zorder=3)
                """
                elif datakey == 'guessBoundaryTimeDual1':
                    print 'a', key, y
                    ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                            label=r'%s: BTD1' % key, zorder=3)
                elif datakey == 'guessBoundaryTimeDual2':
                    print 'b', key, y
                    ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                            label=r'%s: BTD2' % key, zorder=3)
                elif datakey == 'guessBoundaryTimeMono1':
                    print 'c', key, y
                    ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                            label=r'%s: BTM1' % key, zorder=3)
                elif datakey == 'guessBoundaryTimeMono2':
                    print 'd', key, y
                    ax.plot(x, y, '--', marker=None, markeredgecolor='k', color='k',  # colours[idx],
                            label=r'%s: BTM2' % key, zorder=3)
                """

        ax.set_xlabel(r'$N$', fontsize=fs)
        ax.set_ylabel(r'$\tau$', fontsize=fs)
        plt.xticks(fontsize=fs - 2)
        plt.yticks(fontsize=fs - 2)
        plt.legend(bbox_to_anchor=(1.07, 1.00), fontsize=fs-4)
        #plt.legend()
        # log options
        flag_xlog10 = True
        flag_ylog10 = True
        if flag_xlog10:
            # ax.set_xscale("log", nonposx='clip')
            ax.set_xscale("log")
            # ax_dual.set_xscale("log", nonposx='clip')
            ax.set_xlim([np.min(param_set) * 0.9, 1.5 * 1e4])
        if flag_ylog10:
            # ax.set_yscale("log", nonposx='clip')
            ax.set_yscale("log")
            # ax_dual.set_yscale("log", nonposx='clip')
            #ax.set_ylim([6 * 1e-1, 3 * 1e6])
            ax.set_ylim([1*1e3, 3 * 1e7])
        #plt.show()
        plt.savefig(basedir + os.sep + 'mfpt_BL_heuristics.pdf')
