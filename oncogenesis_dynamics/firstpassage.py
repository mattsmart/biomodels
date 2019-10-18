import matplotlib.pyplot as plt
import numpy as np
import seaborn
import time
from os import sep
from multiprocessing import Pool, cpu_count

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, COLOURS_DARK_BLUE
from data_io import read_varying_mean_sd_fpt_and_params, collect_fpt_mean_stats_and_params, read_fpt_and_params,\
                    write_fpt_and_params
from formulae import stoch_gillespie, stoch_tauleap_lowmem, stoch_tauleap, get_physical_fp_stable_and_not, \
    map_init_name_to_init_cond, stoch_gillespie_lowmem, fp_location_fsolve, jacobian_numerical_2d
from params import Params
from presets import presets
from plotting import plot_table_params, plot_simplex2D


def get_fpt(ensemble, init_cond, params, num_steps=1000000, establish_switch=False, brief=True, tauleap=False):
    # TODO could pass simmethod tau or gillespie to params and parse here
    if establish_switch:
        fpt_flag = False
        establish_flag = True
    else:
        fpt_flag = True
        establish_flag = False
    fp_times = np.zeros(ensemble)
    fp_states = np.zeros((ensemble, params.numstates))
    for i in xrange(ensemble):
        if brief:
            if tauleap:
                species_end, times_end = stoch_tauleap_lowmem(init_cond, num_steps, params, fpt_flag=fpt_flag,
                                                              establish_flag=establish_flag)
            else:
                species_end, times_end = stoch_gillespie_lowmem(init_cond, params, fpt_flag=fpt_flag,
                                                                establish_flag=establish_flag)
        else:
            species, times = stoch_gillespie(init_cond, num_steps, params, fpt_flag=fpt_flag,
                                             establish_flag=establish_flag)
            times_end = times[-1]
            species_end = species[-1, :]
            # plotting
            #plt.plot(times, species)
            #plt.show()
        fp_times[i] = times_end
        fp_states[i, :] = species_end
        if establish_switch:
            print "establish time is", fp_times[i]
    return fp_times, fp_states


def get_mean_fpt(init_cond, params, samplesize=32, establish_switch=False):
    fpt, fpt_states = get_fpt(samplesize, init_cond, params, establish_switch=establish_switch)
    return np.mean(fpt)


def wrapper_get_fpt(fn_args_dict):
    np.random.seed()  # TODO double check that this fixes cluster RNG issues
    if fn_args_dict['kwargs'] is not None:
        return get_fpt(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return get_fpt(*fn_args_dict['args'])


def fast_fp_times(ensemble, init_cond, params, num_processes, num_steps='default', establish_switch=False):
    if num_steps == 'default':
        kwargs_dict = {'num_steps': 1000000, 'establish_switch': establish_switch}
    else:
        kwargs_dict = {'num_steps': num_steps, 'establish_switch': establish_switch}

    fn_args_dict = [0]*num_processes
    print "NUM_PROCESSES:", num_processes
    assert ensemble % num_processes == 0
    for i in xrange(num_processes):
        subensemble = ensemble / num_processes
        print "process:", i, "job size:", subensemble, "runs"
        fn_args_dict[i] = {'args': (subensemble, init_cond, params),
                           'kwargs': kwargs_dict}
    t0 = time.time()
    pool = Pool(num_processes)
    results = pool.map(wrapper_get_fpt, fn_args_dict)
    pool.close()
    pool.join()
    print "TIMER:", time.time() - t0

    fp_times = np.zeros(ensemble)
    fp_states = np.zeros((ensemble, params.numstates))
    for i, pair in enumerate(results):
        fp_times[i*subensemble:(i+1)*subensemble] = pair[0]
        fp_states[i*subensemble:(i+1)*subensemble, :] = pair[1]
    return fp_times, fp_states


def fast_mean_fpt_varying(param_vary_name, param_vary_values, params, num_processes, init_name="x_all", samplesize=30, establish_switch=False):
    assert samplesize % num_processes == 0
    mean_fpt_varying = [0]*len(param_vary_values)
    sd_fpt_varying = [0]*len(param_vary_values)
    for idx, pv in enumerate(param_vary_values):
        params_step = params.mod_copy( {param_vary_name: pv} )
        init_cond = map_init_name_to_init_cond(params_step, init_name)
        fp_times, fp_states = fast_fp_times(samplesize, init_cond, params_step, num_processes, establish_switch=establish_switch)
        mean_fpt_varying[idx] = np.mean(fp_times)
        sd_fpt_varying[idx] = np.std(fp_times)
    return mean_fpt_varying, sd_fpt_varying


def sample_exponential(size, scale):
    return np.random.exponential(scale=scale, size=size)


def exponential_scale_estimate(fpt_data):
    """
    Returns maximum likelihood estimator for rate 'a', from exponential: a*e^(-at) distribution
    """
    return len(fpt_data) / np.sum(fpt_data)


def fpt_histogram(fpt_list, params, figname_mod="", flag_show=False, flag_norm=True, flag_xlog10=False,
                  flag_ylog10=False, fs=16, outdir=OUTPUT_DIR, years=True):
    ensemble_size = len(fpt_list)

    if years:
        fpt_list = np.array(fpt_list)/365.0
    bins = np.linspace(np.min(fpt_list), np.max(fpt_list), 50)
    #bins = np.arange(0, 3*1e4, 50)  # to plot against FSP

    # normalize
    if flag_norm:
        y_label = 'Probability'
        weights = np.ones_like(fpt_list) / ensemble_size
    else:
        y_label = 'Frequency'
        weights = np.ones_like(fpt_list)

    # prep fig before axes mod
    fig = plt.figure(figsize=(8,6), dpi=120)
    ax = plt.gca()

    # mod axes (log)
    if flag_xlog10:
        ax.set_xscale("log", nonposx='clip')
        max_log = np.ceil(np.max(np.log10(fpt_list)))  # TODO check this matches multihist
        bins = np.logspace(0.1, max_log, 100)
    if flag_ylog10:
        #ax.set_yscale("log", nonposx='clip')
        ax.set_yscale("log")

    # plot
    plt.hist(fpt_list, bins=bins, alpha=0.6, weights=weights)
    plt.hist(fpt_list, histtype='step', bins=bins, alpha=0.6, label=None, weights=weights, edgecolor='k', linewidth=0.5,
             fill=False)

    # draw mean line
    #plt.axvline(np.mean(fpt_list), color='k', linestyle='dashed', linewidth=2)

    # labels
    plt.title('First-passage time histogram (%d runs) - %s' % (ensemble_size, params.system), fontsize=fs)
    if years:
        ax.set_xlabel('First-passage time (years)', fontsize=fs)
    else:
        ax.set_xlabel('First-passage time (cell division timescale)', fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.tick_params(labelsize=fs)
    # plt.locator_params(axis='x', nbins=4)
    #plt.legend(loc='upper right', fontsize=fs)
    # create table of params
    plot_table_params(ax, params)
    # save and show
    plt_save = "fpt_histogram" + figname_mod
    plt.savefig(outdir + sep + plt_save + '.pdf', bbox_inches='tight')
    if flag_show:
        plt.show()
    return ax


def fpt_histogram_multi(multi_fpt_list, labels, figname_mod="", fs=12, bin_linspace=80, colours=COLOURS_DARK_BLUE,
                        figsize=(8,6), ec='k', lw=0.5, flag_norm=False, flag_show=False, flag_xlog10=False,
                        flag_ylog10=False, flag_disjoint=False):

    # resize fpt lists if not all same size (to the min size)
    fpt_lengths = [len(fpt) for fpt in multi_fpt_list]
    ensemble_size = np.min(fpt_lengths)

    # cleanup data to same size
    if sum(fpt_lengths - ensemble_size) > 0:
        print "Resizing multi_fpt_list elements:", fpt_lengths, "to the min size of:", ensemble_size
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
    plt_save = "fpt_multihistogram" + figname_mod
    fig.savefig(OUTPUT_DIR + sep + plt_save + '.pdf', bbox_inches='tight')
    if flag_show:
        plt.show()


def plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_vary_name, param_set, params, samplesize, SEM_flag=True, show_flag=False, figname_mod=""):
    if SEM_flag:
        sd_fpt_varying = sd_fpt_varying / np.sqrt(samplesize)  # s.d. from CLT since sample mean is approx N(mu, sd**2/n)
    plt.errorbar(param_set, mean_fpt_varying, yerr=sd_fpt_varying, label="sim")
    plt.title("Mean FP Time, %s varying (sample=%d)" % (param_vary_name, samplesize))
    ax = plt.gca()
    ax.set_xlabel(param_vary_name)
    ax.set_ylabel('Mean FP time')

    # log options
    for i in xrange(len(mean_fpt_varying)):
        print i, param_set[i], mean_fpt_varying[i], sd_fpt_varying[i]
    flag_xlog10 = True
    flag_ylog10 = True
    if flag_xlog10:
        ax.set_xscale("log", nonposx='clip')
        #ax.set_xlim([0.8*1e2, 1*1e7])
    if flag_ylog10:
        ax.set_yscale("log", nonposx='clip')
        #ax.set_ylim([0.8*1e2, 3*1e5])

    # create table of params
    plot_table_params(ax, params)
    plt_save = "mean_fpt_varying" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


def simplex_heatmap(fp_times, fp_states, params, ax=None, fp=True, streamlines=True, colour=True, cbar=True,
                    flag_show=True, outdir=OUTPUT_DIR, figname_mod="", save=True, smallfig=False):
    seaborn.reset_orig()
    # plot simplex (as 2D triangle face, equilateral)
    ax = plot_simplex2D(params, smallfig=smallfig, fp=fp, streamlines=streamlines, ax=ax)  # TODO streamlines

    # normalize stochastic x y z st x + y + z = N
    scales = params.N / np.sum(fp_states, axis=1)
    x_norm = fp_states[:, 0] * scales
    y_norm = fp_states[:, 1] * scales
    z_norm = fp_states[:, 2] * scales
    # conversion to 2D
    #conv_x = (params.N + fp_states[:, 1] - fp_states[:, 0]) / 2.0  # old way, but points don't lie ON simplex unless normalized
    conv_x = (params.N + y_norm - x_norm) / 2.0
    conv_y = z_norm#fp_states[:, 2]

    # plot points
    if colour:
        paths = ax.scatter(conv_x, conv_y, c=fp_times, marker='o', s=3, alpha=0.4, zorder=3)  # TODO colour or size change for fp_times
        # LEFT
        #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
        #plt.colorbar(cax=cbaxes, pad=-0.1)
        # RIGHT
        if cbar:
            plt.colorbar(paths, pad=0.18)
    else:
        ax.scatter(conv_x, conv_y, c='RosyBrown', marker='o', s=3, alpha=0.7, zorder=3)

    # save
    if save:
        plt_save = "simplex_heatmap" + figname_mod
        plt.savefig(outdir + sep + plt_save + '.pdf', bbox_inches='tight')
    if flag_show:
        plt.show()
    return ax


def fp_state_zloc_hist(fp_times, fp_states, params, ax=None, normalize=False, fp=True, kde=True,
                       flag_show=True, outdir=OUTPUT_DIR, figname_mod="", save=True):

    N = params.N
    seaborn.set_context("notebook", font_scale=1.9)  # TODO this breaks edges of the markers for FP but it is needed for font size?

    # plot fp_states z coord histogram
    if normalize:
        scales = params.N / np.sum(fp_states, axis=1)
        fp_zcoord = fp_states[:, 2] * scales
        zlabel = r'$z(\tau)/N$'  # how to denote that it is normalized here?
    else:
        fp_zcoord = fp_states[:, 2]
        zlabel = r'$z(\tau)$'
    if kde:
        ax = seaborn.kdeplot(fp_zcoord, shade=True, cut=0.0, vertical=True, ax=ax)
    else:
        ax = seaborn.distplot(fp_zcoord, vertical=True, ax=ax)
    ax.set_ylabel(zlabel)
    ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.set_yticks([0, N])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    # plot fp horizontal line
    if fp:
        stable_fps = []
        unstable_fps = []
        all_fps = fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=35, tol=10e-1, buffer=True)
        for fp in all_fps:
            J = jacobian_numerical_2d(params, fp[0:2])
            eigenvalues, V = np.linalg.eig(J)
            if eigenvalues[0] < 0 and eigenvalues[1] < 0:
                stable_fps.append(fp)
            else:
                unstable_fps.append(fp)
        for fp in stable_fps:
            ax.axhline(fp[2], linestyle='--', linewidth=1.0, color='k')
        """
        for fp in unstable_fps:
            fp_x = (N + fp[1] - fp[0]) / 2.0
            #plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', markerfacecolor="None")
            plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color=(0.902, 0.902, 0.902))
        """

    # save
    if save:
        plt_save = "fp_state_zloc_hist" + figname_mod
        plt.savefig(outdir + sep + plt_save + '.pdf', bbox_inches='tight')
    if flag_show:
        plt.show()
    return ax


def fp_zloc_times_joint(fp_times, fp_states, params, ax=None, normalize=False, fp=True, color='grey', flag_show=True,
                        mean=True, logx=True, outdir=OUTPUT_DIR, figname_mod="", fluxval=None, save=True):
    """
    seaborn documentation: https://seaborn.pydata.org/generated/seaborn.jointplot.html
    "Intended to be a fairly lightweight wrapper; if you need more flexibility, you should use JointGrid directly."
    """
    N = params.N
    seaborn.set_context("notebook", font_scale=1.9)  # TODO this breaks edges of the markers for FP but it is needed for font size?

    # TODO cleanup these, maybe use jointgrid
    fig = plt.figure()
    ax = plt.gca()

    # plot fp_states z coord histogram
    ymin = 0.0
    ymax = 1.0

    """
    if figname_mod == '_TR_N100_icfp':
        Q = len(fp_times)
        a = []
        b = []
        for idx in xrange(Q):
            if fp_times[idx] > 10 ** 3:
                a.append(fp_times[idx])
                b.append(fp_states[idx])
        print Q, len(a)
        fp_times = np.array(a)
        fp_states = np.array(b)
    """

    if normalize:
        scales = 1 / np.sum(fp_states, axis=1)
        fp_zcoord = fp_states[:, 2] * scales
        zlabel = r'$z(\tau)/N$'  # how to denote that it is normalized here?
    else:
        fp_zcoord = fp_states[:, 2]
        zlabel = r'$z(\tau)$'
        ymax = params.N

    nbins = 20
    if logx:
        xbins = np.logspace(np.log10(np.min(fp_times)), np.log10(np.max(fp_times)), nbins)

    g = seaborn.JointGrid(x=fp_times, y=fp_zcoord)
    g = g.plot_joint(plt.scatter, color=color, alpha=0.7, s=40)#, edgecolor="white")
    #g = g.plot_marginals(seaborn.distplot, kde=kde, color="g")
    _ = g.ax_marg_y.hist(fp_zcoord, color="grey", alpha=.7, orientation="horizontal", bins=nbins)
    if logx:
        xbins = np.logspace(np.log10(np.min(fp_times)), np.log10(np.max(fp_times)), nbins)
        _ = g.ax_marg_x.hist(fp_times, color="grey", alpha=.7, bins=xbins)
    else:
        _ = g.ax_marg_x.hist(fp_times, color="grey", alpha=.7)

    if logx:
        g.ax_joint.set_xscale('log')

    g.ax_joint.set_ylabel(zlabel)
    g.ax_joint.set_xlabel(r'$\tau$')

    g.ax_joint.set_xlim((np.min(fp_times), np.max(fp_times)))
    g.ax_joint.set_ylim((ymin, ymax*1.01))

    if fluxval is not None:
        g.ax_joint.axvline(fluxval, linestyle='--', linewidth=1.0, color='k')

    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #ax.set_yticks([0, N])
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    if mean:
        meantau = np.mean(fp_times)
        g.ax_joint.axvline(meantau, linestyle='-', linewidth=2.0, color='k')

    # plot fp horizontal line
    if fp:
        stable_fps = []
        unstable_fps = []
        all_fps = fp_location_fsolve(params, check_near_traj_endpt=True, gridsteps=35, tol=10e-1, buffer=True)
        for fp in all_fps:
            J = jacobian_numerical_2d(params, fp[0:2])
            eigenvalues, V = np.linalg.eig(J)
            if eigenvalues[0] < 0 and eigenvalues[1] < 0:
                stable_fps.append(fp)
            else:
                unstable_fps.append(fp)
        for fp in stable_fps:
            coord = fp[2]
            if normalize:
                coord = coord / params.N
            g.ax_joint.axhline(coord, linestyle='-', linewidth=2.0, color='k')
        for fp in unstable_fps:
            coord = fp[2]
            if normalize:
                coord = coord / params.N
            g.ax_joint.axhline(coord, linestyle='--', linewidth=2.0, color='k')
        """
        for fp in unstable_fps:
            fp_x = (N + fp[1] - fp[0]) / 2.0
            #plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', markerfacecolor="None")
            plt.plot(fp_x, fp[2], marker='o', markersize=ms, markeredgecolor='black', linewidth='3', color=(0.902, 0.902, 0.902))
        """

    # save
    if save:
        plt_save = "fp_zloc_times_joint" + figname_mod
        plt.savefig(outdir + sep + plt_save + '.pdf', bbox_inches='tight')
    if flag_show:
        plt.show()
    return ax


if __name__ == "__main__":
    # SCRIPT FLAGS
    run_compute_fpt = True
    run_read_fpt = False
    run_generate_hist_multi = False
    run_load_hist_multi = False
    run_collect = False
    run_means_read_and_plot = False
    run_means_collect_and_plot = False

    # SCRIPT PARAMETERS
    establish_switch = False
    brief = True
    num_steps = 1000000  # default 1000000
    ensemble = 24  # default 100

    # DYNAMICS PARAMETERS
    params = presets('preset_xyz_constant')  # preset_xyz_constant, preset_xyz_constant_fast, valley_2hit

    # OTHER PARAMETERS
    init_cond = np.zeros(params.numstates, dtype=int)
    init_cond[0] = int(params.N)

    # PLOTTING
    FS = 16
    EC = 'k'
    LW = 0.5
    FIGSIZE=(8,6)

    if run_compute_fpt:
        fp_times, fp_states = get_fpt(ensemble, init_cond, params, num_steps=num_steps, establish_switch=establish_switch, brief=brief)
        write_fpt_and_params(fp_times, fp_states, params)
        fpt_histogram(fp_times, params, flag_show=True, figname_mod="XZ_model_withFeedback_mu1e-1")
        simplex_heatmap(fp_times, fp_states, params, flag_show=True)

    if run_read_fpt:
        dbdir = OUTPUT_DIR
        dbdir_100 = dbdir + sep + "fpt_mean" + sep + "100_c95"
        fp_times_xyz_100, fp_states_xyz_100, params_a = read_fpt_and_params(dbdir_100)
        dbdir_10k = dbdir + sep + "fpt_mean" + sep + "10k_c95"
        fp_times_xyz_10k, fp_states_xyz_10k, params_b = read_fpt_and_params(dbdir_10k)

    if run_generate_hist_multi:
        ensemble = 21
        num_proc = cpu_count() - 1
        param_vary_id = "N"
        param_idx = PARAMS_ID_INV[param_vary_id]
        param_vary_values = [1e2, 1e3, 1e4]
        param_vary_labels = ['A', 'B', 'C']
        params_ensemble = [params.params_list[:] for _ in param_vary_values]
        multi_fpt = np.zeros((len(param_vary_values), ensemble))
        multi_fpt_labels = ['label' for _ in param_vary_values]
        for idx, param_val in enumerate(param_vary_values):
            param_val_string = "%s=%.3f" % (param_vary_id, param_val)
            params_step = params.mod_copy({param_vary_id: param_val})
            #fp_times = get_fpt(ensemble, init_cond, params_set[idx], num_steps=num_steps)
            fp_times, fp_states = fast_fp_times(ensemble, init_cond, params_step, num_proc, establish_switch=establish_switch)
            write_fpt_and_params(fp_times, params_step, filename="fpt_multi", filename_mod=param_val_string)
            multi_fpt[idx,:] = np.array(fp_times)
            multi_fpt_labels[idx] = "%s (%s)" % (param_vary_labels[idx], param_val_string)
        fpt_histogram_multi(multi_fpt, multi_fpt_labels, flag_show=True, flag_ylog10=False)

    if run_load_hist_multi:
        flag_norm = True
        dbdir = OUTPUT_DIR + sep + "may25_100"
        #dbdir_c80 = dbdir + "fpt_feedback_z_ens1040_c0.80_params"
        c80_header = "fpt_feedback_z_ens1040_c80_N100"
        c88_header = "fpt_feedback_z_ens1040_c88_N100"
        c95_header = "fpt_feedback_z_ens1040_c95_N100"
        fp_times_xyz_c80, fp_times_xyz_c80, params_a = read_fpt_and_params(dbdir, filename_times="%s_data.txt" % c80_header, filename_params="%s_params.csv" % c80_header)
        fp_times_xyz_c88, fp_states_xyz_c88, params_b = read_fpt_and_params(dbdir, filename_times="%s_data.txt" % c88_header, filename_params="%s_params.csv" % c88_header)
        fp_times_xyz_c95, fp_states_xyz_c95, params_c = read_fpt_and_params(dbdir, filename_times="%s_data.txt" % c95_header, filename_params="%s_params.csv" % c95_header)
        fpt_histogram(fp_times_xyz_c88, params_b, flag_ylog10=False, figname_mod="_xyz_feedbackz_N10k_c88_may25")
        plt.close('all')
        fpt_histogram(fp_times_xyz_c88, params_b, flag_ylog10=True, figname_mod="_xyz_feedbackz_N10k_c88_may25_logy")
        plt.close('all')
        multi_fpt = [fp_times_xyz_c80, fp_times_xyz_c88, fp_times_xyz_c95]
        labels = ("c=0.80 (Region I)", "c=0.88 (Region IV)", "c=0.95 (Region III)")
        fpt_histogram_multi(multi_fpt, labels, flag_show=True, flag_ylog10=False, flag_norm=flag_norm, fs=FS, ec=EC, lw=LW, figsize=FIGSIZE)
        fpt_histogram_multi(multi_fpt, labels, flag_show=True, flag_ylog10=True, flag_norm=flag_norm, fs=FS, ec=EC, lw=LW, figsize=FIGSIZE)
        fpt_histogram_multi(multi_fpt, labels, flag_show=True, flag_ylog10=True, flag_norm=False, fs=FS, ec=EC, lw=LW, figsize=FIGSIZE, flag_disjoint=True)

    if run_means_read_and_plot:
        datafile = OUTPUT_DIR + sep + "fpt_stats_collected_mean_sd_varying_N.txt"
        paramfile = OUTPUT_DIR + sep + "fpt_stats_collected_mean_sd_varying_N_params.csv"
        samplesize=48
        mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
            read_varying_mean_sd_fpt_and_params(datafile, paramfile)
        plt_axis = plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, samplesize,
                                         SEM_flag=True, show_flag=True, figname_mod="_%s_n%d" % (param_to_vary, samplesize))
        """
        mu = params.mu
        mixed_fp_zinf_at_N = [0.0]*len(param_set)
        for idx, N in enumerate(param_set):
            params_at_N = params.mod_copy( {'N': N} )
            fps = get_physical_and_stable_fp(params_at_N)
            assert len(fps) == 1
            mixed_fp_zinf_at_N[idx] = fps[0][2]
        plt_axis.plot(param_set, [1/(mu*n) for n in param_set], '-o', label="(mu*N)^-1")
        plt_axis.plot(param_set, [1/(mu*zinf) for zinf in mixed_fp_zinf_at_N], '-o', label="(mu*z_inf)^-1")
        plt_axis.set_yscale("log", nonposx='clip')
        plt_axis.set_xscale("log", nonposx='clip')
        plt_axis.legend()
        plt.savefig(OUTPUT_DIR + sep + "theorycompare_loglog" + '.png', bbox_inches='tight')
        plt.show()
        """

    if run_means_collect_and_plot:
        dbdir = OUTPUT_DIR + sep + "tocollect" + sep + "runset_june17_FPT_cvary_44_ens240"
        datafile, paramfile = collect_fpt_mean_stats_and_params(dbdir)
        samplesize=240
        mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
            read_varying_mean_sd_fpt_and_params(datafile, paramfile)
        plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, samplesize,
                              SEM_flag=True, show_flag=True, figname_mod="_%s_n%d" % (param_to_vary, samplesize))

