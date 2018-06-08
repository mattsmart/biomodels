import matplotlib.pyplot as plt
import numpy as np
import time
from os import sep
from multiprocessing import Pool, cpu_count

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, COLOURS_DARK_BLUE
from data_io import read_varying_mean_sd_fpt_and_params, collect_fpt_mean_stats_and_params, read_fpt_and_params,\
                    write_fpt_and_params
from formulae import stoch_gillespie, stoch_bnb, stoch_tauleap, get_physical_and_stable_fp, map_init_name_to_init_cond
from params import Params
from presets import presets
from plotting import plot_table_params


def get_fpt(ensemble, init_cond, params, num_steps=1000000, establish_switch=False, brief=True):
    if establish_switch:
        fpt_flag = False
        establish_flag = True
    else:
        fpt_flag = True
        establish_flag = False
    fp_times = np.zeros(ensemble)
    for i in xrange(ensemble):
        #species, times = stoch_gillespie(init_cond, num_steps, params, fpt_flag=fpt_flag, establish_flag=establish_flag)
        if brief:
            species_end, times_end = stoch_tauleap(init_cond, num_steps, params, fpt_flag=fpt_flag, establish_flag=establish_flag, brief=brief)
        else:
            species, times = stoch_tauleap(init_cond, num_steps, params, fpt_flag=fpt_flag, establish_flag=establish_flag, brief=brief)
            times_end = times[-1]
            # plotting
            #plt.plot(times, species)
            #plt.show()
        fp_times[i] = times_end
        if establish_switch:
            print "establish time is", fp_times[i]
    return fp_times


def get_mean_fpt(init_cond, params, samplesize=32, establish_switch=False):
    fpt = get_fpt(samplesize, init_cond, params, establish_switch=establish_switch)
    return np.mean(fpt)


def wrapper_get_fpt(fn_args_dict):
    if fn_args_dict['kwargs'] is not None:
        return get_fpt(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return get_fpt(*fn_args_dict['args'])


def fast_fp_times(ensemble, init_cond, params, num_processes, num_steps='default',establish_switch=False):
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
    for i, result in enumerate(results):
        fp_times[i*subensemble:(i+1)*subensemble] = result
    return fp_times


def fast_mean_fpt_varying(param_vary_name, param_vary_values, params, num_processes, init_name="x_all", samplesize=30, establish_switch=False):
    assert samplesize % num_processes == 0
    mean_fpt_varying = [0]*len(param_vary_values)
    sd_fpt_varying = [0]*len(param_vary_values)
    for idx, pv in enumerate(param_vary_values):
        params_step = params.mod_copy( {param_vary_name: pv} )
        init_cond = map_init_name_to_init_cond(params, init_name)
        fp_times = fast_fp_times(samplesize, init_cond, params_step, num_processes, establish_switch=establish_switch)
        mean_fpt_varying[idx] = np.mean(fp_times)
        sd_fpt_varying[idx] = np.std(fp_times)
    return mean_fpt_varying, sd_fpt_varying


def fpt_histogram(fpt_list, params, figname_mod="", flag_show=False, flag_norm=True, flag_xlog10=False, flag_ylog10=False, fs=12):
    ensemble_size = len(fpt_list)
    bins = np.linspace(np.min(fpt_list), np.max(fpt_list), 50)

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
        ax.set_yscale("log", nonposx='clip')

    # plot
    plt.hist(fpt_list, bins=bins, alpha=0.6, weights=weights)
    plt.hist(fpt_list, histtype='step', bins=bins, alpha=0.6, label=None, weights=weights, edgecolor='k', linewidth=0.5,
             fill=False)

    # draw mean line
    #plt.axvline(np.mean(fpt_list), color='k', linestyle='dashed', linewidth=2)

    # labels
    plt.title('First-passage time histogram (%d runs) - %s' % (ensemble_size, params.system), fontsize=fs)
    ax.set_xlabel('First-passage time (cell division timescale)', fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.tick_params(labelsize=fs)
    # plt.locator_params(axis='x', nbins=4)
    #plt.legend(loc='upper right', fontsize=fs)
    # create table of params
    plot_table_params(ax, params)
    # save and show
    plt_save = "fpt_histogram" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.pdf', bbox_inches='tight')
    if flag_show:
        plt.show()


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
    flag_xlog10 = True
    flag_ylog10 = True
    if flag_xlog10:
        ax.set_xscale("log", nonposx='clip')
    if flag_ylog10:
        ax.set_yscale("log", nonposx='clip')
        ax.set_ylim([0.5*1e2, 3*1e5])
    # create table of params
    plot_table_params(ax, params)
    plt_save = "mean_fpt_varying" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()
    return ax


if __name__ == "__main__":
    # SCRIPT FLAGS
    run_compute_fpt = False
    run_read_fpt = False
    run_generate_hist_multi = False
    run_load_hist_multi = False
    run_collect = False
    run_means_read_and_plot = False
    run_means_collect_and_plot = True

    # SCRIPT PARAMETERS
    establish_switch = True
    brief=True
    num_steps = 1000000  # default 1000000
    ensemble = 10  # default 100

    # DYNAMICS PARAMETERS
    params = presets('preset_xyz_constant_fast')  # preset_xyz_constant, preset_xyz_constant_fast, valley_2hit

    # OTHER PARAMETERS
    init_cond = np.zeros(params.numstates, dtype=int)
    init_cond[0] = int(params.N)

    # PLOTTING
    FS = 16
    EC = 'k'
    LW = 0.5
    FIGSIZE=(8,6)

    if run_compute_fpt:
        fp_times = get_fpt(ensemble, init_cond, params, num_steps=num_steps, establish_switch=establish_switch, brief=brief)
        write_fpt_and_params(fp_times, params)
        fpt_histogram(fp_times, params, flag_show=True, figname_mod="XZ_model_withFeedback_mu1e-1")

    if run_read_fpt:
        dbdir = OUTPUT_DIR
        dbdir_100 = dbdir + sep + "fpt_mean" + sep + "100_c95"
        fp_times_xyz_100, params_a = read_fpt_and_params(dbdir_100)
        dbdir_10k = dbdir + sep + "fpt_mean" + sep + "10k_c95"
        fp_times_xyz_10k, params_b = read_fpt_and_params(dbdir_10k)

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
            fp_times = fast_fp_times(ensemble, init_cond, params_step, num_proc, establish_switch=establish_switch)
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
        dbdir = OUTPUT_DIR + sep + "tocollect" + sep + "runset_june8_valley2hit_ens80"
        datafile, paramfile = collect_fpt_mean_stats_and_params(dbdir)
        samplesize=48
        mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params = \
            read_varying_mean_sd_fpt_and_params(datafile, paramfile)
        plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, samplesize,
                              SEM_flag=True, show_flag=True, figname_mod="_%s_n%d" % (param_to_vary, samplesize))
