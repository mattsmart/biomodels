import matplotlib.pyplot as plt
import numpy as np
import time
from os import sep
from multiprocessing import Pool

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV
from data_io import read_varying_mean_sd_fpt_and_params
from formulae import stoch_gillespie


def get_fpt(ensemble, init_cond, params, system, num_steps=100000):
    fp_times = np.zeros(ensemble)
    for i in xrange(ensemble):
        species, times = stoch_gillespie(init_cond, num_steps, system, params, fpt_flag=True)
        fp_times[i] = times[-1]
        print i, times[-1]
    return fp_times


def get_mean_fpt(init_cond, params, system, samplesize=32):
    fpt = get_fpt(samplesize, init_cond, params, system)
    return np.mean(fpt)


def wrapper_get_fpt(fn_args_dict):
    if fn_args_dict['kwargs'] is not None:
        return get_fpt(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return get_fpt(*fn_args_dict['args'])


def fast_fp_times(ensemble, init_cond, params, system, num_processes):
    fn_args_dict = [0]*num_processes
    print "NUM_PROCESSES:", num_processes
    assert ensemble % num_processes == 0
    for i in xrange(num_processes):
        subensemble = ensemble / num_processes
        print "process:", i, "job size:", subensemble, "runs"
        fn_args_dict[i] = {'args': (subensemble, init_cond, params, system),
                           'kwargs': None}
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


def fast_mean_fpt_varying(param_vary_name, param_vary_values, params, system, num_processes, samplesize=30):
    assert samplesize % num_processes == 0
    mean_fpt_varying = [0]*len(param_vary_values)
    sd_fpt_varying = [0]*len(param_vary_values)
    for idx, pv in enumerate(param_vary_values):
        init_cond = [int(PARAMS_ID_INV['N']), 0, 0]
        params_step = params
        params_step[PARAMS_ID_INV[param_vary_name]] = pv
        fp_times = fast_fp_times(samplesize, init_cond, params_step, system, num_processes)
        mean_fpt_varying[idx] = np.mean(fp_times)
        sd_fpt_varying[idx] = np.std(fp_times)
    return mean_fpt_varying, sd_fpt_varying


def fpt_histogram(fpt_list, params, system, show_flag=False, figname_mod="", x_log10_flag=False, y_log10_flag=False):
    ensemble_size = len(fpt_list)
    bins = np.linspace(np.min(fpt_list), np.max(fpt_list), 50)
    if x_log10_flag:
        max_log = np.ceil(np.max(np.log10(fpt_list)))
        plt.hist(fpt_list, bins=np.logspace(0.1, max_log, 50))
        ax = plt.gca()
        ax.set_xlabel('log10(fpt)')
        ax.set_ylabel('frequency')
        ax.set_xscale("log", nonposx='clip')
    elif y_log10_flag:
        plt.hist(fpt_list, bins=bins)
        ax = plt.gca()
        ax.set_xlabel('fpt')
        ax.set_ylabel('log10(frequency)')
        ax.set_yscale("log", nonposx='clip')
    else:
        plt.hist(fpt_list, bins=bins)
        ax = plt.gca()
        ax.set_xlabel('fpt')
        ax.set_ylabel('frequency')
    plt.title('First passage time histogram (%d runs) - %s' % (ensemble_size, system))
    # DRAW MEAN LINE
    plt.axvline(np.mean(fpt_list), color='k', linestyle='dashed', linewidth=2)
    # CREATE TABLE OF PARAMS
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals,
                            colWidths=[0.1]*3,
                            rowLabels=row_labels,
                            loc='center right')
    plt_save = "fpt_histogram" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()


def fpt_histogram_multi(multi_fpt_list, labels, show_flag=False, figname_mod="", x_log10_flag=False, y_log10_flag=False):
    # resize fpt lists if not all same size (to the min size)
    fpt_lengths = [len(fpt) for fpt in multi_fpt_list]
    ensemble_size = np.min(fpt_lengths)
    if sum(fpt_lengths - ensemble_size) > 0:
        print "Resizing multi_fpt_list elements:", fpt_lengths, "to the min size of:", ensemble_size
        for idx in xrange(len(fpt_lengths)):
            multi_fpt_list[idx] = multi_fpt_list[idx][:ensemble_size]

    bins = np.linspace(np.min(multi_fpt_list), np.max(multi_fpt_list), 100)
    if x_log10_flag:
        max_log = np.ceil(np.max(np.log10(multi_fpt_list)))
        for idx, fpt_list in enumerate(multi_fpt_list):
            plt.hist(fpt_list, bins=np.logspace(0.1, max_log, 100), alpha=0.5, label=labels[idx])
        ax = plt.gca()
        ax.set_xlabel('log10(fpt)')
        ax.set_ylabel('frequency')
        ax.set_xscale("log", nonposx='clip')
    elif y_log10_flag:
        for idx, fpt_list in enumerate(multi_fpt_list):
            plt.hist(fpt_list, bins=bins, alpha=0.5, label=labels[idx])
        ax = plt.gca()
        ax.set_xlabel('fpt')
        ax.set_ylabel('log10(frequency)')
        ax.set_yscale("log", nonposx='clip')
    else:
        for idx, fpt_list in enumerate(multi_fpt_list):
            plt.hist(fpt_list, bins=bins, alpha=0.5, label=labels[idx])
        ax = plt.gca()
        ax.set_xlabel('fpt')
        ax.set_ylabel('frequency')
    plt.title('First passage time histogram (%d runs)' % (ensemble_size))
    plt.legend(loc='upper right')
    # DRAW MEAN LINE
    #plt.axvline(np.mean(fpt_list), color='k', linestyle='dashed', linewidth=2)
    plt_save = "fpt_multihistogram" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()


def plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_vary_name, param_set, params, system, samplesize, SEM_flag=True, show_flag=False, figname_mod=""):
    if SEM_flag:
        sd_fpt_varying = sd_fpt_varying / np.sqrt(samplesize)  # s.d. from CLT since sample mean is approx N(mu, sd**2/n)
    plt.errorbar(param_set, mean_fpt_varying, yerr=sd_fpt_varying)
    plt.title("Mean FP Time, %s varying (sample=%d)" % (param_vary_name, samplesize))
    ax = plt.gca()
    ax.set_xlabel(param_vary_name)
    ax.set_ylabel('Mean FP time')
    # CREATE TABLE OF PARAMS
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] if PARAMS_ID[i] not in [param_vary_name] else ["None"]
                  for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals,
                            colWidths=[0.1]*3,
                            rowLabels=row_labels,
                            loc='center right')
    plt_save = "mean_fpt_varying" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()


if __name__ == "__main__":
    # SCRIPT PARAMETERS
    system = "feedback_mu_XZ_model"  # "feedback_mu_XZ_model" or "feedback_z"
    num_steps = 100000
    ensemble = 100

    # DYNAMICS PARAMETERS
    alpha_plus = 0.0 #0.2  # 0.05 #0.4
    alpha_minus = 0.0 #0.5  # 4.95 #0.5
    mu = 0.001  # 0.001
    a = 1.0
    b = 0.8
    c = 0.81  # 2.6 #1.2
    N = 100.0  # 100
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = mu*1e-1
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

    # OTHER PARAMETERS
    init_cond = [int(N), 0, 0]

    """
    fp_times = get_fpt(ensemble, init_cond, num_steps, params, system)
    write_fpt_and_params(fp_times, params, system)
    fpt_histogram(fp_times, params, system, show_flag=True, figname_mod="XZ_model_withFeedback_mu1e-1")
    """

    """
    dbdir = OUTPUT_DIR
    dbdir_100 = dbdir + sep + "fpt_mean" + sep + "100_c95"
    fp_times_xyz_100, params_a, system_a = read_fpt_and_params(dbdir_100)
    dbdir_10k = dbdir + sep + "fpt_mean" + sep + "10k_c95"
    fp_times_xyz_10k, params_b, system_b = read_fpt_and_params(dbdir_10k)
    """

    """
    print "DO N100 FIRST"
    import random
    true_mean100 = np.mean(fp_times_xyz_100)
    for k in [1,5,10,15,20,25,30,40,50,75,100,200,500]:
        subsample = random.sample(fp_times_xyz_100, k)
        print len(fp_times_xyz_100), "is, ", true_mean100, "| ", k, "is", np.mean(subsample)

    print "DO N10k now"
    import random
    true_mean10k = np.mean(fp_times_xyz_10k)
    for k in [1,5,10,15,20,25,30,40,50,75,100,200,500]:
        subsample = random.sample(fp_times_xyz_10k, k)
        print len(fp_times_xyz_10k), "is, ", true_mean10k, "| ", k, "is", np.mean(subsample)
    """

    datafile = OUTPUT_DIR + sep + "fpt_stats_N100_c85_n64_4hr_mean_sd_varying_b.txt"
    paramfile = OUTPUT_DIR + sep + "fpt_stats_N100_c85_n64_4hr_params.csv"
    samplesize=64
    mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system = \
        read_varying_mean_sd_fpt_and_params(datafile, paramfile)
    plot_mean_fpt_varying(mean_fpt_varying, sd_fpt_varying, param_to_vary, param_set, params, system, samplesize,
                          SEM_flag=True, show_flag=True, figname_mod="_%s_n%d" % (param_to_vary, samplesize))

    #dbdir_c95 = dbdir + "1000_xyz_feedbackZ_c95"
    #dbdir_c81_xz = dbdir + "1000_xz_feedbackMUBASE_c81"
    #fp_times_xyz_c086, params_a, system_a = read_fpt_and_params(dbdir_c86, "fpt_feedback_z_ens256_N10k_c086_full_data.txt",
    #                                                            "fpt_feedback_z_ens256_N10k_c086_full_params.csv")

    #fp_times_xyz_c095, params_c, system_c = read_fpt_and_params(dbdir_c95, "fpt_xyz_feedbackz_1000_c95_data.txt",
    #                                                            "fpt_xyz_feedbackz_1000_c95_params.csv")
    #fp_times_xz_c081, params_d, system_d = read_fpt_and_params(dbdir_c81_xz, "fpt_xz_1000_c81_data.txt",
    #                                                            "fpt_xz_1000_c81_params.csv")

    #fpt_histogram(fp_times_xyz_c095, params_b, system_b, y_log10_flag=False, figname_mod="_xyz_feedbackz_N10k_c95_ap18")
    #plt.close('all')
    #fpt_histogram(fp_times_xyz_c095, params_b, system_b, y_log10_flag=True, figname_mod="_xyz_feedbackz_N10k_c95_ap18_logy")
    #plt.close('all')

    """
    multi_fpt = [fp_times_xyz_c086, fp_times_xyz_c095]
    labels = ("XYZ_c0.86_N10k", "XYZ_c0.95_N10k")
    fpt_histogram_multi(multi_fpt, labels, show_flag=True, y_log10_flag=False)
    plt.close('all')
    fpt_histogram_multi(multi_fpt, labels, show_flag=True, y_log10_flag=True)
    """

    # print "XZ mean and log10", np.mean(fp_times_xz), np.log10(np.mean(fp_times_xz))
    #dbdir = OUTPUT_DIR + sep + "tocollect"
    #collect_fpt_and_params(dbdir)
