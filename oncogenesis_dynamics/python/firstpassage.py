import csv
import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID, ODE_SYSTEMS
from data_io import write_params, read_params
from formulae import stoch_gillespie


def get_fpt(ensemble, init_cond, num_steps, params, system):
    fp_times = np.zeros(ensemble)
    for i in xrange(ensemble):
        species, times = stoch_gillespie(init_cond, num_steps, system, params, fpt_flag=True)
        fp_times[i] = times[-1]
        print i, times[-1]
    return fp_times


def write_fpt_and_params(fpt, params, system, filename="fpt", filename_mod=""):
    if filename_mod != "":
        filename_params = filename + "_" + filename_mod + "_params.csv"
        filename_fpt = filename + "_" + filename_mod + "_data.txt"
    else:
        filename_params = filename + "_params.csv"
        filename_fpt = filename + "_data.txt"
    write_params(params, system, OUTPUT_DIR, filename_params)
    with open(OUTPUT_DIR + sep + filename_fpt, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for idx in xrange(len(fpt)):
            writer.writerow([str(fpt[idx])])
    return OUTPUT_DIR + sep + filename_fpt


def read_fpt_and_params(filedir, filename_data, filename_params):
    params_with_system = read_params(filedir, filename_params)
    assert params_with_system[-1] in ODE_SYSTEMS
    params = params_with_system[:-1]
    system = params_with_system[-1]
    with open(filedir + sep + filename_data, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
        nn = sum(1 for row in datareader)
        csvfile.seek(0)
        fp_times = [0.0]*nn
        for idx, fpt in enumerate(datareader):
            fp_times[idx] = float(fpt[0])
    return fp_times, params, system


def fpt_histogram(fpt_list, params, system, show_flag=False, figname_mod="", x_log10_flag=False, y_log10_flag=False):
    ensemble_size = len(fpt_list)
    if x_log10_flag:
        max_log = np.ceil(np.max(np.log10(fpt_list)))
        plt.hist(fpt_list, bins=np.logspace(0.1, max_log, 50))
        ax = plt.gca()
        ax.set_xlabel('log10(fpt)')
        ax.set_ylabel('frequency')
        ax.set_xscale("log", nonposx='clip')
    elif y_log10_flag:
        plt.hist(fpt_list, bins='auto')
        ax = plt.gca()
        ax.set_xlabel('fpt')
        ax.set_ylabel('log10(frequency)')
        ax.set_yscale("log", nonposx='clip')
    else:
        plt.hist(fpt_list, bins='auto')
        ax = plt.gca()
        ax.set_xlabel('fpt')
        ax.set_ylabel('frequency')
    plt.title('First passage time histogram (%d runs) - %s' % (ensemble_size, system))
    # DRAW MEAN LINE
    plt.axvline(np.mean(fpt_list), color='k', linestyle='dashed', linewidth=2)
    # CREATE TABLE OF PARAMS
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] for i in xrange(len(PARAMS_ID))]
    print len(row_labels), len(table_vals)
    param_table = plt.table(cellText=table_vals,
                            colWidths=[0.1]*3,
                            rowLabels=row_labels,
                            loc='center right')
    plt_save = "fpt_histogram" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()


def fpt_histogram_multi(multi_fpt_list, labels, show_flag=False, figname_mod="", x_log10_flag=False, y_log10_flag=False):
    ensemble_size = len(multi_fpt_list[0])
    bins = np.linspace(np.min(multi_fpt_list), np.max(multi_fpt_list), 50)
    if x_log10_flag:
        max_log = np.ceil(np.max(np.log10(multi_fpt_list)))
        for idx, fpt_list in enumerate(multi_fpt_list):
            plt.hist(fpt_list, bins=np.logspace(0.1, max_log, 50), alpha=0.5, label=labels[idx])
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
    # CREATE TABLE OF PARAMS
    """
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] for i in xrange(len(PARAMS_ID))]
    print len(row_labels), len(table_vals)
    param_table = plt.table(cellText=table_vals,
                            colWidths=[0.1]*3,
                            rowLabels=row_labels,
                            loc='center right')
    """
    plt_save = "fpt_multihistogram" + figname_mod
    plt.savefig(OUTPUT_DIR + sep + plt_save + '.png', bbox_inches='tight')
    if show_flag:
        plt.show()


if __name__ == "__main__":
    # SCRIPT PARAMETERS
    system = "feedback_mu_XZ_model"  # "feedback_mu_XZ_model" or "feedback_z
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

    fp_times = get_fpt(ensemble, init_cond, num_steps, params, system)
    write_fpt_and_params(fp_times, params, system)
    fpt_histogram(fp_times, params, system, show_flag=True, figname_mod="XZ_model_withFeedback_mu1e-1")
