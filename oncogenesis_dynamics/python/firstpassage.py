import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID
from formulae import stoch_gillespie


def fpt_histogram(fpt_list, params, show_flag=False, figname_mod=""):
    ensemble_size = len(fpt_list)
    plt.hist(fpt_list, bins='auto')
    plt.title('First passage time histogram (%d runs)' % ensemble_size)
    ax = plt.gca()
    ax.set_xlabel('frequency')
    ax.set_ylabel('fpt')
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



if __name__ == "__main__":
    # SCRIPT PARAMETERS
    system = "feedback_z"  # "default" or "feedback_z" etc
    num_steps = 10000
    ensemble = 100

    # DYNAMICS PARAMETERS
    alpha_plus = 0.0 #0.2  # 0.05 #0.4
    alpha_minus = 0.0 #0.5  # 4.95 #0.5
    mu = 0.001  # 0.01
    a = 1.0
    b = 0.8
    c = 0.81  # 2.6 #1.2
    N = 100.0  # 100
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = mu*1e-3
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

    init_cond = [int(N), 0, 0]

    fp_times = np.zeros(ensemble)
    for i in xrange(ensemble):
        species, times = stoch_gillespie(init_cond, num_steps, system, params, fpt_flag=True)
        fp_times[i] = times[-1]
        print i, times[-1]
    fpt_histogram(fp_times, params, show_flag=True, figname_mod="XZ_model")
