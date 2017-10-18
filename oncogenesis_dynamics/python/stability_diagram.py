import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import PARAMS_ID, PARAMS_ID_INV, OUTPUT_DIR
from formulae import is_stable


def get_stability_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range):
    # if flow=0 and no feedback, can use is_stable with fp=[0,0,N]
    #TODO: if flow!=0, need to characterize the shifted "[0,0,N]" fp
    #TODO: how to check stability in feedback case
    #TODO: true/false on stability of fp is one visualization but maybe det J(fp) = order parameter?
    assert param_1_name, param_2_name in PARAMS_ID_INV.keys()
    assert params_general[-3:] == [0.0, 0.0, 0.0]  # currently hard-code non-flow trivial FP location of [0,0,N]
    fp_stationary = [0.0, 0.0, params_general[PARAMS_ID_INV["N"]]]
    stab_array = np.zeros((len(param_1_range), len(param_2_range)), dtype=bool)
    for i, p1 in enumerate(param_1_range):
        for j, p2 in enumerate(param_2_range):
            params_step = params_general
            params_step[PARAMS_ID_INV[param_1_name]] = p1
            params_step[PARAMS_ID_INV[param_2_name]] = p2
            stab_array[i,j] = is_stable(params_step, fp_stationary)
    return stab_array


def plot_stability_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range):
    stability_data_2d = get_stability_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range)
    plt.imshow(stability_data_2d, cmap='Greys', interpolation="none", origin='lower', aspect='auto',
               extent=[param_2_range[0], param_2_range[-1], param_1_range[0], param_1_range[-1]])
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-')
    ax.set_xlabel(param_2_name)
    ax.set_ylabel(param_1_name)
    plt.title("Stability of fp [0,0,N] (black=stable), %s vs %s" % (param_1_name, param_2_name))
    # CREATE TABLE OF PARAMS
    # bbox is x0, y0, height, width
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params[i]] if PARAMS_ID[i] not in [param_1_name, param_2_name] else ["None"] for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals, colWidths=[0.1]*3, rowLabels=row_labels, loc='best',
                            bbox=(1.2, 0.2, 0.1, 0.75))
    #plt.subplots_adjust(left=0.2, bottom=0.2)
    print plt.axis()
    print ax.get_window_extent()
    plt.savefig(OUTPUT_DIR + sep + 'stability_data_2d_%s_%s.png' % (param_1_name, param_2_name), bbox_inches='tight')
    plt.show()
    return plt.gca()


if __name__ == "__main__":
    alpha_plus = 0.2  # 0.05 #0.4
    alpha_minus = 0.5  # 4.95 #0.5
    mu = 0.0001  # 0.01
    a = 1.0
    b = 0.8
    c = 0.6  # 2.6 #1.2
    N = 100.0  # 100
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]


    param_1_name = "mu"
    param_1_start = 0.0
    param_1_stop = 1.0
    param_1_steps = 100
    param_1_range = np.linspace(param_1_start, param_1_stop, param_1_steps)
    param_2_name = "c"
    param_2_start = 0.8
    param_2_stop = 0.95
    param_2_steps = 200
    """
    param_1_name = "b"
    param_1_start = 0.0
    param_1_stop = 2.0
    param_1_steps = 100
    param_1_range = np.linspace(param_1_start, param_1_stop, param_1_steps)
    param_2_name = "c"
    param_2_start = 0.0
    param_2_stop = 2.0
    param_2_steps = 200
    """
    param_2_range = np.linspace(param_2_start, param_2_stop, param_2_steps)
    print get_stability_data_2d(params, param_1_name, param_1_range, param_2_name, param_2_range)
    plot_stability_data_2d(params, param_1_name, param_1_range, param_2_name, param_2_range)
