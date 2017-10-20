import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import PARAMS_ID, PARAMS_ID_INV, STATES_ID_INV, OUTPUT_DIR
from formulae import is_stable, fp_location_general, get_physical_and_stable_fp


def get_stability_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system):
    # assumes flow=0 and no feedback; uses is_stable with fp=[0,0,N]
    #TODO: maybe solve a1 and a0, or just compute and show signs, instead
    #TODO: also show a1 and a0 solutions never intercept (guess/check)
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
            #stab_array[i,j] = is_stable(params_step, fp_stationary, system, method="algebraic_3d")
            stab_array[i, j] = is_stable(params_step, fp_stationary[0:2], system, method="numeric_2d")
    return stab_array


def plot_stability_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system):
    stability_data_2d = get_stability_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system)
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
    table_vals = [[params_general[i]] if PARAMS_ID[i] not in [param_1_name, param_2_name] else ["None"]
                  for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals, colWidths=[0.1]*3, rowLabels=row_labels, loc='best',
                            bbox=(1.2, 0.2, 0.1, 0.75))
    #plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(OUTPUT_DIR + sep + 'stability_data_2d_%s_%s.png' % (param_1_name, param_2_name), bbox_inches='tight')
    plt.show()
    return plt.gca()


def get_jump_dist(params_orig, param_1_name, param_2_name, ode_system, param_1_delta=0.01, param_2_delta=0.01, axis="z"):
    params_shift = params_orig
    params_shift[PARAMS_ID_INV[param_1_name]] += param_1_delta
    params_shift[PARAMS_ID_INV[param_2_name]] += param_2_delta
    fp_orig_list = get_physical_and_stable_fp(params_orig, ode_system)
    fp_shift_list = get_physical_and_stable_fp(params_shift, ode_system)
    assert len(fp_orig_list) == 1
    assert len(fp_shift_list) == 1
    axis_idx = STATES_ID_INV[axis]
    return fp_shift_list[0][axis_idx] - fp_orig_list[0][axis_idx]
    #return 10*np.random.rand()


def get_jump_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, axis_jump):
    assert param_1_name, param_2_name in PARAMS_ID_INV.keys()
    assert params_general[-3:] == [0.0, 0.0, 0.0]  # currently hard-code non-flow trivial FP location of [0,0,N]
    jump_array = np.zeros((len(param_1_range), len(param_2_range)))
    for i, p1 in enumerate(param_1_range):
        for j, p2 in enumerate(param_2_range):
            print i, j, p1, p2
            params_step = params_general
            params_step[PARAMS_ID_INV[param_1_name]] = p1
            params_step[PARAMS_ID_INV[param_2_name]] = p2
            jump_array[i, j] = get_jump_dist(params_step, param_1_name, param_2_name, system, axis=axis_jump)
    return jump_array


def plot_jump_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, axis_jump):
    jump_data_2d = get_jump_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, axis_jump)
    plt.imshow(jump_data_2d, cmap='seismic', interpolation="none", origin='lower', aspect='auto',
               extent=[param_2_range[0], param_2_range[-1], param_1_range[0], param_1_range[-1]])
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-')
    ax.set_xlabel(param_2_name)
    ax.set_ylabel(param_1_name)
    plt.title("Jump in %s when perturbing %s, %s forward" % (axis_jump, param_1_name, param_2_name))
    # CREATE TABLE OF PARAMS
    # bbox is x0, y0, height, width
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params_general[i]] if PARAMS_ID[i] not in [param_1_name, param_2_name] else ["None"]
                  for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals, colWidths=[0.1]*3, rowLabels=row_labels, loc='best',
                            bbox=(1.2, 0.2, 0.1, 0.75))
    #plt.subplots_adjust(left=0.2, bottom=0.2)
    # Now adding the colorbar
    plt.colorbar(orientation='horizontal')
    plt.savefig(OUTPUT_DIR + sep + 'jump_data_2d_%s_%s.png' % (param_1_name, param_2_name), bbox_inches='tight')
    plt.show()
    return plt.gca()


def get_stable_fp_count_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system):
    assert param_1_name, param_2_name in PARAMS_ID_INV.keys()
    assert params_general[-3:] == [0.0, 0.0, 0.0]  # currently hard-code non-flow trivial FP location of [0,0,N]
    fp_count_array = np.zeros((len(param_1_range), len(param_2_range)))
    for i, p1 in enumerate(param_1_range):
        for j, p2 in enumerate(param_2_range):
            params_step = params_general
            params_step[PARAMS_ID_INV[param_1_name]] = p1
            params_step[PARAMS_ID_INV[param_2_name]] = p2
            fp_list = get_physical_and_stable_fp(params_step, system)
            fp_count_array[i, j] = len(fp_list)
        print i, j, p1, p2
    return fp_count_array


def plot_stable_fp_count_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system,
                            figname_mod=""):
    stable_fp_count_2d = get_stable_fp_count_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system)
    plt.imshow(stable_fp_count_2d, cmap='seismic', interpolation="none", origin='lower', aspect='auto',
               extent=[param_2_range[0], param_2_range[-1], param_1_range[0], param_1_range[-1]])
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-')
    ax.set_xlabel(param_2_name)
    ax.set_ylabel(param_1_name)
    plt.title("Physical and Stable FP count (vary %s, %s)" % (param_1_name, param_2_name))
    # CREATE TABLE OF PARAMS
    # bbox is x0, y0, height, width
    row_labels = [PARAMS_ID[i] for i in xrange(len(PARAMS_ID))]
    table_vals = [[params_general[i]] if PARAMS_ID[i] not in [param_1_name, param_2_name] else ["None"]
                  for i in xrange(len(PARAMS_ID))]
    param_table = plt.table(cellText=table_vals, colWidths=[0.1]*3, rowLabels=row_labels, loc='best',
                            bbox=(1.2, 0.2, 0.1, 0.75))
    #plt.subplots_adjust(left=0.2, bottom=0.2)
    # Now adding the colorbar
    plt.colorbar(orientation='horizontal')
    plt.savefig(OUTPUT_DIR + sep + 'fp_count_2d_%s_%s_%s.png' % (param_1_name, param_2_name, figname_mod), bbox_inches='tight')
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
    ode_system = "feedback_z"

    param_1_name = "b"
    param_1_start = 0.6
    param_1_stop = 1.1
    param_1_steps = 200
    param_1_range = np.linspace(param_1_start, param_1_stop, param_1_steps)
    param_2_name = "c"
    param_2_start = 0.7 #1.1 #0.7
    param_2_stop = 1.0 #1.3 #0.95
    param_2_steps = 200
    param_2_range = np.linspace(param_2_start, param_2_stop, param_2_steps)
    plot_stable_fp_count_2d(params, param_1_name, param_1_range, param_2_name, param_2_range, ode_system, figname_mod="mu01_wide")
