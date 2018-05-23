import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from os import sep

from constants import PARAMS_ID, PARAMS_ID_INV, STATES_ID_INV, OUTPUT_DIR, DEFAULT_X_COLOUR, DEFAULT_Y_COLOUR, DEFAULT_Z_COLOUR
from data_io import write_params, write_matrix_data_and_idx_vals, read_matrix_data_and_idx_vals, read_params
from formulae import is_stable, fp_location_general, get_physical_and_stable_fp, get_stable_fp

# TODO: have ONLY 1 plotting script with datatype flags (e.g. fp count flag, stability data flag, other...)


def get_stability_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, flag_write=True):
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
    if flag_write:
        write_matrix_data_and_idx_vals(stab_array, param_1_range, param_2_range, "fpcount2d", param_1_name, param_2_name, output_dir=OUTPUT_DIR)
    return stab_array


def plot_stability_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, flag_show=False):
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
    if flag_show:
        plt.show()
    return plt.gca()


def get_gap_dist(params, system, axis="z", flag_simple=True):
    N = params[PARAMS_ID_INV['N']]
    fp_list = get_physical_and_stable_fp(params, system)
    if len(fp_list) > 2:
        print "WARNING: %d phys/stable fixed points at these params:" % len(fp_list)
        print params, system
        print "FPs:", fp_list
        write_params(params, system, OUTPUT_DIR, "broken_params.csv")
    elif len(fp_list) == 1:
        #return fp_list[0][STATES_ID_INV[axis]]
        #return N - fp_list[0][STATES_ID_INV[axis]]
        #return N
        if flag_simple:
            val = fp_list[0][STATES_ID_INV[axis]]
        else:
            val = (N - fp_list[0][STATES_ID_INV[axis]]) / (N)
    else:
        if flag_simple:
            val = -1.0
            #val = np.abs(fp_list[0][STATES_ID_INV[axis]] - fp_list[1][STATES_ID_INV[axis]])
        else:
            val = (N - (fp_list[0][STATES_ID_INV[axis]] + fp_list[1][STATES_ID_INV[axis]])) / (N)
            #val = np.abs(fp_list[0][STATES_ID_INV[axis]] - fp_list[1][STATES_ID_INV[axis]])  # gap in z-coordinate
    return val



def get_gap_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, axis_gap="z", figname_mod="", flag_write=True):
    # gap between low-z and high-z FPs
    assert param_1_name, param_2_name in PARAMS_ID_INV.keys()
    assert [params_general[PARAMS_ID_INV[x]] for x in ['v_x', 'v_y', 'v_z']] == [0.0, 0.0, 0.0]  # currently hard-code non-flow trivial FP location of [0,0,N]
    gap_array = np.zeros((len(param_1_range), len(param_2_range)))
    for i, p1 in enumerate(param_1_range):
        for j, p2 in enumerate(param_2_range):
            params_step = params_general
            params_step[PARAMS_ID_INV[param_1_name]] = p1
            params_step[PARAMS_ID_INV[param_2_name]] = p2
            gap_array[i, j] = get_gap_dist(params_step, system, axis=axis_gap)
        print i, j, p1, p2
    if flag_write:
        write_matrix_data_and_idx_vals(gap_array, param_1_range, param_2_range, "gap2d", param_1_name, param_2_name, output_dir=OUTPUT_DIR)
    if figname_mod is not None:
        plot_gap_data_2d(gap_array, params_general, param_1_name, param_1_range, param_2_name,
                         param_2_range, system, axis_gap=axis_gap, figname_mod=figname_mod)
    return gap_array


def plot_gap_data_2d(gap_data_2d, params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, axis_gap="z", figname_mod="", flag_show=True):
    # custom cmap for gap diagram
    grey = (169/255.0, 169/255.0, 169/255.0)
    val_to_colour_bistable = [(0.0, grey),
                     (1e-9, DEFAULT_X_COLOUR),
                     (0.5, DEFAULT_Y_COLOUR),
                     (1.0, DEFAULT_Z_COLOUR)]
    xyz_cmap_gradient_bistable = LinearSegmentedColormap.from_list('xyz_cmap_gradient_bistable', val_to_colour_bistable, N=100)
    val_to_colour_3 = [(0.0, DEFAULT_X_COLOUR),
                     (0.5, DEFAULT_Y_COLOUR),
                     (1.0, DEFAULT_Z_COLOUR)]
    xyz_cmap_gradient_3 = LinearSegmentedColormap.from_list('xyz_cmap_gradient_3', val_to_colour_3, N=200)
    xyz_cmap_gradient = val_to_colour_bistable

    # plot image
    plt.imshow(gap_data_2d, cmap=xyz_cmap_gradient_bistable, interpolation="none", origin='lower', aspect='auto',
               extent=[param_2_range[0], param_2_range[-1], param_1_range[0], param_1_range[-1]])
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-')
    ax.set_xlabel(param_2_name)
    ax.set_ylabel(param_1_name)
    plt.title("Gap in %s between FPs, vary %s, %s" % (axis_gap, param_1_name, param_2_name))
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
    plt.savefig(OUTPUT_DIR + sep + 'gap_data_2d_%s_%s_%s.png' % (param_1_name, param_2_name, figname_mod), bbox_inches='tight')
    if flag_show:
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


def get_jump_data_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, axis_jump, figname_mod=None):
    assert param_1_name, param_2_name in PARAMS_ID_INV.keys()
    assert [params_general[PARAMS_ID_INV[x]] for x in ['v_x', 'v_y', 'v_z']] == [0.0, 0.0, 0.0]  # currently hard-code non-flow trivial FP location of [0,0,N]
    jump_array = np.zeros((len(param_1_range), len(param_2_range)))
    for i, p1 in enumerate(param_1_range):
        for j, p2 in enumerate(param_2_range):
            params_step = params_general
            params_step[PARAMS_ID_INV[param_1_name]] = p1
            params_step[PARAMS_ID_INV[param_2_name]] = p2
            jump_array[i, j] = get_jump_dist(params_step, param_1_name, param_2_name, system, axis=axis_jump)
        print i, j, p1, p2
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


def get_stable_fp_count_2d(params_general, param_1_name, param_1_range, param_2_name, param_2_range, system, flag_phys=True, flag_write=True, figname_mod=None):
    if flag_phys:
        fpcollector = get_physical_and_stable_fp
        filestr = "physfpcount2d"
    else:
        fpcollector = get_stable_fp
        filestr = "fpcount2d"

    assert param_1_name, param_2_name in PARAMS_ID_INV.keys()
    fp_count_array = np.zeros((len(param_1_range), len(param_2_range)))
    for i, p1 in enumerate(param_1_range):
        for j, p2 in enumerate(param_2_range):
            params_step = params_general
            params_step[PARAMS_ID_INV[param_1_name]] = p1
            params_step[PARAMS_ID_INV[param_2_name]] = p2
            fp_list = fpcollector(params_step, system)
            fp_count_array[i, j] = len(fp_list)
        print i, j, p1, p2
    if flag_write:
        write_matrix_data_and_idx_vals(fp_count_array, param_1_range, param_2_range, filestr, param_1_name, param_2_name, output_dir=OUTPUT_DIR)
    if figname_mod is not None:
        plot_stable_fp_count_2d(fp_count_array, params_general, param_1_name, param_1_range, param_2_name,
                                param_2_range, system, figname_mod=figname_mod, flag_phys=flag_phys)
    return fp_count_array


def plot_stable_fp_count_2d(fp_count_array, params_general, param_1_name, param_1_range, param_2_name,
                            param_2_range, system, figname_mod="", flag_phys=True, flag_show=False):
    if flag_phys:
        plt_title = "Physical and Stable FP count (vary %s, %s) %dx%d" % (param_1_name, param_2_name, len(fp_count_array), len(fp_count_array[0]))
        filestr = 'physfp_count_2d_%s_%s_%s.png' % (param_1_name, param_2_name, figname_mod)
    else:
        plt_title = "Stable FP count (vary %s, %s) %dx%d" % (param_1_name, param_2_name, len(fp_count_array), len(fp_count_array[0]))
        filestr = 'fp_count_2d_%s_%s_%s.png' % (param_1_name, param_2_name, figname_mod)

    plt.imshow(fp_count_array, cmap='seismic', interpolation="none", origin='lower', aspect='auto',
               extent=[param_2_range[0], param_2_range[-1], param_1_range[0], param_1_range[-1]])
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-')
    ax.set_xlabel(param_2_name)
    ax.set_ylabel(param_1_name)
    plt.title(plt_title)
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
    plt.savefig(OUTPUT_DIR + sep + filestr, bbox_inches='tight')
    if flag_show:
        plt.show()
    plt.close('all')
    return plt.gca()


if __name__ == "__main__":

    flag_generate = False
    flag_load = True

    alpha_plus = 0.2  # 0.05 #0.4
    alpha_minus = 0.5  # 4.95 #0.5
    mu = 0.01  # 0.01
    a = 1.0
    b = 0.8
    c = 0.6  # 2.6 #1.2
    N = 100.0  # 100
    v_x = 0.1
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0
    params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]
    ode_system = "feedback_z"

    param_1_name = "mu"
    param_1_start = 0.0
    param_1_stop = 0.01
    param_1_steps = 40
    param_1_range = np.linspace(param_1_start, param_1_stop, param_1_steps)
    param_2_name = "c"
    param_2_start = 0.8  # 1.1 #0.7
    param_2_stop = 0.9  # 1.3 #0.95
    param_2_steps = 100
    param_2_range = np.linspace(param_2_start, param_2_stop, param_2_steps)

    # generate and plot data
    if flag_generate:
        fp_data = get_stable_fp_count_2d(params, param_1_name, param_1_range, param_2_name, param_2_range, ode_system)
        plot_stable_fp_count_2d(fp_data, params, param_1_name, param_1_range, param_2_name, param_2_range, ode_system, figname_mod="default")

    # loaf data
    if flag_load:
        row_name = 'c'  # aka param 2 is row
        col_name = 'b'  # aka param 1 is col
        datapath = OUTPUT_DIR + sep + "gapdist2d_full.txt"
        rowpath = OUTPUT_DIR + sep + "gapdist2d_full_%s.txt" % row_name
        colpath = OUTPUT_DIR + sep + "gapdist2d_full_%s.txt" % col_name
        paramsname = "gapdist2d_full_params.csv"

        gap_data_2d, param_2_range, param_1_range = read_matrix_data_and_idx_vals(datapath, rowpath, colpath)
        param_1_name = col_name
        param_2_name = row_name

        params_general = read_params(OUTPUT_DIR, paramsname)
        system = params_general[-1]
        print params_general

        plot_gap_data_2d(gap_data_2d, params_general, param_1_name, param_1_range, param_2_name, param_2_range, system,
                         axis_gap="z", figname_mod="", flag_show=True)
