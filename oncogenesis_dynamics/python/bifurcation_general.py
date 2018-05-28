"""
Comments
- current implementation for bifurcation along VALID_BIFURCATION_PARAMS only
- currently supports one bifurcation direction at a time
- no stability calculation implemented (see matlab code for that)

Conventions
- params is 7-vector of the form: params[0] -> alpha_plus
                                  params[1] -> alpha_minus
                                  params[2] -> mu
                                  params[3] -> a           (usually normalized to 1)
                                  params[4] -> b           (b = 1 - delta)
                                  params[5] -> c           (c = 1 + s)
                                  params[6] -> N           (float not int)
- if an element of params is specified as None then a bifurcation range will be found and used

TODO
- implement stability checks
- implement other bifurcation parameters
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from os import sep

from constants import BIFURC_DICT, VALID_BIFURC_PARAMS, OUTPUT_DIR, PARAMS_ID_INV
from data_io import write_bifurc_data, write_params
from formulae import bifurc_value, fp_location_general, is_stable, fp_location_fsolve, jacobian_numerical_2d
from params import Params
from plotting import plot_bifurc_dist, plot_fp_curves_general
from trajectory import trajectory_simulate


# MATPLOTLIB GLOBAL SETTINGS
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(mpl_params)

# SCRIPT PARAMETERS
SEARCH_START = 0.01 #0.5 #0.9  # start at SEARCH_START*bifurcation_point
SEARCH_END = 12.75 #1.1  # end at SEARCH_END*bifurcation_point
SEARCH_AMOUNT = 4000 #10000
SPACING_BIFTEXT = int(SEARCH_AMOUNT/10)
FLAG_BIFTEXT = 1
FLAG_SHOWPLT = 1
FLAG_SAVEPLT = 1
FLAG_SAVEDATA = 1
HEADER_TITLE = 'Fixed Points'
ODE_SYSTEM = "default"
assert ODE_SYSTEM in ["default"]  # TODO: feedback not working in this script bc of unknown number of fp
solver_fsolve = False             # TODO: fsolve solver doesn't find exactly 3 fp.. usually less
check_with_trajectory = False


def get_fp_data_1d(params, param_1_name, param_1_range):
    # return dict of {param vary value: list of [[fixed point i], [eigenvalues i], stability_i]}
    # [fixed point i], [eigenvalues i] are 3-list and 2-list and stability_i is a bool
    # assumes flow=0 and no feedback; uses is_stable with fp=[0,0,N]
    assert param_1_name in PARAMS_ID_INV.keys()
    fp_dict = {p1: [] for p1 in param_1_range}
    for i, p1 in enumerate(param_1_range):
        params_step_list = params.params_list()
        params_step_list[PARAMS_ID_INV[param_1_name]] = p1
        params_step = Params(params_step_list, params.system)
        fp_locs = fp_location_fsolve(params_step, gridsteps=15)
        fp_info_at_p1 = [0]*len(fp_locs)
        for i, fp in enumerate(fp_locs):
            J = jacobian_numerical_2d(params_step, fp[0:2])
            eiglist, V = np.linalg.eig(J)
            fp_info_at_p1[i] = [fp, eiglist, all(eig < 0 for eig in eiglist)]
        fp_dict[p1] = fp_info_at_p1
    return fp_dict


if __name__ == '__main__':

    system = "feedback_z"

    # DYNAMICS PARAMETERS
    alpha_plus = 0.2
    alpha_minus = 0.5
    mu = 0.04
    a = 1.0
    b = 0.8
    c = 0.86
    N = 100.0
    v_x = 0.0
    v_y = 0.0
    v_z = 0.0
    mu_base = 0.0
    params_list = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]
    params = Params(params_list, system)

    print "Specified parameters: \nalpha_plus = " + str(alpha_plus) + "\nalpha_minus = " + str(alpha_minus) + \
          "\nmu = " + str(mu) + "\na = " + str(a) + "\nb = " + str(b) + "\nc = " + str(c) + "\nN = " + str(N) + \
          "\nv_x = " + str(v_x) + "\nv_y = " + str(v_y) + "\nv_z = " + str(v_z) + "\nmu_base = " + str(mu_base)
    #print "Use fsolve solver:", solver_fsolve

    param_vary = 'c'
    param_start = 0.8
    param_stop = 0.82
    param_steps = 50
    param_varying_values = np.linspace(param_start, param_stop, param_steps)
    print param_varying_values

    fp_data_dict = get_fp_data_1d(params, param_vary, param_varying_values)
    print fp_data_dict

    plot_fp_curves_general(fp_data_dict, N, flag_show=True)

    """
    # PLOTTING ON THE SIMPLEX FIGURE
    fig_fp_curves = plot_fp_curves(x0_array, x0_stabilities, x1_array, x1_stabilities, x2_array, x2_stabilities, N,
                                   HEADER_TITLE, False, False)
    if FLAG_BIFTEXT:
        for idx in xrange(0, nn, SPACING_BIFTEXT):
            fig_fp_curves.gca().text(x1_array[idx, 0], x1_array[idx, 1], x1_array[idx, 2], '%.3f' % bifurcation_search[idx])
    if FLAG_SHOWPLT:
        plt.show()
    if FLAG_SAVEPLT:
        fig_fp_curves.savefig(OUTPUT_DIR + sep + 'bifurcation_curves.png')

    # PLOTTING THE BIFURCATION DIAGRAM
    fig_dist_norm = plot_bifurc_dist(x1_array, bifurcation_search, bifurc_id, N, "norm", FLAG_SHOWPLT, FLAG_SAVEPLT)
    fig_dist_z = plot_bifurc_dist(x1_array, bifurcation_search, bifurc_id, N, "z_only", FLAG_SHOWPLT, FLAG_SAVEPLT)

    # DATA OUTPUT
    if FLAG_SAVEDATA:
        write_bifurc_data(bifurcation_search, x0_array, x0_stabilities, x1_array, x1_stabilities, x2_array, x2_stabilities,
                          bifurc_id, OUTPUT_DIR, 'bifurc_data.csv')
        write_params(params, OUTPUT_DIR, 'params.csv')
    """
