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
- if an element of params is specified as None then a bifurcation range will be be found and used

TODO
- implement stability checks
- implement other bifurcation parameters
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from os import sep

from constants import BIFURC_DICT, VALID_BIFURC_PARAMS, OUTPUT_DIR
from formulae import bifurc_value, q_get, fp_location_general, is_stable, write_bifurc_data, write_params
from plotting import plot_fp_curves, plot_bifurc_dist


# MATPLOTLIB GLOBAL SETTINGS
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(mpl_params)

# SCRIPT PARAMETERS
SEARCH_START = 0.1 #0.5 #0.9  # start at SEARCH_START*bifurcation_point
SEARCH_END = 5.6 #1.1  # end at SEARCH_END*bifurcation_point
SEARCH_AMOUNT = 10000
SPACING_BIFTEXT = int(SEARCH_AMOUNT/10)
FLAG_BIFTEXT = 1
FLAG_SHOWPLT = 0
FLAG_SAVEPLT = 1
FLAG_SAVEDATA = 1
HEADER_TITLE = 'Fixed Points'

# DYNAMICS PARAMETERS
alpha_plus = 0.05 #0.4
alpha_minus = 4.95 #0.5
mu = 0.77 #0.77 #0.01
a = 1.0
b = None #1.1
c = 2.6 #1.2
N = 100.0 #100
v_x = 0.001
v_y = 0.0
v_z = 0.0
if b is not None:
    delta = 1 - b
if c is not None:
    s = c - 1
if v_x == 0 and v_y == 0 and v_z == 0:
    solver_numeric = False
else:
    solver_numeric = True
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]
print "Specified parameters: \nalpha_plus = " + str(alpha_plus) + "\nalpha_minus = " + str(alpha_minus) + \
      "\nmu = " + str(mu) + "\na = " + str(a) + "\nb = " + str(b) + "\nc = " + str(c) + "\nN = " + str(N) + \
      "\nv_x = " + str(v_x) + "\nv_y = " + str(v_y) + "\nv_z = " + str(v_z)
print "Use numeric solver: ", solver_numeric

# FP SEARCH SETUP
bifurc_ids = []
for idx in xrange(len(params)):
    if params[idx] is None:
        # identify bifurcation points
        bifurc_idx = idx
        bifurc_id = BIFURC_DICT[idx]
        bifurc_ids.append(bifurc_id)
        assert bifurc_id in VALID_BIFURC_PARAMS
        bifurc_loc = bifurc_value(params, bifurc_id)
        print "Bifurcation in %s possibly at %.5f" % (bifurc_id, bifurc_loc)
        print "Searching in window: %.3f to %.3f with %d points" \
              % (SEARCH_START*bifurc_loc, SEARCH_END*bifurc_loc, SEARCH_AMOUNT)
        bifurcation_search = np.linspace(SEARCH_START*bifurc_loc, SEARCH_END*bifurc_loc, SEARCH_AMOUNT)
nn = len(bifurcation_search)
x0_array = np.zeros((nn, 3))
x1_array = np.zeros((nn, 3))
x2_array = np.zeros((nn, 3))
params_ensemble = np.zeros((nn, len(params)))
for idx in xrange(len(params)):
    if params[idx] is not None:
        params_ensemble[:, idx] = params[idx]
    else:
        params_ensemble[:, idx] = bifurcation_search
x0_stabilities = np.zeros((nn, 1))  # not implemented
x1_stabilities = np.zeros((nn, 1))  # not implemented
x2_stabilities = np.zeros((nn, 1))  # not implemented

# FIND FIXED POINTS
for idx, bifurc_param_val in enumerate(bifurcation_search):
    params_step = params_ensemble[idx, :]
    fp_x0, fp_x1, fp_x2 = fp_location_general(params_step, solver_numeric)
    x0_array[idx, :] = fp_x0
    x1_array[idx, :] = fp_x1
    x2_array[idx, :] = fp_x2
    x0_stabilities[idx, :] = is_stable(params_step, fp_x0)
    x1_stabilities[idx, :] = is_stable(params_step, fp_x1)
    x2_stabilities[idx, :] = is_stable(params_step, fp_x2)

# PLOTTING ON THE SIMPLEX FIGURE
fig_fp_curves = plot_fp_curves(x1_array, x2_array, N, HEADER_TITLE, False, False)
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
    write_bifurc_data(bifurcation_search, x0_array, x1_array, x2_array, bifurc_id, OUTPUT_DIR, 'bifurc_data.csv')
    write_params(params, OUTPUT_DIR, 'params.csv')
