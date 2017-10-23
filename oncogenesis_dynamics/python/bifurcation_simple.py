""" TODO: update this
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

from constants import BIFURC_DICT, VALID_BIFURC_PARAMS, OUTPUT_DIR
from data_io import write_bifurc_data, write_params
from formulae import bifurc_value, fp_location_general, is_stable
from plotting import plot_fp_curves_simple, plot_bifurc_dist
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

# DYNAMICS PARAMETERS
alpha_plus = 0.4#0.05 #0.4
alpha_minus = 0.5#4.95 #0.5
mu = None #0.77 #0.01
a = 1.0
b = 0.8
c = 1.2 #2.6 #1.2
N = 100.0 #100
v_x = 0.0
v_y = 0.0
v_z = 0.0
mu_base = 0.0
if b is not None:
    delta = 1 - b
if c is not None:
    s = c - 1
if v_x == 0 and v_y == 0 and v_z == 0:
    solver_explicit = True
else:
    solver_explicit = False
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

print "Specified parameters: \nalpha_plus = " + str(alpha_plus) + "\nalpha_minus = " + str(alpha_minus) + \
      "\nmu = " + str(mu) + "\na = " + str(a) + "\nb = " + str(b) + "\nc = " + str(c) + "\nN = " + str(N) + \
      "\nv_x = " + str(v_x) + "\nv_y = " + str(v_y) + "\nv_z = " + str(v_z) + "\nmu_base = " + str(mu_base)
print "Use fsolve solver:", solver_fsolve
if solver_explicit:
    print "Use explicit solver:", solver_explicit

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
x0_stabilities = np.zeros((nn, 1), dtype=bool)  # not fully implemented
x1_stabilities = np.zeros((nn, 1), dtype=bool)  # not fully implemented
x2_stabilities = np.zeros((nn, 1), dtype=bool)  # not fully implemented

# FIND FIXED POINTS
for idx, bifurc_param_val in enumerate(bifurcation_search):
    params_step = params_ensemble[idx, :]
    fp_x0, fp_x1, fp_x2 = fp_location_general(params_step, ODE_SYSTEM, solver_fsolve=solver_fsolve,
                                              solver_explicit=solver_explicit)
    x0_array[idx, :] = fp_x0
    x1_array[idx, :] = fp_x1
    x2_array[idx, :] = fp_x2
    x0_stabilities[idx][0] = is_stable(params_step, fp_x0[0:2], ODE_SYSTEM, method="numeric_2d")
    x1_stabilities[idx][0] = is_stable(params_step, fp_x1[0:2], ODE_SYSTEM, method="numeric_2d")
    x2_stabilities[idx][0] = is_stable(params_step, fp_x2[0:2], ODE_SYSTEM, method="numeric_2d")
    print "params:", idx, "of", nn
    if check_with_trajectory:
        r, times, ax_traj, ax_mono = trajectory_simulate(params_step, ODE_SYSTEM, init_cond=[99.9,0.1,0.0], t0=0,
                                                         t1=20000.0, num_steps=2000, flag_showplt=False, flag_saveplt=False)
        print bifurc_param_val, fp_x1, r[-1]

# PLOTTING ON THE SIMPLEX FIGURE
fig_fp_curves = plot_fp_curves_simple(x0_array, x0_stabilities, x1_array, x1_stabilities, x2_array, x2_stabilities, N,
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
    write_params(params, ODE_SYSTEM, OUTPUT_DIR, 'params.csv')
