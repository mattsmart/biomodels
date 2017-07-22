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
                                  params[6] -> N
- if an element of params is specified as None then a bifurcation range will be be found and used

TODO
- implement stability checks
- implement other bifurcation parameters
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
from os import sep

from constants import PARAMS_DICT, VALID_BIFURCATION_PARAMS
from formulae import bifurc_value, q_get, fp_location
from plotting import plot_fp_curves, plot_bifurc_dist


# MATPLOTLIB GLOBAL SETTINGS
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(mpl_params)

# SCRIPT PARAMETERS
SEARCH_START = 0.5 #0.9  # start at SEARCH_START*bifurcation_point
SEARCH_END = 1.6 #1.1  # end at SEARCH_END*bifurcation_point
SEARCH_AMOUNT = 10000
SPACING_BIFTEXT = int(SEARCH_AMOUNT/10)
FLAG_BIFTEXT = 1
FLAG_SHOWPLT = 1
FLAG_SAVEPLT = 1
FLAG_SAVEDATA = 1
OUTPUT_DIR = "output"
HEADER_TITLE = 'Fixed Points'
HEADER_SAVE = 'model_b_fps'

# DYNAMICS PARAMETERS
alpha_plus = 0.05 #0.4
alpha_minus = 4.95 #0.5
mu = 0.77 #0.01
a = 1.0
b = None
c = 2.6 #1.2
N = 100
if b is not None:
    delta = 1 - b
if c is not None:
    s = c - 1
params = [alpha_plus, alpha_minus, mu, a, b, c, N]
print "Specified parameters: \nalpha_plus = " + str(alpha_plus) + "\nalpha_minus = " + str(alpha_minus) + \
      "\nmu = " + str(mu) + "\na = " + str(a) + "\nb = " + str(b) + "\nc = " + str(c) + "\nN = " + str(N)

# FP SEARCH SETUP
bifurc_ids = []
for idx in xrange(len(params)):
    if params[idx] is None:
        # identify bifurcation points
        bifurc_idx = idx
        bifurc_id = PARAMS_DICT[idx]
        bifurc_ids.append(bifurc_id)
        assert bifurc_id in VALID_BIFURCATION_PARAMS
        bifurc_loc = bifurc_value(params, bifurc_id)
        print "Bifurcation in %s possibly at %.5f" % (bifurc_id, bifurc_loc)
        print "Searching in window: %.3f to %.3f with %d points" \
              % (SEARCH_START*bifurc_loc, SEARCH_END*bifurc_loc, SEARCH_AMOUNT)
        bifurcation_search = np.linspace(SEARCH_START*bifurc_loc, SEARCH_END*bifurc_loc, SEARCH_AMOUNT)
nn = len(bifurcation_search)
x1_array = np.zeros((nn, 3))
x2_array = np.zeros((nn, 3))
params_ensemble = np.zeros((nn, len(params)))
for idx in xrange(len(params)):
    if params[idx] is not None:
        params_ensemble[:,idx] = params[idx]
    else:
        params_ensemble[:,idx] = bifurcation_search
# x1_stabilities = np.zeros((nn,1))  # not implemented
# x2_stabilities = np.zeros((nn,1))  # not implemented

# FIND FIXED POINTS
for idx, bifurc_param_val in enumerate(bifurcation_search):
    params_step = params_ensemble[idx, :]
    q1 = q_get(params_step, +1)
    q2 = q_get(params_step, -1)
    x1_array[idx, :] = fp_location(params_step, q1)
    x2_array[idx, :] = fp_location(params_step, q2)

# PLOTTING ON THE SIMPLEX FIGURE
fig_fp_curves = plot_fp_curves(x1_array, x2_array, N, HEADER_TITLE)
if FLAG_BIFTEXT:
    for idx in xrange(0,nn, SPACING_BIFTEXT):
        fig_fp_curves.gca().text(x1_array[idx, 0], x1_array[idx, 1], x1_array[idx, 2], '%.3f' % bifurcation_search[idx])
if FLAG_SHOWPLT:
    plt.show()
if FLAG_SAVEPLT:
    fig_fp_curves.savefig(OUTPUT_DIR + sep + HEADER_SAVE + '.png')

"""
# plot fixed point curves
ax_simplex.scatter(x1_array[:, 0], x1_array[:, 1], x1_array[:, 2], label='q_plus', color=X1_COL)
ax_simplex.scatter(x2_array[:, 0], x2_array[:, 1], x2_array[:, 2], label='q_minus', color=X2_COL)
# plot settings
ax_simplex.view_init(5, 35)  #ax.view_init(-45, -15)
axis_scale = 1
ax_simplex.set_xlim(-N * axis_scale, N * axis_scale)  # may need to flip order
ax_simplex.set_ylim(-N * axis_scale, N * axis_scale)
ax_simplex.set_zlim(-N * axis_scale, N * axis_scale)
ax_simplex.legend()
# plot io
if FLAG_SHOWPLT:
    plt.show()
if FLAG_SAVEPLT:
    fig_simplex.savefig(OUTPUT_DIR + sep + HEADER_SAVE + '.png')
"""

# PLOTTING THE BIFURCATION DIAGRAM
fig_dist_norm = plot_bifurc_dist(x1_array, bifurcation_search, bifurc_id, N, "norm")
if FLAG_SHOWPLT:
    plt.show()
if FLAG_SAVEPLT:
    fig_dist_norm.savefig(OUTPUT_DIR + sep + "bifurcation_dist_norm" + '.png')

fig_dist_z = plot_bifurc_dist(x1_array, bifurcation_search, bifurc_id, N, "z_only")
if FLAG_SHOWPLT:
    plt.show()
if FLAG_SAVEPLT:
    fig_dist_z.savefig(OUTPUT_DIR + sep + "bifurcation_dist_z" + '.png')

# DATA OUTPUT
if FLAG_SAVEDATA:
    with open(OUTPUT_DIR + sep + 'pycsv.csv', "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        csv_header = [bifurc_id, 'x1_x', 'x1_y', 'x1_z', 'x2_x', 'x2_y', 'x2_z']
        writer.writerow(csv_header)
        for idx in xrange(nn):
            line = [bifurcation_search[idx]] + list(x1_array[idx,:]) + list(x2_array[idx,:])
            writer.writerow(line)
