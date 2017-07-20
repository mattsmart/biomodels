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
- csv for data output
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
from os import sep

from constants import PARAMS_DICT, VALID_BIFURCATION_PARAMS
from formulae import bifurc_value, q_get, fp_location
from simplex import plot_simplex


# MATPLOTLIB GLOBAL SETTINGS
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(mpl_params)

# SCRIPT PARAMETERS
SEARCH_START = 0.9  # start at SEARCH_START*bifurcation_point
SEARCH_END = 1.1  # end at SEARCH_END*bifurcation_point
SEARCH_AMOUNT = 10000
SPACING_BIFTEXT = int(SEARCH_AMOUNT/10)
FLAG_BIFTEXT = 1
FLAG_SHOWPLT = 1
FLAG_SAVEPLT = 1
FLAG_SAVEDATA = 1
OUTPUT_DIR = "output"
HEADER_TITLE = 'Fixed Points'
HEADER_SAVE = 'model_b_fps'
X1_COL = "blue"  # blue stable (dashed unstable)
X2_COL = "green"  # green stable (dashed unstable)

# DYNAMICS PARAMETERS
alpha_plus = 0.4
alpha_minus = 0.5
mu = 0.01
a = 1.0
b = None
c = 1.2
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
        print "Bifurcation in %s possibly at %.3f" % (bifurc_id, bifurc_loc)
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

# FIGURE SETUP
fig = plot_simplex(N)
ax = fig.gca()
ax.set_title(HEADER_TITLE)

# FIND FIXED POINTS
for idx, bifurc_param_val in enumerate(bifurcation_search):
    params_step = params_ensemble[idx, :]
    q1 = q_get(params_step, +1)
    q2 = q_get(params_step, -1)
    x1_array[idx, :] = fp_location(params_step, q1)
    x2_array[idx, :] = fp_location(params_step, q2)
    if FLAG_BIFTEXT and idx % SPACING_BIFTEXT == 0:
        #print bifurc_param_val, x1_array[idx,0], x1_array[idx,1], x1_array[idx,2]
        ax.text(x1_array[idx,0], x1_array[idx,1], x1_array[idx,2], '%.3f' % bifurc_param_val)

# PLOTTING
# plot fixed point curves
ax.scatter(x1_array[:,0], x1_array[:,1], x1_array[:,2], label='q_plus', color=X1_COL)
ax.scatter(x2_array[:,0], x2_array[:,1], x2_array[:,2], label='q_minus', color=X2_COL)
# plot settings
ax.view_init(5, 35)  #ax.view_init(-45, -15)
axis_scale = 1
ax.set_xlim(-N*axis_scale, N*axis_scale)  # may need to flip order
ax.set_ylim(-N*axis_scale, N*axis_scale)
ax.set_zlim(-N*axis_scale, N*axis_scale)
ax.legend()
# plot io
if FLAG_SHOWPLT:
    plt.show()
if FLAG_SAVEPLT:
    fig.savefig(OUTPUT_DIR + sep + HEADER_SAVE + '.png')

# DATA OUTPUT
if FLAG_SAVEDATA:
    np.savetxt(OUTPUT_DIR + sep + 'pyx1fp.txt', x1_array)
    np.savetxt(OUTPUT_DIR + sep + 'pyx2fp.txt', x2_array)
    np.savetxt(OUTPUT_DIR + sep + 'pybif.txt', bifurcation_search)
