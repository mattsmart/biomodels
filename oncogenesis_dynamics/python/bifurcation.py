import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
from os import sep

from simplex import plot_simplex


"""
Comments
- current implementation for bifurcation along VALID_BIFURCATION_PARAMS only
- no stability calculation implemented (see matlab)
"""

# MATPLOTLIB GLOBAL SETTINGS
params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# SCRIPT PARAMETERS
BIFURC_ID = "delta"
VALID_BIFURCATION_PARAMS = ["delta"]
assert BIFURC_ID in VALID_BIFURCATION_PARAMS
SEARCH_START = 1.3
SEARCH_END = 1.4
SEARCH_AMOUNT = 10000
SPACING_BIFTEXT = int(SEARCH_AMOUNT/10)
FLAG_BIFTEXT = 1
FLAG_SHOWPLT = 1
FLAG_SAVEPLT = 1
FLAG_SAVEDATA = 0
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
#b = 1.376666 #1.3
#delta = 1-b
c = 1.2
s = c - 1
N = 100

# DYNAMICS SETUP
bifurcation_search = np.linspace(SEARCH_START, SEARCH_END, SEARCH_AMOUNT)
nn = len(bifurcation_search)
x1_array = np.zeros((nn, 3))
x2_array = np.zeros((nn, 3))
# x1_stabilities = np.zeros((nn,1))  # not implemented
# x2_stabilities = np.zeros((nn,1))  # not implemented

# FUNCTIONS
def bifurc_get(bifurc_name):
    #assumes threshold_2 is stronger constraint, atm hardcode rearrange expression for bifruc param
    if bifurc_name == "delta":
        bifurc_val = alpha_minus*alpha_plus/(s + alpha_plus) - (s + alpha_minus + mu)
        return bifurc_val
    else:
        raise ValueError(bifruc_name + ' not valid bifurc_name')

def threshold_1(delta):
    return 2*s + delta + alpha_plus + alpha_minus + mu

def threshold_2(delta):
    return (s + alpha_plus)*(s + delta + alpha_minus + mu) - alpha_minus*alpha_plus

def q_get(sign, delta):
    assert sign in [-1, +1]
    bterm = alpha_plus - alpha_minus - mu - delta
    return 0.5/alpha_minus * (bterm + sign*np.sqrt(bterm**2 + 4*alpha_minus*alpha_plus))
                              
def xvec_get(q, delta):
    xi = N*(s + alpha_plus - alpha_minus*q) / (s + (delta + s)*q)
    yi = q*xi
    zi = N - xi - yi
    return xi, yi, zi 

# FIGURE SETUP
fig = plot_simplex(N)
ax = fig.gca()
ax.set_title(HEADER_TITLE)

# FIND FIXED POINTS
for idx, bif_param in enumerate(bifurcation_search):
    b = bif_param
    delta = 1-b
    q1 = q_get(+1, delta)
    q2 = q_get(-1, delta)
    x1_array[idx, :] = xvec_get(q1, delta)
    x2_array[idx,:] = xvec_get(q2, delta)
    if FLAG_BIFTEXT and idx % SPACING_BIFTEXT == 0:
        #print bif_param, x1_array[idx,0], x1_array[idx,1], x1_array[idx,2]
        ax.text(x1_array[idx,0], x1_array[idx,1], x1_array[idx,2], '%.3f' % bif_param)


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
    fig.savefig(OUTPUT_DIR + sep + HEADER_SAVE + '.pdf')

# =====================
# DATA OUTPUT
# =====================
# note: should use csv or something instead
if FLAG_SAVEDATA:
    np.savetxt(OUTPUT_DIR + sep + 'pyx1fp.txt', x1_array)
    np.savetxt(OUTPUT_DIR + sep + 'pyx2fp.txt', x2_array)
    np.savetxt(OUTPUT_DIR + sep + 'pybif.txt', bifurcation_search)

"""
#threshold1 = 2*s + delta + alpha_plus + alpha_minus + mu
#threshold2 = (s + alpha_plus)*(s + delata + alpha_minus + mu) - alpha_minus*alpha_plus
print "delta thresholds"
print -(2*s + alpha_plus + alpha_minus + mu);
print alpha_minus*alpha_plus / (s + alpha_plus) - (s + alpha_minus + mu);
"""
