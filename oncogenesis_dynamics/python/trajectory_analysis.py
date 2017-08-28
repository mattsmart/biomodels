import matplotlib.pyplot as plt
import numpy as np

from constants import PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from formulae import bifurc_value
from plotting import plot_trajectory_mono
from trajectory import trajectory_simulate


# SCRIPT PARAMS
FLAG_SHOWPLT = False
FLAG_SAVEPLT = False
ODE_METHOD = "libcall"  # see constants.py -- ODE_METHODS
ODE_SYSTEM = "feedback"  #"default"
INIT_COND = [99.0, 1.0, 0.0]
TIME_START = 0.0
TIME_END = 20.0
NUM_STEPS = 2000  # number of timesteps in each trajectory
param_varying_name = "b"
SEARCH_START = 0.55
SEARCH_END = 1.45
SEARCH_AMOUNT = 20

# DYNAMICS PARAMETERS
alpha_plus = 0.05
alpha_minus = 4.95
mu = 0.77
a = 1.0
b = 8.369856428  #1.376666
c = 2.6
N = 100.0
v_x = 1.0
v_y = 0.0
v_z = 0.0
delta = 1 - b
s = c - 1
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]

# VARYING PARAM SPECIFICATION
idx_varying = PARAMS_ID_INV[param_varying_name]
param_varying_bifurcname = BIFURC_DICT[idx_varying]
assert param_varying_bifurcname in VALID_BIFURC_PARAMS
bifurc_loc = bifurc_value(params, param_varying_bifurcname)
print "Bifurcation in %s possibly at %.5f" % (param_varying_bifurcname, bifurc_loc)
print "Searching in window: %.3f to %.3f with %d points" \
      % (SEARCH_START * bifurc_loc, SEARCH_END * bifurc_loc, SEARCH_AMOUNT)
param_varying_values = np.linspace(SEARCH_START * bifurc_loc, SEARCH_END * bifurc_loc, SEARCH_AMOUNT)

# CONSTRUCT PARAM ENSEMBLE
num_param_sets = len(param_varying_values)
param_ensemble = [[elem for elem in params] for _ in xrange(num_param_sets)]
for idx in xrange(num_param_sets):
    param_ensemble[idx][idx_varying] = param_varying_values[idx]
    print param_ensemble[idx]

# GET TRAJECTORIES
ax_comp = None
for params in param_ensemble:
    r, times, ax_traj, ax_mono = trajectory_simulate(init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                                                     params=params, ode_method=ODE_METHOD, ode_system=ODE_SYSTEM,
                                                     flag_showplt=FLAG_SHOWPLT, flag_saveplt=FLAG_SAVEPLT)
    ax_mono = plot_trajectory_mono(r, times, FLAG_SHOWPLT, FLAG_SAVEPLT, ax_mono=ax_comp)
    ax_comp = ax_mono
plt.show()
