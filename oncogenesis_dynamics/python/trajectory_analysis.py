import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from formulae import bifurc_value, fp_from_timeseries
from plotting import plot_trajectory_mono, plot_endpoint_mono
from trajectory import trajectory_simulate


# SCRIPT PARAMS
ODE_METHOD = "libcall"  # see constants.py -- ODE_METHODS
ODE_SYSTEM = "feedback"  #"default"
INIT_COND = [99.9, 0.1, 0.0]
TIME_START = 0.0
TIME_END = 160.0  #20.0
NUM_STEPS = 2000  # number of timesteps in each trajectory
param_varying_name = "b"
SEARCH_START = 0.6  #0.55
SEARCH_END = 0.85  #1.45
SEARCH_AMOUNT = 320  #20

# DYNAMICS PARAMETERS
alpha_plus = 0.05
alpha_minus = 4.95
mu = 0.77
a = 1.0
b = 8.369856428  #1.376666
c = 2.6
N = 100.0
v_x = 0.0#1.0
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
r_inf_list = np.zeros((len(param_varying_values), 3))
for idx, params in enumerate(param_ensemble):
    r, times, ax_traj, ax_mono = trajectory_simulate(init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                                                     params=params, ode_method=ODE_METHOD, ode_system=ODE_SYSTEM,
                                                     flag_showplt=False, flag_saveplt=False)
    ax_mono = plot_trajectory_mono(r, times, False, False, ax_mono=ax_comp)
    ax_comp = ax_mono
    r_inf_list[idx] = fp_from_timeseries(r)
plt.savefig(OUTPUT_DIR + sep + "trajectory_mono_z_composite" + ".png")
plt.show()
ax_endpts = plot_endpoint_mono(r_inf_list, param_varying_values, param_varying_name, True, True)
