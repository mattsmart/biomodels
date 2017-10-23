import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from formulae import bifurc_value, fp_from_timeseries
from plotting import plot_trajectory_mono, plot_endpoint_mono
from trajectory import trajectory_simulate


# SCRIPT PARAMS
SIM_METHOD = "libcall"  # see constants.py -- SIM_METHODS
ODE_SYSTEM = "feedback_z"  # "default" or "feedback_z" or "feedback_yz"
INIT_COND = [5.0, 5.0, 90.0] #[99.9, 0.1, 0.0]
TIME_START = 0.0
TIME_END = 16000.0  #20.0
NUM_STEPS = 2000  # number of timesteps in each trajectory
param_varying_name = "c"
assert param_varying_name in PARAMS_ID_INV.keys()
SEARCH_START = 0.78  #0.55
SEARCH_END = 0.9 #1.45
SEARCH_AMOUNT = 100  #20

# DYNAMICS PARAMETERS
alpha_plus = 0.2#0.05 #0.4
alpha_minus = 0.5#4.95 #0.5
mu = 0.033 #0.01
a = 1.0
b = 0.8
c = 0.901 #0.95 #2.6 #1.2
N = 100.0
v_x = 0.0
v_y = 0.0
v_z = 0.0
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]

# VARYING PARAM SPECIFICATION
"""
idx_varying = PARAMS_ID_INV[param_varying_name]
if param_varying_name in ["b", "c"]:
    param_varying_bifurcname = BIFURC_DICT[idx_varying]
    param_center = bifurc_value(params, param_varying_bifurcname)
    print "Bifurcation in %s possibly at %.5f" % (param_varying_bifurcname, param_center)
else:
    param_center = params[idx_varying]
print "Searching in window: %.8f to %.8f with %d points" \
      % (SEARCH_START * param_center, SEARCH_END * param_center, SEARCH_AMOUNT)
"""
param_varying_values = np.linspace(SEARCH_START, SEARCH_END, SEARCH_AMOUNT)

# CONSTRUCT PARAM ENSEMBLE
num_param_sets = len(param_varying_values)
param_ensemble = [[elem for elem in params] for _ in xrange(num_param_sets)]
for idx in xrange(num_param_sets):
    param_ensemble[idx][PARAMS_ID_INV[param_varying_name]] = param_varying_values[idx]

# GET TRAJECTORIES
ax_comp = None
r_inf_list = np.zeros((len(param_varying_values), 3))
for idx, params in enumerate(param_ensemble):
    r, times, ax_traj, ax_mono = trajectory_simulate(params, ODE_SYSTEM, init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                                                     sim_method=SIM_METHOD, flag_showplt=False, flag_saveplt=False)
    ax_mono = plot_trajectory_mono(r, times, False, False, ax_mono=ax_comp, mono="z")
    ax_comp = ax_mono
    #assert np.abs(np.sum(r[-1, :]) - N) <= 0.001
    if idx % 10 == 0:
        ax_comp.text(times[-1], r[-1, 2], '%.3f' % param_varying_values[idx])
    r_inf_list[idx] = fp_from_timeseries(r, SIM_METHOD)
plt.savefig(OUTPUT_DIR + sep + "trajectory_mono_z_composite" + ".png")
plt.show()
ax_endpts = plot_endpoint_mono(r_inf_list, param_varying_values, param_varying_name, params, True, True, all_axis=True)
