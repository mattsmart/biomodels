import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from formulae import bifurc_value, fp_from_timeseries
from params import Params
from plotting import plot_trajectory_mono, plot_endpoint_mono
from trajectory import trajectory_simulate

# PARAM TO VARY
param_varying_name = "mu"
if param_varying_name == "c":
    flag_log = False
    assert param_varying_name in PARAMS_ID_INV.keys()
    SEARCH_START = 0.7
    SEARCH_END = 1.02
    SEARCH_AMOUNT = 100  #20
elif param_varying_name == "b":
    flag_log = False
    assert param_varying_name in PARAMS_ID_INV.keys()
    SEARCH_START = 0.8
    SEARCH_END = 1.2
    SEARCH_AMOUNT = 100  #20
else:  # assume "mu" in general
    flag_log = True
    assert param_varying_name in PARAMS_ID_INV.keys()
    SEARCH_START = 1e-5  #0.55
    SEARCH_END = 1e-1   #1.45
    SEARCH_AMOUNT = 80  #20

if flag_log:
    param_varying_values = np.logspace(np.log10(SEARCH_START), np.log10(SEARCH_END), num=SEARCH_AMOUNT)  # log axis
else:
    param_varying_values = np.linspace(SEARCH_START, SEARCH_END, SEARCH_AMOUNT)

# SCRIPT PARAMS
system = "feedback_z"  # "default" or "feedback_z" or "feedback_yz"
SIM_METHOD = "libcall"  # see constants.py -- SIM_METHODS_VALID
INIT_COND = [98.0, 1.0, 1.0] #[99.9, 0.1, 0.0]
TIME_START = 0.0
TIME_END = 10*16000.0  #20.0
NUM_STEPS = 2000  # number of timesteps in each trajectory

# DYNAMICS PARAMETERS
alpha_plus = 0.02
alpha_minus = 0.1  # 4.95
mu = 1e-4
a = 1.0
b = 0.92    # 0.98 1.01
c = 0.99  # 0.98
N = 100.0
v_x = 0.0
v_y = 0.0
v_z = 0.0
mu_base = 0.0
params_list = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base]

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

# CONSTRUCT PARAM ENSEMBLE
num_param_sets = len(param_varying_values)
param_ensemble = [[elem for elem in params_list] for _ in xrange(num_param_sets)]
for idx in xrange(num_param_sets):
    param_ensemble[idx][PARAMS_ID_INV[param_varying_name]] = param_varying_values[idx]

# GET TRAJECTORIES
ax_comp = None
r_inf_list = np.zeros((len(param_varying_values), 3))

for idx, params_list in enumerate(param_ensemble):
    params = Params(params_list, system)
    r, times, ax_traj, ax_mono = trajectory_simulate(params, init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                                                     sim_method=SIM_METHOD, flag_showplt=False, flag_saveplt=False)
    ax_mono = plot_trajectory_mono(r, times, False, False, ax_mono=ax_comp, mono="z")
    ax_comp = ax_mono
    #assert np.abs(np.sum(r[-1, :]) - N) <= 0.001
    if idx % 10 == 0:
        ax_comp.text(times[-1], r[-1, 2], '%.3f' % param_varying_values[idx])
    r_inf_list[idx] = fp_from_timeseries(r, SIM_METHOD)
plt.savefig(OUTPUT_DIR + sep + "trajectory_mono_z_composite" + ".png")
plt.show()
ax_endpts = plot_endpoint_mono(r_inf_list, param_varying_values, param_varying_name, params, True, True, all_axis=True,
                               conv_to_fraction=True, flag_log=flag_log, plt_save="endpoint_varying_%s" % param_varying_name)
