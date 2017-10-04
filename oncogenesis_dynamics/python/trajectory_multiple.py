import matplotlib.pyplot as plt
import numpy as np
import random
from os import sep

from constants import OUTPUT_DIR, PARAMS_ID, PARAMS_ID_INV, BIFURC_DICT, VALID_BIFURC_PARAMS
from formulae import bifurc_value, fp_from_timeseries
from plotting import plot_trajectory_mono, plot_endpoint_mono, plot_simplex, plot_trajectory
from trajectory import trajectory_simulate

# SCRIPT PARAMS
SIM_METHOD = "libcall"  # see constants.py -- SIM_METHODS
ODE_SYSTEM = "feedback"  # "default" or "feedback"
INIT_COND = [99.9, 0.1, 0.0]
TIME_START = 0.0
TIME_END = 16000.0  #20.0
NUM_STEPS = 20000  # number of timesteps in each trajectory
NUM_TRAJ = 4

# DYNAMICS PARAMETERS
alpha_plus = 0.2#0.05 #0.4
alpha_minus = 0.5#4.95 #0.5
mu = 0.1 #0.01
a = 1.0
b = 0.8
c = 0.6 #2.6 #1.2
N = 100.0 #100
v_x = 0.0
v_y = 0.0
v_z = 0.0
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]


# GET TRAJECTORIES
ax_comp = None
init_conds = np.zeros((NUM_TRAJ,3))
for k in xrange(NUM_TRAJ):
    ak = N*np.random.random_sample()
    bk = (N-ak)*np.random.random_sample()
    ck = N - ak - bk
    init_cond = [ak, bk, ck]
    init_conds[k,:] = np.array(init_cond)

fig_traj = plot_simplex(N)
ax_traj = fig_traj.gca()
ax_traj.view_init(5, 35)  # ax.view_init(-45, -15)
for idx, init_cond in enumerate(init_conds):
    r, times, _, _ = trajectory_simulate(init_cond=init_cond, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS,
                                         params=params, sim_method=SIM_METHOD, ode_system=ODE_SYSTEM,
                                         flag_showplt=False, flag_saveplt=False)
    ax_traj.plot(r[:, 0], r[:, 1], r[:, 2], label='trajectory')
    #assert np.abs(np.sum(r[-1, :]) - N) <= 0.001
plt.savefig(OUTPUT_DIR + sep + "trajectory_simplex_multi" + ".png")