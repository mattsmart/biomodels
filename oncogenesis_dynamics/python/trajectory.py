import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR
from formulae import simulate_dynamics_general, fp_location_general, is_stable
from plotting import plot_simplex, plot_trajectory_mono, plot_trajectory


# MATPLOTLIB GLOBAL SETTINGS
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(mpl_params)

# SCRIPT PARAMS
SIM_METHOD = "libcall"  # see constants.py -- SIM_METHODS
ODE_SYSTEM = "feedback"  # see constants.py -- ODE_SYSTEMS
INIT_COND = [95.0, 5.0, 0.0]
TIME_START = 0.0
TIME_END = 20.0
NUM_STEPS = 200  # number of timesteps in window
plt_title = 'Trajectory'
plt_save = 'trajectory'

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
PARAMS = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]


def trajectory_infoprint(init_cond, t0, t1, num_steps, params):
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    times = np.linspace(t0, t1, num_steps + 1)
    print "ODE Setup: t0, t1:", t0, t1, "| num_steps, dt:", num_steps, times[1] - times[0]
    print "Init Cond:", init_cond
    print "Specified parameters: \nalpha_plus = " + str(alpha_plus) + " | alpha_minus = " + str(alpha_minus) + \
          " | mu = " + str(mu) + " | a = " + str(a) + " | b = " + str(b) + " | c = " + str(c) + " | N = " + str(N) + \
          " | v_x = " + str(v_x) + " | v_y = " + str(v_y) + " | v_z = " + str(v_z)


def trajectory_simulate(init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS, params=PARAMS,
                        sim_method=SIM_METHOD, ode_system=ODE_SYSTEM, flag_showplt=False, flag_saveplt=True):
    # SIMULATE SETUP
    alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z = params
    display_spacing = int(num_steps / 10)
    times = np.linspace(t0, t1, num_steps + 1)
    trajectory_infoprint(init_cond, t0, t1, num_steps, params)

    # SIMULATE
    r, times = simulate_dynamics_general(init_cond, times, params, method=sim_method, system=ode_system)
    print 'Done trajectory\n'

    # FP COMPARISON
    """
    if v_x == 0 and v_y == 0 and v_z == 0:
        solver_numeric = False
    else:
        solver_numeric = True
    predicted_fps = fp_location_general(params, solver_numeric=solver_numeric, solver_fast=False)
    print "Predicted FPs:"
    for i in xrange(3):
        print "FP", i, predicted_fps[i], "Stable:", is_stable(params, predicted_fps[i])
    """

    # PLOTTING
    if flag_showplt or flag_saveplt:
        fig_traj = plot_simplex(N)
        ax_traj = plot_trajectory(fig_traj, r, times, flag_showplt, flag_saveplt)
        ax_mono_z = plot_trajectory_mono(r, times, flag_showplt, flag_saveplt)
    else:
        ax_traj = None
        ax_mono_z = None
    return r, times, ax_traj, ax_mono_z


if __name__ == "__main__":
    print "main functionality not implemented"
    r, times, ax_traj, ax_mono_z = trajectory_simulate(init_cond=[20,40,40], sim_method="gillespie",num_steps=NUM_STEPS*10)
    for i in xrange(len(r)):
        print r[i,:]