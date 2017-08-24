import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR
from formulae import ode_general, fp_location_general, is_stable
from plotting import plot_simplex, plot_trajectory_mono, plot_trajectory


# MATPLOTLIB GLOBAL SETTINGS
mpl_params = {'legend.fontsize': 'x-large', 'figure.figsize': (8, 5), 'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
pylab.rcParams.update(mpl_params)

# SCRIPT PARAMS
FLAG_SHOWPLT = False
FLAG_SAVEPLT = True
plt_title = 'Trajectory'
plt_save = 'trajectory'
ode_method = "libcall"  # see constants.py -- ODE_METHODS = ["euler", "rk4", "libcall"]

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

# =====================
# SIMULATE SETUP
# =====================
init_cond = [95.0, 5.0, 0.0]
time_start = 0.0
time_end = 20.0
num_steps = 200  # number of timesteps in window
display_spacing = int(num_steps/10)
times = np.linspace(time_start, time_end, num_steps + 1)
print "ODE Setup: \nt0, t1:", time_start, time_end, "\nnum_steps, dt:", num_steps, times[1] - times[0], "\n"
print "Specified parameters: \nalpha_plus = " + str(alpha_plus) + "\nalpha_minus = " + str(alpha_minus) + \
      "\nmu = " + str(mu) + "\na = " + str(a) + "\nb = " + str(b) + "\nc = " + str(c) + "\nN = " + str(N) + \
      "\nv_x = " + str(v_x) + "\nv_y = " + str(v_y) + "\nv_z = " + str(v_z), "\n"

# =====================
# SIMULATE
# =====================
print "Trajectory loop..."
#r = ode_euler(init_cond, times, params)
r = ode_general(init_cond, times, params, method=ode_method)
print 'Done trajectory\n'

# =====================
# FP COMPARISON
# =====================
if v_x == 0 and v_y == 0 and v_z == 0:
    solver_numeric = False
else:
    solver_numeric = True
predicted_fps = fp_location_general(params, solver_numeric=solver_numeric, solver_fast=False)
print "Predicted FPs:"
for i in xrange(3):
    print "FP", i, predicted_fps[i], "Stable:", is_stable(params, predicted_fps[i])

# =====================
# PLOTTING
# =====================
fig_traj = plot_simplex(N)
fig_traj = plot_trajectory(fig_traj, r, times, FLAG_SHOWPLT, FLAG_SAVEPLT, plt_title=plt_title)
fig_mono_z = plot_trajectory_mono(r, times, FLAG_SHOWPLT, FLAG_SAVEPLT)
