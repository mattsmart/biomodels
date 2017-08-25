import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR
from formulae import ode_general, fp_location_general, is_stable
from plotting import plot_simplex, plot_trajectory_mono, plot_trajectory
from trajectory import trajectory_simulate


# SCRIPT PARAMS
FLAG_SHOWPLT = False
FLAG_SAVEPLT = True
ODE_METHOD = "libcall"  # see constants.py -- ODE_METHODS
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


r = trajectory_simulate(init_cond=INIT_COND, t0=TIME_START, t1=TIME_END, num_steps=NUM_STEPS, params=PARAMS,
                        ode_method=ODE_METHOD, flag_showplt=FLAG_SHOWPLT, flag_saveplt=FLAG_SAVEPLT)