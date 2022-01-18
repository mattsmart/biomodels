import os
import sys

# IO: Default directories
OSCILLATOR_NETWORKS = os.path.dirname(__file__)
DIR_RUNS = 'runs'  # store timestamped runs here
sys.path.append(OSCILLATOR_NETWORKS)

# DEFAULTS: Integrating the ODE trajectory
DYNAMICS_METHOD = 'libcall'
DYNAMICS_METHODS_VALID = ['libcall', 'rk4', 'euler']
INIT_COND = [60.0, 0.0, 0.0]  # default initial cond for trajectory
NUM_TRAJ = 200  # number of trajectories for trajectory_multiple
TIME_START = 0.0  # standard trajectory start time
TIME_END = 800.0  # standard trajectory end time
NUM_STEPS = 2000  # number of time steps in each trajectory (non-gillespie)

# DEFAULTS: dynamical system
VALID_STYLE_ODE = ['Yang2013', 'PWL']
STYLE_ODE = 'PWL'
