import os
import sys

# IO: Default directories
OSCILLATOR_NETWORKS = os.path.dirname(__file__)
DIR_RUNS = 'runs'      # store timestamped runs here
DIR_OUTPUT = 'output'  # misc output like simple plots
sys.path.append(OSCILLATOR_NETWORKS)

# DEFAULTS: Integrating the ODE trajectory
DYNAMICS_METHOD = 'solve_ivp'
DYNAMICS_METHODS_VALID = ['solve_ivp', 'libcall', 'rk4', 'euler']

# DEFAULTS: single cell dynamical system
VALID_STYLE_ODE = ['Yang2013', 'PWL', 'toy_flow']
DEFAULT_STYLE_ODE = 'PWL'

# Cell-cell coupling:
DIFFUSION_RATE = 0.0

# PLOTTING
PLOT_XLABEL = r'Cyc$_{act}$'
PLOT_YLABEL = r'Cyc$_{tot}$'
