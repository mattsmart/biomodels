import os
import sys

# IO: Default directories
OSCILLATOR_NETWORKS = os.path.dirname(__file__)
DIR_RUNS = 'runs'      # store timestamped runs here
DIR_OUTPUT = 'output'  # misc output like simple plots
sys.path.append(OSCILLATOR_NETWORKS)

# PLOTTING
PLOT_XLABEL = r'Cyc$_{act}$'
PLOT_YLABEL = r'Cyc$_{tot}$'

# DEFAULTS: module 0 - Integrating the ODE trajectory
STYLE_DYNAMICS = 'solve_ivp'
STYLE_DYNAMICS_VALID = ['solve_ivp', 'libcall', 'rk4', 'euler']

# DEFAULTS: module 1 - Single cell dynamical system
STYLE_ODE = 'PWL3_swap'
STYLE_ODE_VALID = ['Yang2013', 'PWL2', 'PWL3', 'PWL3_swap', 'PWL4_auto_wz', 'PWL4_auto_ww', 'PWL4_auto_linear', 'toy_flow']

# DEFAULTS: module 2 - Oscillation detection
STYLE_DETECTION = 'scipy_peaks'
STYLE_DETECTION_VALID = ['ignore', 'scipy_peaks', 'manual_crossings']

# DEFAULTS: module 3 - Coupled cell graph dynamical system - division rules
STYLE_DIVISION = 'copy'
STYLE_DIVISION_VALID = ['copy', 'partition_equal']  # TODO partition_ndiv; partition_random; more?

# DEFAULTS: module 4 - Coupled cell graph dynamical system - diffusion rules
DIFFUSION_RATE = 0.0
