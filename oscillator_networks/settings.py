import os
import sys

# IO: Default directories
OSCILLATOR_NETWORKS = os.path.dirname(__file__)
DIR_RUNS = 'runs'  # store timestamped runs here
sys.path.append(OSCILLATOR_NETWORKS)

# DEFAULTS: Integrating the ODE trajectory
DYNAMICS_METHOD = 'libcall'
DYNAMICS_METHODS_VALID = ['libcall', 'rk4', 'euler']

# DEFAULTS: dynamical system
VALID_STYLE_ODE = ['Yang2013', 'PWL']
STYLE_ODE = 'PWL'

# PLOTTING
PLOT_XLABEL = r'Cyc$_{act}$'
PLOT_YLABEL = r'Cyc$_{tot}$'
