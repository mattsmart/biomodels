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
STYLE_DYNAMICS_VALID = ['solve_ivp', 'numba_lsoda', 'diffeqpy', 'libcall', 'rk4', 'euler']

# DEFAULTS: module 1 - Single cell dynamical system
STYLE_ODE = 'PWL3_swap'
STYLE_ODE_VALID = ['Yang2013', 'bpj2017',
                   'PWL2', 'PWL3', 'PWL3_swap', 'PWL4_auto_wz', 'PWL4_auto_ww', 'PWL4_auto_linear',
                   'toy_flow', 'toy_clock']

# DEFAULTS: module 2 - Oscillation detection
STYLE_DETECTION = 'manual_crossings_1d_mid'
STYLE_DETECTION_VALID = ['ignore', 'scipy_peaks', 'manual_crossings_1d_mid', 'manual_crossings_1d_hl', 'manual_crossings_2d']

# DEFAULTS: module 3 - Coupled cell graph dynamical system - division rules
STYLE_DIVISION = 'partition_ndiv_bam'
STYLE_DIVISION_VALID = ['copy', 'partition_equal', 'partition_ndiv_all', 'partition_ndiv_bam']  # TODO partition_random; more?

# DEFAULTS: module 4 - Coupled cell graph dynamical system - diffusion rules
DIFFUSION_RATE = 0.0  # default diffusion_rate attribute of cellgraph; also induces the diffusion vector attribute
STYLE_DIFFUSION = 'xy'
STYLE_DIFFUSION_VALID = ['all', 'xy']

# Parameter sweep settings
SWEEP_VARIETY_VALID = ['meta_cellgraph', 'sc_ode']
