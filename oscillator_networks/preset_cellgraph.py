import numpy as np

from preset_solver import PRESET_SOLVER

# Initialize the base CellGraph which will be varied during the sweep
# A) High-level initialization & graph settings
style_ode = 'PWL3_swap'  # styles: ['PWL2', 'PWL3', 'PWL3_swap', 'Yang2013', 'toy_flow', 'toy_clock']
style_detection = 'manual_crossings'  # styles: ['ignore', 'scipy_peaks', 'manual_crossings', 'manual_crossings_2d']
style_division = 'copy'  # styles: ['copy', 'partition_equal']

# B) Initialization modifications for different cases
if style_ode == 'PWL2':
    state_history = np.array([[100, 100]]).T  # None or array of shape (NM x times)

# D) Setup solver kwargs for the graph trajectory wrapper
solver_kwargs = {}
solver_kwargs['t_eval'] = None  # None or np.linspace(0, 50, 2000)  np.linspace(15, 50, 2000)
solver_kwargs['max_step'] = np.Inf  # try 1e-1 or 1e-2 if division time-sequence is buggy as a result of large adaptive steps


PRESET_CELLGRAPH = {
    'PWL3_swap_copy': dict(
        num_cells=1,
        style_ode='PWL3_swap',
        style_detection='manual_crossings',
        style_division='copy',
        t0=0,
        t1=65,
        state_history=np.array([[0, 0, 0]]).T,
        verbosity=0,
        mods_params_ode={}
    )
}
