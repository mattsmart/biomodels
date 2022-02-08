import numpy as np

from class_sweep_cellgraph import SweepCellGraph
from preset_sweep import PRESET_SWEEP


if __name__ == '__main__':
    flag_preset = True  # use preset defined in SWEEP_PRESETS, or prepare own custom sweep run

    if flag_preset:
        preset_choice = '1d_diffusion_ndiv_bam'
        sweep_preset = PRESET_SWEEP[preset_choice]
        sweep_cellgraph = SweepCellGraph(**sweep_preset)

    else:
        sweep_label = 'sweep_custom'   #%s_%.2f_%.2f_%d' % (
        params_name = ['t0']
        params_variety = ['meta_cellgraph']  # must be in ['meta_cellgraph', 'sc_ode']
        params_values = [
            [0, 2.0, 10.0]
        ]

        # Initialize the base CellGraph which will be varied during the sweep
        # A) High-level initialization & graph settings
        style_ode = 'PWL3_swap'                # styles: ['PWL2', 'PWL3', 'PWL3_swap', 'Yang2013', 'toy_flow', 'toy_clock']
        style_detection = 'manual_crossings'   # styles: ['ignore', 'scipy_peaks', 'manual_crossings', 'manual_crossings_2d']
        style_division = 'copy'                # styles: ['copy', 'partition_equal']
        style_diffusion = 'xy'                 # styles: ['all', 'xy']
        M = 1
        diffusion_rate = 0.0
        # B) Initialization modifications for different cases
        if style_ode == 'PWL2':
            state_history = np.array([[100, 100]]).T     # None or array of shape (NM x times)
        elif style_ode == 'PWL3_swap':
            state_history = np.array([[0, 0, 0]]).T  # None or array of shape (NM x times)
        else:
            state_history = None
        # C) Specify time interval which is separate from solver kwargs (used in graph_trajectory explicitly)
        t0 = 00  # None ot float
        t1 = 65  # None ot float
        # D) Setup solver kwargs for the graph trajectory wrapper
        solver_kwargs = {}
        solver_kwargs['t_eval'] = None  # None or np.linspace(0, 50, 2000)  np.linspace(15, 50, 2000)
        solver_kwargs['max_step'] = np.Inf  # try 1e-1 or 1e-2 if division time-sequence is buggy as a result of large adaptive steps
        base_kwargs = dict(
            style_ode=style_ode,
            style_detection=style_detection,
            style_division=style_division,
            style_diffusion=style_diffusion,
            M=M,
            diffusion_rate=diffusion_rate,
            t0=t0,
            t1=t1,
            state_history=state_history,
            verbosity=0)

        # Initialize the sweep object
        sweep_cellgraph = SweepCellGraph(
            sweep_label=sweep_label,
            base_cellgraph_kwargs=base_kwargs,
            params_name=params_name,
            params_values=params_values,
            params_variety=params_variety,
            solver_kwargs=solver_kwargs)

    # Perform the sweep
    sweep_cellgraph.sweep()
