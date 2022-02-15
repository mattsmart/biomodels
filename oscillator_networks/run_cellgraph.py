import numpy as np
import os

from class_cellgraph import CellGraph
from utils_io import run_subdir_setup
from preset_solver import PRESET_SOLVER
from preset_cellgraph import PRESET_CELLGRAPH

"""
To run a cellgraph trajectory, need
1) the cellgraph kwargs (can get from preset_cellgraph.py)
2) the solver kwargs (can get from preset_solver.py) -- currently hard-coded to be used with solve_ivp() by default
3) an io_dict instance, which can be tacked on to cellgraph_kwargs before passing to create_cellgraph()
"""


def create_cellgraph(style_ode=None,
                     style_detection=None,
                     style_division=None,
                     style_diffusion=None,
                     num_cells=None,
                     diffusion_rate=None,
                     t0=None,
                     t1=None,
                     state_history=None,
                     io_dict=False,
                     verbosity=0,
                     mods_params_ode={}):
    # Instantiate the graph and modify ode params if needed
    cellgraph = CellGraph(
        num_cells=num_cells,
        style_ode=style_ode,
        style_detection=style_detection,
        style_division=style_division,
        style_diffusion=style_diffusion,
        state_history=state_history,
        diffusion_rate=diffusion_rate,
        t0=t0,
        t1=t1,
        io_dict=io_dict,
        verbosity=verbosity)

    # Post-processing as in class_cellgraph.py main() call
    for k, v in mods_params_ode.items():
        cellgraph.sc_template.params_ode[k] = v

    return cellgraph


def mod_cellgraph_ode_params(base_cellgraph, mods_params_ode):
    """
    Args:
        base_cellgraph    - an instance of CellGraph
        mods_params_ode   - dict of form {single cell params ode name: new_attribute_value}
    Returns:
        new CellGraph with all attirbutes same as base except those specified in attribute_mods
    Currently, unused in favor of simply recreating CellGraph each loop (this would be faster, just more bug risk)
    """
    for k, v in mods_params_ode.items():
        # setattr(base_cellgraph.sc_template.params_ode[k] = v, k, v)
        base_cellgraph.sc_template.params_ode[k] = v
    return base_cellgraph


if __name__ == '__main__':

    flag_preset = False

    if flag_preset:
        cellgraph_preset_choice = 'PWL3_swap_partition_ndiv_bam'  # PWL3_swap_partition_ndiv_bam, PWL3_swap_copy
        io_dict = run_subdir_setup(run_subfolder='cellgraph')
        solver_kwargs = PRESET_SOLVER['solve_ivp_radau_default']['kwargs']

        cellgraph_preset = PRESET_CELLGRAPH[cellgraph_preset_choice]
        cellgraph_preset['io_dict'] = io_dict
        cellgraph_preset['mods_params_ode']['epsilon'] = 0.20
        #cellgraph_preset['style_detection'] = 'manual_crossings_1d_mid'
        cellgraph_preset['style_diffusion'] = 'xy'
        cellgraph_preset['diffusion_rate'] = 10.5
        cellgraph = create_cellgraph(**cellgraph_preset)

    else:
        # High-level initialization & graph settings
        style_ode = 'PWL3_swap'                      # styles: ['PWL2', 'PWL3', 'PWL3_swap', 'Yang2013', 'toy_flow', 'toy_clock']
        style_detection = 'manual_crossings_1d_mid'  # styles: ['ignore', 'scipy_peaks', 'manual_crossings_1d_mid', 'manual_crossings_1d_hl', 'manual_crossings_2d']
        style_division = 'copy'        # styles: ['copy', 'partition_equal', 'partition_ndiv_all', 'partition_ndiv_bam']
        style_diffusion = 'all'                      # styles: ['all', 'xy']
        M = 1
        diffusion_rate = 0
        verbosity = 0  # in 0, 1, 2 (highest)
        # TODO - GLOBAL initiliazation style (predefined, random, other? -- the if else below is just an override to predefined ones)
        # TODO external parameter/init arguments for this DIFFUSION style

        # Main-loop-specific settings
        add_init_cells = 0

        # Initialization modifications for different cases
        if style_ode == 'PWL2':
            state_history = np.array([[100, 100]]).T     # None or array of shape (NM x times)
        elif style_ode == 'PWL3_swap':
            state_history = np.array([[0, 0, 0]]).T  # None or array of shape (NM x times)
        else:
            state_history = None

        # Specify time interval which is separate from solver kwargs (used in graph_trajectory explicitly)
        #time_interval = [10, 100]  # None or [t0, t1]
        t0 = 00
        t1 = 65

        # Prepare io_dict
        io_dict = run_subdir_setup(run_subfolder='cellgraph')

        # Instantiate the graph and modify ode params if needed
        cellgraph = CellGraph(
            num_cells=M,
            style_ode=style_ode,
            style_detection=style_detection,
            style_division=style_division,
            style_diffusion=style_diffusion,
            state_history=state_history,
            diffusion_rate=diffusion_rate,
            t0=t0,
            t1=t1,
            io_dict=io_dict,
            verbosity=verbosity)
        if cellgraph.style_ode in ['PWL2', 'PWL3', 'PWL3_swap']:
            #pass
            cellgraph.sc_template.params_ode['epsilon'] = 0.15
            cellgraph.sc_template.params_ode['C'] = 1e-2

        # Add some cells through manual divisions (two different modes - linear or random) to augment initialization
        for idx in range(add_init_cells):
            dividing_idx = np.random.randint(0, cellgraph.num_cells)
            print("Division event (idx, div idx):", idx, dividing_idx)
            # Mode choice (divide linearly or randomly)
            cellgraph = cellgraph.division_event(idx, 0)  # Mode 1 - linear division idx
            #cellgraph.division_event(dividing_idx, 0)    # Mode 2 - random division idx
            # Output plot & print
            #cellgraph.plot_graph()
            cellgraph.print_state()
            print()

        # Setup solver kwargs for the graph trajectory wrapper
        # TODO ensure solver kwargs can be passed properly -- note wrapper is recursive so some kwargs MUST be updated...
        # TODO resolve issue where if t0 != 0, then we have gap in times history [0, t0, ...] -- another issue where t0 is skipped, start t0+dt
        # TODO one option is to use dense_output to interpolate... another is to use non-adaptive stepping in the vicinity of an event
        solver_kwargs = {}  # assume passing to solve_ivp for now
        solver_kwargs['method'] = 'Radau'
        solver_kwargs['t_eval'] = None  # None or np.linspace(0, 50, 2000)  np.linspace(15, 50, 2000)
        solver_kwargs['max_step'] = 1e-1   # np.Inf ; try 1e-1 or 1e-2 if division time-sequence is buggy as a result of large adaptive steps

    # Write initial CellGraph info to file
    cellgraph.print_state()
    cellgraph.write_metadata()
    cellgraph.write_state(fmod='init')
    cellgraph.plot_graph(fmod='init')

    # From the initialized graph (after all divisions above), simulate graph trajectory
    print('\nExample trajectory for the graph...')
    event_detected, cellgraph = cellgraph.wrapper_graph_trajectory(**solver_kwargs)
    print("\n in main: num cells after wrapper trajectory =", cellgraph.num_cells)

    # Plot the timeseries for each cell
    """
    cellgraph.plot_state_unified(arrange_vertical=True, fmod='final')
    cellgraph.plot_graph(fmod='final')
    if cellgraph.sc_dim_ode > 1:
        cellgraph.plot_xy_separate(fmod='final')
    cellgraph.plotly_traj(fmod='final', show=True, write=True)

    # Save class state as pickle object
    cellgraph.pickle_save('classdump.pkl')
    """