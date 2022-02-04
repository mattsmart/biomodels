import matplotlib.pyplot as plt
import numpy as np
import os

from class_cellgraph import CellGraph
from file_io import run_subdir_setup
from settings import SWEEP_VARIETY_VALID


def create_cellgraph(style_ode=None,
                     style_detection=None,
                     style_division=None,
                     M=None,
                     t0=None,
                     t1=None,
                     state_history=None,
                     mods_params_ode={},
                     io_dict=False,
                     verbosity=0):
    # Instantiate the graph and modify ode params if needed
    base_cellgraph = CellGraph(
        num_cells=M,
        style_ode=style_ode,
        style_detection=style_detection,
        style_division=style_division,
        state_history=state_history,
        t0=t0,
        t1=t1,
        io_dict=io_dict,
        verbosity=verbosity)

    # Post-processing as in class_cellgraph.py main() call
    for k, v in mods_params_ode.items():
        base_cellgraph.sc_template.params_ode[k] = v

    return base_cellgraph


def mod_cellgraph(base_cellgraph, attribute_mods):
    """
    Args:
        base_cellgraph    - an instance of CellGraph
        attribute_mods    - dict of form {attribute_name: new_attribute_value}
    Returns:
        new CellGraph with all attirbutes same as base except those specified in attribute_mods
    Currently, unused in favor of simply recreating CellGraph each loop (this would be faster, just more bug risk)
    """
    ATTIBUTE_MOD_LIST = ['t1']
    for k, v in attribute_mods.items():
        assert k in ATTIBUTE_MOD_LIST  # Note that some attributes cannot simply be set, e.g. t0, due to complex init
        setattr(base_cellgraph, k, v)
    return base_cellgraph


def basic_run(cellgraph, solver_kwargs):
    """
    Given a cellgraph, performs some basic operations (e.g. a trajectory) and outputs a 'results' array
    Returns:
        output_results - 1D array of size k
    """
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
    cellgraph.plot_state_unified(arrange_vertical=True, fmod='final')
    cellgraph.plot_graph(fmod='final')
    if cellgraph.sc_dim_ode > 1:
        cellgraph.plot_xy_separate(fmod='final')
    cellgraph.plotly_traj(fmod='final', show=False, write=True)

    # Save class state as pickle object
    cellgraph.pickle_save('classdump.pkl')

    output_results = np.array([cellgraph.num_cells])
    return output_results


def sweep_1D(base_kwargs, param_variety, param_name, param_values, solver_kwargs):

    assert param_variety in SWEEP_VARIETY_VALID
    # Prepare runs subfolder in which the sweep will be stored
    run_subfolder = 'sweep_1d_%s_%.2f_%.2f_%d' % (param_name, param_values[0], param_values[-1], len(param_values))
    path_run_subfolder = 'runs' + os.sep + run_subfolder
    assert not os.path.exists(path_run_subfolder)
    os.mkdir(path_run_subfolder)

    sweep_data = np.zeros((len(param_values), 2))
    print('Beginning 1D sweep for %s with %d values in [%.4f, %.4f]' %
          (param_name, len(param_values), param_values[0], param_values[-1]))
    print('Output will be saved to:', path_run_subfolder)
    for p_idx, p_val in enumerate(param_values):

        # 1) Prepare io_dict - this is unique to each run
        timedir_override = str(p_idx)
        io_dict = run_subdir_setup(run_subfolder=run_subfolder, timedir_override=timedir_override)

        # 2) create modified form of the base_cellgraph
        """
        attribute_mods = {}
        attribute_mods['io_dict'] = io_dict
        attribute_mods[param_name] = p_val
        modified_cellgraph = mod_cellgraph(base_cellgraph, attribute_mods)"""
        modified_kwargs = base_kwargs.copy()
        modified_kwargs['io_dict'] = io_dict
        if param_variety == 'sc_ode':
            mods_params_ode = {param_name: p_val}
            modified_kwargs['mods_params_ode'] = mods_params_ode
        else:
            assert param_variety == 'meta_cellgraph'
            modified_kwargs[param_name] = p_val
        modified_cellgraph = create_cellgraph(**modified_kwargs)

        # 3) Perform the run
        output_results = basic_run(modified_cellgraph, solver_kwargs)

        # 4) Extract relevant output
        sweep_data[p_idx, :] = [output_results[:]]

    fpath_results = path_run_subfolder + os.sep + 'results' + '.txt'
    fpath_params = path_run_subfolder + os.sep + 'param_values' + '.txt'
    print('1D sweep is done. Writing results to file...')
    np.savetxt(fpath_results, sweep_data)
    np.savetxt(fpath_params, param_values)

    return sweep_data


if __name__ == '__main__':
    param_name = 't0'
    param_variety = 'meta_cellgraph'  # must be in ['meta_cellgraph', 'sc_ode']
    param_values = [0, 2.0, 10.0]

    # Initialize the base CellGraph which will be varied during the sweep
    # A) High-level initialization & graph settings
    style_ode = 'PWL3_swap'                # styles: ['PWL2', 'PWL3', 'PWL3_swap', 'Yang2013', 'toy_flow', 'toy_clock']
    style_detection = 'manual_crossings'   # styles: ['ignore', 'scipy_peaks', 'manual_crossings', 'manual_crossings_2d']
    style_division = 'copy'                # styles: ['copy', 'partition_equal']
    M = 1
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
        M=M,
        t0=t0,
        t1=t1,
        state_history=state_history)

    # Perform a 1D sweep
    sweep_data = sweep_1D(base_kwargs, param_variety, param_name, param_values, solver_kwargs)
