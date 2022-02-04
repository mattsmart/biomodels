import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from class_cellgraph import CellGraph
from file_io import run_subdir_setup
from settings import SWEEP_VARIETY_VALID

"""
class SweepCellGraph structure

self.sweep_label     - name of the subdir within 'runs' dir which identifies the sweep
self.sweep_dir       - [inferred] directory in which all runs are stored as the params are varied
self.base_kwargs     - core kwargs for CllGraph generation which will be tweaked to create phase diagram
self.params_name     - ordered k-list of param names which are to be varied
self.params_values   - ordered k-list of array like for each param
self.params_variety  - ordered k-list with elements in ['meta_cellgraph', 'sc_ode']
self.k_vary          - [inferred] int k >= 1; number of parameters which are to be varied
self.sizes           - [inferred] k-list with elements len(params_values[j])
self.total_runs      - [inferred] int = prod_j sizes[j]) 
self.results_dict    - main output object; dict of the form 
    - k_tuple ----> output dictionary for a single cellgraph trajectory
    - e.g. if there are two parameters being swept, it is: 
      (i, j, ...): output_dict
      
Important notes:
    - output_dict currently has the form:
        {'num_cells': int,
         'adjacency': num_cells x num_cells array}
    - each run creates a directory uniquely named i_j_... in self.sweep_dir
"""


class SweepCellGraph():

    def __init__(self,
                 sweep_label,
                 base_cellgraph_kwargs,
                 params_name,
                 params_values,
                 params_variety,
                 solver_kwargs):
        self.sweep_label = sweep_label
        self.base_kwargs = base_cellgraph_kwargs
        self.solver_kwargs = solver_kwargs
        self.params_name = params_name
        self.params_values = params_values
        self.params_variety = params_variety
        self.results_dict = {}

        # asserts
        k = len(self.params_name)
        assert k == len(params_values)
        assert k == len(params_variety)
        assert all(a in SWEEP_VARIETY_VALID for a in params_variety)

        self.fast_mod_mode = False
        if all([a == 'sc_ode' for a in params_variety]):
            self.fast_mod_mode = True

        path_run_subfolder = 'runs' + os.sep + self.sweep_label
        assert not os.path.exists(path_run_subfolder)
        os.mkdir(path_run_subfolder)
        self.sweep_dir = path_run_subfolder

        # create base cellgraph for sweep speedups
        self.base_cellgraph = self.create_cellgraph(**self.base_kwargs)

        # set inferred attributes
        self.k_vary = k
        self.sizes = [len(params_values[j]) for j in range(k)]
        self.total_runs = np.prod(self.sizes)
        return

    def create_cellgraph(self,
                         style_ode=None,
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
        cellgraph = CellGraph(
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
            cellgraph.sc_template.params_ode[k] = v

        return cellgraph

    def mod_cellgraph_ode_params(self, base_cellgraph, mods_params_ode):
        """
        Args:
            base_cellgraph    - an instance of CellGraph
            mods_params_ode   - dict of form {single cell params ode name: new_attribute_value}
        Returns:
            new CellGraph with all attirbutes same as base except those specified in attribute_mods
        Currently, unused in favor of simply recreating CellGraph each loop (this would be faster, just more bug risk)
        """
        for k, v in mods_params_ode.items():
            #setattr(base_cellgraph.sc_template.params_ode[k] = v, k, v)
            base_cellgraph.sc_template.params_ode[k] = v
        return base_cellgraph

    def basic_run(self, cellgraph):
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
        event_detected, cellgraph = cellgraph.wrapper_graph_trajectory(**self.solver_kwargs)
        print("\n in main: num cells after wrapper trajectory =", cellgraph.num_cells)

        # Plot the timeseries for each cell
        cellgraph.plot_state_unified(arrange_vertical=True, fmod='final')
        cellgraph.plot_graph(fmod='final')
        if cellgraph.sc_dim_ode > 1:
            cellgraph.plot_xy_separate(fmod='final')
        cellgraph.plotly_traj(fmod='final', show=False, write=True)

        # Save class state as pickle object
        cellgraph.pickle_save('classdump.pkl')

        output_results = {
            'num_cells': cellgraph.num_cells,
            'adjacency': cellgraph.adjacency,
        }
        return output_results

    """  TODO... these are currently replaced by np.ndindex(3, 2, 1):
    def convert_run_id_list_to_run_int(self, run_id_list):

        return run_int

    def convert_run_int_to_run_id_list(self, run_int):
        run_id_list = [0 for _ in self.k_vary]
        for j in self.k_vary:
            run_id_list[j] = ...
        return run_id_list
    """

    def sweep(self):
        """
        sweep_data = np.zeros((len(param_values), 2))
        print('Beginning 1D sweep for %s with %d values in [%.4f, %.4f]' %
              (param_name, len(param_values), param_values[0], param_values[-1]))
        print('Output will be saved to:', self.sweep_dir)
        """
        #for run_int in range(self.total_runs):
        run_int = 0
        for run_id_list in np.ndindex(*self.sizes):
            print('On sweep %d/%d' % (run_int, self.total_runs))

            #run_id_list = self.convert_run_int_to_run_id_list(run_int)
            # 1) Prepare io_dict - this is unique to each run
            timedir_override = '_'.join([str(i) for i in run_id_list])
            io_dict = run_subdir_setup(run_subfolder=self.sweep_label, timedir_override=timedir_override)

            # 2) create modified form of the base_cellgraph -- approach depends on self.fast_mod_mode
            if self.fast_mod_mode:
                mods_params_ode = {}
                for j in range(self.k_vary):
                    mod_key = self.params_name[j]
                    mod_val = self.params_values[j][run_id_list[j]]
                    mods_params_ode[mod_key] = mod_val
                modified_cellgraph = self.mod_cellgraph_ode_params(self.base_cellgraph, mods_params_ode)
                modified_cellgraph['io_dict'] = io_dict
            else:
                mods_params_ode = {}
                modified_cellgraph_kwargs = base_kwargs.copy()
                modified_cellgraph_kwargs['io_dict'] = io_dict
                for j in range(self.k_vary):
                    pname = self.params_name[j]
                    pval = self.params_values[j]
                    pvariety = self.params_variety[j]
                    if pvariety == 'sc_ode':
                        mods_params_ode[pname] = pval
                        modified_cellgraph_kwargs['mods_params_ode'] = mods_params_ode
                    else:
                        assert pvariety == 'meta_cellgraph'
                        modified_cellgraph_kwargs[pname] = pval
                modified_cellgraph = self.create_cellgraph(**modified_cellgraph_kwargs, mods_params_ode=mods_params_ode)

            # 3) Perform the run
            output_results = self.basic_run(modified_cellgraph)

            # 4) Extract relevant output
            self.results_dict[run_id_list] = output_results
            run_int += 1

        print('Sweep done. Saving pickle file...')
        self.pickle_save()
        return

    def pickle_save(self, fname='sweep.pkl'):
        fpath = self.sweep_dir + os.sep + fname
        with open(fpath, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
        return


if __name__ == '__main__':
    params_name = ['t0']
    params_variety = ['meta_cellgraph']  # must be in ['meta_cellgraph', 'sc_ode']
    params_values = [
        [0, 2.0, 10.0]
    ]
    sweep_label = 'sweeps_A'   #%s_%.2f_%.2f_%d' % (

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

    # Initialize the sweep object
    sweep_cellgraph = SweepCellGraph(
        sweep_label,
        base_kwargs,
        params_name,
        params_values,
        params_variety,
        solver_kwargs)

    # Perform the sweep
    sweep_cellgraph.sweep()
