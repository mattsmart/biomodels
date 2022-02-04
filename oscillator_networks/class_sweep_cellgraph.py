import numpy as np
import os
import pickle

from file_io import run_subdir_setup
from settings import SWEEP_VARIETY_VALID
from run_cellgraph import create_cellgraph, mod_cellgraph_ode_params

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
                 sweep_label=None,
                 base_cellgraph_kwargs=None,
                 params_name=None,
                 params_values=None,
                 params_variety=None,
                 solver_kwargs=None):
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
        self.base_cellgraph = create_cellgraph(**self.base_kwargs)

        # set inferred attributes
        self.k_vary = k
        self.sizes = [len(params_values[j]) for j in range(k)]
        self.total_runs = np.prod(self.sizes)
        return

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
                modified_cellgraph = mod_cellgraph_ode_params(self.base_cellgraph, mods_params_ode)
                modified_cellgraph['io_dict'] = io_dict
            else:
                mods_params_ode = {}
                modified_cellgraph_kwargs = self.base_kwargs.copy()
                modified_cellgraph_kwargs['io_dict'] = io_dict
                for j in range(self.k_vary):
                    pname = self.params_name[j]
                    pvariety = self.params_variety[j]
                    pval = self.params_values[j][run_id_list[j]]
                    if pvariety == 'sc_ode':
                        mods_params_ode[pname] = pval
                        modified_cellgraph_kwargs['mods_params_ode'] = mods_params_ode
                    else:
                        assert pvariety == 'meta_cellgraph'
                        modified_cellgraph_kwargs[pname] = pval
                modified_cellgraph = create_cellgraph(**modified_cellgraph_kwargs, mods_params_ode=mods_params_ode)

            # 3) Perform the run
            output_results = self.basic_run(modified_cellgraph)

            # 4) Extract relevant output
            self.results_dict[run_id_list] = output_results
            run_int += 1

        print('Sweep done. Saving pickle file...')
        self.pickle_save()
        return

    def printer(self):
        print('self.sweep_label -', self.sweep_label)
        print('self.sweep_dir -', self.sweep_dir)
        print('self.k_vary -', self.k_vary)
        print('self.sizes -', self.sizes)
        print('self.total_runs -', self.total_runs)
        print('Parameters in sweep:')
        for idx in range(self.k_vary):
            pname = self.params_name[idx]
            pvar = self.params_variety[idx]
            pv = self.params_values[idx]
            print('\tname: %s, variety: %s, npts: %d, low: %.4f, high: %.4f' % (pname, pvar, len(pv), pv[0], pv[-1]))

    def pickle_save(self, fname='sweep.pkl'):
        fpath = self.sweep_dir + os.sep + fname
        with open(fpath, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
        return
