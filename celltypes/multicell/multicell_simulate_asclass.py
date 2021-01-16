import numpy as np
import os
import random
import matplotlib.pyplot as plt

from multicell.graph_adjacency import \
    lattice_square_int_to_loc, adjacency_lattice_square, adjacency_lattice_general
from multicell.multicell_constants import \
    GRIDSIZE, SEARCH_RADIUS_CELL, NUM_LATTICE_STEPS, VALID_BUILDSTRINGS, VALID_EXOSOME_STRINGS, \
    BUILDSTRING, EXOSTRING, LATTICE_PLOT_PERIOD, MEANFIELD, EXOSOME_REMOVE_RATIO, \
    BLOCK_UPDATE_LATTICE, AUTOCRINE
from multicell.multicell_lattice import \
    build_lattice_main, get_cell_locations, prep_lattice_data_dict, write_state_all_cells, \
    write_grid_state_int
from multicell.multicell_metrics import \
    calc_lattice_energy, calc_compression_ratio, get_state_of_lattice
from multicell.multicell_visualize import \
    lattice_uniplotter, reference_overlap_plotter, lattice_projection_composite
from singlecell.singlecell_constants import FIELD_SIGNAL_STRENGTH, FIELD_APPLIED_STRENGTH, BETA
from singlecell.singlecell_fields import construct_app_field_from_genes
from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import run_subdir_setup, runinfo_append, write_general_arr, read_general_arr


# TODO indicate _private methods (vs non-private, like get_cell_state(idx))
class Multicell:
    """
    Primary object for multicell simulations.
    Current state of the graph is tracked using attribute: graph_state
     - graph_state is an N*M long 1D array
     - get state of a specific cell using get_cell_state(idx)

    simsetup: dictionary created using singlecell_simsetup() in singlecell_simsetup.py

    Expected kwargs for __init__
        num_cells
        graph_style
        initialization_style
        gamma
        autocrine
        flag_housekeeping
        beta

    Optional kwargs for __init__
        graph_kwargs
        exosome_string
        exosome_remove_ratio
        field_applied
        kappa
        seed

    Attributes
        simsetup:          (dict) simsetup with internal and external gene regulatory rules
        num_genes:         (int) aka N -- internal dimension of each cell
        num_celltypes:     (int) aka P -- patterns encoded in each cell
        num_cells:         (int) aka M -- number of nodes in cell-cell graph
        total_spins:       (int) N * M
        matrix_J:          (arr) N x N -- governs internal dynamics
        matrix_W:          (arr) N x N -- governs cell-cell signalling
        matrix_A:          (arr) M x M -- adjacency matrix
        matrix_J_multicell:(arr) NM x NM -- ising interaction matrix for the entire graph
        graph_style:       (str) style of adjacency matrix for cell-cell interactions
            supported: meanfield, general, or lattice_square
        graph_kwargs:      (dict) optional kwargs for call to self.build_adjacency(...)
            Case: meanfield:       N/A
            Case: general:         expects 'prebuilt_adjacency'
            Case: lattice_square:  expects 'initialization_style', 'search_radius'
        autocrine:         (bool) do cells signal/interact with themselves?
        gamma:             (float) cell-cell signalling field strength
        exosome_string:    (str) see valid exosome strings; adds exosomes to field_signal
        exosome_remove_ratio: (float) if exosomes act, how much of the cell state to subsample?
        field_applied:     (arr) NM x T (total_steps) -- the external/manual applied field
        kappa:             (float) scales overall strength of applied/manual field
        flag_housekeeping: (bool) is there a housekeeping component to the manual field?
        seed:              (int) controls all random calls

    Attributes specific to graph state and its dynamics
        graph_kwargs:    (dict) see above
        beta:            (float or arr T) inverse temperature for dynamics
        total_steps:     (int) aka T - total 'lattice steps' to simulate
        current_step:    (int) step counter
        dynamics_blockparallel: (bool) synchronized lattice updates (can use GPU when True)
        plot_period:     (int) lattice plot period


    Data storage attributes
        flag_state_int:  (bool) track and plot the int rep of cell state (asserts low N)
        io_dict:         (dict) stores output file paths according to utils.file_io
        data_dict:       (dict) live data storage for graph state and computed properties
            ...
            ...
            ...TODO

    External methods
        TODO docs

    # TODO what if applied field is different on different nodes? (e.g. top half of lattice?)
    """
    def __init__(self, simsetup, verbose=True, **kwargs):
        if verbose:
            print('Initializing Multicell class object...',)
        # core parameters
        self.simsetup = simsetup
        self.matrix_J = simsetup['J']
        self.matrix_W = simsetup['FIELD_SEND']
        self.num_genes = simsetup['N']
        self.num_celltypes = simsetup['P']
        self.num_cells = kwargs['num_cells']
        self.graph_style = kwargs['graph_style']
        self.autocrine = kwargs.get('autocrine', AUTOCRINE)
        # random seed
        # see https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
        self.seed = kwargs.get('seed',
                               np.random.randint(low=0, high=1e5))    # TODO use throughout and test
        # graph initialization
        # TODO replace lattice by graph everywhere?
        self.graph_kwargs = kwargs.get('graph_kwargs', {})
        self.initialization_style = self.graph_kwargs.get('initialization_style', None)
        self.matrix_A = self.build_adjacency(**self.graph_kwargs)
        self.graph_state_as_cells = self.init_graph_state()
        # initialize matrix_J_multicell (used explicitly for parallel dynamics)
        self.total_spins = self.num_cells * self.num_genes
        self.matrix_J_multicell = self.build_J_multicell()
        # simulation/dynamics properties
        self.beta = kwargs['beta']
        self.current_step = 0
        self.total_steps = kwargs['total_steps']
        self.plot_period = kwargs['plot_period']
        self.dynamics_blockparallel = kwargs.get('flag_blockparallel', BLOCK_UPDATE_LATTICE)    # bool: can use GPU

        # TODO remove?
        self.dynamics_meanfield = ...        # bool: speedup dynamics (check vs graph attributes)
        # field 'signal': cell-cell signalling
        self.gamma = kwargs['gamma']  # aka field_signal_strength
        self.exosome_string = kwargs.get('exosome_string', EXOSTRING)
        self.exosome_remove_ratio = kwargs.get('exosome_remove_ratio', EXOSOME_REMOVE_RATIO)
        # field 'applied': manual/applied field including possible housekeeping gene portion
        self.field_applied = kwargs.get('field_applied', None)
        self.kappa = kwargs.get('kappa', 0.0)
        self.flag_housekeeping = kwargs['flag_housekeeping']
        self.num_housekeeping = self.simsetup['K']
        # metadata
        self.flag_state_int = kwargs.get('flag_state_int', False)
        self.io_dict = self.init_io_dict()      # TODO
        self.data_dict = self.init_data_dict()  # TODO
        # final assertion of attributes
        self.init_assert_and_sanitize()         # TODO
        if verbose:
            print('done')

    # TODO
    def init_assert_and_sanitize(self):

        # beta check
        if isinstance(self.beta, np.ndarray):
            assert self.beta.shape == (self.total_steps,)
        else:
            assert isinstance(self.beta, float)

        # field signal
        assert self.exosome_string in VALID_EXOSOME_STRINGS
        assert 0.0 <= self.exosome_remove_ratio < 1.0
        assert 0.0 <= self.gamma < 10.0

        # field applied (not it has length NM
        if self.num_housekeeping > 0:
            assert self.flag_housekeeping

        if self.field_applied is not None:
            # first: rescale by kappa
            self.field_applied = self.field_applied * self.kappa
            # check that shape is as expected (N*M x timesteps)
            assert len(np.shape(self.field_applied)) in (1, 2)
            if len(self.field_applied.shape) == 2:
                assert self.field_applied.shape[1] == self.total_steps
                if self.field_applied.shape[0] != self.total_spins:
                    # if size N we duplicate it for each cell if needed)
                    assert self.field_applied.shape[0] == self.num_genes
                    print('Warning: field_applied is size N x T but expect NM x T,'
                          'it will be duplicated onto each cell')
                    self.field_applied = np.array(
                        [self.field_applied for _ in range(self.num_cells)]). \
                        reshape(self.total_spins, self.total_steps)
            else:
                if self.field_applied.shape[0] != self.total_spins:
                    # if size N we duplicate it for each cell if needed)
                    assert self.field_applied.shape[0] == self.num_genes
                    print('Warning: field_applied is size N but expect NM x T,'
                          'it will be duplicated onto each cell and each timestep')
                    self.field_applied = np.array([self.field_applied for _ in range(self.num_cells)]).\
                        reshape(self.total_spins)
                self.field_applied = np.array([self.field_applied for _ in range(self.total_steps)]).T
        else:
            self.field_applied = np.zeros((self.total_spins, self.total_steps))
        # TODO remove?
        self.field_applied_current = self.field_applied[:, 0]
        print('field_applied.shape:', self.field_applied.shape)

        # temperature
        if isinstance(self.beta, np.ndarray):
            assert self.beta.shape == self.total_steps
        else:
            assert isinstance(self.beta, float)

        # TODO other checks to reimplement
        # misc checks
        assert type(self.num_steps) is int
        assert type(self.plot_period) is int

        # graph = lattice square case
        if self.graph_style == 'lattice_square':
            self.graph_kwargs['sidelength'] = int(np.sqrt(self.num_cells) + 0.5)
            assert self.graph_kwargs['search_radius'] < 0.5 * self.graph_kwargs['sidelength']
            assert self.graph_kwargs['initialization_style'] in VALID_BUILDSTRINGS

        assert self.graph_style in ['general', 'meanfield', 'lattice_square']
        return

    # TODO cleanup
    def init_io_dict(self):
        io_dict = run_subdir_setup(run_subfolder='multicell_sim')
        info_list = [['memories_path', self.simsetup['memories_path']],
                     ['script', 'multicell_simulate.py'],
                     ['num_cells', self.num_cells],
                     ['total_steps', self.total_steps],
                     ['graph_style', self.graph_style],
                     ['initialization_style', self.initialization_style],
                     ['search_radius', self.graph_kwargs.get('search_radius', None)],
                     ['gamma', self.gamma],
                     ['autocrine', AUTOCRINE],
                     ['exosome_string', self.exosome_string],
                     ['exosome_remove_ratio', self.exosome_remove_ratio],
                     ['kappa', self.kappa],
                     ['field_applied', field_applied],
                     ['flag_housekeeping', self.flag_housekeeping],
                     ['num_housekeeping', self.num_housekeeping],
                     ['beta', beta],
                     ['random_mem', simsetup['random_mem']],
                     ['random_W', simsetup['random_W']],
                     ['dynamics_blockparallel', BLOCK_UPDATE_LATTICE],
                     ]
        runinfo_append(io_dict, info_list, multi=True)
        # conditionally store random mem and W
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_XI.txt',
                   self.simsetup['XI'], delimiter=',', fmt='%d')
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_W.txt',
                   self.matrix_W, delimiter=',', fmt='%.4f')
        return io_dict

    # TODO support for non lattice too ?
    # TODO this can be achieved by hiding the grid aspect of cells -- just use their graph node index
    #  and convert to loc i, j on the fly (or as cell class attribute...)
    def init_data_dict(self):
        # currently has keys like
        """
        data_dict['memory_proj_arr']
        data_dict['grid_state_int']
        data_dict['lattice_energy']
        data_dict['compressibility_full']
        data_dict['memory_proj_arr'][idx] = np.zeros((n * n, duration))"""
        data_dict = {}

        # TODO
        data_dict = prep_lattice_data_dict(
            gridsize, num_steps, list_of_type_idx, buildstring, data_dict)
        # TODO
        if self.flag_state_int:
            data_dict['state_int'] = np.zeros((self.num_cells, self.num_steps), dtype=int)
        return data_dict

    def build_adjacency(self, plot=False):
        """
        Builds node adjacency matrix arr_A based on graph style and hyperparameters
        Supported graph styles
            'meanfield': all nodes are connected
                kwargs: N/A
            'general': will use the supplied array 'prebuilt_adjacency'
                kwargs: 'prebuilt_adjacency'
            'lattice_square': adjacency style for square lattice with r-nearest neighbour radius
                kwargs: 'search_radius'
        """
        if self.graph_style == 'meanfield':
            arr_A = np.ones((self.num_cells, self.num_cells))

        elif self.graph_style == 'general':
            arr_A = self.graph_kwargs.get('prebuilt_adjacency', None)
            if arr_A is None:
                arr_A = adjacency_lattice_general(self.num_cells)

        else:
            assert self.graph_style == 'lattice_square'
            # check the number is a perfect square
            sidelength = int(np.sqrt(self.num_cells) + 0.5)
            self.graph_kwargs['sidelength'] = sidelength
            assert sidelength ** 2 == self.num_cells
            search_radius = self.graph_kwargs['search_radius']
            arr_A = adjacency_lattice_square(sidelength, self.num_cells, search_radius)

        # autocrine loop before returning adjacency matrix (set all diags to 1 or 0)
        # note this will override 'prebuilt_adjacency' in case of self.graph_style == 'general'
        if self.autocrine:
            np.fill_diagonal(arr_A, 1)
        else:
            np.fill_diagonal(arr_A, 0)

        if plot:
            plt.imshow(arr_A)
            plt.title('Cell-cell adjacency matrix')
            plt.show()

        return arr_A

    def build_J_multicell(self, plot=True):
        # TODO test for small num_cell cases e.g. 1 cell, 2 cells up tri lowtri may fail
        assert 1==2

        W_scaled = self.gamma * self.matrix_W
        W_scaled_sym = 0.5 * (W_scaled + W_scaled.T)

        # Term A: self interactions for each cell (diagonal blocks of multicell J_block)
        if self.autocrine:
            J_diag_blocks = np.kron(np.eye(self.num_cells), simsetup['J'] + W_scaled_sym)
        else:
            J_diag_blocks = np.kron(np.eye(self.num_cells), simsetup['J'])

        # Term B of J_multicell (cell-cell interactions)
        # TODO what about exosomes?
        #  should it be accounted for as a Term C here; or
        #  should it be accounted for dynamically
        adjacency_arr_lowtri = np.tril(self.arr_A, k=-1)
        adjacency_arr_uptri = np.triu(self.arr_A, k=1)
        J_offdiag_blocks = np.kron(adjacency_arr_lowtri, W_scaled.T) \
                           + np.kron(adjacency_arr_uptri, W_scaled)

        # build final J multicell matrix
        J_multicell = J_diag_blocks + J_offdiag_blocks

        if plot:
            plt.imshow(J_multicell)
            plt.title('Multicell gene interaction matrix')
            plt.show()

        return J_multicell


    # TODO
    def init_graph_state(self):
        """"
        Initialize the state of each cell in the Multicell dynamical system
        """
        initialization_style = self.graph_kwargs['initialization_style']
        if self.graph_style == 'general':
        elif self.graph_style == 'meanfield':
        else:
            assert self.graph_style == 'lattice_square'

            # setup lattice IC
            flag_uniplots = False
            if buildstring == "mono":
                type_1_idx = 0
                list_of_type_idx = [type_1_idx]
            if buildstring == "dual":
                type_1_idx = 0
                type_2_idx = 1
                list_of_type_idx = [type_1_idx, type_2_idx]
            if buildstring == "memory_sequence":
                flag_uniplots = False
                list_of_type_idx = list(range(simsetup['P']))
                # random.shuffle(list_of_type_idx)  # TODO shuffle or not?
            if buildstring == "random":
                flag_uniplots = False
                list_of_type_idx = list(range(simsetup['P']))
            lattice = build_lattice_main(gridsize, list_of_type_idx, buildstring, simsetup)
            # print list_of_type_idx
        return

    # TODO
    def dynamics_step_parallel(self, applied_field, beta):
        """
        Performs one "graph step": each cell has an opportunity to update its state
        Returns None (operates directly on self.graph_state)
        """
        total_field = np.zeros(self.total_spins)
        internal_field = np.dot(self.matrix_J_multicell, self.graph_state)
        total_field += internal_field
        total_field += applied_field

        # probability that site i will be "up" after the timestep
        prob_on_after_timestep = 1 / (1 + np.exp(-2 * beta * total_field))
        rsamples = np.random.rand(self.total_spins)
        for idx in range(self.total_spins):
            if prob_on_after_timestep[idx] > rsamples[idx]:
                self.graph_state[idx] = 1.0
            else:
                self.graph_state[idx] = -1.0
        return

    # TODO how to support different graph types without using the SpatialCell class?
    def dynamics_step_async(self, applied_field, beta):
        cell_indices = random.shuffle(list(range(self.num_cells)))  # TODO random SEED

        for idx, loc in enumerate(cell_indices):
            cell = lattice[loc[0]][loc[1]]

            if meanfield:
                cellstate_pre = np.copy(cell.get_current_state())
                cell.update_with_meanfield(
                    simsetup['J'], field_global, beta=beta, app_field=app_field_step,
                    field_signal_strength=ext_field_strength,
                    field_app_strength=app_field_strength)
                # TODO update field_avg based on new state TODO test
                state_total += (cell.get_current_state() - cellstate_pre)
                state_total_01 = (state_total + num_cells) / 2
                field_global = np.dot(simsetup['FIELD_SEND'], state_total_01)
                print(field_global)
                print(state_total)
            else:
                cell.update_with_signal_field(
                    lattice, SEARCH_RADIUS_CELL, n, simsetup['J'], simsetup, beta=beta,
                    exosome_string=exosome_string,
                    exosome_remove_ratio=exosome_remove_ratio,
                    field_signal_strength=ext_field_strength, field_app=app_field_step,
                    field_app_strength=app_field_strength)

            # update cell specific datdict entries for the current timestep
            cell_proj = update_datadict_timestep_cell(lattice, loc, memory_idx_list, turn)

            if turn % (
                    120 * plot_period) == 0:  # proj vis of each cell (slow; every k steps)
                fig, ax, proj = cell. \
                    plot_projection(simsetup['A_INV'], simsetup['XI'], proj=cell_proj,
                                    use_radar=False, pltdir=io_dict['latticedir'])

        # compute lattice properties (assess global state)
        # TODO 1 - consider lattice energy at each cell update (not lattice update)
        # TODO 2 - speedup lattice energy calc by using info from state update calls...
        update_datadict_timestep_global(lattice, turn)

        if turn % plot_period == 0:  # plot the lattice
            lattice_projection_composite(
                lattice, turn, n, io_dict['latticedir'], simsetup, state_int=state_int)
            reference_overlap_plotter(
                lattice, turn, n, io_dict['latticedir'], simsetup, state_int=state_int)
            # if flag_uniplots:
            #    for mem_idx in memory_idx_list:
            #        lattice_uniplotter(lattice, turn, n, io_dict['latticedir'], mem_idx, simsetup)

        # update class attributes TODO
        self.current_step += 1
        self.

    # TODO
    def dynamics_step_async(self):...

    # TODO periodic storage of state properties in data_dict
    def datastore_state(self):
        return

    # TODO periodic saving to file
    def save_state(self):
        return

    # TODO periodic plot to file
    def visualize_state(self):
        return

    # TODO total_steps handles cases of parallel and async
    def dynamics_full(self):
        """
        Form of data_dict:
            {'memory_proj_arr':
                {memory_idx: np array [N x num_steps] of projection each grid cell onto memory idx}
             'grid_state_int': n x n x num_steps of int at each site
               (int is inverse of binary string from state)
        Notes:
            -can replace update_with_signal_field with update_state to simulate ensemble
            of non-interacting n**2 cells
        """

        def update_datadict_timestep_cell(lattice, loc, memory_idx_list, timestep_idx):
            cell = lattice[loc[0]][loc[1]]
            # store the projections
            proj = cell.get_memories_projection(simsetup['A_INV'], simsetup['XI'])
            for mem_idx in memory_idx_list:
                data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], timestep_idx] = proj[mem_idx]
            # store the integer representation of the state
            if state_int:
                data_dict['grid_state_int'][loc[0], loc[1], timestep_idx] = cell.get_current_label()
            return proj

        def update_datadict_timestep_global(lattice, timestep_idx):
            data_dict['lattice_energy'][timestep_idx, :] = calc_lattice_energy(
                lattice, simsetup, app_field_step, app_field_strength, ext_field_strength,
                SEARCH_RADIUS_CELL,
                exosome_remove_ratio, exosome_string, meanfield)
            data_dict['compressibility_full'][timestep_idx, :] = calc_compression_ratio(
                get_state_of_lattice(lattice, simsetup, datatype='full'),
                eta_0=None, datatype='full', elemtype=np.int, method='manual')

        def lattice_plot_init(lattice, memory_idx_list):
            lattice_projection_composite(lattice, 0, n, io_dict['latticedir'], simsetup,
                                         state_int=state_int)
            reference_overlap_plotter(lattice, 0, n, io_dict['latticedir'], simsetup,
                                      state_int=state_int)
            if flag_uniplots:
                for mem_idx in memory_idx_list:
                    lattice_uniplotter(lattice, 0, n, io_dict['latticedir'], mem_idx, simsetup)

        def meanfield_global_field():
            # TODO careful: not clear best way to update exo field as cell state changes in a time step,
            #  refactor exo fn?
            assert exosome_string == 'no_exo_field'
            print('Initializing mean field...')
            # TODO decide if want scale factor to be rescaled by total popsize (i.e. *mean*field or total field?)
            state_total = np.zeros(simsetup['N'])
            field_global = np.zeros(simsetup['N'])
            # TODO ok that cell is neighbour with self as well? should remove diag
            neighbours = [[a, b] for a in range(len(lattice[0])) for b in range(len(lattice))]
            if simsetup['FIELD_SEND'] is not None:
                for loc in neighbours:
                    state_total += lattice[loc[0]][loc[1]].get_current_state()
                state_total_01 = (state_total + num_cells) / 2
                field_paracrine = np.dot(simsetup['FIELD_SEND'], state_total_01)
                field_global += field_paracrine
            if exosome_string != 'no_exo_field':
                field_exo, _ = lattice[0][0]. \
                    get_local_exosome_field(lattice, None, None, exosome_string=exosome_string,
                                            exosome_remove_ratio=exosome_remove_ratio,
                                            neighbours=neighbours)
                field_global += field_exo
            return field_global

        def parallel_block_update_lattice(J_block, s_block_current, applied_field_block,
                                          total_spins):


        def build_block_state_from_lattice(lattice, n, num_cells, simsetup):
            N = simsetup['N']
            total_spins = num_cells * N
            s_block = np.zeros(total_spins)
            for a in range(num_cells):
                arow, acol = lattice_square_int_to_loc(a, n)
                cellstate = np.copy(
                    lattice[arow][acol].get_current_state())
                s_block[a * N: (a + 1) * N] = cellstate
            return s_block

        def update_lattice_using_state_block(lattice, n, num_cells, simsetup, s_block):
            N = simsetup['N']
            total_spins = num_cells * N
            for a in range(num_cells):
                arow, acol = lattice_square_int_to_loc(a, n)
                cell = lattice[arow][acol]
                cellstate = np.copy(s_block[a * N: (a + 1) * N])
                # update cell state specifically
                lattice[arow][acol].state = cellstate
                # update whole cell state array (append new state for the current timepoint)
                state_array_ext = np.zeros((N, np.shape(cell.state_array)[1] + 1))
                state_array_ext[:, :-1] = cell.state_array  # TODO: make sure don't need array copy
                state_array_ext[:, -1] = cellstate
                cell.state_array = state_array_ext
                # update steps attribute
                cell.steps += 1
            return lattice

        # input processing
        n, num_cells, app_field, app_field_step = input_checks(app_field)
        cell_locations = get_cell_locations(lattice, n)
        loc_to_idx = {pair: idx for idx, pair in enumerate(cell_locations)}
        memory_idx_list = list(data_dict['memory_proj_arr'].keys())

        # assess & plot initial state
        for loc in cell_locations:
            update_datadict_timestep_cell(lattice, loc, memory_idx_list, 0)
        update_datadict_timestep_global(lattice, 0)  # measure initial state
        lattice_plot_init(lattice, memory_idx_list)  # plot initial state

        # special update method for meanfield case (infinite search radius)
        if meanfield:
            state_total, field_global = meanfield_global_field()

        if BLOCK_UPDATE_LATTICE:
            # Pseudo 3: applied_field_block timeseries or None
            # TODO

        if self.dynamics_blockparallel:

            for turn in range(1, self.total_steps):
                print('Dynamics step: ', self.current_step + 1)
                self.dynamics_step_parallel()

                # TODO applied field block
                # block update rule for the lattice (represented by state_block)
                state_block = parallel_block_update_lattice(J_block, state_block, None, total_spins)

                # TODO applied field block

                # TODO not needed anymore?
                #  better usage of the lattice object, this refilling is inefficient
                #  especially the state array part
                # fill lattice object based on updated state_block
                lattice = update_lattice_using_state_block(
                    lattice, n, num_cells, simsetup, state_block)

            else:
                random.shuffle(cell_locations)
                for idx, loc in enumerate(cell_locations):
                    cell = lattice[loc[0]][loc[1]]
                    if app_field is not None:
                        app_field_step = app_field[:, turn]
                    if meanfield:
                        cellstate_pre = np.copy(cell.get_current_state())
                        cell.update_with_meanfield(
                            simsetup['J'], field_global, beta=beta, app_field=app_field_step,
                            field_signal_strength=ext_field_strength,
                            field_app_strength=app_field_strength)
                        # TODO update field_avg based on new state TODO test
                        state_total += (cell.get_current_state() - cellstate_pre)
                        state_total_01 = (state_total + num_cells) / 2
                        field_global = np.dot(simsetup['FIELD_SEND'], state_total_01)
                        print(field_global)
                        print(state_total)
                    else:
                        cell.update_with_signal_field(
                            lattice, SEARCH_RADIUS_CELL, n, simsetup['J'], simsetup, beta=beta,
                            exosome_string=exosome_string,
                            exosome_remove_ratio=exosome_remove_ratio,
                            field_signal_strength=ext_field_strength, field_app=app_field_step,
                            field_app_strength=app_field_strength)

                    # update cell specific datadict entries for the current timestep
                    cell_proj = update_datadict_timestep_cell(lattice, loc, memory_idx_list, turn)

                    if turn % (
                            120 * plot_period) == 0:  # proj vis of each cell (slow; every k steps)
                        fig, ax, proj = cell. \
                            plot_projection(simsetup['A_INV'], simsetup['XI'], proj=cell_proj,
                                            use_radar=False, pltdir=io_dict['latticedir'])

            # compute lattice properties (assess global state)
            # TODO 1 - consider lattice energy at each cell update (not lattice update)
            # TODO 2 - speedup lattice energy calc by using info from state update calls...
            update_datadict_timestep_global(lattice, turn)

            if turn % self.plot_period == 0:  # plot the lattice
                lattice_projection_composite(
                    lattice, turn, n, self.io_dict['latticedir'], self.simsetup,
                    state_int=self.flag_state_int)
                reference_overlap_plotter(
                    lattice, turn, n, self.io_dict['latticedir'], self.simsetup,
                    state_int=self.flag_state_int)
                # if flag_uniplots:
                #    for mem_idx in memory_idx_list:
                #        lattice_uniplotter(lattice, turn, n, io_dict['latticedir'], mem_idx, simsetup)

    # TODO
    def standard_simulation(self):
        # run the simulation
        self.dynamics_full()

        # """check the data dict"""
        self.plot_datadict_memory_proj()

        # """write and plot cell state timeseries"""
        # TODO convert to 'write data dict' and 'plot data dict' fn calls
        # write_state_all_cells(lattice, io_dict['datadir'])
        self.mainloop_data_dict_write()
        self.mainloop_data_dict_plot()

        print("\nMulticell simulation complete - output in %s" % self.io_dict['basedir'])
        return

    def get_cell_state(self, cell_idx):
        assert 0 < cell_idx < self.num_cells
        a = self.num_genes * cell_idx
        b = self.num_genes * (cell_idx + 1)
        return self.graph_state[a:b]

    def plot_datadict_memory_proj(self):
        # check the data dict
        for data_idx, memory_idx in enumerate(self.data_dict['memory_proj_arr'].keys()):
            print(self.data_dict['memory_proj_arr'][memory_idx])
            plt.plot(self.data_dict['memory_proj_arr'][memory_idx].T)
            plt.ylabel(
                'Projection of all cells onto type: %s' % self.simsetup['CELLTYPE_LABELS'][memory_idx])
            plt.xlabel('Time (full lattice steps)')
            plt.savefig(
                self.io_dict['plotdatadir'] + os.sep + '%s_%s_t%d_proj%d_remove%.2f_exo%.2f.png' %
                (self.exosome_string, self.initialization_style, self.total_steps, memory_idx,
                 self.exosome_remove_ratio, field_signal_strength))
            plt.clf()  # plt.show()
        return

    def mainloop_data_dict_write(self):
        # TODO make savestring property to append to files and plots?
        if self.flag_state_int:
            write_grid_state_int(self.data_dict['grid_state_int'], self.io_dict['datadir'])

        if 'lattice_energy' in list(self.data_dict.keys()):
            write_general_arr(self.data_dict['lattice_energy'], self.io_dict['datadir'],
                              'lattice_energy', txt=True, compress=False)

        if 'compressibility_full' in list(data_dict.keys()):
            write_general_arr(self.data_dict['compressibility_full'], self.io_dict['datadir'],
                              'compressibility_full', txt=True, compress=False)

    def mainloop_data_dict_plot(self):
        if 'lattice_energy' in list(self.data_dict.keys()):
            plt.plot(self.data_dict['lattice_energy'][:, 0], '--ok', label=r'$H_{\mathrm{total}}$')
            plt.plot(self.data_dict['lattice_energy'][:, 1], '--b', alpha=0.7,
                     label=r'$H_{\mathrm{self}}$')
            plt.plot(self.data_dict['lattice_energy'][:, 2], '--g', alpha=0.7,
                     label=r'$H_{\mathrm{app}}$')
            plt.plot(self.data_dict['lattice_energy'][:, 3], '--r', alpha=0.7,
                     label=r'$H_{\mathrm{pairwise}}$')
            plt.plot(self.data_dict['lattice_energy'][:, 0] -
                     self.data_dict['lattice_energy'][:, 2], '--o',
                     color='gray', label=r'$H_{\mathrm{total}} - H_{\mathrm{app}}$')
            plt.title(r'Multicell hamiltonian over time')
            plt.ylabel(r'Lattice energy')
            plt.xlabel(r'$t$ (lattice steps)')
            plt.legend()
            plt.savefig(self.io_dict['plotdatadir'] + os.sep +'hamiltonian.png')

            # zoom on relevant part
            ylow = min(np.min(self.data_dict['lattice_energy'][:, [1, 3]]),
                       np.min(self.data_dict['lattice_energy'][:, 0] -
                              self.data_dict['lattice_energy'][:, 2]))
            yhigh = max(np.max(self.data_dict['lattice_energy'][:, [1, 3]]),
                        np.max(self.data_dict['lattice_energy'][:, 0] -
                               self.data_dict['lattice_energy'][:, 2]))
            plt.ylim(ylow - 0.1, yhigh + 0.1)
            plt.savefig(self.io_dict['plotdatadir'] + os.sep + 'hamiltonianZoom.png')
            plt.clf()  # plt.show()

        if 'compressibility_full' in list(self.data_dict.keys()):
            assert 1==2
            # TODO non lattice

            plt.plot(self.data_dict['compressibility_full'][:, 0], '--o', color='orange')
            plt.title(r'File compressibility ratio of the full lattice spin state')
            plt.ylabel(r'$\eta(t)/\eta_0$')
            plt.axhline(y=1.0, ls='--', color='k')

            ref_0 = calc_compression_ratio(
                x=np.zeros((len(lattice), len(lattice[0]), simsetup['N']), dtype=int),
                method='manual',
                eta_0=data_dict['compressibility_full'][0, 2], datatype='full', elemtype=np.int)
            ref_1 = calc_compression_ratio(
                x=np.ones((len(lattice), len(lattice[0]), simsetup['N']), dtype=int),
                method='manual',
                eta_0=data_dict['compressibility_full'][0, 2], datatype='full', elemtype=np.int)
            plt.axhline(y=ref_0[0], ls='-.', color='gray')
            plt.axhline(y=ref_1[0], ls='-.', color='blue')
            print(ref_0, ref_0, ref_0, ref_0, 'is', ref_0, 'vs', ref_1)
            plt.xlabel(r'$t$ (lattice steps)')
            plt.ylim(-0.05, 1.01)
            plt.savefig(self.io_dict['plotdatadir'] + os.sep + 'compresibility.png')
            plt.clf()  # plt.show()

def run_mc_sim(lattice, num_lattice_steps, data_dict, io_dict, simsetup, exosome_string=EXOSTRING,
               exosome_remove_ratio=0.0, ext_field_strength=FIELD_SIGNAL_STRENGTH, app_field=None,
               app_field_strength=FIELD_APPLIED_STRENGTH, beta=BETA, plot_period=LATTICE_PLOT_PERIOD,
               flag_uniplots=False, state_int=False, meanfield=MEANFIELD):

    return lattice, data_dict, io_dict


if __name__ == '__main__':
    curated = True
    random_mem = False
    random_W = True
    simsetup = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)

    # setup: lattice sim core parameters
    n = 6                 # global GRIDSIZE
    steps = 10            # global NUM_LATTICE_STEPS
    buildstring = "mono"  # init condition: mono/dual/memory_sequence/random
    meanfield = False     # True: infinite signal distance (no neighbor search; track mean field)
    plot_period = 1
    state_int = True
    beta = BETA  # 2.0

    # setup: signalling field (exosomes + cell-cell signalling via W matrix)
    exosome_string = "no_exo_field"   # on/off/all/no_exo_field; 'off' = send info only 'off' genes
    fieldprune = 0.0                  # amount of exo field idx to randomly prune from each cell
    field_signal_strength = 90 * 0.1  #  / (n*n) * 8   # global GAMMA = field_strength_signal tunes exosomes AND sent field

    # setup: applied/manual field (part 1)
    #field_applied = construct_app_field_from_genes(
    #    IPSC_EXTENDED_GENES_EFFECTS, simsetup['GENE_ID'], num_steps=steps)  # size N x steps or None
    field_applied = None
    field_applied_strength = 1.0

    # setup: applied/manual field (part 2) -- optionally add housekeeping field with strength Kappa
    flag_housekeeping = False
    field_housekeeping_strength = 0.0  # aka Kappa
    assert not flag_housekeeping
    if flag_housekeeping:
        assert field_housekeeping_strength > 0
        # housekeeping auto (via model extension)
        field_housekeeping = np.zeros(simsetup['N'])
        if simsetup['K'] > 0:
            field_housekeeping[-simsetup['K']:] = 1.0
            print(field_applied)
        else:
            print('Note gene 0 (on), 1 (on), 2 (on) are HK in A1 memories')
            print('Note gene 4 (off), 5 (on) are HK in C1 memories')
            field_housekeeping[4] = 1.0
            field_housekeeping[5] = 1.0
        if field_applied is not None:
            field_applied += field_housekeeping_strength * field_housekeeping
        else:
            field_applied = field_housekeeping_strength * field_housekeeping
    else:
        field_housekeeping = None
        field_housekeeping_strength = 0.0

    mc_sim_wrapper(
        simsetup, gridsize=n, num_steps=steps, buildstring=buildstring,
        exosome_string=exosome_string, exosome_remove_ratio=fieldprune,
        field_signal_strength=field_signal_strength,
        field_applied=field_applied,  field_applied_strength=field_applied_strength,
        flag_housekeeping=flag_housekeeping,
        beta=beta, plot_period=plot_period, state_int=state_int,
        meanfield=meanfield)
    """
    for beta in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 4.0, 5.0, 10.0, 100.0]:
        mc_sim_wrapper(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
               field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
               app_field_strength=app_field_strength, beta=beta, plot_period=plot_period, state_int=state_int, meanfield=meanfield)
    """
