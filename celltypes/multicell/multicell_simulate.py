import numpy as np
import os
import random
import shutil
import matplotlib.pyplot as plt

from multicell.graph_adjacency import \
    lattice_square_int_to_loc, adjacency_lattice_square, adjacency_general, general_exosome_field, \
    general_paracrine_field
from multicell.multicell_constants import \
    VALID_BUILDSTRINGS, VALID_EXOSOME_STRINGS, EXOSTRING, EXOSOME_REMOVE_RATIO, \
    BLOCK_UPDATE_LATTICE, AUTOCRINE
from multicell.multicell_lattice import \
    build_lattice_main, write_grid_state_int_alt
from multicell.multicell_metrics import \
    calc_compression_ratio, calc_graph_energy
from multicell.multicell_visualize import \
    graph_lattice_uniplotter, graph_lattice_reference_overlap_plotter, graph_lattice_projection_composite
from singlecell.singlecell_class import Cell
from singlecell.singlecell_functions import \
    state_memory_overlap_alt, state_memory_projection_alt, state_to_label
from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import run_subdir_setup, runinfo_append, write_general_arr


class Multicell:
    """
    Primary object for multicell simulations.
    Current state of the graph is tracked using attribute: graph_state
     - graph_state_arr is an [N*M x T] 2D array
     - get state of a specific cell using get_cell_state(idx)

    simsetup: dictionary created using singlecell_simsetup() in singlecell_simsetup.py

    Expected kwargs for __init__
        num_cells
        graph_style
        gamma
        autocrine
        flag_housekeeping
        beta
        num_steps
        plot_period

    Optional kwargs for __init__
        graph_kwargs
        exosome_string
        exosome_remove_ratio
        field_applied
        kappa
        flag_state_int
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
        run_label:       (str) override the datetime format for multicell_sim sub-directories
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
        self.total_spins = self.num_cells * self.num_genes
        self.graph_style = kwargs['graph_style']
        self.autocrine = kwargs.get('autocrine', AUTOCRINE)
        # random seed
        # see https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752
        self.seed = kwargs.get('seed',
                               np.random.randint(low=0, high=1e5))    # TODO use throughout and test
        # field 'signal': cell-cell signalling
        self.gamma = kwargs['gamma']  # aka field_signal_strength
        self.exosome_string = kwargs.get('exosome_string', EXOSTRING)
        self.exosome_remove_ratio = kwargs.get('exosome_remove_ratio', EXOSOME_REMOVE_RATIO)
        # field 'applied': manual/applied field including possible housekeeping gene portion
        self.field_applied = kwargs.get('field_applied', None)
        self.kappa = kwargs.get('kappa', 0.0)
        self.flag_housekeeping = kwargs['flag_housekeeping']
        self.num_housekeeping = self.simsetup['K']
        # simulation/dynamics properties
        self.beta = kwargs['beta']
        self.current_step = 0
        self.total_steps = kwargs['total_steps']
        self.plot_period = kwargs['plot_period']
        self.dynamics_blockparallel = kwargs.get('flag_blockparallel', BLOCK_UPDATE_LATTICE)  # GPU
        # bool: speedup dynamics (check vs graph attributes) TODO reimplement?
        #self.dynamics_meanfield = ...
        # graph initialization
        # TODO replace lattice by graph everywhere?
        self.graph_kwargs = kwargs.get('graph_kwargs', {})
        self.initialization_style = self.graph_kwargs.get('initialization_style', None)
        self.matrix_A = self.build_adjacency()
        self.graph_state_arr = np.zeros((self.total_spins, self.total_steps), dtype=int)
        self.graph_state_arr[:, 0] = self.init_graph_state()
        # initialize matrix_J_multicell (used explicitly for parallel dynamics)
        self.matrix_J_multicell = self.build_J_multicell()
        # metadata
        self.run_label = kwargs.get('run_label', None)
        self.flag_state_int = kwargs.get('flag_state_int', False)
        self.io_dict = self.init_io_dict()      # TODO
        self.data_dict = self.init_data_dict()  # TODO
        # final assertion of attributes
        self.init_assert_and_sanitize()         # TODO
        if verbose:
            print('done')

    # TODO cleanup
    def init_assert_and_sanitize(self):

        # field signal
        assert self.exosome_string in VALID_EXOSOME_STRINGS
        assert 0.0 <= self.exosome_remove_ratio < 1.0
        assert 0.0 <= self.gamma <= 100.0

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
        print('field_applied.shape:', self.field_applied.shape)         # TODO remove?

        # beta (temperature) check
        if isinstance(self.beta, np.ndarray):
            assert self.beta.shape == (self.total_steps,)
        else:
            assert isinstance(self.beta, float)
            self.beta = np.array([self.beta for _ in range(self.total_steps)])

        # TODO other checks to reimplement
        # misc checks
        assert type(self.total_steps) is int
        assert type(self.plot_period) is int

        # graph = lattice square case
        if self.graph_style == 'lattice_square':
            self.graph_kwargs['sidelength'] = int(np.sqrt(self.num_cells) + 0.5)
            assert self.graph_kwargs['search_radius'] <= 0.5 * self.graph_kwargs['sidelength']
            assert self.graph_kwargs['initialization_style'] in VALID_BUILDSTRINGS

        assert self.graph_style in ['general', 'meanfield', 'lattice_square']
        return

    # TODO cleanup
    def init_io_dict(self):
        io_dict = run_subdir_setup(run_subfolder='multicell_sim', timedir_override=self.run_label)
        info_list = [['seed', self.seed],
                     ['memories_path', self.simsetup['memories_path']],
                     ['script', 'multicell_simulate_old.py'],
                     ['num_cells', self.num_cells],
                     ['total_steps', self.total_steps],
                     ['graph_style', self.graph_style],
                     ['initialization_style', self.initialization_style],
                     ['search_radius', self.graph_kwargs.get('search_radius', None)],
                     ['gamma', self.gamma],
                     ['autocrine', self.autocrine],
                     ['exosome_string', self.exosome_string],
                     ['exosome_remove_ratio', self.exosome_remove_ratio],
                     ['kappa', self.kappa],
                     ['field_applied', field_applied],
                     ['flag_housekeeping', self.flag_housekeeping],
                     ['num_housekeeping', self.num_housekeeping],
                     ['beta', beta],
                     ['random_mem', self.simsetup['random_mem']],
                     ['random_W', self.simsetup['random_W']],
                     ['dynamics_blockparallel', self.dynamics_blockparallel],
                     ]
        runinfo_append(io_dict, info_list, multi=True)
        # conditionally store random mem and W
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_XI.txt',
                   self.simsetup['XI'], delimiter=',', fmt='%d')
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_W.txt',
                   self.matrix_W, delimiter=',', fmt='%.4f')
        return io_dict

    # TODO full + support for non lattice too
    # TODO this can be achieved by hiding the grid aspect of cells -- just use their graph node index
    #  and convert to loc i, j on the fly (or as cell class attribute...)
    def init_data_dict(self):
        """
        Currently has the core keys
            memory_proj_arr:           arr M x P x T
            memory_overlap_arr:        arr M x P x T
            graph_energy:              arr T x 5
            compressibility_full:      arr T x 3
        Optional keys
            cell_state_int:            arr 1 x T
        """
        data_dict = {}
        # stores: H_multi, H_self, H_app, H_pairwise_scaled
        data_dict['graph_energy'] = np.zeros((self.total_steps, 5))
        # stores: compressibility ratio, eta, eta0
        data_dict['compressibility_full'] = np.zeros((self.total_steps, 3))
        # stores: memory projection/overlap for each memory for each cell
        data_dict['memory_proj_arr'] = np.zeros(
            (self.num_cells, self.simsetup['P'], self.total_steps))
        data_dict['memory_overlap_arr'] = np.zeros(
            (self.num_cells, self.simsetup['P'], self.total_steps))
        # stores: memory projection for each memory for each cell
        if self.flag_state_int:
            data_dict['cell_state_int'] = np.zeros((self.num_cells, self.total_steps), dtype=int)
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
                arr_A = adjacency_general(self.num_cells)

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

    def build_J_multicell(self, plot=False):

        W_scaled = self.gamma * self.matrix_W
        W_scaled_sym = 0.5 * (W_scaled + W_scaled.T)

        # Term A: self interactions for each cell (diagonal blocks of multicell J_block)
        if self.autocrine:
            J_diag_blocks = np.kron(np.eye(self.num_cells), self.simsetup['J'] + W_scaled_sym)
        else:
            J_diag_blocks = np.kron(np.eye(self.num_cells), self.simsetup['J'])

        # Term B of J_multicell (cell-cell interactions)
        # TODO what about exosomes?
        #  should it be accounted for as a Term C here; or
        #  should it be accounted for dynamically
        adjacency_arr_lowtri = np.tril(self.matrix_A, k=-1)
        adjacency_arr_uptri = np.triu(self.matrix_A, k=1)
        J_offdiag_blocks = np.kron(adjacency_arr_lowtri, W_scaled.T) \
                           + np.kron(adjacency_arr_uptri, W_scaled)

        # build final J multicell matrix
        J_multicell = J_diag_blocks + J_offdiag_blocks

        if plot:
            plt.imshow(J_multicell)
            plt.title('Multicell gene interaction matrix')
            plt.show()
        return J_multicell

    # TODO fix this old approach
    def init_graph_state(self):
        """"
        Initialize the state of each cell in the Multicell dynamical system
        """
        assert self.graph_style == 'lattice_square' # TODO not this
        initialization_style = self.graph_kwargs['initialization_style']

        if self.graph_style == 'general':
            pass # TODO
        elif self.graph_style == 'meanfield':
            pass # TODO
        else:
            assert self.graph_style == 'lattice_square'
            sidelength = self.graph_kwargs['sidelength']
            buildstring = initialization_style

            # 1) use old lattice initializer
            if buildstring == "mono":
                type_1_idx = 0
                list_of_type_idx = [type_1_idx]
            if buildstring == "dual":
                type_1_idx = 0
                type_2_idx = 1
                list_of_type_idx = [type_1_idx, type_2_idx]
            if buildstring == "memory_sequence":
                list_of_type_idx = list(range(self.simsetup['P']))
                # random.shuffle(list_of_type_idx)  # TODO shuffle or not?
            if buildstring == "random":
                list_of_type_idx = list(range(self.simsetup['P']))
            lattice = build_lattice_main(
                sidelength, list_of_type_idx, initialization_style, self.simsetup, seed=self.seed)
            # print list_of_type_idx

            # 2) now convert the lattice to a graph state (tall NM vector)
            graph_state = self.TEMP_graph_state_from_lattice(lattice, sidelength)

        return graph_state

    # TODO remove asap
    def TEMP_graph_state_from_lattice(self, lattice, sidelength):
        print('call to TEMP_graph_state_from_lattice() -- remove this function')
        assert self.graph_style == 'lattice_square'
        N = self.num_genes
        s_block = np.zeros(self.total_spins)
        for a in range(self.num_cells):
            arow, acol = lattice_square_int_to_loc(a, sidelength)
            cellstate = np.copy(
                lattice[arow][acol].get_current_state())
            s_block[a * N: (a+1) * N] = cellstate
        return s_block

    def step_dynamics_parallel(self, next_step, field_applied, beta):
        """
        Performs one "graph step": each cell has an opportunity to update its state
        Returns None (operates directly on self.graph_state_arr)
        """
        total_field = np.zeros(self.total_spins)
        internal_field = np.dot(self.matrix_J_multicell, self.graph_state_arr[:, next_step - 1])
        total_field += internal_field
        total_field += field_applied

        # probability that site i will be "up" after the timestep
        prob_on_after_timestep = 1 / (1 + np.exp(-2 * beta * total_field))
        np.random.seed(self.seed); rsamples = np.random.rand(self.total_spins)
        for idx in range(self.total_spins):
            if prob_on_after_timestep[idx] > rsamples[idx]:
                self.graph_state_arr[idx, next_step] = 1.0
            else:
                self.graph_state_arr[idx, next_step] = -1.0
        return

    # TODO how to support different graph types without using the SpatialCell class?
    #  currently all old code from multicell+simulate
    # TODO Meanfield setting previously used for speedups, but removed now:
    def step_dynamics_async(self, next_step, field_applied, beta):

        cell_indices = list(range(self.num_cells))
        random.seed(self.seed); random.shuffle(cell_indices)

        # need to move current state to next step index, then operate on that column 'in place'
        self.graph_state_arr[:, next_step] = np.copy(self.graph_state_arr[:, next_step - 1])

        for idx, node_idx in enumerate(cell_indices):
            cell_state = self.get_cell_state(node_idx, next_step)
            spin_idx_low = self.num_genes * node_idx
            spin_idx_high = self.num_genes * (node_idx + 1)

            # extract applied field specific to cell at given node
            field_applied_on_cell = field_applied[spin_idx_low: spin_idx_high]

            #  note that a cells neighboursa are the ones which 'send' to it
            #  if A_ij = 1, then there is a connection from i to j
            #  to get all the senders to cell i, we need to look at col i
            graph_neighbours_col = self.matrix_A[:, node_idx]
            graph_neighbours = [node for node, i in enumerate(graph_neighbours_col) if i == 1]

            # signaling field part 1
            field_signal_exo, _ = general_exosome_field(
                self, node_idx, next_step, neighbours=graph_neighbours)
            # signaling field part 2
            field_signal_W = general_paracrine_field(
                self, node_idx, next_step, flag_01=False, neighbours=graph_neighbours)
            # sum the two field contributions
            field_signal_unscaled = field_signal_exo + field_signal_W
            field_signal = self.gamma * field_signal_unscaled

            dummy_cell = Cell(np.copy(cell_state), 'fake_cell',
                              self.simsetup['CELLTYPE_LABELS'],
                              self.simsetup['GENE_LABELS'],
                              state_array=None,
                              steps=None)
            # TODO pass seed to update call or pass to new attribute Cell
            dummy_cell.update_state(
                beta=beta,
                intxn_matrix=self.matrix_J,
                field_signal=field_signal,
                field_signal_strength=1.0,
                field_applied=field_applied_on_cell,
                field_applied_strength=1.0,
                seed=self.seed)

            self.graph_state_arr[spin_idx_low:spin_idx_high, next_step] = \
                dummy_cell.get_current_state()
            """
            if turn % (
                    120 * plot_period) == 0:  # proj vis of each cell (slow; every k steps)
                fig, ax, proj = cell. \
                    plot_projection(simsetup['A_INV'], simsetup['XI'], proj=cell_proj,
                                    use_radar=False, pltdir=io_dict['latticedir'])"""
        return

    def step_datadict_update_global(self, step, fill_to_end=False):
        """
        Following a simulation multicell step, update the data dict.
          fill_to_end: if True, fill all steps after 'step' with the calculated value (for early FP)
        See init_data_dict() for additional documentation.
        """
        if fill_to_end:
            step_indices = range(step, self.total_steps)
        else:
            step_indices = step
        # 1) compressibility statistics
        graph_state_01 = ((1 + self.graph_state_arr[:, step])/2).astype(int)
        comp_ratio = calc_compression_ratio(
            graph_state_01, eta_0=None, datatype='full', elemtype=np.int, method='manual')
        self.data_dict['compressibility_full'][step_indices, :] = comp_ratio
        # 2) energy statistics
        energy_values = calc_graph_energy(self, step, norm=True)
        self.data_dict['graph_energy'][step_indices, :] = energy_values
        # 3) node-wise projection on the encoded singlecell types
        for i in range(self.num_cells):
            cell_state = self.get_cell_state(i, step)
            # get the projections/overlaps
            overlap_vec = state_memory_overlap_alt(cell_state, self.num_genes, self.simsetup['XI'])
            proj_vec = state_memory_projection_alt(
                cell_state, self.simsetup['A_INV'], self.num_genes, self.simsetup['XI'],
                overlap_vec=overlap_vec)
            # store the projections/overlaps
            self.data_dict['memory_overlap_arr'][i, :, step_indices] = overlap_vec
            self.data_dict['memory_proj_arr'][i, :, step_indices] = proj_vec
            # 4) node-wise storage of the integer representation of the state
            if self.flag_state_int:
                self.data_dict['cell_state_int'][i, step_indices] = state_to_label(tuple(cell_state))
        return

    def step_state_save(self, step, prefix='X', suffix='', cells_as_cols=True):
        """
        Saves graph state at timestep as a txt file
            cells_as_cols: reshape to 2D arr so each column reps a cell
        To load:
            X = np.loadtxt(fpath)
            num_genes, num_cells = X.shape
            X = X.reshape((num_genes*num_cells), order='F')
        """
        outdir = self.io_dict['statesdir']
        X = self.graph_state_arr[:, step]
        fpath = outdir + os.sep + '%s_%d%s.txt' % (prefix, step, suffix)
        fmt = '%d'
        if cells_as_cols:
            np.savetxt(fpath, X.reshape((self.num_genes, self.num_cells), order='F'), fmt=fmt)
        else:
            np.savetxt(fpath, X, fmt=fmt)
        return fpath

    # TODO remove lattice square assert (generalize)
    def step_state_visualize(self, step, flag_uniplots=False):
        assert self.graph_style == 'lattice_square'
        nn = self.graph_kwargs['sidelength']

        # plot type A
        graph_lattice_projection_composite(self, step, use_proj=False)
        graph_lattice_projection_composite(self, step, use_proj=True)
        # plot type B
        graph_lattice_reference_overlap_plotter(self, step)
        # plot type C
        if flag_uniplots:
            for mu in range(self.simsetup['P']):
                graph_lattice_uniplotter(self, step, nn, self.io_dict['latticedir'],
                                         mu, use_proj=False)
                graph_lattice_uniplotter(self, step, nn, self.io_dict['latticedir'],
                                         mu, use_proj=True)
        return

    def check_if_fixed_point(self, state_to_compare, step, msg=None):
        """
        :param state_to_compare: to check array equality with current graph state
        :return: bool
        """
        if np.all(self.graph_state_arr[:, step] == state_to_compare):
            if msg is not None:
                print(msg)
            return True
        else:
            return False

    # TODO handle case of dynamics_async
    def dynamics_full(self, flag_visualize=True, flag_datastore=True, end_at_fp=True):
        """
            flag_visualize: optionally force off datadict updates (speedup)
            flag_datastore: optionally force off calls to step_state_visualize updates (speedup)
            end_at_fp:      terminate before self.total_steps if a fixed point is reached
        Notes:
            -can replace update_with_signal_field with update_state to simulate ensemble
            of non-interacting n**2 cells
        """
        # TODO removed features
        #  - meanfield_global_field() subfunction recreate if needed (cost savings if async)
        #  - cell_locations = get_cell_locations(lattice, n)
        #  - for loc in cell_locations:
        #        update_datadict_timestep_cell(lattice, loc, memory_idx_list, 0)
        #  - cell specific datastorage call

        # 1) initial data storage and plotting
        self.step_state_save(0)
        if flag_datastore:
            self.step_datadict_update_global(0)  # measure initial state
        if flag_visualize:
            self.step_state_visualize(0)

        # 2) main loop
        for step in range(1, self.total_steps):
            print('Dynamics step: ', step)

            # applied field and beta schedule
            field_applied_step = self.field_applied[:, step - 1]
            beta_step = self.beta[step - 1]

            if self.dynamics_blockparallel:
                self.step_dynamics_parallel(step, field_applied_step, beta_step)
            else:
                self.step_dynamics_async(step, field_applied_step, beta_step)

            # compute lattice properties (assess global state)
            # TODO 1 - consider lattice energy at each cell update (not lattice update)
            # TODO 2 - speedup lattice energy calc by using info from state update calls...
            self.step_state_save(step)
            if flag_datastore:
                self.step_datadict_update_global(step)

            # periodic plotting call
            if flag_visualize and step % self.plot_period == 0:  # plot the lattice
                self.step_state_visualize(step)
                #self.step_state_save(step, memory_idx_list)  # TODO call to save

            # update class attributes TODO any others to increment?
            self.current_step += 1

            # (optional) simulation termination condition: fixed point reached
            if end_at_fp:
                fp_msg = 'FP reached early (step %d of %d), terminating dynamics' % \
                         (step, self.total_steps)
                fp_reached = self.check_if_fixed_point(
                    self.graph_state_arr[:, step - 1], step, msg=fp_msg)
                fp_reached_flicker = self.check_if_fixed_point(
                    self.graph_state_arr[:, step - 2], step, msg='Flicker (2-state) ' + fp_msg)
                if fp_reached or fp_reached_flicker:
                    break

    def simulation_standard(self):
        # run the simulation
        self.dynamics_full()

        # """check the data dict"""
        self.plot_datadict_memory(use_proj=False)
        self.plot_datadict_memory(use_proj=True)

        # """write and plot cell state timeseries"""
        # write_state_all_cells(lattice, io_dict['datadir'])
        self.mainloop_data_dict_write()
        self.mainloop_data_dict_plot()

        print("\nMulticell simulation complete - output in %s" % self.io_dict['basedir'])
        return

    def simulation_fast_to_fp(self):

        # """get data dict from initial state"""
        # TODO in flicker case should analyze & plot the final two states
        self.step_datadict_update_global(0, fill_to_end=False)    # measure initial state
        self.step_state_visualize(0)                            # visualize initial state

        # run the simulation
        self.dynamics_full(flag_visualize=False, flag_datastore=False, end_at_fp=True)

        # """get data dict from final two states"""
        self.step_datadict_update_global(self.current_step - 1, fill_to_end=True)  # measure
        self.step_state_visualize(self.current_step - 1)                           # visualize
        self.step_datadict_update_global(self.current_step, fill_to_end=True)      # measure
        self.step_state_visualize(self.current_step)                               # visualize

        # """make copies of relevant save states"""
        sdir = self.io_dict['statesdir']
        # copy final - 1 to X_secondlast.txt
        shutil.copyfile(sdir + os.sep + 'X_%d.txt' % (self.current_step - 1),
                        sdir + os.sep + 'X_secondlast.txt')
        # copy final to X_last.txt
        shutil.copyfile(sdir + os.sep + 'X_%d.txt' % (self.current_step),
                        sdir + os.sep + 'X_last.txt')

        # """check the data dict"""
        self.plot_datadict_memory(use_proj=False)
        self.plot_datadict_memory(use_proj=True)

        # """write and plot cell state timeseries"""
        # write_state_all_cells(lattice, io_dict['datadir'])
        self.mainloop_data_dict_write()
        self.mainloop_data_dict_plot()

        print("\nMulticell simulation complete - output in %s" % self.io_dict['basedir'])
        return

    def get_cell_state(self, cell_idx, step):
        assert 0 <= cell_idx < self.num_cells  # TODO think this is not needed
        a = self.num_genes * cell_idx
        b = self.num_genes * (cell_idx + 1)
        return self.graph_state_arr[a:b, step]

    def get_field_on_cell(self, cell_idx, step):
        assert 0 <= cell_idx < self.num_cells  # TODO think this is not needed
        a = self.num_genes * cell_idx
        b = self.num_genes * (cell_idx + 1)
        return self.field_applied[a:b, step]

    def cell_cell_overlap(self, idx_a, idx_b, step):
        s_a = self.get_cell_state(idx_a, step)
        s_b = self.get_cell_state(idx_b, step)
        return np.dot(s_a.T, s_b) / self.num_genes

    def plot_datadict_memory(self, use_proj=True):
        if use_proj:
            datakey = 'memory_proj_arr'
            datatitle = 'projection'
        else:
            datakey = 'memory_overlap_arr'
            datatitle = 'overlap'
        # check the data dict
        for mu in range(self.simsetup['P']):
            #print(self.data_dict[datakey][:, mu, :])
            plt.plot(self.data_dict[datakey][:, mu, :].T)
            plt.ylabel('%s of all cells onto type: %s' %
                       (datatitle, self.simsetup['CELLTYPE_LABELS'][mu]))
            plt.xlabel('Time (full lattice steps)')
            plt.savefig(self.io_dict['plotdatadir'] + os.sep + '%s%d.png' % (datatitle, mu))
            plt.clf()  # plt.show()
        return

    # TODO make savestring property to append to files and plots?
    """e.g 
    '%s_%s_t%d_proj%d_remove%.2f_exo%.2f.png' %
    (self.exosome_string, self.initialization_style, self.total_steps, memory_idx,
     self.exosome_remove_ratio, field_signal_strength)"""
    def mainloop_data_dict_write(self):
        if self.flag_state_int:
            write_grid_state_int_alt(self.data_dict['cell_state_int'], self.io_dict['datadir'])
        if 'graph_energy' in list(self.data_dict.keys()):
            write_general_arr(self.data_dict['graph_energy'], self.io_dict['datadir'],
                              'graph_energy', txt=True, compress=False)
        if 'compressibility_full' in list(self.data_dict.keys()):
            write_general_arr(self.data_dict['compressibility_full'], self.io_dict['datadir'],
                              'compressibility_full', txt=True, compress=False)

    def mainloop_data_dict_plot(self):
        if 'graph_energy' in list(self.data_dict.keys()):
            plt.plot(self.data_dict['graph_energy'][:, 0], '-ok', label=r'$H_{\mathrm{quad}}$',
                     alpha=0.5)
            plt.plot(self.data_dict['graph_energy'][:, 1], '--ok', label=r'$H_{\mathrm{total}}$')
            plt.plot(self.data_dict['graph_energy'][:, 2], '--b', alpha=0.7,
                     label=r'$H_{\mathrm{self}}$')
            plt.plot(self.data_dict['graph_energy'][:, 3], '--g', alpha=0.7,
                     label=r'$H_{\mathrm{app}}$')
            plt.plot(self.data_dict['graph_energy'][:, 4], '--r', alpha=0.7,
                     label=r'$H_{\mathrm{pairwise}}$')
            plt.plot(self.data_dict['graph_energy'][:, 1] -
                     self.data_dict['graph_energy'][:, 3], '--o',
                     color='gray', label=r'$H_{\mathrm{total}} - H_{\mathrm{app}}$')
            plt.title(r'Multicell hamiltonian over time')
            plt.ylabel(r'Graph energy')
            plt.xlabel(r'$t$ (graph steps)')
            plt.legend()
            plt.savefig(self.io_dict['plotdatadir'] + os.sep +'hamiltonian.png')

            # zoom on relevant part
            ylow = min(np.min(self.data_dict['graph_energy'][:, [2, 4]]),
                       np.min(self.data_dict['graph_energy'][:, 1] -
                              self.data_dict['graph_energy'][:, 3]))
            yhigh = max(np.max(self.data_dict['graph_energy'][:, [2, 4]]),
                        np.max(self.data_dict['graph_energy'][:, 1] -
                               self.data_dict['graph_energy'][:, 3]))
            plt.ylim(ylow - 0.1, yhigh + 0.1)
            plt.savefig(self.io_dict['plotdatadir'] + os.sep + 'hamiltonianZoom.png')
            plt.clf()  # plt.show()

        # TODO check validity or remove
        if 'compressibility_full' in list(self.data_dict.keys()):

            assert self.graph_style == 'lattice_square'
            nn = self.graph_kwargs['sidelength']

            plt.plot(self.data_dict['compressibility_full'][:, 0], '--o', color='orange')
            plt.title(r'File compressibility ratio of the full lattice spin state')
            plt.ylabel(r'$\eta(t)/\eta_0$')
            plt.axhline(y=1.0, ls='--', color='k')

            ref_0 = calc_compression_ratio(
                x=np.zeros((nn, nn, self.simsetup['N']), dtype=int),
                method='manual',
                eta_0=self.data_dict['compressibility_full'][0, 2], datatype='full', elemtype=np.int)
            ref_1 = calc_compression_ratio(
                x=np.ones((nn, nn, self.simsetup['N']), dtype=int),
                method='manual',
                eta_0=self.data_dict['compressibility_full'][0, 2], datatype='full', elemtype=np.int)
            plt.axhline(y=ref_0[0], ls='-.', color='gray')
            plt.axhline(y=ref_1[0], ls='-.', color='blue')
            #print(ref_0, ref_0, ref_0, ref_0, 'is', ref_0, 'vs', ref_1)
            plt.xlabel(r'$t$ (lattice steps)')
            plt.ylim(-0.05, 1.01)
            plt.savefig(self.io_dict['plotdatadir'] + os.sep + 'compresibility.png')
            plt.clf()  # plt.show()


if __name__ == '__main__':

    # 1) create simsetup
    main_seed = 12410
    curated = False
    random_mem = False        # TODO incorporate seed in random XI
    random_W = False          # TODO incorporate seed in random W
    simsetup_main = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)
    print("simsetup checks:")
    print("\tsimsetup['N'],", simsetup_main['N'])
    print("\tsimsetup['P'],", simsetup_main['P'])

    # setup 2.1) multicell sim core parameters
    num_cells = 10**2          # global GRIDSIZE
    total_steps = 40           # global NUM_LATTICE_STEPS
    plot_period = 1
    flag_state_int = True
    flag_blockparallel = True
    beta = 2000.0
    gamma = 20.0               # i.e. field_signal_strength
    kappa = 0.0                # i.e. field_applied_strength

    # setup 2.2) graph options
    autocrine = False
    graph_style = 'lattice_square'
    graph_kwargs = {'search_radius': 1,
                    'initialization_style': 'random'}

    # setup 2.3) signalling field (exosomes + cell-cell signalling via W matrix)
    # Note: consider rescale gamma as gamma / num_cells * num_plaquette
    # global gamma acts as field_strength_signal, it tunes exosomes AND sent field
    # TODO implement exosomes for dynamics_blockparallel case
    exosome_string = "no_exo_field"  # on/off/all/no_exo_field; 'off' = send info only 'off' genes
    exosome_remove_ratio = 0.0       # amount of exo field idx to randomly prune from each cell

    # setup 2.4) applied/manual field (part 1)
    # size [N x steps] or size [NM x steps] or None
    # field_applied = construct_app_field_from_genes(
    #    IPSC_EXTENDED_GENES_EFFECTS, simsetup['GENE_ID'], num_steps=steps)
    field_applied = None

    # setup 2.5) applied/manual field (part 2) add housekeeping field with strength kappa
    flag_housekeeping = False
    field_housekeeping_strength = 0.0  # aka Kappa
    assert not flag_housekeeping
    if flag_housekeeping:
        assert field_housekeeping_strength > 0
        # housekeeping auto (via model extension)
        field_housekeeping = np.zeros(simsetup_main['N'])
        if simsetup_main['K'] > 0:
            field_housekeeping[-simsetup_main['K']:] = 1.0
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

    # 2) prep args for Multicell class instantiation
    multicell_kwargs = {
        'beta': beta,
        'total_steps': total_steps,
        'num_cells': num_cells,
        'flag_blockparallel': flag_blockparallel,
        'graph_style': graph_style,
        'graph_kwargs': graph_kwargs,
        'autocrine': autocrine,
        'gamma': gamma,
        'exosome_string': exosome_string,
        'exosome_remove_ratio': exosome_remove_ratio,
        'kappa': kappa,
        'field_applied': field_applied,
        'flag_housekeeping': flag_housekeeping,
        'flag_state_int': flag_state_int,
        'plot_period': plot_period,
        'seed': main_seed,
        'run_label': 's%d' % main_seed
    }

    # 3) instantiate
    multicell = Multicell(simsetup_main, verbose=True, **multicell_kwargs)

    # 4) run sim
    #multicell.simulation_standard()
    multicell.simulation_fast_to_fp()
