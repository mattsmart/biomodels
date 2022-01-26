import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from class_singlecell import SingleCell
from dynamics_vectorfields import set_ode_attributes, ode_integration_defaults
from plotting_networkx import draw_from_adjacency
from settings import DEFAULT_STYLE_ODE, VALID_STYLE_ODE, DIFFUSION_RATE, DYNAMICS_METHOD

"""
The collection of coupled cells is represented by
- an adjacency matrix defining connections, and
- a corresponding array of state variables

Attributes:
- self.num_cells     - integer         - denoted by "M" = number of cells in the system
- self.sc_dim_ode    - integer         - denoted by "N" = number of dynamic variables tracked in a single cell
- self.graph_dim_ode - integer         - sc_dim_ode * self.num_cells
- self.adjacency     - array (M x M)   - cell-cell adjacency matrix
- self.diffusion     - array (N)       - rate of diffusion may be distinct for each of the N internal sc variables
- self.labels        - list of strings - unique name for each node on the graph e.g. 'cell_%d'
- self.style_ode     - string          - determines single cell ODE
- self.state_history - array (NM x t)  - state history of the graph
- self.times_history - array (t)       - timepoints on which state was integrated
- self.sc_template   - SingleCell      - instance of custom class which exposes dx/dt=f(x) (where x is one cell)

Utility methods:
- self.state_to_stacked(x):    converts array x from shape [N x M] to [NM] 
- self.state_to_rectangle(x):  converts array x from shape [NM] to [N x M]

Issues:
- state_init and state history may need to be reinitialized following a division event, unless we do zero or NaN fill 
"""


class CellGraph():

    def __init__(self, num_cells=1, adjacency=None, labels=None, state_history=None, times_history=None, style_ode=None,
                 sc_template=None):
        self.num_cells = num_cells
        self.adjacency = adjacency
        self.labels = labels
        self.style_ode = style_ode
        self.state_history = state_history
        self.times_history = times_history
        self.sc_template = sc_template

        if adjacency is None:
            self.adjacency = np.zeros((self.num_cells, self.num_cells))
        if labels is None:
            self.labels = ['c%d' % c for c in range(1, self.num_cells+1)]
        if style_ode is None:
            self.style_ode = DEFAULT_STYLE_ODE

        # initialize single cell template which exposes dx/dt=f(x) for internal gene regulation components of network
        if sc_template is None:
            self.sc_template = SingleCell(style_ode=self.style_ode, label='template')
        assert self.sc_template.style_ode == self.style_ode

        # construct graph matrices based on adjacency
        self.degree = np.diag(np.sum(self.adjacency, axis=1))
        self.laplacian = self.degree - self.adjacency

        sc_dim_ode, sc_dim_misc, variables_short, variables_long = set_ode_attributes(style_ode)
        self.graph_dim_ode = sc_dim_ode * self.num_cells
        self.sc_dim_ode = sc_dim_ode
        # TODO external parameter/init arguments for this
        self.diffusion = np.array([DIFFUSION_RATE for i in range(sc_dim_ode)])  # internal variable have own rates

        if state_history is None:
            # Approach 1: use default init cond from single cell ode code
            _, _, _, sc_init_cond = ode_integration_defaults(self.style_ode)
            state_init = np.tile(sc_init_cond, self.num_cells)
            # Approach 2: just zeros
            #state_init = np.zeros(self.graph_dim_ode)
            # Approach 3: random
            #state_init = 10 * np.random.rand(self.graph_dim_ode)

            self.state_history = np.zeros((self.graph_dim_ode, 1))
            self.state_history[:, 0] = state_init

        if self.times_history is None:
            t0, t1, _, _ = ode_integration_defaults(style_ode)
            self.times_history = np.array([t0])

        assert self.num_cells > 0
        assert self.adjacency.shape == (self.num_cells, self.num_cells)
        assert len(self.labels) == self.num_cells
        assert self.state_history.shape[0] == self.graph_dim_ode
        assert self.style_ode in VALID_STYLE_ODE
        assert all([c >= 0.0 for c in self.diffusion])

    def division_event(self, idx_dividing_cell, copy_exact=False):
        """
        Returns a new instance of the CellGraph with updated state variables (as a result of adding one cell)
        """
        def split_mother_and_daughter_state():
            # TODO what choices here for splitting materials? copy or divide by two?
            current_graph_state = self.state_history[:, -1]
            mother_idx_low = self.sc_dim_ode * idx_dividing_cell
            mother_idx_high = self.sc_dim_ode * (idx_dividing_cell + 1)
            current_mother_state = current_graph_state[mother_idx_low : mother_idx_high]

            if copy_exact:
                print('TODO - currently mother_orig=daughter exactly during division event... try partition states into 2')
                post_mother_state = current_mother_state
                post_daughter_state = current_mother_state
            else:
                print("TODO - currently dividing perfectly by two")
                post_mother_state = current_mother_state / 2.0
                post_daughter_state = current_mother_state / 2.0

            return post_mother_state, post_daughter_state, mother_idx_low, mother_idx_high

        updated_num_cells = self.num_cells + 1
        updated_graph_dim_ode = int(self.sc_dim_ode * updated_num_cells)

        updated_labels = self.labels + ['c%d' % updated_num_cells]
        # generate update adjaency matrix
        updated_adjacency = np.zeros((updated_num_cells, updated_num_cells))
        # fill A[0:M-1, 0:M-1] entries with old adjacency matrix
        updated_adjacency[0:updated_num_cells - 1, 0:updated_num_cells - 1] = self.adjacency
        # add new row/column with index corresponding k to the generating cell
        # i.e. for input i = idx_dividing_cell, set A[i, k] = 1 and # set A[k, i] = 1
        updated_adjacency[idx_dividing_cell, -1] = 1
        updated_adjacency[-1, idx_dividing_cell] = 1

        # update new initial state based on extra cell and dispersion of maternal variables
        updated_state_history = np.zeros((updated_graph_dim_ode, len(self.times_history)))
        updated_state_history[0:self.graph_dim_ode, :] = self.state_history
        post_mother_state, post_daughter_state, mlow, mhigh = split_mother_and_daughter_state()
        print(updated_graph_dim_ode, self.graph_dim_ode, len(post_daughter_state))
        print(post_mother_state, post_daughter_state, mlow, mhigh)
        updated_state_history[mlow:mhigh, -1] = post_mother_state
        updated_state_history[self.graph_dim_ode:, -1] = post_daughter_state

        new_cellgraph = CellGraph(
            num_cells=updated_num_cells,
            adjacency=updated_adjacency,
            labels=updated_labels,
            state_history=updated_state_history,
            times_history=self.times_history,
            style_ode=self.style_ode,
            sc_template=self.sc_template)
        return new_cellgraph

    def graph_trajectory_TOYFLOW(self, init_cond=None, t0=None, t1=None, **solver_kwargs):
        single_cell = None
        assert self.style_ode == 'toy_flow'

        def graph_ode_system(t_scalar, xvec, single_cell):
            dxvec_dt = -1 * self.diffusion * np.dot(self.laplacian, xvec)
            return dxvec_dt

        fn = graph_ode_system
        time_interval = [t0, t1]

        if 'vectorized' not in solver_kwargs.keys():
            solver_kwargs['vectorized'] = True
        sol = solve_ivp(fn, time_interval, init_cond, method='Radau', args=(single_cell,), **solver_kwargs)
        r = np.transpose(sol.y)
        times = sol.t
        return r, times

    def graph_trajectory(self, init_cond=None, time_interval=None, update_state=True, **solver_kwargs):
        """
        In principle, can simulate starting from the current state of the graph to some arbitrary timepoint,
        However, we'd like to "pause" after the first cell completes a cycle (call this time "t_div").
          - Issue #1: how to detect this given a timeseries of state changes for the whole graph.
            TODO - see handlers in scipy solve_ivp() for event detection
          - Issue #2: what if no limit cycles are observed (answer - we keep simulating until quiescence).
          - Immediately at t_div, we record the graph state, pause it, create a new instance of the graph with M + 1
            cells, and decide how to split the material between the dividing cell and its daughter cell.
        TODO For now, we will not pause at cycles and just simulate the graph trajectory and return it.

        Suppose the ODE for a single cell with state x_1 in R^N is given by
            d{x_1}/dt = F({x_1})
        Then the ODE for M coupled cells with a diffusive coupling of all N variables is
            dX/dt = F(X) - LX
        Where
            X = [x_1, x_2, ..., x_M]^T a stacked vector of length NM representing the state of all cells
        """
        N = self.sc_dim_ode
        M = self.num_cells
        single_cell = self.sc_template

        def f_of_x_single_cell(t_scalar, init_cond, single_cell):
            # Gene regulatory dynamics internal to one cell based on its state variables (dx/dt = f(x))
            dxdt = single_cell.ode_system_vector(init_cond, t_scalar)
            return dxdt

        def graph_ode_system(t_scalar, xvec, single_cell):
            xvec_matrix = self.state_to_rectangle(xvec)
            # Term 1: stores the single cell gene regulation (for each cell)
            #         [f(x_1) f(x_2) ... f(x_M)] as a stacked NM long 1D array
            term_1 = np.zeros(self.graph_dim_ode)
            for cell_idx in range(M):
                a = N * cell_idx
                b = N * (cell_idx + 1)
                term_1[a:b] = f_of_x_single_cell(t_scalar, xvec_matrix[:, cell_idx], single_cell)

            # TODO check that slicing is correct
            # TODO this can be parallelized as one liner Dvec * np.dot(X, L^T)
            # Term 2: stores the cell-cell coupling which is just laplacian diffusion -c * L * x
            # Note: we consider each reactant separately with own diffusion rate
            term_2 = np.zeros(self.graph_dim_ode)
            for gene_idx in range(N):
                indices_for_specific_gene = np.arange(gene_idx, self.graph_dim_ode, N)
                xvec_specific_gene = xvec[indices_for_specific_gene]
                diffusion_specific_gene = - self.diffusion[gene_idx] * np.dot(self.laplacian, xvec_specific_gene)
                term_2[indices_for_specific_gene] = diffusion_specific_gene

            dxvec_dt = term_1 + term_2
            return dxvec_dt

        fn = graph_ode_system
        if init_cond is None:
            init_cond = self.state_history[:, -1]
        if time_interval is None:
            t0, t1, _, _ = ode_integration_defaults(self.style_ode)
            tshift = t1 - t0
            time_interval = [self.times_history[-1], tshift]
        if 't_eval' in solver_kwargs.keys():
            time_interval[0] = min(time_interval[0], solver_kwargs['t_eval'][0])
            time_interval[1] = max(time_interval[1], solver_kwargs['t_eval'][-1])

        if 'vectorized' not in solver_kwargs.keys():
            solver_kwargs['vectorized'] = False  # TODO how to vectorize our graph ODE?
        sol = solve_ivp(fn, time_interval, init_cond, method='Radau', args=(single_cell,), **solver_kwargs)
        r = sol.y
        times = sol.t

        if update_state:
            print("update_state r.shape t.shape", r.shape, times.shape, times[0], self.times_history[0])
            self.state_history = np.concatenate((self.state_history, r[:, 1:]), axis=1)
            self.times_history = np.concatenate((self.times_history, times[1:]), axis=0)
            print("confirm", self.times_history.shape)

        return r, times

    def print_state(self):
        print("self.adjacency\n", self.adjacency)
        print("self.labels", self.labels)
        print("self.num_cells", self.num_cells)
        print("self.graph_dim_ode", self.graph_dim_ode)
        print("self.sc_dim_ode", self.sc_dim_ode)
        print("self.style_ode", self.style_ode)
        print("self.diffusion_rate", self.diffusion)
        print("self.state_history.shape", self.state_history.shape)
        print("timepoints: t_0, t_1, npts:", self.times_history[0], self.times_history[-1], self.times_history.shape)
        print("Current state:")
        X = self.state_to_rectangle(self.state_history)
        for cell in range(self.num_cells):
            print('\tCell #%d' % cell, X[:, cell, :].flatten())
        return

    def plot_graph(self):
        # TODO more detailed node coloring / state info / labels (str)?
        draw_from_adjacency(self.adjacency)
        return

    def plot_state_each_cell(self):

        assert self.num_cells <= 16  # for now
        ncols = 4
        nrows = 1 + (self.num_cells - 1) // ncols
        print("in plot_state():", self.num_cells, ncols, nrows)

        # TODO other function which up to 16 cells does 4x4 grid of xyz traj plots  ---- or ----- x,y phase plots - SUBPLOTS or GRIDSPEC
        fig, axarr = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 8), constrained_layout=True, squeeze=False)
        state_tensor = self.state_to_rectangle(self.state_history)
        times = self.times_history

        for idx in range(self.num_cells):
            i = idx // 4
            j = idx % 4
            print("idx, i, j", idx, i, j)

            r = np.transpose(state_tensor[:, idx, :])

            axarr[i, j].plot(
                times, r, label=[self.sc_template.variables_short[i] for i in range(self.sc_dim_ode)])
            axarr[i, j].set_xlabel(r'$t$ [min]')
            axarr[i, j].set_ylabel(r'concentration [nM]')

        plt.legend()

        plt.suptitle('plot_state_each_cell()')
        plt.show()
        return

    def state_to_stacked(self, x):
        """
        converts array x from shape [N x M] to [NM]
        converts array x from shape
         - [N x M]      to  [NM]
         - [N x M x t]  to  [NM x t]

        E.g.: suppose 2 cells each with 2 components
              the first two components belong to cell one, the next two to cell two
              in: [1,2,3,4]   out: [[1, 3],
                                    [2, 4]]
        """
        d = len(x.shape)
        assert d in [2, 3]
        assert x.shape[0:2] == (self.sc_dim_ode, self.num_cells)
        if d == 2:
            out = x.reshape(self.graph_dim_ode, order='F')
        else:
            out = x.reshape((self.graph_dim_ode, -1), order='F')
        return out

    def state_to_rectangle(self, x):
        """
        converts array x from shape
         - [NM]      to  [N x M]
         - [NM x t]  to  [N x M x t]
        """
        d = len(x.shape)
        assert d in [1,2]
        assert x.shape[0] == self.graph_dim_ode
        if d == 1:
            out = x.reshape((self.sc_dim_ode, self.num_cells), order='F')
        else:
            out = x.reshape((self.sc_dim_ode, self.num_cells, -1), order='F')
        return out


if __name__ == '__main__':

    # Misc. setting
    style_ode = 'PWL3'  # styles: ['PWL2', 'PWL3', 'Yang2013', 'toy_flow']
    copy_exact = False  # if True, divide cell contents 100%/100% between mother/daughter (else 50%/50%)
    M = 1
    if style_ode == 'PWL2':
        state_history = np.array([[100, 100]]).T     # None or array of shape (NM x times)
    else:
        state_history = np.array([[100, 100, 10]]).T  # None or array of shape (NM x times)

    # Initialization
    cellgraph = CellGraph(num_cells=M, style_ode=style_ode, state_history=state_history)
    if cellgraph.style_ode in ['PWL2', 'PWL3']:
        cellgraph.sc_template.params_ode['epsilon'] = 0.3

    # Initial state output
    cellgraph.plot_graph()
    cellgraph.print_state()

    # Add some cells through manual divisions (two different modes - linear or random)
    for idx in range(15):
        dividing_idx = np.random.randint(0, cellgraph.num_cells)
        print("Division event (idx, div idx):", idx, dividing_idx)
        cellgraph = cellgraph.division_event(idx, copy_exact=copy_exact)  # Mode 1 - linear division idx
        #cellgraph.division_event(dividing_idx, copy_exact=copy_exact)    # Mode 2 - random division idx
        cellgraph.plot_graph()
        cellgraph.print_state()
        print()

    # From the final graph (after all divisions above), simulate graph trajectory
    print('Example trajectory for the graph...')
    #t_eval = None  # None or np.linspace(0, 50, 2000)
    t_eval = np.linspace(15, 50, 2000)
    solver_kwargs = {
        't_eval': t_eval
    }
    r, times = cellgraph.graph_trajectory(**solver_kwargs)

    # Plot the timeseries for each cell
    #print(r)
    #print(r.shape)
    #print(times)
    #print(times.shape)
    cellgraph.plot_state_each_cell()

    # TODO division event detection handling
