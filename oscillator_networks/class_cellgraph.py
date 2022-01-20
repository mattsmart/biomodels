import numpy as np
from scipy.integrate import solve_ivp

from dynamics_vectorfields import set_ode_attributes, ode_integration_defaults
from plotting_networkx import draw_from_adjacency
from settings import DEFAULT_STYLE_ODE, VALID_STYLE_ODE, DIFFUSION_RATE, DYNAMICS_METHOD

"""
The collection of coupled cells is represented by
- an adjacency matrix defining connections, and
- a corresponding array of state variables

Attributes:
- self.num_cells     - integer         - denoted by "M"
- self.adjacency     - array (M x M)   - cell-cell adjacency matrix
- self.labels        - list of strings - unique name for each node on the graph e.g. 'cell_%d'
- self.style_ode     - string          - determines single cell ODE
- self.state_init    - array (NM x 1)  - initial condition for the graph
- self.state_history - array (NM x t)  - state history of the graph

Issues:
- state_init and state history may need to be reinitialized following a division event, unless we do zero or NaN fill 
"""


class CellGraph():

    def __init__(self, num_cells=1, adjacency=None, labels=None, state_init=None, state_history=None, style_ode=None):
        self.num_cells = num_cells
        self.adjacency = adjacency
        self.labels = labels
        self.style_ode = style_ode
        self.state_init = state_init
        self.state_history = state_history
        self.diffusion_rate = DIFFUSION_RATE

        if adjacency is None:
            self.adjacency = np.zeros((self.num_cells, self.num_cells))
        if labels is None:
            self.labels = ['c%d' % c for c in range(1, self.num_cells+1)]
        if style_ode is None:
            self.style_ode = DEFAULT_STYLE_ODE

        # construct graph matrices based on adjacency
        self.degree = np.diag(np.sum(self.adjacency, axis=1))
        self.laplacian = self.degree - self.adjacency

        sc_dim_ode, sc_dim_misc, variables_short, variables_long = set_ode_attributes(style_ode)
        self.graph_dim_ode = sc_dim_ode * self.num_cells
        self.sc_dim_ode = sc_dim_ode

        if state_init is None:
            # Approach 1: use default init cond from single cell ode code
            _, _, _, sc_init_cond = ode_integration_defaults(self.style_ode)
            self.state_init = np.tile(sc_init_cond, self.num_cells)
            # Approach 2: just zeros
            #self.state_init = np.zeros(self.graph_dim_ode)
            # Approach 3: random
            #self.state_init = 10 * np.random.rand(self.graph_dim_ode)

        if state_init is None:
            self.state_history = np.zeros((self.graph_dim_ode, 1))
            self.state_history[:, 0] = self.state_init

        assert self.num_cells > 0
        assert self.adjacency.shape == (self.num_cells, self.num_cells)
        assert len(self.labels) == self.num_cells
        assert self.state_init.shape[0] == self.graph_dim_ode and len(self.state_init.shape) == 1
        assert self.state_history.shape[0] == self.graph_dim_ode
        assert self.style_ode in VALID_STYLE_ODE
        assert self.diffusion_rate >= 0.0

    def division_event(self, idx_dividing_cell):

        def split_mother_and_daughter_state(copy_exact=False):
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
        updated_state_init = np.zeros(updated_graph_dim_ode)
        updated_state_init[0:self.graph_dim_ode] = self.state_init
        post_mother_state, post_daughter_state, mlow, mhigh = split_mother_and_daughter_state()

        print(updated_graph_dim_ode, self.graph_dim_ode, len(post_daughter_state))
        print(post_mother_state, post_daughter_state, mlow, mhigh)
        updated_state_init[mlow:mhigh] = post_mother_state
        updated_state_init[self.graph_dim_ode:] = post_daughter_state

        # TODO handler for updating graph state history after division -- currently history is not being stored
        updated_state_history = np.zeros((updated_graph_dim_ode, 1))
        updated_state_history[:, 0] = updated_state_init

        new_cellgraph = CellGraph(
            num_cells=updated_num_cells,
            adjacency=updated_adjacency,
            labels=updated_labels,
            state_init=updated_state_init,
            state_history=updated_state_history,
            style_ode=self.style_ode)
        return new_cellgraph

    def graph_trajectory(self, init_cond=None, t0=None, t1=None, **solver_kwargs):
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

        # TODO implement
        single_cell = None
        assert self.style_ode == 'toy_flow'

        def graph_ode_system(t_scalar, xvec, single_cell):
            dxvec_dt = -1 * self.diffusion_rate * np.dot(self.laplacian, xvec)
            return dxvec_dt

        fn = graph_ode_system
        time_interval = [t0, t1]
        if init_cond is None:
            init_cond = self.state_init

        if 'vectorized' not in solver_kwargs.keys():
            solver_kwargs['vectorized'] = True
        sol = solve_ivp(fn, time_interval, init_cond, method='Radau', args=(single_cell,), **solver_kwargs)
        r = np.transpose(sol.y)
        times = sol.t
        return r, times

    def print_state(self):
        print("self.adjacency\n", self.adjacency)
        print("self.labels", self.labels)
        print("self.num_cells", self.num_cells)
        print("self.graph_dim_ode", self.graph_dim_ode)
        print("self.sc_dim_ode", self.sc_dim_ode)
        print("self.style_ode", self.style_ode)
        print("self.diffusion_rate", self.diffusion_rate)
        print("self.state_init", self.state_init)
        return

    def plot_state(self):
        draw_from_adjacency(self.adjacency)
        return


if __name__ == '__main__':
    cellgraph = CellGraph(num_cells=1, style_ode='toy_flow')
    cellgraph.plot_state()
    cellgraph.print_state()

    for idx in range(2):
        dividing_idx = np.random.randint(0, cellgraph.num_cells)
        print(idx, dividing_idx)
        cellgraph = cellgraph.division_event(idx)
        #cellgraph.division_event(dividing_idx)
        cellgraph.plot_state()
        cellgraph.print_state()
        print()

    print("Final state")
    cellgraph.print_state()

    print('Example trajectory for the graph...')
    t1 = 10.0
    t_eval = np.linspace(0, t1, 7)
    r, times = cellgraph.graph_trajectory(t0=0, t1=t1, t_eval=t_eval)

    print(r)
    print(r.shape)
    print(times)
    print(times.shape)
