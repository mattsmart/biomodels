import numpy as np

from dynamics_vectorfields import set_ode_attributes
from plotting_networkx import draw_from_adjacency
from settings import DEFAULT_STYLE_ODE, VALID_STYLE_ODE

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

        if adjacency is None:
            self.adjacency = np.zeros((self.num_cells, self.num_cells))
        if labels is None:
            self.labels = ['c%d' % c for c in range(1, self.num_cells+1)]
        if style_ode is None:
            self.style_ode = DEFAULT_STYLE_ODE

        sc_dim_ode, sc_dim_misc, variables_short, variables_long = set_ode_attributes(style_ode)
        self.dim_ode = sc_dim_ode * self.num_cells

        if state_init is None:
            self.state_init = np.zeros((self.dim_ode, 1))

        assert self.num_cells > 0
        assert self.adjacency.shape == (self.num_cells, self.num_cells)
        assert len(self.labels) == self.num_cells
        assert self.state_init.shape == (self.dim_ode, 1)
        assert self.state_history.shape[0] == self.dim_ode
        assert self.style_ode in VALID_STYLE_ODE

    def division_event(self, idx_dividing_cell):
        self.num_cells += 1

        updated_labels = self.labels + ['c%d' % self.num_cells]

        # generate update adjaency matrix
        updated_adjacency = np.zeros((self.num_cells, self.num_cells))
        # fill A[0:M-1, 0:M-1] entries with old adjacency matrix
        updated_adjacency[0:self.num_cells - 1, 0:self.num_cells - 1] = self.adjacency
        # add new row/column with index corresponding k to the generating cell
        # i.e. for input i = idx_dividing_cell, set A[i, k] = 1 and # set A[k, i] = 1
        updated_adjacency[idx_dividing_cell, -1] = 1
        updated_adjacency[-1, idx_dividing_cell] = 1

        self.adjacency = updated_adjacency
        self.labels = updated_labels

    def print_state(self):
        print(self.labels)
        print(self.adjacency)
        return

    def plot_state(self):
        draw_from_adjacency(self.adjacency)
        return


if __name__ == '__main__':
    cellgraph = CellGraph(num_cells=1)
    cellgraph.plot_state()
    cellgraph.print_state()

    for idx in range(10):
        dividing_idx = np.random.randint(0, cellgraph.num_cells)
        print(idx, dividing_idx)
        cellgraph.division_event(idx)
        #cellgraph.division_event(dividing_idx)
        cellgraph.plot_state()
        cellgraph.print_state()
