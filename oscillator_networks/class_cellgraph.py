import numpy as np

from plotting_networkx import draw_from_adjacency

"""
The collection of coupled cells is represented by
- an adjacency matrix defining connections, and
- a corresponding array of state variables
"""


class CellGraph():

    def __init__(self, num_cells=1, adjacency=None, labels=None):
        self.num_cells = num_cells
        self.adjacency = adjacency
        self.labels = labels

        if adjacency is None:
            self.adjacency = np.zeros((self.num_cells, self.num_cells))
        if labels is None:
            self.labels = ['c%d' % c for c in range(1, self.num_cells+1)]

        assert self.num_cells > 0
        assert self.adjacency.shape == (self.num_cells, self.num_cells)
        assert len(self.labels) == self.num_cells

    def division_event(self, idx_dividing_cell):
        self.num_cells += 1
        # TODO
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
