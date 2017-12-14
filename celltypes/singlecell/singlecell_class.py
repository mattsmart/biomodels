import numpy as np
from random import shuffle

from singlecell_data_io import state_write
from singlecell_functions import glauber_dynamics_update, state_memory_projection, state_memory_overlap, hamiltonian
from singlecell_simsetup import GENE_LABELS, CELLTYPE_LABELS
from singlecell_visualize import plot_as_radar, save_manual

"""
TODO:
    -steps is redundant: np.shape(state_array)[1]
    -state is redundant: state_array[:,-1]
"""

class Cell(object):
    def __init__(self, state, label, memories_list=CELLTYPE_LABELS, gene_list=GENE_LABELS, state_array=None,
                 steps=None):
        self.state = state  # this should be N x 1 array
        self.label = label  # label represents it's init cond
        self.memories_list = memories_list  # may not be needed here
        self.gene_list = gene_list         # may not be needed here
        self.P = len(memories_list)
        self.N = len(gene_list)
        assert len(self.state) == self.N
        if state_array is None:
            state_array_temp = np.zeros((self.N, 1))
            state_array_temp[:, 0] = state
            self.state_array = state_array_temp  # this should be N x time array
            self.steps = 0  # columns of state_array
        else:
            self.state_array = state_array
            assert steps == len(state_array[0])
            self.steps = steps

    def __str__(self):
        return self.label

    def get_current_state(self):
        return self.state

    def get_state_array(self):
        return self.state_array

    def get_energy(self):
        return hamiltonian(self.state)

    def get_memories_overlap(self):
        return state_memory_overlap(self.state_array, self.steps)

    def plot_overlap(self, pltdir=None):
        overlap = self.get_memories_overlap()
        fig, ax = plot_as_radar(overlap)
        if pltdir is not None:
            save_manual(fig, pltdir, "sc_state_radar_%s_%d" % (self.label, self.steps))
        return fig, ax

    def get_memories_projection(self):
        return state_memory_projection(self.state_array, self.steps)

    def plot_projection(self, pltdir=None):
        proj = self.get_memories_projection()
        fig, ax = plot_as_radar(proj)
        if pltdir is not None:
            save_manual(fig, pltdir, "state_radar_%s_%d" % (self.label, self.steps))
        return fig, ax

    def update_state(self, field=None):
        randomized_sites = range(self.N)
        shuffle(randomized_sites)  # randomize site ordering each timestep updates
        state_array_ext = np.zeros((self.N, np.shape(self.state_array)[1] + 1))
        state_array_ext[:, :-1] = self.state_array  # TODO: make sure don't need array copy
        for idx, site in enumerate(randomized_sites):  # TODO: parallelize
            state_array_ext = glauber_dynamics_update(state_array_ext, site, self.steps, external_field=field)
        self.state_array = state_array_ext
        self.steps += 1
        self.state = state_array_ext[:, -1]
        return self.state

    def write_state(self, datadir):
        state_write(self.state_array, range(self.steps), self.gene_list, "sc_state_%s" % self.label, "times", "gene_labels", datadir)
