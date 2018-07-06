import numpy as np
from random import shuffle

from singlecell_data_io import state_write
from singlecell_constants import BETA, EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH
from singlecell_functions import glauber_dynamics_update, state_memory_projection, state_memory_overlap, hamiltonian, state_burst_errors, state_to_label
from singlecell_simsetup import GENE_LABELS, CELLTYPE_LABELS, J
from singlecell_visualize import plot_as_bar, plot_as_radar, save_manual

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

    def get_current_state(self):  # TODO maybe make this and other copy instead of pass pointer np.zeros(self.N).. etc
        return self.state

    def get_current_label(self):
        state = self.state
        return state_to_label(tuple(state))

    def get_state_array(self):
        return self.state_array

    def get_energy(self, intxn_matrix=J):
        return hamiltonian(self.state, intxn_matrix=intxn_matrix)

    def get_memories_overlap(self):
        return state_memory_overlap(self.state_array, self.steps)

    def plot_overlap(self, use_radar=False, pltdir=None):
        overlap = self.get_memories_overlap()
        if use_radar:
            fig, ax = plot_as_radar(overlap)
            if pltdir is not None:
                save_manual(fig, pltdir, "state_overlap_radar_%s_%d" % (self.label, self.steps))
        else:
            fig, ax = plot_as_bar(overlap)
            if pltdir is not None:
                save_manual(fig, pltdir, "state_overlap_bar_%s_%d" % (self.label, self.steps))
        return fig, ax, overlap

    def get_memories_projection(self):
        return state_memory_projection(self.state_array, self.steps)

    def plot_projection(self, use_radar=False, pltdir=None):
        proj = self.get_memories_projection()
        if use_radar:
            fig, ax = plot_as_radar(proj)
            if pltdir is not None:
                save_manual(fig, pltdir, "state_%d_proj_radar_%s" % (self.steps, self.label))
        else:
            fig, ax = plot_as_bar(proj)
            if pltdir is not None:
                save_manual(fig, pltdir, "state_%d_proj_bar_%s" % (self.steps, self.label))
        return fig, ax, proj

    def apply_burst_errors(self, ratio_to_flip=0.02):
        burst_errors = state_burst_errors(self.state, ratio_to_flip=ratio_to_flip)
        self.state[:] = burst_errors[:]
        self.state_array[:, -1] = burst_errors[:]
        return burst_errors

    def update_state(self, intxn_matrix=J, beta=BETA, ext_field=None, ext_field_strength=EXT_FIELD_STRENGTH, app_field=None,
                     app_field_strength=APP_FIELD_STRENGTH, randomize=False):
        """
        ext_field - N x 1 - field external to the cell in a signalling sense; exosome field in multicell sym
        ext_field_strength  - scaling factor for ext_field
        app_field - N x 1 - unnatural external field (e.g. force TF on for some time period experimentally)
        app_field_strength - scaling factor for appt_field
        """
        sites = range(self.N)
        if randomize:
            shuffle(sites)  # randomize site ordering each timestep updates
        state_array_ext = np.zeros((self.N, np.shape(self.state_array)[1] + 1))
        state_array_ext[:, :-1] = self.state_array  # TODO: make sure don't need array copy
        state_array_ext[:,-1] = self.state_array[:,-1]
        for idx, site in enumerate(sites):  # TODO: parallelize
            state_array_ext = glauber_dynamics_update(state_array_ext, site, self.steps + 1,
                                                      intxn_matrix=intxn_matrix, beta=beta, ext_field=ext_field,
                                                      ext_field_strength=ext_field_strength, app_field=app_field,
                                                      app_field_strength=app_field_strength)
        self.state_array = state_array_ext
        self.steps += 1
        self.state = state_array_ext[:, -1]
        return self.state

    def write_state(self, datadir):
        state_write(self.state_array, range(self.steps), self.gene_list, "sc_state_%s" % self.label, "times", "gene_labels", datadir)
