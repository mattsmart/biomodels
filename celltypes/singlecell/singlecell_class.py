import numpy as np
from random import shuffle, random

from singlecell_data_io import state_write
from singlecell_constants import BETA, EXT_FIELD_STRENGTH, APP_FIELD_STRENGTH, ASYNC_BATCH
from singlecell_functions import glauber_dynamics_update, state_memory_projection, state_memory_overlap, hamiltonian, \
    state_burst_errors, state_to_label
from singlecell_visualize import plot_as_bar, plot_as_radar, save_manual

"""
TODO:
    -steps is redundant: np.shape(state_array)[1]
    -state is redundant: state_array[:,-1]
"""


class Cell(object):
    def __init__(self, state, label, memories_list, gene_list, state_array=None,
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
        """
        Converts binary array into unique integer via bitwise representation
        """
        state = self.state
        return state_to_label(tuple(state))

    def get_state_array(self):
        return self.state_array

    def get_energy(self, intxn_matrix):
        return hamiltonian(self.state, intxn_matrix=intxn_matrix)

    def get_memories_overlap(self, xi):
        return state_memory_overlap(self.state_array, self.steps, self.N, xi)

    def plot_overlap(self, xi, use_radar=False, pltdir=None):
        overlap = self.get_memories_overlap(xi)
        if use_radar:
            fig, ax = plot_as_radar(overlap, self.memories_list)
            if pltdir is not None:
                save_manual(fig, pltdir, "state_overlap_radar_%s_%d" % (self.label, self.steps))
        else:
            fig, ax = plot_as_bar(overlap, self.memories_list)
            if pltdir is not None:
                save_manual(fig, pltdir, "state_overlap_bar_%s_%d" % (self.label, self.steps))
        return fig, ax, overlap

    def get_memories_projection(self, a_inv, xi):
        return state_memory_projection(self.state_array, self.steps, a_inv, self.N, xi)

    def plot_projection(self, a_inv, xi, use_radar=False, pltdir=None, proj=None):
        if proj is None:
            proj = self.get_memories_projection(a_inv, xi)
        if use_radar:
            fig, ax = plot_as_radar(proj, self.memories_list)
            if pltdir is not None:
                save_manual(fig, pltdir, "state_%d_proj_radar_%s" % (self.steps, self.label))
        else:
            fig, ax = plot_as_bar(proj, self.memories_list)
            if pltdir is not None:
                save_manual(fig, pltdir, "state_%d_proj_bar_%s" % (self.steps, self.label))
        return fig, ax, proj

    def apply_burst_errors(self, ratio_to_flip=0.02):
        burst_errors = state_burst_errors(self.state, ratio_to_flip=ratio_to_flip)
        self.state[:] = burst_errors[:]
        self.state_array[:, -1] = burst_errors[:]
        return burst_errors

    def update_state(self, intxn_matrix, beta=BETA, ext_field=None, ext_field_strength=EXT_FIELD_STRENGTH, app_field=None,
                     app_field_strength=APP_FIELD_STRENGTH, async_batch=ASYNC_BATCH, async=True):
        """
        async_batch: if True, sample from 0 to N with replacement, else each step will be 'fully random'
                     i.e. can update same site twice in a row, vs time gap of at least N substeps
                     these produce different short term behaviour, but should reach same steady state
        ext_field - N x 1 - field external to the cell in a signalling sense; exosome field in multicell sym
        ext_field_strength  - scaling factor for ext_field
        app_field - N x 1 - unnatural external field (e.g. force TF on for some time period experimentally)
        app_field_strength - scaling factor for appt_field
        """
        if async:
            sites = list(range(self.N))
            rsamples = np.random.rand(self.N)  # optimized: pass one to each of the N single spin update calls  TODO: benchmark vs intels
            if async_batch:
                shuffle(sites)  # randomize site ordering each timestep updates
            else:
                #sites = np.random.choice(self.N, self.N, replace=True)
                #sites = [int(self.N*np.random.random()) for _ in xrange(self.N)]  # this should be same and faster
                sites = [int(self.N * u) for u in np.random.rand(self.N)]  # this should be 5-10% percent faster

            state_array_ext = np.zeros((self.N, np.shape(self.state_array)[1] + 1))
            state_array_ext[:, :-1] = self.state_array  # TODO: make sure don't need array copy
            state_array_ext[:,-1] = self.state_array[:,-1]
            for idx, site in enumerate(sites):          # TODO: parallelize approximation
                state_array_ext = glauber_dynamics_update(state_array_ext, site, self.steps + 1, intxn_matrix, rsamples[idx],
                                                          beta=beta, ext_field=ext_field, app_field=app_field,
                                                          ext_field_strength=ext_field_strength,
                                                          app_field_strength=app_field_strength)

        else:
            #print "WARNING: experimental sync update (can use gpu)"
            state_array_ext = np.zeros((self.N, np.shape(self.state_array)[1] + 1))
            state_array_ext[:, :-1] = self.state_array  # TODO: make sure don't need array copy
            state_array_ext[:,-1] = self.state_array[:,-1]

            total_field = np.zeros(self.N)
            internal_field = np.dot(intxn_matrix, state_array_ext[:, self.steps + 1])
            total_field += internal_field
            if ext_field is not None:
                ext_field_vec = ext_field_strength * ext_field
                total_field += ext_field_vec
            if app_field is not None:
                app_field_vec = app_field_strength * app_field
                total_field += app_field_vec
            # probability that site i will be "up" after the timestep
            prob_on_after_timestep = 1 / (1 + np.exp(-2 * beta * total_field))
            rsamples = np.random.rand(self.N)  # optimized: pass one to each of the N single spin update calls  TODO: benchmark vs intels
            for idx in range(self.N):
                if prob_on_after_timestep[idx] > rsamples[idx]:
                    state_array_ext[idx, self.steps + 1] = 1.0
                else:
                    state_array_ext[idx, self.steps + 1] = -1.0

        self.state_array = state_array_ext
        self.steps += 1
        self.state = state_array_ext[:, -1]

        return self.state

    def write_state(self, datadir):
        state_write(self.state_array, list(range(self.steps)), self.gene_list, "sc_state_%s" % self.label, "times", "gene_labels", datadir)
