import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from random import shuffle

from noneq_data_io import state_write
from noneq_functions import glauber_dynamics_update
from noneq_settings import BETA


class State(object):
    def __init__(self, state, label, state_array=None, steps=None):
        self.state = state  # this should be N x 1 array
        self.label = label  # label represents it's init cond
        self.N = len(state)
        self.spin_labels = range(self.N)
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

    def get_state_array(self):
        return self.state_array

    def update_state(self, intxn_matrix, app_field=None, beta=BETA, randomize=False):
        sites = range(self.N)
        if randomize:
            shuffle(sites)  # randomize site ordering each timestep updates
        state_array_ext = np.zeros((self.N, np.shape(self.state_array)[1] + 1))
        state_array_ext[:, :-1] = self.state_array  # TODO: make sure don't need array copy
        state_array_ext[:,-1] = self.state_array[:,-1]
        for idx, site in enumerate(sites):  # TODO: parallelize
            state_array_ext = glauber_dynamics_update(state_array_ext, site, self.steps + 1, intxn_matrix, app_field=app_field, beta=beta)
        self.state_array = state_array_ext
        self.steps += 1
        self.state = state_array_ext[:, -1]
        return self.state

    def visualize(self, style='spring'):
        # TODO want like Petrie polygon Orthographic projections on wikipedia
        # TODO make G an updating property of state (need to label nodes appropriately etc)
        G = nx.hypercube_graph(self.N)
        if style is None:
            nx.draw(G)
        elif style == 'circular':
            nx.draw_circular(G)
        elif style == 'spectral':
            nx.draw_spectral(G)
        elif style == 'spring':
            nx.draw_spring(G)
        elif style == 'shell':
            nx.draw_shell(G)
        plt.show()
        return


    def write_state(self, datadir):
        state_write(self.state_array, range(self.steps), self.spin_labels, "sc_state_%s" % self.label, "times", "spin_labels", datadir)
