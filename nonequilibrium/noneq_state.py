import numpy as np
from random import shuffle

from noneq_data_io import state_write
from noneq_functions import glauber_dynamics_update


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

    def update_state(self, app_field=None):
        randomized_sites = range(self.N)
        shuffle(randomized_sites)  # randomize site ordering each timestep updates
        state_array_ext = np.zeros((self.N, np.shape(self.state_array)[1] + 1))
        state_array_ext[:, :-1] = self.state_array  # TODO: make sure don't need array copy
        for idx, site in enumerate(randomized_sites):  # TODO: parallelize
            state_array_ext = glauber_dynamics_update(state_array_ext, site, self.steps, app_field=app_field)
        self.state_array = state_array_ext
        self.steps += 1
        self.state = state_array_ext[:, -1]
        return self.state

    def write_state(self, datadir):
        state_write(self.state_array, range(self.steps), self.spin_labels, "sc_state_%s" % self.label, "times", "spin_labels", datadir)
