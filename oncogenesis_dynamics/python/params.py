import numpy as np

from constants import ODE_SYSTEMS, PARAMS_ID
from data_io import read_params, write_params


class Params(object):

    def __init__(self, params, system, init_cond=None):
        alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params
        # vector of params
        self.params = params
        # params as scalars
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.mu = mu
        self.a = a
        self.b = b
        self.c = c
        self.N = N
        self.v_x = v_X
        self.v_y = v_y
        self.v_z = v_z
        self.mu_base = mu_base
        # init_cond as x, y, z
        self.init_cond = init_cond
        # system as defined in constants.pu (e.g. 'default', 'feedback_z')
        assert system in ODE_SYSTEMS
        self.system = system

    def __str__(self):
        return self.params

    def printer(self):
        for idx in xrange(len(PARAMS_ID.keys())):
            print "Param %d: (%s) = %.3f" % (idx, PARAMS_ID[idx], self.params[idx])

    def params_copy(self):  # TODO maybe make this and other copy instead of pass pointer np.zeros(self.N).. etc
        params_copy = self.params[:]
        return params_copy

    def write(self, filedir, filename):
        filepath = write_params(self.params, self.system, filedir, filename)
        return filepath

    def read(filedir, filedir, filename):
        params_with_system = read_params(filedir, filename)
        params = params_with_system[:-1]
        system = params_with_system[-1]
        assert system in ODE_SYSTEMS
        return Params(params, system, init_cond=None)
