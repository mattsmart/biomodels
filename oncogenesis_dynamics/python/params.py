import csv
from os import sep

from constants import ODE_SYSTEMS, PARAMS_ID, PARAMS_ID_INV
from data_io import read_params, write_params


class Params(object):

    def __init__(self, params_list, system, init_cond=None):
        alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z, mu_base = params_list
        # vector of params_list
        self.params = params_list
        # params_list as scalars
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.mu = mu
        self.a = a
        self.b = b
        self.c = c
        self.N = N
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z
        self.mu_base = mu_base
        # init_cond as x, y, z
        self.init_cond = init_cond  # TODO not fully implemented
        # system as defined in constants.pu (e.g. 'default', 'feedback_z')
        assert system in ODE_SYSTEMS
        self.system = system

    def __str__(self):
        return str(self.params)

    def __iter__(self):
        return self.params

    def get(self, param_label):
        # TODO implement (also modify params list attribute
        return self.params[PARAMS_ID_INV[param_label]]

    def mod_copy(self, new_values):
        """
        new_values is list of pairs of form (param id, val)
        return new params instance
        """
        params_shift_list = self.params_list()
        for pair in new_values:
            params_shift_list[PARAMS_ID_INV[pair[0]]] = pair[1]
        return Params(params_shift_list, self.system)

    def printer(self):
        for idx in xrange(len(PARAMS_ID.keys())):
            print "Param %d: (%s) = %.3f" % (idx, PARAMS_ID[idx], self.params[idx])

    def params_list(self):
        params_list = self.params[:]
        return params_list

    def write(self, filedir, filename):
        filepath = filedir + sep + filename
        with open(filepath, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for idx in xrange(len(PARAMS_ID.keys())):
                if self.params[idx] is None:
                    self.params[idx] = 'None'
                writer.writerow([PARAMS_ID[idx], self.params[idx]])
            # any extra non-dynamics params
            writer.writerow(['system', self.system])
        return filepath

    @staticmethod
    def read(filedir, filename):
        with open(filedir + sep + filename, 'rb') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
            num_params = sum(1 for row in datareader)
            csvfile.seek(0)
            params_with_system = [0.0] * num_params
            for idx, pair in enumerate(datareader):
                if idx < num_params - 1:
                    assert pair[0] == PARAMS_ID[idx]
                    if pair[1] != 'None':
                        params_with_system[idx] = float(pair[1])
                    else:
                        params_with_system[idx] = None
                else:
                    assert pair[0] == 'system'
                    params_with_system[idx] = pair[1]
        params_list = params_with_system[:-1]
        system = params_with_system[-1]
        assert system in ODE_SYSTEMS
        return Params(params_list, system, init_cond=None)
