import numpy as np

# project level constants
FOLDER_OUTPUT = "output"

# default model parameters (master genes)
DIM_MASTER = 2
BETA = 7.0
TAU = 1.8
HILL_COEFF = 1.0                                         # TODO not sure how to incorporate, fix at 1.0 for now
GAMMA = 1.3

# default model parameters (slave genes)
DIM_SLAVE = 2
ALPHAS = [0.2 for _ in xrange(DIM_SLAVE)]
BETAS = [5.4 for _ in xrange(DIM_SLAVE)]
TAUS = [3.2 for _ in xrange(DIM_SLAVE)]

# param dicts
PARAMS_ID = {0: "dim_master",
             1: "dim_slave",
             2: "beta",
             3: "tau",
             4: "hill_coeff",
             5: "gamma",
             6: "alphas",
             7: "betas",
             8: "taus"}
PARAMS_DEFAULTS_DICT = {'dim_master': DIM_MASTER,
                       'dim_slave': DIM_SLAVE,
                       'beta': BETA,
                       'tau': TAU,
                       'hill_coeff': HILL_COEFF,
                       'gamma': GAMMA,
                       'alphas': ALPHAS,
                       'betas': BETAS,
                       'taus': TAUS}

# misc
assert DIM_MASTER == 2
INIT_COND = [7.0, 2.0] + [5.0 for _ in xrange(DIM_SLAVE)]
TIMESTEP = 0.1
NUM_TRAJ = 30
NUM_STEPS = 2000


class Params:

    def __init__(self, params_dict=PARAMS_DEFAULTS_DICT):
        # model dimension
        self.dim_master = None
        self.dim_slave = None
        # master gene params
        self.beta = None                                 # Note: this is new
        self.tau = None                                   # TODO consider split into tau_x, tau_y params
        self.hill_coeff = None
        self.gamma = None
        # slave gene params
        self.alphas = None
        self.betas = None
        self.taus = None
        # assign all default params
        for k, v in params_dict.iteritems():
            setattr(self, k, v)
        # add the leftover elements to params_dict
        self.params_dict = params_dict
        keys_to_add = set(PARAMS_ID.values()) - set(params_dict.keys())
        for key in keys_to_add:
            self.params_dict[key] = getattr(self, key)        # housekeeping
        self.params_dict = params_dict
        # housekeeping
        assert self.hill_coeff == 1.0  # unclear how to put in model otherwise (based on pdf)
        self.dim = self.dim_master + self.dim_slave
        if self.dim_master == 2:
            self.state_dict = {idx: 'v_%d' % (idx-1) for idx in xrange(2, self.dim)}
            self.state_dict.update({0: 'x', 1: 'y'})
        else:
            self.state_dict = None

    def printer(self):
        print "Dimension: (total = master + slave): %d = %d + %d" % (self.dim, self.dim_master, self.dim_slave)
        print "Master params:"
        print "\ttau = %.3f" % self.tau
        print "\thill_coeff = %.3f" % self.hill_coeff
        print "\tgamma = %.3f" % self.gamma
        print "Slave params:"
        print "\talphas = %s" % self.alphas
        print "\tbetas = %s" % self.betas
        print "\ttaus = %s" % self.taus

    def mod_copy(self, new_values):
        """
        new_values is dict of pairs of form param id: val
        return new params instance
        """
        params_dict_new = dict(self.params_dict)
        for k,v in new_values.iteritems():
            params_dict_new[k] = v
        return Params(params_dict_new)


# prep default params for typical simulation
DEFAULT_PARAMS = Params()
