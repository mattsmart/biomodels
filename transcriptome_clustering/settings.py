import numpy as np

# project level constants
FOLDER_OUTPUT = "output"

# model specification
SYSTEM = np.array([[-1.0, 4],
                   [-0.5, 0.2]])

# default model parameters (master genes)
STATE_MASTER_DIM = 2
TAU = 1.9
HILL_COEFF = 1.0
GAMMA = 1.3

# default model parameters (slave genes)
STATE_SLAVE_DIM = 2
ALPHAS = [0.2 for _ in xrange(STATE_SLAVE_DIM)]
BETAS = [0.4 for _ in xrange(STATE_SLAVE_DIM)]
TAUS = [0.1 for _ in xrange(STATE_SLAVE_DIM)]


class Params:

    def __init__(self):
        # model dimension
        self.dim_master = STATE_MASTER_DIM
        self.dim_slave = STATE_SLAVE_DIM
        self.dim = self.dim_master + self.dim_slave
        # master gene params
        self.tau = TAU                                   # TODO consider split into tau_x, tau_y params
        self.hill_coeff = HILL_COEFF                     # TODO not sure how to incorporate, fix at 1.0 for now
        self.gamma = GAMMA
        # slave gene params
        self.alphas = ALPHAS
        self.betas = BETAS
        self.taus = TAUS
        assert self.hill_coeff == 1.0  # unclear how to put in model otherwise (based on pdf)

    def printer(self):
        print "Dimension: (total = master + slave)"
        print "%d = %d + %d" % (self.dim, self.dim_master, self.dim_slave)
        print "Master params:"
        print "tau = %.3f" % self.tau
        print "hill_coeff = %.3f" % self.hill_coeff
        print "gamma = %.3f" % self.gamma
        print "Slave params:"
        print "alphas = %s" % self.alphas
        print "betas = %s" % self.betas
        print "taus = %s" % self.taus


# prep default params for typical simulation
DEFAULT_PARAMS = Params()
