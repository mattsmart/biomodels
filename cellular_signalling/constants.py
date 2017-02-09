import numpy as np

# global constants
NUM_TYPE_LIGAND = 2
NUM_TYPE_RECEPTOR = 1
NUM_TYPE_SIGNAL = 1
KBT = 0.1  # TODO for approp temp

# concentration (mol/L) of ligands
# TODO: remove these and solve for them?cant have these and mu_i as given
#l = [2.0, 0.5]
#assert len(l) == NUM_TYPE_LIGAND

# number of receptors available of each type
m = [100]
assert len(m) == NUM_TYPE_RECEPTOR

# steady state concentratons (mol/L) of signal molecules
s = [0.2]
assert len(s) == NUM_TYPE_SIGNAL

# production matrix A_ij = production rate of s_i when receptor type j bound
# note: dim A = NUM_TYPE_SIGNAL x NUM_TYPE_RECEPTOR
A = [3.0]

# diagonal degredation matrix W entries (W_ii = gamma_i = deg rate of s_i)
# note: dim W = NUM_TYPE_SIGNAL x NUM_TYPE_SIGNAL
gamma0 = 6.0
W = [gamma0]

# ligand i -> receptor j binding affinities eps_ij
eps = [[4.0], [50.0]]

# chemical potential mu_i for ligand i
# TODO: these should be function of concentrations above..
mu_ref = 0.75
mu = [mu_ref + KBT * np.log(l[i]) for i in xrange(NUM_TYPE_LIGAND)]
assert len(mu) == NUM_TYPE_LIGAND
