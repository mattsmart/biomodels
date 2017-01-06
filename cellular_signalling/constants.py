# global constants
NUM_TYPE_LIGAND = 2
NUM_TYPE_RECEPTOR = 1
NUM_TYPE_SIGNAL = 1

# concentration (mol/L) of ligands
# TODO: remove these and solve for them?
l0 = 2.0 
l1 = 0.5

# number of receptors available of each type
m0 = 100

# steady state concentratons (mol/L) of signal molecules
s0 = 0.2

# production matrix A_ij = production rate of s_i when receptor type j bound
# note: dim A = NUM_TYPE_SIGNAL x NUM_TYPE_RECEPTOR
A = [3.0]

# diagonal degredation matrix W entries (W_ii = gamma_i = deg rate of s_i)
# note: dim W = NUM_TYPE_SIGNAL x NUM_TYPE_SIGNAL
gamma0 = 6.0
W = [gamm0]

# ligand i -> receptor j binding affinities eps_ij
eps_00 = 4.0
eps_10 = 50.0

# chemical potential mu_i for ligand i
# TODO: these should be function of concentrations above..
mu_0 = 0.5
mu_1 = 1.0
