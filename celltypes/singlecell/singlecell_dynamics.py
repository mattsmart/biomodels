import numpy as np
from random import random, shuffle

from singlecell_data_io import load_singlecell_data, binarize_data


def interaction_matrix(xi, method="projection"):  # note may be off by scaling factor of 1/N
    if method == "hopfield":
        return np.dot(xi, xi.T)
    elif method == "projection":
        A = np.dot(xi.T, xi)
        return reduce(np.dot, [xi, np.linalg.inv(A), xi.T])
    else:
        raise ValueError("method arg invalid, must be one of %s" % ["projection", "hopfield"])


def hamiltonian(J, state, t):  # plus some other field terms... do we care for these?
    return -0.5*reduce(np.dot, [state[:,t].T, J, state[:,t]])


def local_field(J, state, idx, t):
    h_i = 0
    intxn_list = range(0,idx) + range(idx+1,N)
    for j in intxn_list:
        h_i += J[idx,j]*state[j,t]  # plus some other field terms... do we care for these?
    return h_i


def glauber_dynamics_update(J, state, idx, t):
    r1 = random()
    beta_h = BETA * local_field(J, state, idx, t)
    prob_on_after_timestep = 1 / (1 + np.exp(-2*beta_h))  # probability that site i will be on after the timestep
    if prob_on_after_timestep > r1:
        state[idx,t+1] = 1
    else:
        state[idx, t + 1] = -1
    return state


# Constants
BETA = 2.2  # value used in Mehta 2014
METHOD = "projection"
TIMESTEPS = 20

# Data loading
xi, celltype_labels, gene_labels = load_singlecell_data()
xi_bool = binarize_data(xi)
N = len(gene_labels)
p = len(celltype_labels)

# Variable setup
J = interaction_matrix(xi_bool, method=METHOD)
init_state = -1 + np.zeros((N,1))
randomized_sites = range(N)
state = np.zeros((N,TIMESTEPS))
state[:,0] = init_state[:,0]

# Simulate
for t in xrange(TIMESTEPS-1):
    shuffle(randomized_sites)  # randomize site ordering each timestep updates
    for idx, site in enumerate(randomized_sites):  # TODO: parallelize
        state = glauber_dynamics_update(J, state, idx, t)
    print t
print state