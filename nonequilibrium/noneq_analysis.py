import numpy as np

from noneq_constants import BETA
from noneq_functions import state_to_label, label_to_state, hamiltonian, hamming, get_adjacent_labels


def get_states_energies_prob(N, J, beta=BETA):
    num_states = 2 ** N
    Z = 0.0
    states = np.zeros((num_states, N), dtype=int)
    energies = np.zeros(num_states)
    propensities = np.zeros(num_states)
    for label in xrange(num_states):
        state = label_to_state(label, N)
        energy = hamiltonian(state, J)
        propensity = np.exp(-beta*energy)
        states[label, :] = state
        energies[label] = energy
        propensities[label] = propensity
        Z += np.exp(-beta*energy)
    probabilities = propensities/Z
    return states, energies, probabilities


def get_transition_matrix(N, J, beta=BETA):
    # p 111 Amit divide by N eah probability because of a priori probability of picking specific spin to flip
    num_states = 2**N
    states, energies, probabilities = get_states_energies_prob(N, J, beta=beta)
    M = np.zeros((num_states, num_states))
    for i in xrange(num_states):
        for j in xrange(num_states):
            hamming_dist = hamming(states[i], states[j])
            if hamming_dist == 1:
                energy_diff = energies[j] - energies[i]
                M[i,j] = 1.0/(1.0 + np.exp(-beta*energy_diff)) / N  # prob jump from j to i
            elif hamming_dist == 0:  # here i == j
                prob_leave = 0.0
                adjacent_labels = get_adjacent_labels(states[i])
                for k in adjacent_labels:
                    energy_diff = energies[i] - energies[k]
                    prob_leave += 1.0 / (1.0 + np.exp(-beta * energy_diff)) / N
                M[i,j] = 1 - prob_leave  # prob stay at site (diagonals) is 1 - prob_leave
            else:
                M[i,j] = 0
    return M


if __name__ == '__main__':
    beta=2.0 #0.2
    N = 3
    J = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])
    states, energies, probabilities = get_states_energies_prob(N, J, beta=beta)
    print "The system's state labels, energies, boltzmann probabilities"
    for label in xrange(2**N):
        print "State:", label, "is", states[label], "H(state):", energies[label], "exp(-beta*H(x))/Z:", probabilities[label]
    print "Normality", np.sum(probabilities)

    print "The transition matrix is:"
    M = get_transition_matrix(N, J, beta=beta)
    np.set_printoptions(precision=3)
    print M
    eigenval, eigenvec = np.linalg.eig(M)
    print "eigenvalues are:"
    print eigenval
    print "Pss: third eigenvector (check that it corresp. eigenvalue 1)"
    Pss = eigenvec[:,2]
    print Pss
    print "MPss = Pss"
    print np.dot(M, Pss)
    print "normalized 1-eigenvector:"
    print Pss / np.sum(Pss)