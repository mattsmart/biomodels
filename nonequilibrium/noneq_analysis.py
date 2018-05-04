import numpy as np
from scipy.linalg import logm, expm

from noneq_settings import BETA, build_J
from noneq_functions import state_to_label, label_to_state, hamiltonian, hamming, get_adjacent_labels


def get_states_energies_prob(N, J, beta=BETA):
    assert (J.transpose() == J).all()  # assert J is symm
    # only makes sense for symmetric J
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
    Q = np.zeros((num_states, num_states))
    for i in xrange(num_states):
        for j in xrange(num_states):
            hamming_dist = hamming(states[i], states[j])
            if hamming_dist == 1:
                energy_diff = energies[j] - energies[i]
                Q[i,j] = 1.0/(1.0 + np.exp(-beta*energy_diff)) / N  # prob jump from j to i
            elif hamming_dist == 0:  # here i == j
                prob_leave = 0.0
                adjacent_labels = get_adjacent_labels(states[i])
                for k in adjacent_labels:
                    energy_diff = energies[i] - energies[k]
                    prob_leave += 1.0 / (1.0 + np.exp(-beta * energy_diff)) / N
                Q[i,j] = 1 - prob_leave  # prob stay at site (diagonals) is 1 - prob_leave
            else:
                Q[i,j] = 0
    return Q


def construct_flux_matrix(Q):
    # based on: Wang, 2017, PLOS
    eigenpairs = eigen(Q)
    pss_pair = eigenpairs[-1]
    assert 1.0 - 1e-4 < pss_pair[0] < 1.0 + 1e-4
    pss = pss_pair[1]

    F = np.zeros(Q.shape)
    for i in xrange(Q.shape[0]):
        for j in xrange(Q.shape[1]):
            F[i,j] = -Q[j,i]*pss[i] + Q[i,j]*pss[j]
    return F


def decompose_matrix(Q):
    # TODO: don't think this works for stoch matrix Q, only stoch rate (W-) matrices (LITERATURE UNCLEAR)
    # TODO: fix this.. if possible
    # based on: Wang, 2017, PLOS
    # Q = C + D
    # C is irreversible and does not satisfy detailed balance
    # D is detailed-balnce prserving part
    eigenpairs = eigen(Q)
    pss_pair = eigenpairs[-1]
    assert 1.0 - 1e-4 < pss_pair[0] < 1.0 + 1e-4
    pss = pss_pair[1]

    C = np.zeros(Q.shape)
    D = np.zeros(Q.shape)
    for i in xrange(Q.shape[0]):
        for j in xrange(Q.shape[1]):
            C[i, j] = max(Q[j,i]*pss[i] - Q[i,j]*pss[j], 0)
            D[i, j] = min(Q[j,i]*pss[i],  Q[i,j]*pss[j])
    # set diagonals
    np.fill_diagonal(C, 0)
    np.fill_diagonal(D, 0)
    for i in xrange(Q.shape[0]):
        C[i, i] = 1 - np.sum([C[j,i] for j in xrange(Q.shape[1])])  # columns sum to 1
        D[i, i] = 1 - np.sum([D[j,i] for j in xrange(Q.shape[1])])  # columns sum to 1
    return C, D


def get_transition_rate_matrix(N, J, tau=1.0, beta=BETA):
    # TODO: fix this.. if possible
    # suppose p_k+1 = Q * p_k for some known Q
    # then determine corresponding generator M
    # if one knew M, could construct Q ~ I + M*tau (tau = timestep)
    # here we take: scipy.linalg.logm(Q) since Q = e^(M tau)
    Q = get_transition_matrix(N, J, beta=BETA)
    M = logm(Q) / tau
    print expm(M)
    return M


def eigen(A):
    eigenval, eigenvec = np.linalg.eig(A)
    #eigenval, eigenvec = zip(*sorted(zip(eigenval, eigenvec)))
    pairs = [(eigenval[idx], eigenvec[:,idx]) for idx in xrange(len(eigenval))]
    pairs.sort(key=lambda pair: pair[0])
    return pairs


def analyze_transition_rate_matrices(N, J, beta=BETA, tau=1.0):
    states, energies, probabilities = get_states_energies_prob(N, J, beta=beta)
    print "The system's state labels, energies, boltzmann probabilities"
    for label in xrange(2**N):
        print "State:", label, "is", states[label], "H(state):", energies[label], "exp(-beta*H(x))/Z:", probabilities[label]
    print "Normality", np.sum(probabilities)

    print "The transition matrix Q is:"
    Q = get_transition_matrix(N, J, beta=beta)
    M = get_transition_rate_matrix(N, J, tau=tau, beta=beta)
    np.set_printoptions(precision=3)
    print Q
    print "The transition rate matrix M is:"
    print M
    Q_eigenpairs = eigen(Q)
    M_eigenpairs = eigen(M)
    print "Q eigenvalues are:"
    print Q_eigenpairs
    print "M eigenvalues are:"
    print M_eigenpairs
    print "Q Pss: third eigenvector (check that it corresp. eigenvalue 1)"
    Q_Pss = Q_eigenpairs[-1][1]
    print Q_eigenpairs[-1][0], Q_Pss
    print "QPss = Pss"
    print np.dot(Q, Q_Pss)
    print "M Pss: third eigenvector (check that it corresp. eigenvalue 0)"
    M_Pss = M_eigenpairs[-1][1]
    print M_eigenpairs[-1][0], M_Pss
    print "MPss = Pss"
    print np.dot(M, M_Pss)
    print "normalized 1-eigenvector of Q:"
    print Q_Pss / np.sum(Q_Pss)
    print "normalized 0-eigenvector of M:"
    print M_Pss / np.sum(M_Pss)


if __name__ == '__main__':
    # settings
    beta=0.2 #0.2
    tau=1e-3
    N = 3
    J = build_J(N, id='symm')
    print "J is\n", J

    # stoch matrix decompositions
    Q = get_transition_matrix(N, J, beta=beta)
    F = construct_flux_matrix(Q)
    C, D = decompose_matrix(Q)
    np.set_printoptions(precision=2)
    print F
    print C
    print D
    print "commute?"
    print np.allclose(np.dot(C,D), np.dot(D,C), atol=1e-04)

    # stoch rate matrix construction checks
    #analyze_transition_rate_matrices(N, J, beta=beta, tau=tau)