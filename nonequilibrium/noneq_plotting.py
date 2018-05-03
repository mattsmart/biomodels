import matplotlib.pyplot as plt
import numpy as np

from noneq_constants import BETA
from noneq_functions import state_to_label, label_to_state, hamiltonian


def plot_steadystate_dist(pss):
    state_space = len(pss)
    plt.bar(range(state_space), pss)
    plt.title('Steady state probability distribution')
    plt.ylabel('Probability')
    plt.xlabel('State label (bit<->int method)')
    plt.show()
    return 0


def plot_boltzmann_dist(N, J, beta=BETA):
    energy_vals = [hamiltonian(label_to_state(idx, N), J) for idx in xrange(2 ** N)]
    propensities = np.exp([-beta*ev for ev in energy_vals])
    partition_fn = np.sum(propensities)
    plt.bar(range(2 ** N), propensities / partition_fn)
    plt.title('H(x) = - 0.5 x^T J x determined boltzmann dist')
    plt.ylabel('e^-beta*H(x) / Z')
    plt.xlabel('State label')
    plt.show()
    return 0
