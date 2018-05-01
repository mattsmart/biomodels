import matplotlib.pyplot as plt
import numpy as np

from noneq_constants import BETA
from noneq_functions import state_to_label, label_to_state, hamiltonian
from noneq_simulate import state_simulate

# TODO optimization-dictionary in memory of state labels to spins (and vice versa? invert how) and see if runtime improves
# TODO why does PSS_sim not match PSS_boltzmann? especially for higher temperatures


def get_steadystate_dist(ensemble_size, total_steps, N, J):
    labels_to_states = {idx:label_to_state(idx, N) for idx in xrange(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.iteritems()}

    occupancy_counts = np.zeros(2**N)
    for system in xrange(ensemble_size):
        state_array, _, _, _, _ = state_simulate(init_state=None, init_id=None, N=N, iterations=total_steps, intxn_matrix=J,
                                                 flag_makedir=False, app_field=None, flag_write=False, analysis_subdir="ensemble", plot_period=10)
        end_state = state_array[:, -1]
        end_label = states_to_labels[tuple(end_state)] #state_to_label(end_state)
        print system, end_state, end_label
        occupancy_counts[end_label] += 1
    return occupancy_counts / float(ensemble_size)

if __name__ == '__main__':
    # settings
    ensemble_size = int(1e3)
    total_steps = int(1e5)
    N = 3
    J = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])

    pss = get_steadystate_dist(ensemble_size, total_steps, N, J)
    plt.bar(range(2**N), pss)
    plt.title('Steady state probability distribution')
    plt.ylabel('Probability')
    plt.xlabel('State label')
    plt.show()

    energy_vals = [hamiltonian(label_to_state(idx, N), J) for idx in xrange(2 ** N)]
    propensities = np.exp([-BETA*ev for ev in energy_vals])
    partition_fn = np.sum(propensities)
    plt.bar(range(2 ** N), propensities / partition_fn)
    plt.title('H(x) = - x^T J x determined boltzmann dist')
    plt.ylabel('e^beta*H(x) / Z')
    plt.xlabel('State label')

    plt.show()
