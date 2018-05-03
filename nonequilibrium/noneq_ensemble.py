import matplotlib.pyplot as plt
import numpy as np

from noneq_constants import BETA
from noneq_functions import state_to_label, label_to_state, hamiltonian
from noneq_plotting import plot_steadystate_dist, plot_boltzmann_dist
from noneq_simulate import state_simulate

# TODO optimization-dictionary in memory of state labels to spins (and vice versa? invert how) and see if runtime improves


def get_steadystate_dist_simple(ensemble_size, total_steps, N, J):
    # get steady state of ensemble without wasting resources collecting other statistics
    labels_to_states = {idx:label_to_state(idx, N) for idx in xrange(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.iteritems()}

    occupancy_counts = np.zeros(2**N)
    for system in xrange(ensemble_size):
        state_array, _, _, _, _ = state_simulate(init_state=None, init_id=None, N=N, iterations=total_steps, intxn_matrix=J,
                                                 flag_makedir=False, app_field=None, flag_write=False, analysis_subdir="ensemble", plot_period=10)
        end_state = state_array[:, -1]
        end_label = states_to_labels[tuple(end_state)] #state_to_label(end_state)
        if system % 1000 == 0:
            print system, end_state, end_label
        occupancy_counts[end_label] += 1
    return occupancy_counts / float(ensemble_size)


def get_ensemble_label_timeseries(ensemble_size, total_steps, N, J):
    labels_to_states = {idx:label_to_state(idx, N) for idx in xrange(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.iteritems()}
    ensemble_label_timeseries = np.zeros((ensemble_size, total_steps))
    for system in xrange(ensemble_size):
        state_array, _, _, _, _ = state_simulate(init_state=None, init_id=None, N=N, iterations=total_steps, intxn_matrix=J,
                                                 flag_makedir=False, app_field=None, flag_write=False, analysis_subdir="ensemble", plot_period=10)
        label_timeseries = [states_to_labels[tuple(state_array[:,step])] for step in xrange(total_steps)]
        ensemble_label_timeseries[system, :] = label_timeseries
    return ensemble_label_timeseries


def get_ensemble_statistics(ensemble_size, total_steps, N, J):
    """
    labels_to_states = {idx:label_to_state(idx, N) for idx in xrange(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.iteritems()}

    occupancy_counts = np.zeros(2**N)
    for system in xrange(ensemble_size):
        state_array, _, _, _, _ = state_simulate(init_state=None, init_id=None, N=N, iterations=total_steps, intxn_matrix=J,
                                                 flag_makedir=False, app_field=None, flag_write=False, analysis_subdir="ensemble", plot_period=10)
        end_state = state_array[:, -1]
        end_label = states_to_labels[tuple(end_state)] #state_to_label(end_state)
        if system % 1000 == 0:
            print system, end_state, end_label
        occupancy_counts[end_label] += 1
    return occupancy_counts / float(ensemble_size)
    """
    return 0


if __name__ == '__main__':
    # settings
    ensemble_size = 3 #int(1e5)
    total_steps = 10000 #int(1e2)
    N = 3
    J = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])
    ensemble_label_timeseries = get_ensemble_label_timeseries(ensemble_size, total_steps, N, J)
    print ensemble_label_timeseries

    # periodicity analysis
    for traj in xrange(ensemble_size):
        f_components = np.fft.rfft(ensemble_label_timeseries[traj,:])
        f_axis = np.fft.rfftfreq(total_steps, d=1.0)
        plt.plot(f_axis, np.abs(f_components))
        plt.show()
