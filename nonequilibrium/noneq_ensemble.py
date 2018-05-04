import matplotlib.pyplot as plt
import numpy as np

from noneq_settings import BETA
from noneq_functions import state_to_label, label_to_state, hamiltonian
from noneq_plotting import plot_steadystate_dist, plot_boltzmann_dist, autocorr_label_timeseries, fft_label_timeseries, \
                           plot_label_timeseries, visualize_ensemble_label_timeseries
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


def get_ensemble_label_timeseries(ensemble_size, total_steps, N, J, visual=False):
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


def mean_label_timeseries(ensemble_size, N, J, total_steps=10000):
    ensemble_label_timeseries = get_ensemble_label_timeseries(ensemble_size, total_steps, N, J)
    mean_label_timeseries = np.mean(ensemble_label_timeseries, axis=0)
    assert len(mean_label_timeseries) == total_steps
    plt.plot(range(total_steps), mean_label_timeseries)
    plt.title('Mean label timeseries (%d steps, %d ensemble size)' % (total_steps, ensemble_size))
    plt.xlabel('steps')
    plt.ylabel('mean state label over all trajectories at step t')
    plt.show()
    return 0


def periodicity_analysis(ensemble_label_timeseries, endratio=0.01):
    ensemble_size, total_steps = np.shape(ensemble_label_timeseries)
    steadystate_steps = int(endratio * total_steps)
    print "total steps:", total_steps
    print "steps at end we are calling steady state:", steadystate_steps
    for traj in xrange(ensemble_size):
        steadystate_traj = ensemble_label_timeseries[traj,-steadystate_steps:]
        fft_label_timeseries(steadystate_traj)
        autocorr_label_timeseries(steadystate_traj)
    return 0


if __name__ == '__main__':
    # settings
    N = 3
    """
    J_symm = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]])
    J_broken1 = np.array([[0, 1, 2],
                          [0.5, 0, 1],
                          [1, 1, 0]])
    J_broken2 = np.array([[0, 1, 2],
                          [-0.5, 0, 1],
                          [1, 1, 0]])
    J_broken3 = np.array([[0, 0.1, 6],
                          [-1, 0, -0.1],
                          [-4, 10, 0]])
    J_general = np.array([[0, -61, -100],
                          [-9, 0, -1],
                          [87, 11, 0]])
    J = J_broken2
    """


    N = 4
    factor_asymm = 0.1
    mem = [[1 for i in xrange(N)]]
    XI = np.transpose(np.array(mem))
    J = np.dot(XI, np.transpose(XI)) - np.eye(N)
    J = J + factor_asymm*np.random.uniform(-1.0,1.0,(N,N))
    np.fill_diagonal(J, 0)
    print XI
    print np.transpose(XI)
    print J


    # flags
    flag_timeseries_periodicity = False
    flag_mean_timeseries = False
    flag_visualize = True

    # analysis (plot label timeseries, periodicity plots)
    if flag_timeseries_periodicity:
        small_ensemble_size = 1
        total_steps = 10000
        steadystate_fraction = 0.1  # last x percent of the time
        ensemble_label_timeseries = get_ensemble_label_timeseries(small_ensemble_size, total_steps, N, J)
        plot_label_timeseries(ensemble_label_timeseries, endratio=steadystate_fraction)
        periodicity_analysis(ensemble_label_timeseries, endratio=steadystate_fraction)

    # analysis (mean label timeseries)
    if flag_mean_timeseries:
        ensemble_size_mean = 100
        mean_label_timeseries(ensemble_size_mean, N, J)

    # visualize few steps of big ensemble
    if flag_visualize:
        ensemble_label_timeseries = get_ensemble_label_timeseries(1000, 10, N, J)
        visualize_ensemble_label_timeseries(ensemble_label_timeseries, N)
