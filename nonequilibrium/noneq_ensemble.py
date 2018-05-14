import matplotlib.pyplot as plt
import numpy as np

from noneq_analysis import get_flux_dict
from noneq_settings import BETA, build_J
from noneq_functions import state_to_label, label_to_state, hamiltonian
from noneq_plotting import plot_steadystate_dist, plot_boltzmann_dist, autocorr_label_timeseries, fft_label_timeseries, \
                           plot_label_timeseries, visualize_ensemble_label_timeseries
from noneq_simulate import state_simulate

# TODO optimization-dictionary in memory of state labels to spins (and vice versa? invert how) and see if runtime improves


def get_steadystate_dist_simple(ensemble_size, total_steps, N, J, beta=BETA):
    # get steady state of ensemble without wasting resources collecting other statistics
    labels_to_states = {idx:label_to_state(idx, N) for idx in xrange(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.iteritems()}
    occupancy_counts = np.zeros(2**N)
    for system in xrange(ensemble_size):
        state_array, _, _, _, _ = state_simulate(init_state=None, init_id=None, N=N, iterations=total_steps, intxn_matrix=J, beta=beta,
                                                 flag_makedir=False, app_field=None, flag_write=False, analysis_subdir="ensemble", plot_period=10)
        end_state = state_array[:, -1]
        end_label = states_to_labels[tuple(end_state)] #state_to_label(end_state)
        if system % 1000 == 0:
            print system, end_state, end_label
        occupancy_counts[end_label] += 1
    return occupancy_counts / float(ensemble_size)


def get_ensemble_label_timeseries(ensemble_size, total_steps, N, J, beta=BETA):
    labels_to_states = {idx:label_to_state(idx, N) for idx in xrange(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.iteritems()}
    ensemble_label_timeseries = np.zeros((ensemble_size, total_steps))
    for system in xrange(ensemble_size):
        state_array, _, _, _, _ = state_simulate(init_state=None, init_id=None, N=N, iterations=total_steps, intxn_matrix=J, beta=beta,
                                                 flag_makedir=False, app_field=None, flag_write=False, analysis_subdir="ensemble", plot_period=10)
        label_timeseries = [states_to_labels[tuple(state_array[:,step])] for step in xrange(total_steps)]
        ensemble_label_timeseries[system, :] = label_timeseries
    return ensemble_label_timeseries


def get_ensemble_spin_statistics(ensemble_size, N, J, total_steps=10000, beta=BETA):
    labels_to_states = {idx:label_to_state(idx, N) for idx in xrange(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.iteritems()}
    ensemble_label_timeseries = get_ensemble_label_timeseries(ensemble_size, total_steps, N, J, beta=beta)
    ensemble_size, total_steps = np.shape(ensemble_label_timeseries)
    spin_mean_timeseries = np.zeros((N, total_steps))
    for step in xrange(total_steps):
        ensemble_labels_at_t = ensemble_label_timeseries[:,step]
        for spin_idx in xrange(N):
            spin_sum_at_t = 0.0
            for sample in xrange(ensemble_size):
                state = labels_to_states[ensemble_labels_at_t[sample]]
                spin_sum_at_t += state[spin_idx]
            spin_mean_timeseries[spin_idx, step] = spin_sum_at_t / ensemble_size
    return spin_mean_timeseries


def mean_label_timeseries(ensemble_size, N, J, total_steps=10000, beta=BETA):
    ensemble_label_timeseries = get_ensemble_label_timeseries(ensemble_size, total_steps, N, J, beta=beta)
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
    beta=1.0
    N = 3
    J = build_J(N, id='asymm_2')
    #J = build_J(N, id='asymm_2')
    #J = build_J(N, id='asymm_1')

    # flags
    flag_timeseries_periodicity = False
    flag_mean_label_timeseries = False
    flag_mean_spin_timeseries = False
    flag_visualize = False
    flag_steadystate_simple = True

    # analysis (plot label timeseries, periodicity plots)
    if flag_timeseries_periodicity:
        small_ensemble_size = 1
        total_steps = 10000
        steadystate_fraction = 0.1  # last x percent of the time
        ensemble_label_timeseries = get_ensemble_label_timeseries(small_ensemble_size, total_steps, N, J, beta=beta)
        plot_label_timeseries(ensemble_label_timeseries, endratio=steadystate_fraction)
        periodicity_analysis(ensemble_label_timeseries, endratio=steadystate_fraction)

    # analysis (mean label timeseries)
    if flag_mean_label_timeseries:
        ensemble_size_mean = 100
        mean_label_timeseries(ensemble_size_mean, N, J, beta=beta)

    # analysis (mean spin timeseries)
    if flag_mean_spin_timeseries:
        ensemble_size_mean = 1000
        spin_mean_steps = 100
        spin_mean_timeseries = get_ensemble_spin_statistics(ensemble_size_mean, N, J, beta=beta, total_steps=spin_mean_steps)
        plt.plot(range(spin_mean_steps), spin_mean_timeseries[0, :])
        plt.plot(range(spin_mean_steps), spin_mean_timeseries[1, :])
        plt.show()

    # visualize few steps of big ensemble
    if flag_visualize:
        ensemble_label_timeseries = get_ensemble_label_timeseries(1000, 10, N, J, beta=beta)
        flux_dict = get_flux_dict(N, J, beta=beta)
        flux_dict_str = {key:"%.3f" % flux_dict[key] for key in flux_dict.keys()}
        visualize_ensemble_label_timeseries(ensemble_label_timeseries, N, flux_dict=flux_dict_str)

    if flag_steadystate_simple:
        pss = get_steadystate_dist_simple(10000, 100, N,J,beta=beta)
        plot_steadystate_dist(pss)
        plot_boltzmann_dist(N,J,beta=beta)
