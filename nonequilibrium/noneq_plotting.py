import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from noneq_settings import BETA
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


def plot_label_timeseries(ensemble_label_timeseries, endratio=1.0):
    assert 0.0 < endratio <= 1.0
    ensemble_size, T = np.shape(ensemble_label_timeseries)
    start = int((1-endratio)*T)
    stop = T
    ensemble_label_timeseries = ensemble_label_timeseries[:,start:stop]
    plt.plot(range(start, stop), ensemble_label_timeseries.transpose())
    plt.title('%d trajectories (defined by integer state labels) truncated to last %.2f steps' % (ensemble_size, endratio))
    plt.xlabel('steps')
    plt.ylabel('label (state) at time t')
    plt.show()
    return 0


def autocorr_label_timeseries(traj):
    T = len(traj)
    vertical_scale = np.dot(traj, traj)
    f_components = np.fft.rfft(traj)
    Cw = f_components.conj() * f_components
    autocorr = np.fft.fftshift(np.fft.irfft(Cw))
    lag_axis = np.arange(-T/2, T/2)
    #plt.figure(figsize=(16, 8))
    plt.plot(lag_axis, autocorr / vertical_scale, 'r--')
    plt.title('autocorrelation of one traj, %d steps' % T)
    plt.xlabel('step lag')
    plt.ylabel('x(t)*x(t+tau) overlap')
    plt.show()


def fft_label_timeseries(traj):
    T = len(traj)
    f_components = np.fft.rfft(traj)
    f_components_abs = np.abs(f_components)
    f_axis = np.fft.rfftfreq(T, d=1.0)
    plt.plot(f_axis, f_components_abs)
    plt.title('magnitude of rfft of one traj, steps %d' % T)
    plt.xlim(f_axis[1], f_axis[-1])
    plt.ylim(-0.1, np.max(f_components_abs[1:]))
    plt.show()


def visualize_ensemble_label_timeseries(ensemble_label_timeseries, N):
    ensemble_size, T = np.shape(ensemble_label_timeseries)

    # dictionaries
    labels_to_states01 = {idx: tuple(label_to_state(idx, N, use_neg=False)) for idx in xrange(2 ** N)}
    states01_to_labels = {tuple(v): k for k, v in labels_to_states01.iteritems()}
    states01_to_states = {state: tuple([2 * v - 1 for v in state]) for state in states01_to_labels.keys()}
    #states01_to_colors = {state01: np.random.rand(3,) for state01, label in states01_to_labels.iteritems()}

    # initialize graph
    G = nx.hypercube_graph(N)
    pos = nx.spring_layout(G)

    # initialize ensemble plot properties
    ensemble_pos = np.zeros((ensemble_size, 2))
    ensemble_colors = [np.random.rand(3,) for _ in xrange(ensemble_size)]
    delta = 0.05
    x_perturb = np.random.uniform(-delta, delta, ensemble_size)
    y_perturb = np.random.uniform(-delta, delta, ensemble_size)

    for step in xrange(T):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=G.nodes(),node_size=150)
        nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges())
        nx.draw_networkx_labels(G, pos=pos, font_size=12)
        plt.gca().axis('off')

        for idx, label in enumerate(ensemble_label_timeseries[:,step]):
            state01 = labels_to_states01[label]
            val_xy = pos[state01]
            val_xy_perturbed = val_xy + np.array([x_perturb[idx], y_perturb[idx]])
            ensemble_pos[idx, :] = val_xy_perturbed
            #ensemble_colors[idx] = states01_to_colors[state01]
        plt.scatter(ensemble_pos[:, 0], ensemble_pos[:, 1], c=ensemble_colors, alpha=0.7)
        plt.title('Step %d' % step)
        plt.show()

    return 0

