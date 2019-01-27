import matplotlib.pyplot as plt
import numpy as np
import os

from inference import choose_J_from_general_form, infer_interactions, error_fn
from settings import FOLDER_OUTPUT
from visualize_matrix import plot_matrix


def get_spectrum_from_arr(J, real=True, sort=True):
    # TODO deal with massive complex part if necessary
    eig, V = np.linalg.eig(J)
    if real:
        eig = np.real(eig)
        if sort:
            eig = np.sort(eig)
    return eig


def get_spectrums(C, D, num_spectrums=10, method='U_data', print_errors=True):
    """
    Returns J's (generated from method) and their spectrums and their labels
        J's returned as list of arrays
        Shape is num_spectrums X dim_spectrum
    """
    assert method in ['U_data', 'U_lyap', 'infer_data', 'infer_lyap']
    spectrums = np.zeros((num_spectrums, D.shape[0]))
    list_of_J = [0]*num_spectrums
    # generate spectrum labels
    if method[0] == 'U':
        scales = np.linspace(0, 0.09, num_spectrums)
        labels = ['scale_%.3f' % i for i in scales]
    else:
        alphas = np.logspace(-10, -1, num_spectrums)
        labels = ['alpha_%.2e' % a for a in alphas]
    for idx in xrange(num_spectrums):
        if method[0] == 'U':
            J = choose_J_from_general_form(C, D, scale=scales[idx])
        else:
            J = infer_interactions(C, D, alpha=alphas[idx])
            if print_errors:
                err = error_fn(C, D, J)
                print "Error in method %s, idx %d, is %.3f (alpha=%.2e)" % (method, idx, err, alphas[idx])
        list_of_J[idx] = J
        spectrums[idx, :] = get_spectrum_from_arr(J, real=True)
    return list_of_J, spectrums, labels


def get_J_truncated_spectrum(J, idx):
    """
    Given an idx, removes row/col idx of J and computes the spectrum of the new (n-1)*(n-1) array
    """
    J_reduce = J.copy()
    J_reduce = np.delete(J_reduce, (idx), axis=0)
    J_reduce = np.delete(J_reduce, (idx), axis=1)
    return get_spectrum_from_arr(J_reduce, real=True)


def scan_J_truncations(J, verbose=False, spectrum_unperturbed=None):
    """
    Given a Jacobian matrix J
    (1) compute the spectrum
    (2) assess if the spectrum is a suitable starting point
    (3) iteratively delete all row/col pairs and compute spectrum of each
    (4) for each row/col pair, report if the spectrum has been sufficiently perturbed
    """
    assert J.shape[0] == J.shape[1]
    n = J.shape[0]
    if spectrum_unperturbed is None:
        spectrum_unperturbed = get_spectrum_from_arr(J, real=True)
    spectrums_perturbed = np.zeros((n, n-1))
    if verbose:
        print 'unperturbed', '\n', spectrum_unperturbed
    for idx in xrange(n):
        spectrum_idx = get_J_truncated_spectrum(J, idx)
        spectrums_perturbed[idx, :] = spectrum_idx
        if verbose:
            print idx, '\n', spectrum_idx
    return spectrum_unperturbed, spectrums_perturbed


def gene_control_scores(spectrum_unperturbed, spectrums_perturbed, use_min=True):
    """
    See Sid 2018 draft for idea
    """
    if use_min:
        cg = np.min(spectrums_perturbed, axis=1)  # numerator left term
        cg = cg - np.min(spectrum_unperturbed)  # numerator full
    else:
        cg = np.max(spectrums_perturbed, axis=1)  # numerator left term
        cg = np.max(spectrum_unperturbed) - cg  # numerator full (note swap for index positivity)
    cg = cg / np.sqrt( np.mean( (spectrum_unperturbed ** 2) ) )  # denominator
    return cg


def plot_spectrum_hists(spectrums, labels, method='U', hist='default', title_mod='', show=False):
    # TODO fix x axis range -6 6
    # TODO remove method from title since not used

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')

    f = plt.figure(figsize=(10, 6))
    if hist == 'default':
        # plot first spectrum to get bins
        _, bins, _ = plt.hist(spectrums[0, :], bins=10, range=[-6, 6], alpha=0.5, normed=True, label=labels[0])
        for idx in xrange(1, len(labels)):
            _ = plt.hist(spectrums[idx, :], bins=bins, alpha=0.5, normed=True, label=labels[idx])
        plt.xlabel('Re(lambda)')
        plt.ylabel('Spectrums')
    elif hist == 'violin':
        print 'hist type %s not yet implemented in plot_spectrum_hists(...)' % hist
        plt.violinplot(spectrums.T, showmeans=False, showmedians=True)
        set_axis_style(plt.gca(), labels)
        plt.ylabel('Re(lambda)')
    else:
        print 'hist type %s not supported in plot_spectrum_hists(...)' % hist
        assert 1==2
    plt.title('Spectrums from %s %s' % (method, title_mod))
    plt.legend()
    plt.savefig(FOLDER_OUTPUT + os.sep + 'spectrum_hist_%s_%s_%s.png' % (hist, method, title_mod))
    if show:
        plt.show()
    return


def plot_rank_order_spectrum(spectrum, method='U', title_mod='', show=False):
    f = plt.figure(figsize=(10, 6))
    sorted_spectrums_low_to_high = np.sort(spectrum)
    sorted_spectrums_high_to_low = sorted_spectrums_low_to_high[::-1]
    plt.bar(range(len(sorted_spectrums_high_to_low)), sorted_spectrums_high_to_low)
    plt.axhline(0.0, linewidth=1.0, color='k')
    plt.ylabel('Re(lambda)')
    plt.xlabel('Eigenvalue ranking')
    plt.title('Spectrum from %s %s %s' % (method, title_mod))
    plt.savefig(FOLDER_OUTPUT + os.sep + 'spectrum_ranking_%s_%s.pdf' % (method, title_mod))
    if show:
        plt.show()
    return


def plot_spectrum_extremes(spectrum_unperturbed, spectrums_perturbed, method='U', title_mod='', show=False, max=True):
    n = len(spectrum_unperturbed)
    bar_width = 0.45
    plt.close('all')
    f = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    if max:
        spectrum_unperturbed_max = np.max(spectrum_unperturbed)
        spectrums_perturbed_maxes = np.max(spectrums_perturbed, axis=1)
        plt.bar(np.arange(n), spectrums_perturbed_maxes, bar_width)
        plt.axhline(spectrum_unperturbed_max, linewidth=1.0, color='g')
        #plt.ylim(np.min(spectrums_perturbed_maxes) * 1.05, np.max(spectrums_perturbed_maxes) * 1.05)
        plt.ylabel('Max Re(lambda)')
        plt.title('Largest eigenvalue after row/col deletion (green = no deletion) from %s %s' % (method, title_mod))
        figpath = FOLDER_OUTPUT + os.sep + 'spectrum_perturbed_max_%s_%s.pdf' % (method, title_mod)
    else:
        spectrum_unperturbed_min = np.min(spectrum_unperturbed)
        spectrums_perturbed_mins = np.min(spectrums_perturbed, axis=1)
        ax.bar(np.arange(n), spectrums_perturbed_mins, bar_width)
        plt.axhline(spectrum_unperturbed_min, linewidth=1.0, color='g')
        #plt.ylim(np.min(spectrums_perturbed_mins) * 1.05, np.max(spectrums_perturbed_mins) * 1.05)
        plt.ylabel('Min Re(lambda)')
        plt.title('Lowest eigenvalue after row/col deletion (green = no deletion) from %s %s' % (method, title_mod))
        figpath = FOLDER_OUTPUT + os.sep + 'spectrum_perturbed_min_%s_%s.pdf' % (method, title_mod)
    plt.axhline(0.0, linewidth=1.0, color='k')
    ax.set_xticks(np.arange(n))
    plt.xlabel('Index of deleted row/col')
    plt.savefig(figpath)
    if show:
        plt.show()
    return


def plot_sliding_tau_scores(tau_range, gene_score_arr, gene_score_label, score_type, show=False):
    assert gene_score_arr.shape[0] == tau_range.shape[0]
    plt.close('all')
    f = plt.figure(figsize=(12, 7))
    plt.plot(tau_range, gene_score_arr, '--ok', alpha=0.3)
    # add vertical line at tau = 2.0 bifurcation
    plt.axvline(2.0, linewidth=1.0, color='k', alpha=0.7)
    # highlight top k curves
    top_k = 2
    sorted_top_curves = np.argsort(np.sum(gene_score_arr, axis=0))[::-1]
    for rank, k in enumerate(sorted_top_curves[0:top_k]):
        plt.plot(tau_range, gene_score_arr[:, k], '--o', alpha=0.7, label='rank%d = gene %d' % (rank, k))
    plt.legend()
    plt.xlabel('tau')
    plt.ylabel('%s index' % score_type)
    plt.title('%s index from %s over all genes, approaching bifurcation (tau=2.0)' % (score_type, gene_score_label))
    figpath = FOLDER_OUTPUT + os.sep + 'score_%s_%s.pdf' % (gene_score_label, score_type)
    plt.savefig(figpath)
    if show:
        plt.show()
    return


if __name__ == '__main__':
    num_spectrum = 10
    fake_spectrums = np.random.normal(0.0, 2.0, (num_spectrum, 500))
    fake_labels = [str(a) for a in range(num_spectrum)]
    plot_spectrum_hists(fake_spectrums, fake_labels, hist='default', title_mod='(fake_main)', show=True)
    plot_spectrum_hists(fake_spectrums, fake_labels, hist='violin', title_mod='(fake_main)', show=True)
