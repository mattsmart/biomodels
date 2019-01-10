import matplotlib.pyplot as plt
import numpy as np
import os

from inference import choose_J_from_general_form, infer_interactions, error_fn
from settings import FOLDER_OUTPUT

def get_spectrum_from_J(J, real=True):
    # TODO deal with massive complex part if necessary
    eig, V = np.linalg.eig(J)
    if real:
        eig = np.real(eig)
    return eig


def get_spectrums(C, D, num_spectrums=10, method='U', print_errors=True):
    """
    Returns spectrums of J's generated from method and their labels
        Shape is num_spectrums X dim_spectrum
    """
    assert method in ['U', 'infer']
    spectrums = np.zeros((num_spectrums, D.shape[0]))
    # generate spectrum labels
    if method == 'U':
        labels = ['scale_%d' % i for i in xrange(num_spectrums)]
        scales = [i for i in xrange(num_spectrums)]
    else:
        alphas = np.linspace(1e-10, 1e-1, num_spectrums)
        labels = ['alpha_%.3f' % a for a in alphas]
    for idx in xrange(num_spectrums):
        if method == 'U':
            J = choose_J_from_general_form(C, D, scale=scales[idx])
        else:
            J = infer_interactions(C, D, alpha=alphas[idx])
            if print_errors:
                err = error_fn(C, D, J)
                print "Error in method %s, idx %d, is %.3f (alpha=%.2e)" % (method, idx, err, alphas[idx])
        spectrums[idx, :] = get_spectrum_from_J(J, real=True)
    return spectrums, labels


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


if __name__ == '__main__':
    num_spectrum = 10
    fake_spectrums = np.random.normal(0.0, 2.0, (num_spectrum, 500))
    fake_labels = [str(a) for a in range(num_spectrum)]
    plot_spectrum_hists(fake_spectrums, fake_labels, hist='default', title_mod='(fake_main)', show=True)
    plot_spectrum_hists(fake_spectrums, fake_labels, hist='violin', title_mod='(fake_main)', show=True)
