import matplotlib.pyplot as plt
import numpy as np

from inference import choose_J_from_general_form, infer_interactions, error_fn


def get_spectrum_from_J(J, real=True):
    eig, V = np.linalg.eig(J)
    if real:
        print eig
        eig = np.real(eig)
        print eig
    return eig


def get_spectrums(C, D, num_spectrums=10, method='U', print_errors=True):
    """
    Returns spectrums of J's generated from method and their labels
        Shape is num_spectrums X dim_spectrum
    """
    assert method in ['choose', 'infer']
    spectrums = np.zeros((num_spectrums, D.shape[0]))
    for idx in xrange(num_spectrums):
        if method == 'choose':
            J = choose_J_from_general_form(C, D, scale=10.0)
        else:
            J = infer_interactions(C, D, alpha=0.1)
            if print_errors:
                err = error_fn(C, D, J)
                print "Error in method %s, idx %d, is %.3f" % (method, idx, err)
        spectrums[idx, :] = get_spectrum_from_J(J, real=True)
    labels = [i for i in xrange(num_spectrums)]  # TODO more meaningful? e.g. vary scales or alphas
    return spectrums, labels


def plot_spectrum_hists(spectrums, labels, method='U', hist='default'):
    # TODO violin plot of real part? others...
    if hist == 'default':
        _, bins, _ = plt.hist(spectrums[0, :], bins=10, range=[-6, 6], normed=True, label=labels[0])  # plot first to get bins
        for idx in xrange(1, len(labels)):
            _ = plt.hist(spectrums[idx, :], bins=bins, alpha=0.5, normed=True, label=labels[idx])
    elif hist == 'violin':
        print 'hist type %s not yet implemented in plot_spectrum_hists(...)' % hist
        assert 1==2
    else:
        print 'hist type %s not supported in plot_spectrum_hists(...)' % hist
        assert 1==2
    plt.title('Spectrums from %s' % method)
    plt.xlabel('Spectrums')
    plt.show()
    return
