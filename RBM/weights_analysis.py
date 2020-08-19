import matplotlib.pyplot as plt
import numpy as np
import os

from plotting import image_fancy
from settings import MNIST_BINARIZATION_CUTOFF, DIR_OUTPUT, CLASSIFIER, BETA


def plot_weights_timeseries(weights_timeseries, outdir, mode='eval', extra=False):
    assert mode in ['eval', 'minmax']
    N, p, num_steps = weights_timeseries.shape

    if mode == 'eval':
        evals_timeseries = np.zeros((p, num_steps)) # TODO square or not
        lsv_timeseries = np.zeros((N, p, num_steps))
        rsv_timeseries = np.zeros((p, p, num_steps))
        for idx in range(num_steps):
            weights = weights_timeseries[:, :, idx]
            u, s, vh = np.linalg.svd(weights, full_matrices=False)
            evals_timeseries[:, idx] = s
            lsv_timeseries[:, :, idx] = u
            rsv_timeseries[:, :, idx] = vh
        ret = (evals_timeseries, lsv_timeseries, rsv_timeseries)

        plt.plot(range(num_steps), evals_timeseries.T)
        plt.ylabel('weights: singular values')
        #plt.xlim(0, 20)
    else:
        min_arr = np.amin(weights_timeseries, axis=(0, 1))
        max_arr = np.amax(weights_timeseries, axis=(0, 1))
        plt.plot(range(num_steps), min_arr, label=r'min $W(t)$')
        plt.plot(range(num_steps), max_arr, label=r'max $W(t)$')
        plt.ylabel('weights min, max')
        ret = (min_arr, max_arr)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    if extra and mode == 'eval':
        for col in range(p):
            for epoch in range(20):
                lsv = lsv_timeseries[:, col, epoch]
                plt.imshow(lsv.reshape(28, 28), interpolation='none')
                plt.colorbar()
                plot_title = 'training_lsv_col%d_%d' % (col, epoch)
                plt.title(plot_title)
                plt.savefig(outdir + os.sep + plot_title + '.jpg')
                plt.close()
    return ret


if __name__ == '__main__':
    epoch_indices = None

    bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs'

    # specify dir
    runtype = 'normal'  # hopfield or normal
    alt_names = False  # some weights had to be run separately with different naming convention
    hidden_units = 10
    use_fields = True
    maindir = bigruns + os.sep + 'rbm' + os.sep + \
              '%s_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % (
                  runtype, hidden_units, use_fields, 2.00, 100, 100, 20, 1e-4, 200)
    run = 0
    if alt_names:
        rundir = maindir
        prepend = '%d_' % run
        weights_ais = 0
    if not alt_names:
        rundir = maindir + os.sep + 'run%d' % run
        prepend = ''
        weights_ais = 200

    # load weights
    obj_title = '%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta' % (hidden_units, use_fields, 20, weights_ais, 2.00)
    weights_obj = np.load(rundir + os.sep + prepend + 'weights_%s.npz' % obj_title)
    weights_timeseries = weights_obj['weights']
    #  (try to) load visible and hidden biases
    visible_loaded = False
    hidden_loaded = False
    if use_fields:
        try:
            visible_obj = np.load(rundir + os.sep + prepend + 'visiblefield_%s.npz' % obj_title)
            visiblefield_timeseries = visible_obj['visiblefield']
            visible_loaded = True
        except:
            print("Visible bias file not found")
        try:
            hidden_obj = np.load(rundir + os.sep + prepend + 'hiddenfield_%s.npz' % obj_title)
            hiddenfield_timeseries = hidden_obj['hiddenfield']
            hidden_loaded = True
        except:
            print("Hidden bias file not found")

    # analysis
    outdir = DIR_OUTPUT + os.sep + 'evals' + os.sep + \
             '%s_%dhidden_%dfields' % (runtype, hidden_units, use_fields)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plot_weights_timeseries(weights_timeseries, outdir, mode='minmax')
    plot_weights_timeseries(weights_timeseries, outdir, mode='eval', extra=False)

    """
    for idx in epoch_indices:

        if visible_loaded:
            visiblefield = visiblefield_timeseries[:, idx]
        if hidden_loaded:
            hiddenfield = hiddenfield_timeseries[:, idx]
    """
