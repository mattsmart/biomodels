import numpy as np
import os
import matplotlib.pyplot as plt
from random import shuffle, random, sample

from RBM_train import load_rbm_hopfield
from recall_functions import update_state_deterministic, update_state_noise, update_site_glauber, hamming, plot_confusion_matrix_recall, build_J_from_xi
from data_process import data_mnist, binarize_image_data, image_data_collapse
from settings import MNIST_BINARIZATION_CUTOFF


"""
Recall percent v1

For each data point
    - measure ensemble fraction of time it converges to the correct basin
    - keep tabs on which basin it converges to otherwise
    - this will give either scalar (convergence to correct basin) or p+1 dim vector (p patterns + 'other' category)
    
"""


# Globals
OUTDIR = 'output' + os.sep + 'recall'
N = 28 ** 2
P = 10

# Local hyperparameters
BETA = None  # None means deterministic
REMOVE_DIAG = False  # TODO care remove self interactions or not? customary... improves radius of attraction
BASIC_MODEL = True
FORCE_BINARIZE_RMAP = True
SUBSAMPLE = None
MAX_STEPS = 10


def basin_convergence_check_exact(sample, patterns_scaled):
    overlaps = np.dot(patterns_scaled.T, sample)
    #print(overlaps)
    if any(overlaps > 0.8):
        out = (True, np.argmax(overlaps))
    else:
        out = (False, None)
    return out


def converge_datapoint_deterministic(sample, patterns_scaled, intxn_matrix, nsteps):
    converge_basin = -1  # this is e.g. mixed or not converged

    nsteps = 10
    for step in range(nsteps):
        has_converged, basin = basin_convergence_check_exact(sample, patterns_scaled)
        if has_converged:
            converge_basin = basin
            break

        sample = update_state_deterministic(sample, intxn_matrix)
    return converge_basin, step


def score_dataset_deterministic(dataset, patterns_scaled, intxn_matrix, nsteps=MAX_STEPS):
    # statistics to gether
    required_steps = [0] * nsteps
    confusion = np.zeros((P, P + 1), dtype=int)  #confusion_matrix_10 = np.zeros((10, 10), dtype=int)

    for sample, label in dataset:
        sample = sample.reshape(N)
        converge_basin, steps = converge_datapoint_deterministic(sample, patterns_scaled, intxn_matrix, nsteps)
        # harvest statistics
        required_steps[steps] += 1
        confusion[label, converge_basin] += 1
        if label != converge_basin:
            print('a %d sample went to' % (label), converge_basin)

    return confusion, required_steps



if __name__ == '__main__':
    ##############################################################
    # LOAD HOPFIELD INITIAL PATTERNS
    ##############################################################
    if BASIC_MODEL:
        fname = 'hopfield_mnist_10.npz'
        rbm = load_rbm_hopfield(npzpath='models' + os.sep + 'saved' + os.sep + fname)

        patterns_images = rbm.xi_image
        patterns = patterns_images.reshape(N, -1)
        patterns_scaled = patterns / float(N)
        rbm_weights = rbm.internal_weights
        #intxn_matrix = np.dot(rbm_weights, rbm_weights.T)
        intxn_matrix = build_J_from_xi(patterns, remove_diag=REMOVE_DIAG)

    else:
        reversemap_dir = 'models' + os.sep + 'reversemap'
        epoch = 30
        fname = 'binarized_rbm_hopfield_iter%d_star1_lowdin0_num50_beta200.0_eta5.0E-02_noise0.0E+00.npz' % epoch
        reverse_obj = np.load(reversemap_dir + os.sep + fname)
        WX_final = reverse_obj['WX_final']
        patterns = WX_final
        if FORCE_BINARIZE_RMAP:
            patterns = np.sign(patterns)
        intxn_matrix = build_J_from_xi(patterns, remove_diag=REMOVE_DIAG)

    ##############################################################
    # LOAD DATASET
    ##############################################################
    dataset, _ = data_mnist(binarize=True)
    if SUBSAMPLE is not None:
        dataset = dataset[:SUBSAMPLE]  # or randomly, e.g. sample(dataset, SUBSAMPLE)
        num_samples = len(dataset)
    else:
        num_samples = len(dataset)

    ##############################################################
    # SCORE DATASET
    ##############################################################
    if BETA == None:
        cm, required_steps = score_dataset_deterministic(dataset, patterns_scaled, intxn_matrix, nsteps=MAX_STEPS)
    else:
        print('TODO')

    ##############################################################
    # ASSESS
    ##############################################################
    matches = sum(cm[i,i] for i in range(P))
    unlabelled = sum(cm[i,-1] for i in range(P))
    score = matches / float(num_samples)
    print('score = %.3f (%d/%d)' % (score, matches, num_samples))
    print('unlabelled = %.3f (%d/%d)' % (unlabelled / float(num_samples), unlabelled, num_samples))

    plt.bar(range(MAX_STEPS), required_steps, width=0.8)
    plt.title('required steps hist'); plt.xlabel('steps to converge'); plt.ylabel('freq')
    plt.show(); plt.close()
    plot_confusion_matrix_recall(cm, classlabels=list(range(10)), title='', save=None)
