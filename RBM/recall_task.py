import numpy as np
import os
import matplotlib.pyplot as plt
from random import shuffle, random, sample
from scipy.linalg import qr

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
    
Findings (AIS):
 - epoch 0, AIS score: ~ -254
 - epoch 10, AIS score: ~ -243
 - epoch 20, AIS score: ~ -243
 - epoch 30, AIS score: ~ -244
 - worse after
"""

# TODO questions:
#  - how to define overlap when patterns are non-binary? specifically, meaning of param "CONV_CHECK"

# Globals
OUTDIR = 'output' + os.sep + 'recall'
N = 28 ** 2
P = 10
NUM_HIDDEN = P

# Which model to load (and whether to force binarize)
BASIC_MODEL = False
EPOCH = 20
FORCE_BINARIZE_RMAP = True

# Local hyperparameters
HIJACK_AIS = False
BETA = None          # None means deterministic
ENSEMBLE = 10
REMOVE_DIAG = True  # TODO care remove self interactions or not? customary... improves radius of attraction
SUBSAMPLE = None
USE_PROJ = False
MAX_STEPS = 30
CONV_CHECK = 0.9

# Sanity checks
if HIJACK_AIS:
    if not FORCE_BINARIZE_RMAP and not BASIC_MODEL:
        print('Note: Fig. 4 already has AIS for FORCE_BINARIZE_RMAP=False, so we should set FORCE_BINARIZE_RMAP=True')
        #assert FORCE_BINARIZE_RMAP


def basin_convergence_check_exact(sample, patterns_measurement):
    measure = np.dot(patterns_measurement, sample)  # either the 'overlaps' or 'projection'
    # print(measure)
    if any(measure > CONV_CHECK):
        out = (True, np.argmax(measure))
    else:
        out = (False, None)
    return out


def converge_datapoint_deterministic(sample, patterns_measurement, intxn_matrix, nsteps):
    converge_basin = -1  # this is e.g. mixed or not converged

    for step in range(nsteps):
        has_converged, basin = basin_convergence_check_exact(sample, patterns_measurement)
        if has_converged:
            converge_basin = basin
            break
        sample = update_state_deterministic(sample, intxn_matrix)
        """
        if all([sample[i] == sample_prev[i] for i in range(N)]):
            print('Note stuck at step %d' % step)
            plt.imshow(sample.reshape(28, 28)); plt.colorbar(); plt.title(step), plt.show()
            measure = np.dot(patterns_measurement, sample)  # either the 'overlaps' or 'projection'
            print(measure)
        """
    return converge_basin, step


def score_dataset_deterministic(dataset, patterns_measurement, intxn_matrix, nsteps=MAX_STEPS):

    # statistics to gether
    required_steps = [0] * nsteps
    confusion = np.zeros((P, P + 1), dtype=int)  #confusion_matrix_10 = np.zeros((10, 10), dtype=int)

    for sample, label in dataset:
        sample = sample.reshape(N)
        converge_basin, steps = converge_datapoint_deterministic(sample, patterns_measurement, intxn_matrix, nsteps)
        # harvest statistics
        required_steps[steps] += 1
        confusion[label, converge_basin] += 1
        if label != converge_basin:
            print('a %d sample went to' % (label), converge_basin)

    return confusion, required_steps


def converge_datapoint_stochastic(sample_init, patterns_measurement, intxn_matrix, nsteps, beta, ensemble):
    # converge_basins, steps are arrays of size ensemble

    converge_basins = -1 * np.ones(ensemble, dtype=int)  # this is e.g. mixed or not converged
    steps_req = np.zeros(ensemble, dtype=int)
    sites = list(range(N))

    for traj in range(ensemble):
        sample = sample_init.copy()
        for step in range(nsteps):
            has_converged, basin = basin_convergence_check_exact(sample, patterns_measurement)
            if has_converged:
                converge_basins[traj] = basin
                break
            sample = update_state_noise(sample, intxn_matrix, beta, sites, async_batch=True)
        steps_req[traj] = step

    return converge_basins, steps_req


def score_dataset_stochastic(dataset, patterns_measurement, intxn_matrix, beta, nsteps=MAX_STEPS, ensemble=10):
    """
    Difference vs deterministic:
    - each sample is run for ensemble trajectories
    - each trajectory will hit one basin (or not)
        - each contributes (1/ensemble) to the confusion matrix (e.g. 0.01 amount for ensemble size 100)
    """
    token = 1 / float(ensemble)
    # statistics to gether
    required_steps = [0] * nsteps
    confusion = np.zeros((P, P + 1), dtype=float)  #confusion_matrix_10 = np.zeros((10, 10), dtype=int)

    for data_idx, pair in enumerate(dataset):
        sample = pair[0]
        label = pair[1]

        sample = sample.reshape(N)
        # converge_basins, steps are arrays of size ensemble
        converge_basins, steps_req = converge_datapoint_stochastic(sample, patterns_measurement, intxn_matrix, nsteps, beta, ensemble)
        # harvest statistics
        for traj in range(ensemble):
            required_steps[steps_req[traj]] += token
            confusion[label, converge_basins[traj]] += token
            #if label != converge_basins[traj]:
            #    print('a %d sample went to' % (label), converge_basins[traj])

        # TROUBLESHOOTING OPTIONS
        # >>>>>>>> simple:
        #print(data_idx, label, converge_basins)
        # >>>>>>>> heavy:
        converge_basin_determ, step = converge_datapoint_deterministic(sample, patterns_measurement, intxn_matrix, nsteps)
        print(data_idx, 'true:', label, 'determ:', converge_basin_determ, converge_basins)

    return confusion, required_steps


if __name__ == '__main__':
    ##############################################################
    # LOAD HOPFIELD INITIAL PATTERNS
    ##############################################################
    if BASIC_MODEL:
        fname = 'hopfield_mnist_10.npz'
        rbm = load_rbm_hopfield(npzpath='models' + os.sep + 'saved' + os.sep + fname)
        EPOCH = 0
        patterns_images = rbm.xi_image
        patterns = patterns_images.reshape(N, -1)
        rbm_weights = rbm.internal_weights
        resulting_rbm_weights = rbm_weights
        #intxn_matrix = np.dot(rbm_weights, rbm_weights.T)
        intxn_matrix = build_J_from_xi(patterns, remove_diag=REMOVE_DIAG)
    else:
        reversemap_dir = 'models' + os.sep + 'reversemap'
        fname = 'binarized_rbm_hopfield_iter%d_star1_lowdin0_num50_beta200.0_eta5.0E-02_noise0.0E+00.npz' % EPOCH
        reverse_obj = np.load(reversemap_dir + os.sep + fname)
        WX_final = reverse_obj['WX_final']
        W = reverse_obj['W']
        unsigned_patterns = WX_final
        patterns = np.sign(unsigned_patterns)

        if FORCE_BINARIZE_RMAP:
            intxn_matrix = build_J_from_xi(patterns, remove_diag=REMOVE_DIAG)
            Qprime, Rprime = qr(patterns, mode='economic')
            resulting_rbm_weights = Qprime
        else:
            intxn_matrix = build_J_from_xi(unsigned_patterns, remove_diag=REMOVE_DIAG)
            resulting_rbm_weights = W

    A = np.dot(patterns.T, patterns)
    A_inv = np.linalg.inv(A)
    if USE_PROJ:
        patterns_measure = np.dot(A_inv, patterns.T)
    else:
        patterns_measure = patterns.T / float(N)

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
    if HIJACK_AIS:
        from custom_rbm import RBM_gaussian_custom
        from RBM_assess import get_X_y_dataset
        from AIS import get_obj_term_A, manual_AIS
        import torch

        BETA_AIS = 2.0
        AIS_STEPS = 1000
        nchains = 500

        X, _ = get_X_y_dataset(dataset, dim_visible=N, binarize=True)

        rbm = RBM_gaussian_custom(N, NUM_HIDDEN, 0, init_weights=None, use_fields=False, learning_rate=0)
        rbm.weights = torch.from_numpy(resulting_rbm_weights).float()

        print('Estimating term A...', )
        logP_termA = get_obj_term_A(X, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=BETA_AIS)
        print('Estimating log Z (homemade AIS)...', )
        logP_termB_manual, chains_state = manual_AIS(rbm, BETA_AIS, nchains=nchains, nsteps=AIS_STEPS)
        print('\tlogP_termB_manual:', logP_termB_manual)
        print('Manual AIS - Term A:', logP_termA, '| Log Z:', logP_termB_manual, '| Score:', logP_termA - logP_termB_manual)

    else:
        if BETA == None:
            noisestr = 'None'
            cm, required_steps = score_dataset_deterministic(dataset, patterns_measure, intxn_matrix, nsteps=MAX_STEPS)
        else:
            noisestr = '%.2fens%d' % (BETA, ENSEMBLE)
            cm, required_steps = score_dataset_stochastic(dataset, patterns_measure, intxn_matrix, BETA,
                                                          nsteps=MAX_STEPS, ensemble=ENSEMBLE)

        out_local = OUTDIR + os.sep + 'recall_basemodel%d_epoch%d_forceBinarize%d_subsample%d_noise%s_removeDiag%d_nsteps%d_convChk%.2f_projMode%d' % \
                    (BASIC_MODEL, EPOCH, FORCE_BINARIZE_RMAP, SUBSAMPLE is not None, noisestr, REMOVE_DIAG, MAX_STEPS, CONV_CHECK, USE_PROJ)
        if not os.path.exists(out_local): os.makedirs(out_local)

        ##############################################################
        # ASSESS
        ##############################################################
        # stats
        matches = sum(cm[i,i] for i in range(P))
        unlabelled = sum(cm[i,-1] for i in range(P))
        score = matches / float(num_samples)
        title = 'score = %.3f (%d/%d)' % (score, matches, num_samples)
        subtitle = 'unlabelled = %.3f (%d/%d)' % (unlabelled / float(num_samples), unlabelled, num_samples)
        title_3 = 'max possible = %.3f' % ((matches + unlabelled) / float(num_samples))
        print(title); print(subtitle); print(title_3)
        # plots
        plt.bar(range(MAX_STEPS), required_steps, width=0.8)
        plt.title('required steps hist'); plt.xlabel('steps to converge'); plt.ylabel('freq')
        plt.savefig(out_local + os.sep + 'step_hist.pdf'); plt.close()
        # cm plot
        plot_confusion_matrix_recall(cm, classlabels=list(range(10)), title='%s; %s; %s' % (title, subtitle, title_3),
                                     save=out_local + os.sep + 'cm.pdf', annot=False)
