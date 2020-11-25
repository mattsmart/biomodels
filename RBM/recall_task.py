import numpy as np
import os
import matplotlib.pyplot as plt
from random import shuffle, random, sample
from scipy.linalg import qr

from RBM_train import load_rbm_hopfield
from recall_functions import update_state_deterministic, update_state_noise, update_site_glauber, hamming, \
    plot_confusion_matrix_recall, build_J_from_xi, update_state_noise_parallel
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
#  - parallelize updates to run faster?
#  - run given HN as RBM to sample faster (assumes REMOVE_DIAG = False though)?
#  - cleaner if we set MAX_STEPS to (effectively) inf sao that the confusion matrix is 10 x 10 (easier to show)

# Globals
OUTDIR = 'output' + os.sep + 'recall'
N = 28 ** 2
P = 10
NUM_HIDDEN = P
USE_TEST = True

# Which model to load (and whether to force binarize)
BASIC_MODEL = False
EPOCH = 10
FORCE_BINARIZE_RMAP = False

# Local hyperparameters
HIJACK_MEASURE = False
HIJACK_AIS = False
BETA = 2.0          # None means deterministic TODO hybridmode flag?
ENSEMBLE = 20
REMOVE_DIAG = True  # TODO care remove self interactions or not? customary... improves radius of attraction
SUBSAMPLE = 1000
USE_PROJ = False
MAX_STEPS = 102    #100 or 100000
CONV_CHECK = 0.7

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
            ####################################
            #sample = update_state_noise(sample, intxn_matrix, beta, sites, async_batch=True)
            sample = update_state_noise_parallel(sample, intxn_matrix, beta)
            ####################################
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

        if data_idx % 10 == 0:
            print('data_idx', data_idx, 'true:', label, 'pred:\n', converge_basins)

        # TROUBLESHOOTING OPTIONS
        # >>>>>>>> simple:
        #print(data_idx, label, converge_basins)
        # >>>>>>>> heavy:
        #converge_basin_determ, step = converge_datapoint_deterministic(sample, patterns_measurement, intxn_matrix, nsteps)
        #print(data_idx, 'true:', label, 'determ:', converge_basin_determ, converge_basins)

    return confusion, required_steps



def score_dataset_hybrid(dataset, patterns_measurement, intxn_matrix, beta, nsteps=MAX_STEPS, ensemble=10):
    """
    Difference vs deterministic/stochastic:
    - tried deterministic
    - if -1 (classified as other), then do stochastic

    """
    token = 1 / float(ensemble)
    # statistics to gether
    required_steps = [0] * nsteps
    confusion = np.zeros((P, P + 1), dtype=float)  #confusion_matrix_10 = np.zeros((10, 10), dtype=int)

    for data_idx, pair in enumerate(dataset):
        sample = pair[0]
        label = pair[1]

        sample = sample.reshape(N)

        converge_basin, steps = converge_datapoint_deterministic(sample, patterns_measurement, intxn_matrix, nsteps)

        if converge_basin == -1:
            for traj in range(ensemble):
                converge_basins, steps_req = converge_datapoint_stochastic(sample, patterns_measurement, intxn_matrix,
                                                                           nsteps, beta, ensemble)
                required_steps[steps_req[traj]] += token
                confusion[label, converge_basins[traj]] += token
        else:
            required_steps[steps] += 1
            confusion[label, converge_basin] += 1

        if data_idx % 100 == 0:
            print('data_idx', data_idx, 'true:', label)

        # TROUBLESHOOTING OPTIONS
        # >>>>>>>> simple:
        #print(data_idx, label, converge_basins)
        # >>>>>>>> heavy:
        #converge_basin_determ, step = converge_datapoint_deterministic(sample, patterns_measurement, intxn_matrix, nsteps)
        #print(data_idx, 'true:', label, 'determ:', converge_basin_determ, converge_basins)

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

    if HIJACK_MEASURE:
        # attempts to rescale the non-binary patterns to obtain a useful thresholding measure for convergence
        print('Note setting: HIJACK_MEASURE HIJACK_MEASURE HIJACK_MEASURE')
        assert not BASIC_MODEL
        assert not FORCE_BINARIZE_RMAP

        if USE_PROJ:
            # variant of projection measure
            A = np.dot(unsigned_patterns.T, unsigned_patterns)
            A_inv = np.linalg.inv(A)
            patterns_measure_unscaled = np.dot(A_inv, unsigned_patterns.T)

            # scaling so diags are 1 for sigfned patterns
            scales = np.array([
                np.dot(patterns_measure_unscaled[mu, :],
                       patterns[:, mu])
                for mu in range(NUM_HIDDEN)])

            #patterns_measure = patterns_measure_unscaled                 # pick scaled version
            patterns_measure = (patterns_measure_unscaled.T / scales).T  # pick UNscaled version

            # projection measure plots
            plt.imshow(A); plt.colorbar(); plt.show(); plt.close()
            I_p = np.dot(patterns_measure, patterns)
            plt.imshow(I_p); plt.colorbar(); plt.show(); plt.close()

        else:
            # variant of overlap measure
            scales = np.array([
                np.dot(unsigned_patterns[:, mu],
                       patterns[:, mu])
                for mu in range(NUM_HIDDEN)])
            patterns_measure = (patterns / scales).T
            print('scales', scales)

            # overlap plots
            I_p = np.dot(patterns_measure, unsigned_patterns)
            plt.imshow(I_p, vmin=0.0, vmax=1.2); plt.colorbar(); plt.show(); plt.close()

            I_p_alt = np.dot(patterns_measure, patterns)
            plt.imshow(I_p_alt, vmin=0.0, vmax=1.2); plt.colorbar(); plt.show(); plt.close()

            I_p_orig = np.dot(patterns.T / float(N), patterns)
            plt.imshow(I_p_orig, vmin=0.0, vmax=1.2); plt.colorbar(); plt.show(); plt.close()

        for mu in range(NUM_HIDDEN):

            print('Raw magnitude overlaps (unsigned):', mu)
            print(np.dot(unsigned_patterns.T, unsigned_patterns[:, mu]))

            print('Raw magnitude overlaps (signed):', mu)
            print(np.dot(unsigned_patterns.T, patterns[:, mu]))

            print('Measure on signed version:', mu)
            print(np.dot(patterns_measure, patterns[:, mu]))

    ##############################################################
    # LOAD DATASET
    ##############################################################
    dataset_train, dataset_test = data_mnist(binarize=True)
    if USE_TEST:
        dataset = dataset_test
    else:
        dataset = dataset_train

    if SUBSAMPLE is not None:
        subsamplestr = '%d' % SUBSAMPLE
        dataset = dataset[:SUBSAMPLE]  # or randomly, e.g. sample(dataset, SUBSAMPLE)
        num_samples = len(dataset)
    else:
        subsamplestr = 'Full'
        num_samples = len(dataset)

    ##############################################################
    # SCORE DATASET
    ##############################################################
    if HIJACK_AIS:
        assert (not USE_TEST) and (SUBSAMPLE is None)

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
            cm, required_steps = score_dataset_hybrid(dataset, patterns_measure, intxn_matrix, BETA,
                                                      nsteps=MAX_STEPS, ensemble=ENSEMBLE)

        out_local = OUTDIR + os.sep + 'recall_basemodel%d_epoch%d_forceBinarize%d_sample%s_noise%s_removeDiag%d_nsteps%d_convChk%.2f_projMode%d_test%d' % \
                    (BASIC_MODEL, EPOCH, FORCE_BINARIZE_RMAP, subsamplestr, noisestr, REMOVE_DIAG, MAX_STEPS, CONV_CHECK, USE_PROJ, USE_TEST)
        if not os.path.exists(out_local): os.makedirs(out_local)

        ##############################################################
        # ASSESS
        ##############################################################
        # save cm
        fname = out_local + os.sep + 'recall_data.npz'
        np.savez(fname, cm=cm, reqsteps=required_steps)
        # stats
        matches = sum(cm[i,i] for i in range(P))
        unlabelled = sum(cm[i,-1] for i in range(P))
        score = matches / float(num_samples)
        title = 'score = %.5f (%d/%d)' % (score, matches, num_samples)
        subtitle = 'unlabelled = %.5f (%d/%d)' % (unlabelled / float(num_samples), unlabelled, num_samples)
        title_3 = 'max possible = %.5f (est %.5f)' % ((matches + unlabelled) / float(num_samples),
                                                      (matches) / float(num_samples - unlabelled))
        print(title); print(subtitle); print(title_3)
        # plots
        plt.bar(range(MAX_STEPS), required_steps, width=0.8)
        plt.title('required steps hist'); plt.xlabel('steps to converge'); plt.ylabel('freq')
        plt.savefig(out_local + os.sep + 'step_hist.pdf'); plt.close()
        # cm plot
        plot_confusion_matrix_recall(cm, classlabels=list(range(10)), title='%s; %s; %s' % (title, subtitle, title_3),
                                     save=out_local + os.sep + 'cm.pdf', annot=False)
