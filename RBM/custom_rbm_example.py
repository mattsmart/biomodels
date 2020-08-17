import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms

from AIS import esimate_logZ_with_AIS, get_obj_term_A
from custom_rbm import RBM_custom, RBM_gaussian_custom
from data_process import image_data_collapse, binarize_image_data, data_mnist
from RBM_train import build_rbm_hopfield
from RBM_assess import plot_confusion_matrix, rbm_features_MNIST, get_X_y_dataset
from settings import MNIST_BINARIZATION_CUTOFF, DIR_OUTPUT, CLASSIFIER, BETA


"""
WHAT'S CHANGED:
- removed cuda lines
- removed momentum
- updates actually sample hidden and visible values (instead of passing bernoulli probabilities) 
- flip state to +1, -1
- TODO: remove or change regularization: none, L1 (and scale), L2 (and scale)
- (?) remove weight decay
- (?) remove momentum
- (?) remove applied fields?
- (?) augment learning rate
- (?) augment logistic regression
- (?) option for binary or gaussian hidden nodes
"""

########## CONFIGURATION ##########
BATCH_SIZE = 100  # default 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 10  # was 128 but try 10
CD_K = 20
LEARNING_RATE = 1e-4  # default 1e-3
EPOCHS = 100  # was 10
AIS_STEPS = 200 #200
LOAD_INIT_WEIGHTS = True
USE_FIELDS = True
PLOT_WEIGHTS = True

GAUSSIAN_RBM = True
if RBM_gaussian_custom:
    RBM = RBM_gaussian_custom
else:
    RBM = RBM_custom

########## LOADING DATASET ##########
print('Loading dataset...')
DATA_FOLDER = 'data'
train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True,
                                           transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

TRAINING, _ = data_mnist(binarize=True)
X, _ = get_X_y_dataset(TRAINING, dim_visible=VISIBLE_UNITS, binarize=True)


def custom_RBM_loop(epochs=EPOCHS, cdk=CD_K, load_weights=LOAD_INIT_WEIGHTS, use_fields=USE_FIELDS, beta=BETA,
                    outdir=None, classify=True):
    assert beta == BETA  # TODO uncouple global STDEV in rbm class to make beta passable

    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        trainingdir = outdir + os.sep + 'training'
        if not os.path.exists(trainingdir):
            os.makedirs(trainingdir)
    else:
        trainingdir = DIR_OUTPUT + os.sep + 'training'

    ########## RBM INIT ##########
    rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, cdk, load_init_weights=load_weights, use_fields=use_fields,
              learning_rate=LEARNING_RATE)

    weights_timeseries = np.zeros((rbm.num_visible, rbm.num_hidden, epochs + 1))
    weights_timeseries[:, :, 0] = rbm.weights
    if use_fields:
        visible_bias_timeseries = np.zeros((rbm.num_visible, epochs + 1))
        visible_bias_timeseries[:, 0] = rbm.visible_bias.numpy()
        hidden_bias_timeseries = np.zeros((rbm.num_hidden, epochs + 1))
        hidden_bias_timeseries[:, 0] = rbm.hidden_bias.numpy()

    rbm.plot_model(title='epoch_0', outdir=trainingdir)

    obj_reconstruction = np.zeros(epochs)
    obj_logP_termA = np.zeros(epochs + 1)
    obj_logP_termB = np.zeros(epochs + 1)

    if AIS_STEPS > 0:
        obj_logP_termA[0] = get_obj_term_A(X, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta)
        print('Estimating log Z...', )
        obj_logP_termB[0], _ = esimate_logZ_with_AIS(rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta,
                                                     num_steps=AIS_STEPS)

    print('INIT obj - A:', obj_logP_termA[0], '| Log Z:', obj_logP_termB[0], '| Score:',
          obj_logP_termA[0] - obj_logP_termB[0])

    ########## TRAINING RBM ##########
    print('Training RBM...')
    for epoch in range(epochs):
        epoch_recon_error = 0.0
        for batch, _ in train_loader:
            batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
            batch = (batch > MNIST_BINARIZATION_CUTOFF).float()  # convert to 0,1 form
            batch = -1 + batch * 2  # convert to -1,1 form
            batch_recon_error = rbm.contrastive_divergence(batch)
            epoch_recon_error += batch_recon_error
        if PLOT_WEIGHTS:
            rbm.plot_model(title='epoch_%d' % (epoch + 1), outdir=trainingdir)
        print('Epoch (Reconstruction) Error (epoch=%d): %.4f' % (epoch + 1, epoch_recon_error))
        obj_reconstruction[epoch] = epoch_recon_error
        if AIS_STEPS > 0:
            obj_logP_termA[epoch + 1] = get_obj_term_A(X, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta)
            print('Estimating log Z...', )
            obj_logP_termB[epoch + 1], _ = esimate_logZ_with_AIS(rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta,
                                                                 num_steps=AIS_STEPS)

        # save parameters each epoch
        weights_timeseries[:, :, epoch + 1] = rbm.weights.numpy()
        if use_fields:
            visible_bias_timeseries[:, epoch + 1] = rbm.visible_bias.numpy()
            hidden_bias_timeseries[:, epoch + 1] = rbm.hidden_bias.numpy()

        print('Term A:', obj_logP_termA[epoch + 1], '| Log Z:', obj_logP_termB[epoch + 1], '| Score:',
              obj_logP_termA[epoch + 1] - obj_logP_termB[epoch + 1])

    ########## PLOT AND SAVE TRAINING INFO ##########
    score_arr = obj_logP_termA - obj_logP_termB

    if outdir is None:
        scoredir = DIR_OUTPUT + os.sep + 'logZ' + os.sep + 'rbm'
    else:
        scoredir = outdir

    # save parameters
    title_mod = '%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta' % (HIDDEN_UNITS, use_fields, cdk, AIS_STEPS, beta)
    fpath = scoredir + os.sep + 'objective_%s' % title_mod
    np.savez(fpath,
             epochs=range(epochs + 1),
             termA=obj_logP_termA,
             logZ=obj_logP_termB,
             score=score_arr)
    fpath = scoredir + os.sep + 'weights_%s' % title_mod
    np.savez(fpath, epochs=range(epochs + 1), weights=weights_timeseries)
    if use_fields:
        np.savez(scoredir + os.sep + 'visiblefield_%s' % title_mod, epochs=range(epochs + 1),
                 visiblefield=visible_bias_timeseries)
        np.savez(scoredir + os.sep + 'hiddenfield_%s' % title_mod, epochs=range(epochs + 1),
                 hiddenfield=hidden_bias_timeseries)

    plot_scores(epochs, obj_logP_termA, obj_logP_termB, score_arr, scoredir, title_mod, 'epochs',
                obj_reconstruction=obj_reconstruction)

    if classify:
        classify_with_rbm(rbm, outdir=outdir)

    return rbm


def plot_scores(timesteps, obj_logP_termA, obj_logP_termB, score_arr, scoredir, title_mod, xlabel,
                obj_reconstruction=None):
    if obj_reconstruction is not None:
        plt.plot(range(timesteps), obj_reconstruction)
        plt.xlabel(xlabel);
        plt.ylabel('reconstruction error')
        plt.savefig(scoredir + os.sep + 'rbm_recon_%s.pdf' % (title_mod)); plt.close()

    plt.plot(range(timesteps + 1), obj_logP_termA)
    plt.xlabel(xlabel);
    plt.ylabel(r'$- \langle H(s) \rangle$')
    plt.savefig(scoredir + os.sep + 'rbm_termA_%s.pdf' % (title_mod)); plt.close()

    plt.plot(range(timesteps + 1), obj_logP_termB)
    plt.xlabel(xlabel);
    plt.ylabel(r'$\ln \ Z$')
    plt.savefig(scoredir + os.sep + 'rbm_logZ_%s.pdf' % (title_mod)); plt.close()

    plt.plot(range(timesteps + 1), score_arr)
    plt.xlabel(xlabel);
    plt.ylabel(r'$\langle\ln \ p(x)\rangle$')
    plt.savefig(scoredir + os.sep + 'rbm_score_%s.pdf' % (title_mod)); plt.close()
    return


def classify_with_rbm(rbm, outdir=None, beta=BETA):
    stdev = 1.0/np.sqrt(beta)

    print('Extracting features...')
    # TODO: check classification error after each epoch
    train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
    train_labels = np.zeros(len(train_dataset))
    test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
    test_labels = np.zeros(len(test_dataset))
    for i, (batch, labels) in enumerate(train_loader):
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
        batch = (batch > MNIST_BINARIZATION_CUTOFF).float()  # convert to 0,1 form
        batch = 2 * batch - 1  # convert to -1,1 form
        train_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch), :] = rbm.sample_hidden(batch, stdev=stdev)
        train_labels[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = labels.numpy()

    for i, (batch, labels) in enumerate(test_loader):
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
        batch = (batch > MNIST_BINARIZATION_CUTOFF).float()  # convert to 0,1 form
        batch = 2 * batch - 1  # convert to -1,1 form
        test_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch), :] = rbm.sample_hidden(batch, stdev=stdev)
        test_labels[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = labels.numpy()

    print('Training Classifier...')
    CLASSIFIER.fit(train_features, train_labels)
    print('Classifying...')
    predictions = CLASSIFIER.predict(test_features).astype(int)

    confusion_matrix = np.zeros((10, 10), dtype=int)
    matches = [False for _ in test_dataset]
    for idx, pair in enumerate(test_dataset):
        if pair[1] == predictions[idx]:
            matches[idx] = True
        confusion_matrix[pair[1], predictions[idx]] += 1
    title = "Successful test cases: %d/%d (%.3f)" % (
        matches.count(True), len(matches), float(matches.count(True) / len(matches)))
    if outdir is None:
        fpath = DIR_OUTPUT + os.sep + 'training' + os.sep + 'cm.jpg'
    else:
        fpath = outdir + os.sep + 'cm.jpg'
    cm = plot_confusion_matrix(confusion_matrix, title=title, save=fpath)
    print(title)
    return


if __name__ == '__main__':
    num_runs = 5
    hopfield_runs = True
    random_runs = False
    load_scores = False
    load_weights = False
    # TODO print settings file for each run?

    bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs'
    if hopfield_runs:
        for idx in range(num_runs):
            outdir = bigruns + os.sep + 'rbm' + os.sep + 'hopfield_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % (HIDDEN_UNITS, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS)
            rundir = outdir + os.sep + 'run%d' % idx
            custom_RBM_loop(load_weights=True, outdir=rundir)
    if random_runs:
        for idx in range(num_runs):
            outdir = bigruns + os.sep + 'rbm' + os.sep + 'normal_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % (HIDDEN_UNITS, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS)
            rundir = outdir + os.sep + 'run%d' % idx
            custom_RBM_loop(load_weights=False, outdir=rundir)

    if load_scores:
        outdir = bigruns + os.sep + 'rbm' + os.sep + 'C_beta2duringTraining_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % (BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS)
        fname = 'objective_10hidden_0fields_20cdk_200stepsAIS_2.00beta.npz'
        dataobj = np.load(outdir + os.sep + fname)

        obj_logP_termA = dataobj['termA']
        obj_logP_termB = dataobj['logZ']
        score_arr = dataobj['score']
        epochs = dataobj['epochs']
        timesteps = len(epochs) - 1

        title_mod = '10hidden_0fields_20cdk_200stepsAIS_2.00beta'
        plot_scores(timesteps, obj_logP_termA, obj_logP_termB, score_arr, outdir, title_mod,
                    'epochs', obj_reconstruction=None)

    if load_weights:
        # Note: looks like generative training does not help with classification at first glance
        local_beta = 200
        hidden = 20
        epoch_idx = 20

        outdir = bigruns + os.sep + 'rbm' + os.sep + 'E_extra_beta2duringTraining_100batch_20epochs_20cdk_1.00E-04eta_2ais'
        fname = 'weights_20hidden_0fields_20cdk_2stepsAIS_2.00beta.npz'
        dataobj = np.load(outdir + os.sep + fname)
        arr = dataobj['weights'][:, :, epoch_idx]

        rbm = RBM(VISIBLE_UNITS, hidden, 0, load_init_weights=False, use_fields=False, learning_rate=0)
        rbm.weights = torch.from_numpy(arr).float()

        classify_with_rbm(rbm, outdir=outdir, beta=local_beta)
