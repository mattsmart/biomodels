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
from RBM_assess import plot_confusion_matrix, confusion_matrix_from_pred, get_X_y_dataset
from settings import MNIST_BINARIZATION_CUTOFF, DIR_OUTPUT, CLASSIFIER, BETA, DIR_MODELS


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
CD_K = 1 # TODO revert
LEARNING_RATE = 1e-4  # default 1e-3
EPOCHS = 200  # was 10
AIS_STEPS = 0  #200
USE_FIELDS = False
PLOT_WEIGHTS = False

GAUSSIAN_RBM = True
if RBM_gaussian_custom:
    RBM = RBM_gaussian_custom
else:
    RBM = RBM_custom


def torch_data_loading(batch_size=BATCH_SIZE):
    print('Loading dataset...')
    DATA_FOLDER = 'data'
    train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True,
                                               transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False,
                                              transform=torchvision.transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return train_dataset, train_loader, test_dataset, test_loader


def get_classloader(global_dataset, class_name):
    # see https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276/12
    def get_indices(dataset):
        indices = []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == class_name:
                indices.append(i)
        return indices

    idx = get_indices(global_dataset)
    loader = torch.utils.data.DataLoader(global_dataset, batch_size=BATCH_SIZE, sampler=torch.utils.data.sampler.SubsetRandomSampler(idx))

    #for idx, (data, target) in enumerate(loader):
    #    print(idx, target)
    return loader


def custom_RBM_loop(train_loader, train_data_as_arr, hidden_units=HIDDEN_UNITS, init_weights=None,
                    use_fields=USE_FIELDS, beta=BETA, epochs=EPOCHS, cdk=CD_K,
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
    rbm = RBM(VISIBLE_UNITS, hidden_units, cdk, init_weights=init_weights, use_fields=use_fields,
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
        obj_logP_termA[0] = get_obj_term_A(train_data_as_arr, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta)
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
            obj_logP_termA[epoch + 1] = get_obj_term_A(train_data_as_arr, rbm.weights, rbm.visible_bias, rbm.hidden_bias, beta=beta)
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
    title_mod = '%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta' % (hidden_units, use_fields, cdk, AIS_STEPS, beta)
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
        classify_with_rbm_hidden(rbm, outdir=outdir)

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


def classify_with_rbm_hidden(rbm, train_dataset, train_loader, test_dataset, test_loader, outdir=None, beta=BETA):
    stdev = 1.0/np.sqrt(beta)

    print('Extracting features...')
    # TODO: check classification error after each epoch
    train_features = np.zeros((len(train_dataset), rbm.num_hidden))
    train_labels = np.zeros(len(train_dataset))
    test_features = np.zeros((len(test_dataset), rbm.num_hidden))
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


def classifier_on_poe_scores(models, dataset_train, dataset_test, outpath, clfs=None, beta=2.0):
    """
    models: dict of idx: rbm for idx in {0, ..., 9} i.e. the experts on each digit class
    clfs: list of classifiers
    """

    def score_1digit_model(model, img):
        """
        see hinton2002 ... .py for details
        """
        # should just be 0.5 * beta * Jij si sj
        beta = 1.0  # TODO 2?
        W = model.weights
        dotp = np.dot(W.T, img)
        # score = np.dot(dotp, dotp)  #
        score = beta * np.dot(dotp, dotp) / 2.0
        return score

    if clfs is None:
        clfs = [CLASSIFIER]

    features_order = list(models.keys())
    feature_dim = len(features_order)

    def get_X_y_features(dataset):
        X = np.zeros((len(dataset), feature_dim))
        y = np.zeros(len(dataset), dtype=int)
        for idx, pair in enumerate(dataset):
            elem_arr, elem_label = pair
            preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
            features = np.array([score_1digit_model(models[key], preprocessed_input) for key in features_order])
            #features = np.array([score_1digit_model(models[idx], preprocessed_input) for idx in range(10)])
            X[idx, :] = features
            y[idx] = elem_label
        return X, y

    print("[classifier_on_rbm_features] Step 1: get features for training")
    X_train_reduced, y_train = get_X_y_features(dataset_train)
    print("\tTraining data dimension", X_train_reduced.shape, y_train.shape)

    print("[classifier_on_rbm_features] Step 2: train classifier layer")
    for clf in clfs:
        print('fitting...')
        clf.fit(X_train_reduced, y_train)  # fit data

    print("[classifier_on_rbm_features] Step 3: get features for testing")
    X_test_reduced, y_test = get_X_y_features(dataset_test)

    print("[classifier_on_rbm_features] Step 4: classification metrics and confusion matrix")
    cms = [0] * len(clfs)
    accs = [0] * len(clfs)
    for idx, clf in enumerate(clfs):
        print('predicting...')
        predictions = clf.predict(X_test_reduced).astype(int)
        confusion_matrix, matches = confusion_matrix_from_pred(predictions, y_test)
        acc = float(matches.count(True) / len(matches))
        cms[idx] = confusion_matrix
        accs[idx] = acc
        title = "Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), acc)
        cm = plot_confusion_matrix(confusion_matrix, title=title, save=outpath)
        plt.close()
        print(title)
    return cms, accs


if __name__ == '__main__':

    train_dataset, train_loader, test_dataset, test_loader = torch_data_loading()
    TRAINING, TESTING = data_mnist(binarize=True)
    X, _ = get_X_y_dataset(TRAINING, dim_visible=VISIBLE_UNITS, binarize=True)  # TODO digit specification

    test_data_loader = False

    num_runs = 5
    hopfield_runs = False
    random_runs = False
    load_scores = False
    load_weights = False

    poe_mode_train = False
    poe_mode_classify = True
    # TODO print settings file for each run?

    if test_data_loader:
        seven_loader = get_classloader(train_dataset, 7)

    bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs'
    if hopfield_runs:
        for idx in range(num_runs):
            outdir = bigruns + os.sep + 'rbm' + os.sep + 'hopfield_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % \
                     (HIDDEN_UNITS, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS)
            rundir = outdir + os.sep + 'run%d' % idx
            # load hopfield weights
            npzpath = DIR_MODELS + os.sep + 'saved' + os.sep + 'hopfield_mnist_%d.npz' % HIDDEN_UNITS
            print("Loading weights from %s" % npzpath)
            arr = np.load(npzpath)['Q']
            init_weights = torch.from_numpy(arr).float()
            custom_RBM_loop(train_loader, X, init_weights=init_weights, outdir=rundir)
    if random_runs:
        for idx in range(num_runs):
            outdir = bigruns + os.sep + 'rbm' + os.sep + 'normal_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % \
                     (HIDDEN_UNITS, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS)
            rundir = outdir + os.sep + 'run%d' % idx
            custom_RBM_loop(train_loader, X, init_weights=None, outdir=rundir)

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

        rbm = RBM(VISIBLE_UNITS, hidden, 0, init_weights=None, use_fields=False, learning_rate=0)
        rbm.weights = torch.from_numpy(arr).float()

        classify_with_rbm_hidden(rbm, train_dataset, train_loader, test_dataset, test_loader, outdir=outdir, beta=local_beta)

    if poe_mode_train:
        # TODO X_for_digit make (will be called if AIS steps > 0 ) -- second arg to custom_RBM_loop()
        # TODO beta in scoring
        use_hopfield = True
        k_range = [100]  # range(1, 110)
        loader_dict = {idx: get_classloader(train_dataset, idx) for idx in range(10)}

        for k in k_range:
            print("Training POE for k=%d" % k)
            for digit in range(10):
                print("Training POE for k=%d (digit: %d)" % (k, digit))
                dir_modifier = '%ddigit_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % \
                               (digit, k, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS)
                if use_hopfield:
                    outdir = bigruns + os.sep + 'poe' + os.sep + 'poe_hopfield_%s' % dir_modifier
                    rundir = outdir
                    # load hopfield weights
                    npzpath = DIR_MODELS + os.sep + 'poe' + os.sep + 'hopfield_digit%d_p%d.npz' % (digit, 10*k)
                    print("Loading weights from %s" % npzpath)
                    arr = np.load(npzpath)['Q']
                    init_weights = torch.from_numpy(arr).float()

                    custom_RBM_loop(loader_dict[digit], None, hidden_units=k, init_weights=init_weights, outdir=rundir, classify=False)
                else:
                    outdir = bigruns + os.sep + 'poe' + os.sep + 'poe_normal_%s' % dir_modifier
                    rundir = outdir
                    custom_RBM_loop(loader_dict[digit], None, hidden_units=k, init_weights=None, outdir=rundir, classify=False)

    if poe_mode_classify:
        init_weights_type = 'hopfield'  # hopfield or normal
        k_range = [100] # range(1, 110)  # 110
        epochs = [0] + list(range(19, 200, 20)) #[0, 5, 10, 15, 19]
        accs = np.zeros((len(epochs), len(k_range)))

        for epoch_idx, epoch in enumerate(epochs):
            for k_idx, k in enumerate(k_range):
                hidden_units = k
                dir_modifier = '0fields_2.00beta_100batch_200epochs_%dcdk_%.2Eeta_0ais' % (CD_K, LEARNING_RATE)
                models10 = {}
                for digit in range(10):
                    rbm_digit = RBM(VISIBLE_UNITS, hidden_units, 0, init_weights=None, use_fields=False, learning_rate=0)
                    # load weights for given epoch
                    npzpdir = bigruns + os.sep + 'poe' + os.sep + '%s' % (init_weights_type) + os.sep + \
                              'poe_%s_%ddigit_%dhidden_%s' % (init_weights_type, digit, hidden_units, dir_modifier)
                    weights_path = npzpdir + os.sep + 'weights_%dhidden_0fields_%dcdk_0stepsAIS_2.00beta.npz' % (hidden_units, CD_K)
                    arr = np.load(weights_path)['weights'][:, :, epoch]
                    rbm_digit.weights = torch.from_numpy(arr).float()
                    # set as model for that digit
                    models10[digit] = rbm_digit
                fpath = DIR_OUTPUT + os.sep + 'training' + os.sep + 'cm_%s_k%d_epoch%d.jpg' % (init_weights_type, k, epoch)
                cm_list, acc_list = classifier_on_poe_scores(models10, TRAINING, TESTING, fpath, clfs=None, beta=2.0)
                accs[epoch_idx, k_idx] = acc_list[0]
        # save data
        fpath = DIR_OUTPUT + os.sep + 'training' + os.sep + 'poe_%s_scores_kNum%d_epochsNum%d.npz' % (init_weights_type, len(k_range), len(epochs))
        np.savez(fpath, accs_epoch_by_k=accs, epochs=epochs, k_range=k_range)
        print(accs)
        # plot data
        plt.figure()
        error_pct = 100 * (1 - accs)
        for idx, epoch in enumerate(epochs):
            plt.plot(k_range, error_pct[idx, :], label='Epoch: %d' % epoch)
        plt.xlabel(r'$k$')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(DIR_OUTPUT + os.sep + 'training' + os.sep + 'score_vs_k%d_vs_epoch%d.jpg' % (len(k_range), len(epochs)))
        plt.show()
        # TODO check if hinton use logreg or SVM
        # TODO increase max iter to remove warning: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
        #   "the coef_ did not converge", ConvergenceWarning)
        # TODO more epochs on whichever k worked best? depends on how the error scales with epoch at large k
        # TODO compare vs normal dist (note this run of 20epochs x 110 k values was ~11GB)