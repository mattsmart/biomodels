import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms

from AIS import get_obj_term_A, manual_AIS
from custom_rbm import RBM_custom, RBM_gaussian_custom
from data_process import image_data_collapse, binarize_image_data, data_mnist
from RBM_assess import plot_confusion_matrix, confusion_matrix_from_pred, get_X_y_dataset
from settings import MNIST_BINARIZATION_CUTOFF, DIR_OUTPUT, CLASSIFIER, BETA, DIR_MODELS


########## CONFIGURATION ##########
BATCH_SIZE = 100  # default 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 10  # was 128 but try 10
CD_K = 20
EPOCHS = 100  # was 10, or 51

LEARNING_RATE = 1*1e-4               # default was 1e-3, new base is 1e-4
learningrate_schedule = False         # swap from LEARNING_RATE to diff value at specified epoch
learningrate_schedule_value = 1*1e-4
learningrate_schedule_epoch = 25

AIS_STEPS = 0 #1000      # 0 or 1000 typically
AIS_CHAINS = 0 #100      # 100 or 500
USE_FIELDS = False
PLOT_WEIGHTS = False
POINTS_PER_EPOCH = 1

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
    num_class_samples = len(idx)
    loader = torch.utils.data.DataLoader(global_dataset, batch_size=BATCH_SIZE, sampler=torch.utils.data.sampler.SubsetRandomSampler(idx))

    return loader, num_class_samples


def custom_RBM_loop(train_loader, train_data_as_arr, hidden_units=HIDDEN_UNITS, init_weights=None,
                    use_fields=USE_FIELDS, beta=BETA, epochs=EPOCHS, cdk=CD_K,
                    outdir=None, classify=False, restart=False, points_per_epoch=POINTS_PER_EPOCH):
    assert beta == BETA         # TODO uncouple global STDEV in rbm class to make beta passable
    assert classify is False    # TODO need to add support for classify_with_rbm_hidden(...) at end

    if restart:
        fmod = ''  # was '_restart'; now use rundir name alone to store restart label
    else:
        fmod = ''

    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        trainingdir = outdir + os.sep + 'training' + fmod
        if not os.path.exists(trainingdir):
            os.makedirs(trainingdir)
    else:
        trainingdir = DIR_OUTPUT + os.sep + 'training'

    # initialize RBM class
    rbm = RBM(VISIBLE_UNITS, hidden_units, cdk, init_weights=init_weights, use_fields=use_fields,
              learning_rate=LEARNING_RATE)

    # prep object timeseries to store over training
    num_samples = train_data_as_arr.shape[0]
    batches_per_epoch = num_samples / BATCH_SIZE
    total_timepoints = points_per_epoch * epochs + 1
    iterations_per_pt = batches_per_epoch / points_per_epoch
    if POINTS_PER_EPOCH != 1:
        assert num_samples % BATCH_SIZE == 0
        assert batches_per_epoch % points_per_epoch == 0
    print("epochs:", epochs)
    print("num_samples:", num_samples)
    print("batches_per_epoch:", batches_per_epoch)
    print("iterations_per_pt:", iterations_per_pt)
    print("total_timepoints:", total_timepoints)
    iteration_arr = np.arange(total_timepoints) * iterations_per_pt  # iteration pt of the saved data
    print(iteration_arr)

    weights_timeseries = np.zeros((rbm.num_visible, rbm.num_hidden, total_timepoints))
    weights_timeseries[:, :, 0] = rbm.weights
    if use_fields:
        visible_bias_timeseries = np.zeros((rbm.num_visible, total_timepoints))
        visible_bias_timeseries[:, 0] = rbm.visible_bias.numpy()
        hidden_bias_timeseries = np.zeros((rbm.num_hidden, total_timepoints))
        hidden_bias_timeseries[:, 0] = rbm.hidden_bias.numpy()

    if PLOT_WEIGHTS:
        rbm.plot_model(title='epoch_0', outdir=trainingdir)

    obj_reconstruction = np.zeros(total_timepoints - 1)
    obj_logP_termA = np.zeros(total_timepoints)
    obj_logP_termB = np.zeros(total_timepoints)


    def loop_updates(iteration_idx, iteration_counter, epoch_recon_error=None):
        if PLOT_WEIGHTS:
            rbm.plot_model(title='iteration_%d' % (iteration_counter), outdir=trainingdir)

        if epoch_recon_error is not None:
            print('Epoch (Reconstruction) Error (epoch=%d) (iteration_idx:%d): %.4f' % (
                epoch + 1, iteration_idx, epoch_recon_error))
            obj_reconstruction[iteration_idx - 1] = epoch_recon_error

        if AIS_STEPS > 0:
            obj_logP_termA[iteration_idx] = get_obj_term_A(train_data_as_arr, rbm.weights, rbm.visible_bias,
                                                           rbm.hidden_bias, beta=beta)
            print('Estimating log Z...', )
            obj_logP_termB[iteration_idx], _ = manual_AIS(rbm, beta, nchains=AIS_CHAINS, nsteps=AIS_STEPS, CDK=1,
                                                          joint_mode=True)
        # save parameters each epoch
        weights_timeseries[:, :, iteration_idx] = rbm.weights.numpy()
        if use_fields:
            visible_bias_timeseries[:, iteration_idx] = rbm.visible_bias.numpy()
            hidden_bias_timeseries[:, iteration_idx] = rbm.hidden_bias.numpy()

        print('Term A:', obj_logP_termA[iteration_idx],
              '| Log Z:', obj_logP_termB[iteration_idx],
              '| Score:', obj_logP_termA[iteration_idx] - obj_logP_termB[iteration_idx])

    loop_updates(0, 0)

    print('Training RBM...')
    iteration_counter = 0
    for epoch in range(epochs):
        if epoch == learningrate_schedule_epoch and learningrate_schedule:
            rbm.learning_rate = learningrate_schedule_value
        print('LEARNING RATE: %.2E' % rbm.learning_rate)

        epoch_recon_error = 0.0
        for batch, _ in train_loader:
            iteration_counter += 1
            batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
            batch = (batch > MNIST_BINARIZATION_CUTOFF).float()  # convert to 0,1 form
            batch = -1 + batch * 2  # convert to -1,1 form
            batch_recon_error = rbm.contrastive_divergence(batch)
            epoch_recon_error += batch_recon_error

            if POINTS_PER_EPOCH != 1:
                if iteration_counter % iterations_per_pt == 0:
                    iteration_idx = int(iteration_counter / iterations_per_pt)
                    loop_updates(iteration_idx, iteration_counter, epoch_recon_error=epoch_recon_error)

        if POINTS_PER_EPOCH == 1:
            iteration_idx = epoch + 1
            loop_updates(iteration_idx, iteration_counter, epoch_recon_error=epoch_recon_error)

    score_arr = obj_logP_termA - obj_logP_termB

    if outdir is None:
        scoredir = DIR_OUTPUT + os.sep + 'logZ' + os.sep + 'rbm'
    else:
        scoredir = outdir

    # save parameters
    title_mod = '%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta%s' % (hidden_units, use_fields, cdk, AIS_STEPS, beta, fmod)
    fpath = scoredir + os.sep + 'objective_%s' % title_mod
    np.savez(fpath,
             epochs=range(epochs + 1),
             iterations=iteration_arr,
             termA=obj_logP_termA,
             logZ=obj_logP_termB,
             score=score_arr)

    fpath = scoredir + os.sep + 'weights_%s' % title_mod
    np.savez(fpath,
             epochs=range(epochs + 1),
             weights=weights_timeseries)  #TODO add iterations arr
    if use_fields:
        np.savez(scoredir + os.sep + 'visiblefield_%s' % title_mod,
                 epochs=range(epochs + 1),
                 iterations=iteration_arr,
                 visiblefield=visible_bias_timeseries)
        np.savez(scoredir + os.sep + 'hiddenfield_%s' % title_mod,
                 epochs=range(epochs + 1),
                 iterations=iteration_arr,
                 hiddenfield=hidden_bias_timeseries)

    plot_scores(iteration_arr, obj_logP_termA, obj_logP_termB, score_arr, scoredir, title_mod, 'iterations',
                obj_reconstruction=obj_reconstruction)

    if classify:
        classify_with_rbm_hidden(rbm, outdir=outdir)

    return rbm


def plot_scores(iteration_arr, obj_logP_termA, obj_logP_termB, score_arr, scoredir, title_mod, xlabel,
                obj_reconstruction=None):
    if obj_reconstruction is not None:
        plt.plot(iteration_arr[1:], obj_reconstruction)
        plt.xlabel(xlabel);
        plt.ylabel('reconstruction error')
        plt.savefig(scoredir + os.sep + 'rbm_recon_%s.pdf' % (title_mod)); plt.close()

    plt.plot(iteration_arr, obj_logP_termA)
    plt.xlabel(xlabel);
    plt.ylabel(r'$- \langle H(s) \rangle$')
    plt.savefig(scoredir + os.sep + 'rbm_termA_%s.pdf' % (title_mod)); plt.close()

    plt.plot(iteration_arr, obj_logP_termB)
    plt.xlabel(xlabel);
    plt.ylabel(r'$\ln \ Z$')
    plt.savefig(scoredir + os.sep + 'rbm_logZ_%s.pdf' % (title_mod)); plt.close()

    plt.plot(iteration_arr, score_arr)
    plt.xlabel(xlabel);
    plt.ylabel(r'$\langle\ln \ p(x)\rangle$')
    plt.savefig(scoredir + os.sep + 'rbm_score_%s.pdf' % (title_mod)); plt.close()
    return


def classify_with_rbm_hidden(rbm, train_dataset, train_loader, test_dataset, test_loader, outdir=None, beta=BETA):
    stdev = 1.0/np.sqrt(beta)

    print('Extracting features...')
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


def classifier_on_poe_scores(models, dataset_train, dataset_test, outpath, clfs=None, beta=2.0, plot=False):
    """
    models: dict of idx: rbm for idx in {0, ..., 9} i.e. the experts on each digit class
    clfs: list of classifiers
    """

    def score_1digit_model(model, img):
        """
        see hinton2002 ... .py for details
        """
        # should just be 0.5 * beta * Jij si sj
        beta = 1.0
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
        if plot:
            cm = plot_confusion_matrix(confusion_matrix, title=title, save=outpath)
            plt.close()
        print(title)
    return cms, accs


if __name__ == '__main__':

    train_dataset, train_loader, test_dataset, test_loader = torch_data_loading()
    TRAINING, TESTING = data_mnist(binarize=True)
    X, _ = get_X_y_dataset(TRAINING, dim_visible=VISIBLE_UNITS, binarize=True)

    test_data_loader = False

    num_runs = 5
    hopfield_runs = False
    random_runs = False
    restart_random_runs = False

    load_scores = False
    load_weights = False

    poe_mode_train = False
    poe_mode_classify = True
    # TODO print settings file for each run

    rescore_ais_trained_rbms = False

    if test_data_loader:
        seven_loader = get_classloader(train_dataset, 7)

    bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs'
    if hopfield_runs:

        HEBBIAN = False
        PCA = False
        MOD_SVD = False
        MOD_SQRT = False
        if HEBBIAN:
            hmod = '_hebbian'
            assert not PCA
        elif PCA:
            hmod = '_PCA'
        else:
            hmod = ''

        # load hopfield weights
        npzpath = DIR_MODELS + os.sep + 'saved' + os.sep + 'hopfield_mnist_%d%s.npz' % (HIDDEN_UNITS, hmod)
        print("Loading weights from %s" % npzpath)
        arr = np.load(npzpath)['Q']

        # modify initial weight matrix
        if MOD_SVD or MOD_SQRT:
            assert HEBBIAN
            import scipy as sp
            A = np.dot(arr.T, arr)
            A_inv = np.linalg.inv(A)
            A_sqrt = sp.linalg.sqrtm(A)  # A_sqrt = mtrx_sqrt(A)
            A_sqrt_inv = np.linalg.inv(A_sqrt)

            if MOD_SVD:
                # svd of XI
                U, Sigma, V = np.linalg.svd(arr, full_matrices=False)

                init_weights = torch.from_numpy(U).float()
            else:
                assert MOD_SQRT
                K = np.dot(arr, A_sqrt_inv)
                init_weights = torch.from_numpy(K).float()
        else:
            init_weights = torch.from_numpy(arr).float()

        outdir = bigruns + os.sep + 'rbm' + os.sep + 'hopfield%s_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais_%dppEpoch' % \
                 (hmod, HIDDEN_UNITS, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS, POINTS_PER_EPOCH)

        for idx in range(num_runs):
            rundir = outdir + os.sep + 'run%d' % idx
            custom_RBM_loop(train_loader, X, init_weights=init_weights.clone(), outdir=rundir, classify=False)

    if random_runs:
        for idx in range(num_runs):
            outdir = bigruns + os.sep + 'rbm' + os.sep + 'normal_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais_%dppEpoch' % \
                     (HIDDEN_UNITS, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS, POINTS_PER_EPOCH)
            rundir = outdir + os.sep + 'run%d' % idx
            custom_RBM_loop(train_loader, X, init_weights=None, outdir=rundir, classify=False)

    if restart_random_runs:
        EPOCHS_RESTART = 100
        HIDDEN_RESTART = 50
        ETA_RESTART = 1e-4
        AIS_STEPS_RESTART = 200
        EPOCHS_TO_EXTEND = 100

        for idx in range(num_runs):
            # load pre-trained weights
            indir = bigruns + os.sep + 'rbm' + os.sep + 'normal_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % \
                     (HIDDEN_RESTART, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS_RESTART, CD_K, ETA_RESTART, AIS_STEPS_RESTART)
            rundir = indir + os.sep + 'run%d' % idx
            print("Loading PRE_TRAINED weights from %s" % rundir)
            npzpath = rundir + os.sep + 'weights_%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta.npz' % (HIDDEN_RESTART, USE_FIELDS, CD_K, AIS_STEPS_RESTART, BETA)
            loaded_weights_timeseries_np = np.load(npzpath)['weights']
            init_weights_np = loaded_weights_timeseries_np[:, :, -1]
            init_weights = torch.from_numpy(init_weights_np).float()
            # specify new filename modifier
            outdir = rundir + '_restart'
            custom_RBM_loop(train_loader, X, init_weights=init_weights, outdir=outdir, classify=False, restart=True, epochs=EPOCHS_TO_EXTEND)

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
        hidden = 1000
        epoch_idx = 199

        outdir = bigruns + os.sep + 'rbm' + os.sep + 'normal_%dhidden_0fields_2.00beta_100batch_200epochs_1cdk_1.00E-04eta_0ais' % hidden \
                 + os.sep + 'run0'
        fname = 'weights_%dhidden_0fields_1cdk_0stepsAIS_2.00beta.npz' % hidden
        dataobj = np.load(outdir + os.sep + fname)
        arr = dataobj['weights'][:, :, epoch_idx]

        rbm = RBM(VISIBLE_UNITS, hidden, 0, init_weights=None, use_fields=False, learning_rate=0)
        rbm.weights = torch.from_numpy(arr).float()

        classify_with_rbm_hidden(rbm, train_dataset, train_loader, test_dataset, test_loader, outdir=outdir, beta=local_beta)

    if poe_mode_train:
        # TODO X_for_digit make (will be called if AIS steps > 0 ) -- second arg to custom_RBM_loop()
        # TODO beta in scoring
        use_hopfield = True
        k_range = [10, 20, 100] #, 200, 500, 250, 300]
        runs = 3

        HEBBIAN = False
        PCA = True
        MOD_SVD = False
        MOD_SQRT = False
        if HEBBIAN:
            hmod = '_hebbian'
            assert not PCA
        elif PCA:
            hmod = '_PCA'
        else:
            hmod = ''

        loader_dict = {idx: get_classloader(train_dataset, idx)[0] for idx in range(10)}
        loader_num_samples_dict = {idx: get_classloader(train_dataset, idx)[1] for idx in range(10)}

        for idx in range(runs):
            for k in k_range:
                print("Training POE for k=%d" % k)
                for digit in range(10):
                    print("Training POE for k=%d (digit: %d)" % (k, digit))
                    dir_modifier = '%ddigit_%dhidden_%dfields_%.2fbeta_%dbatch_%depochs_%dcdk_%.2Eeta_%dais' % \
                                   (digit, k, USE_FIELDS, BETA, BATCH_SIZE, EPOCHS, CD_K, LEARNING_RATE, AIS_STEPS)

                    fake_empty_data = np.zeros((loader_num_samples_dict[digit], 28**2))

                    if use_hopfield:
                        outdir = bigruns + os.sep + 'poe' + os.sep + 'run%d' % idx + os.sep + \
                                 'poe_hopfield%s_%s' % (hmod, dir_modifier)
                        rundir = outdir
                        # load hopfield weights
                        npzpath = DIR_MODELS + os.sep + 'poe' + os.sep + 'hopfield_digit%d_p%d%s.npz' % (digit, 10*k, hmod)
                        print("Loading weights from %s" % npzpath)
                        arr = np.load(npzpath)['Q']
                        init_weights = torch.from_numpy(arr).float()

                        custom_RBM_loop(loader_dict[digit], fake_empty_data, hidden_units=k, init_weights=init_weights.clone(), outdir=rundir, classify=False)
                    else:
                        outdir = bigruns + os.sep + 'poe' + os.sep + 'run%d' % idx + os.sep + \
                                 'poe_normal_%s' % dir_modifier
                        rundir = outdir
                        custom_RBM_loop(loader_dict[digit], fake_empty_data, hidden_units=k, init_weights=None, outdir=rundir, classify=False)

    if poe_mode_classify:
        init_weights_type = 'hopfield'  # hopfield or normal

        HEBBIAN = False
        PCA = False
        if HEBBIAN:
            assert init_weights_type == 'hopfield'
            assert not PCA
            init_weights_type += '_hebbian'
        elif PCA:
            assert init_weights_type == 'hopfield'
            init_weights_type += '_PCA'
        else:
            hmod = ''

        k_range = [10, 20, 100]    #, 200, 250, 300, 500]  # range(1, 110)  # 110
        epochs = [0, 1, 2, 3, 4] + list(range(5, 101, 5))
        runs = [2] # [0,1,2]          # TODO care
        accs = np.zeros((len(epochs), len(k_range)))

        for run in runs:
            for epoch_idx, epoch in enumerate(epochs):
                for k_idx, k in enumerate(k_range):
                    print('epoch, k:', epoch, k)
                    hidden_units = k
                    dir_modifier = '0fields_2.00beta_100batch_100epochs_%dcdk_1.00E-04eta_0ais' % (CD_K)
                    models10 = {}
                    for digit in range(10):
                        rbm_digit = RBM(VISIBLE_UNITS, hidden_units, 0, init_weights=None, use_fields=False, learning_rate=0)
                        # load weights for given epoch
                        run_dir = bigruns + os.sep + 'poe' + os.sep + '%s' % init_weights_type + os.sep + 'run%d' % run
                        npzpdir = run_dir + os.sep + \
                                  'poe_%s_%ddigit_%dhidden_%s' % (init_weights_type, digit, hidden_units, dir_modifier)
                        weights_path = npzpdir + os.sep + 'weights_%dhidden_0fields_%dcdk_0stepsAIS_2.00beta.npz' % (hidden_units, CD_K)
                        arr = np.load(weights_path)['weights'][:, :, epoch]
                        rbm_digit.weights = torch.from_numpy(arr).float()
                        # set as model for that digit
                        models10[digit] = rbm_digit
                    fpath = DIR_OUTPUT + os.sep + 'training' + os.sep + 'cm_%s_k%d_epoch%d.jpg' % (init_weights_type, k, epoch)
                    cm_list, acc_list = classifier_on_poe_scores(models10, TRAINING, TESTING, fpath, clfs=None, beta=2.0, plot=False)
                    accs[epoch_idx, k_idx] = acc_list[0]

            # save data
            fpath = DIR_OUTPUT + os.sep + 'training' + os.sep + 'poe_scores_kNum%d_epochsNum%d_%s%d.npz' % \
                    (len(k_range), len(epochs), init_weights_type, run)
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
            plt.savefig(DIR_OUTPUT + os.sep + 'training' + os.sep + 'score_vs_k%d_vs_epoch%d_%s%d.jpg' %
                        (len(k_range), len(epochs), init_weights_type, run))
            plt.show()
            # TODO check if hinton use logreg or SVM
            # TODO increase max iter to remove warning: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
            #   "the coef_ did not converge", ConvergenceWarning)
            # TODO more epochs on whichever k worked best? depends on how the error scales with epoch at large k
            # TODO compare vs normal dist (note this run of 20epochs x 110 k values was ~11GB)

    if rescore_ais_trained_rbms:

        # AIS settings
        beta = 2.0
        nsteps = 1000
        nchains = 100
        ntest = 1
        nsteps_rev = 0
        nchains_rev = 0
        runs = 1
        assert runs == 1
        # hebbian = False
        """
        if hebbian:
            strmod = '_hebbian'
        else:
            strmod = '' """

        # prep dataset
        training_subsample = TRAINING[:]
        X, _ = get_X_y_dataset(training_subsample, dim_visible=28 ** 2, binarize=True)

        # prep models to load
        epoch_list = list(range(0, 51))  #71
        VISIBLE_UNITS = 28 ** 2
        hidden_units = 50
        CD_K = 20
        bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'rbm'
        model_dir = 'hopfield_PCA_%dhidden_0fields_2.00beta_100batch_50epochs_20cdk_1.00E-04eta_0ais_1ppEpoch' % hidden_units
        model_runs = [2,3,4] #[0,1,2,3,4]  #[0, 1, 2, 3, 4]
        model_paths = [
            bigruns + os.sep + model_dir + os.sep + 'run%d' % a + os.sep +
            'weights_%dhidden_0fields_20cdk_0stepsAIS_2.00beta.npz' % hidden_units
            for a in model_runs
        ]

        for training_run_idx, weights_path in enumerate(model_paths):

            # load weights for given epoch
            weights_timeseries_np = np.load(weights_path)['weights']

            rbm = RBM(VISIBLE_UNITS, hidden_units, 0, init_weights=None, use_fields=False, learning_rate=0)
            N = rbm.num_visible
            p = rbm.num_hidden
            zero_field_visible = np.zeros(N)
            zero_field_hidden = np.zeros(p)

            termA_arr = np.zeros(len(epoch_list))
            logZ_arr = np.zeros(len(epoch_list))
            score_arr = np.zeros(len(epoch_list))

            for idx in range(len(epoch_list)):
                epoch = epoch_list[idx]
                # specify new local rbm class for this epoch only
                weights_epoch_np = weights_timeseries_np[:, :, epoch]
                rbm.weights = torch.from_numpy(weights_epoch_np).float()

                obj_term_A = get_obj_term_A(X, weights_epoch_np, zero_field_visible, zero_field_hidden, beta=beta)
                termA_arr[idx] = obj_term_A

                # Forward AIS
                logZ_fwd, _ = manual_AIS(rbm, beta, nchains=nchains, nsteps=nsteps)
                score_fwd = obj_term_A - logZ_fwd
                print('training_run_idx, epoch:', training_run_idx, epoch)
                print('mean log p(data):', score_fwd,
                      '(run 1 only, beta=%.2f, A=%.2f, B=%.2f)' % (beta, obj_term_A, logZ_fwd))
                logZ_arr[idx] = logZ_fwd
                score_arr[idx] = score_fwd

            # save updated AIS data
            out_dir = os.path.dirname(weights_path)
            title_mod = '%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta' % (hidden_units, False, CD_K, nsteps, beta)
            fpath = out_dir + os.sep + 'objective_%s' % title_mod
            np.savez(fpath,
                     epochs=epoch_list,
                     termA=termA_arr,
                     logZ=logZ_arr,
                     score=score_arr)
