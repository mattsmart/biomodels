import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.decomposition import PCA

from data_process import image_data_collapse, torch_image_to_numpy, binarize_image_data
from RBM_train import RBM, TRAINING, TESTING, build_rbm_hopfield, load_rbm_hopfield
from settings import BETA, USE_BETA_SCHEDULER, DIR_OUTPUT, VISIBLE_FIELD, HRBM_CLASSIFIER_STEPS, HRBM_MANUAL_MAXSTEPS, \
    MNIST_BINARIZATION_CUTOFF, DEFAULT_HOPFIELD, DIR_MODELS, DIR_CLASSIFY, CLASSIFIER


# TODO
#   Question: how to perform classification with binary-binary or binary-gaussian RBM? append module to end?


def setup_MNIST_classification(rbm):
    label_dict = {}
    for idx in range(rbm.dim_hidden):

        idx_to_patternlabel_exact = rbm.pattern_labels[idx]
        idx_to_patternlabel_class = idx_to_patternlabel_exact[0]  # i.e. if its '7_0', take '7'

        key_tuple = tuple([0 if a != idx else 1 for a in range(rbm.dim_hidden)])
        label_dict[key_tuple] = idx_to_patternlabel_class
        neg_key_tuple = tuple([0 if a != idx else -1 for a in range(rbm.dim_hidden)])
        label_dict[neg_key_tuple] = 'anti-%s' % idx_to_patternlabel_class
    return label_dict


def extend_init_visible_state(visible_state, dim_hidden, init_mode='A'):
    # mode A - initial class label is all -1
    # mode B - initial class label is coin flips
    assert init_mode in ['A', 'B']
    if init_mode == 'A':
        visible_state_appendage = np.ones(dim_hidden) * -1
    else:
        visible_state_appendage = np.random.binomial(size=dim_hidden, n=1, p=0.5) * 2 - 1  # convert to -1, 1 form
    visible_state = np.concatenate((visible_state, visible_state_appendage))
    return visible_state


def classify_MNIST(rbm, visual_init, dataset_idx, MNIST_output_to_label, sum_mode=False, onehot=False):
    visual_step = visual_init

    if onehot:
        visual_step = extend_init_visible_state(visual_init, rbm.dim_hidden, init_mode='A')

        def conv_class_vector_to_label_onehot(visual_step):
            # TODO alternate condensing
            pool = True
            class_label = rbm.onehot_class_label(visual_step, condense=True, pool=pool)
            if class_label is None:
                return False, None
            else:
                return True, class_label

        for idx in range(HRBM_MANUAL_MAXSTEPS):
            visual_step, hidden_step, output_step = rbm.RBM_step(visual_step)
            classified, classification = conv_class_vector_to_label_onehot(visual_step)
            if classified:
                break
        if idx == HRBM_MANUAL_MAXSTEPS - 1:
            print("******************** Edge case unclassified: (%d) step == MAX_STEPS_CLASSIFY - 1" % dataset_idx)
        #else:
        #    print('Step %d (data #%d) = %s' % (idx, dataset_idx, classification))

    else:
        def conv_class_vector_to_label(output_as_ints):
            if tuple(output_as_ints) in MNIST_output_to_label.keys():
                return True, MNIST_output_to_label[tuple(output_as_ints)]
            else:
                return False, output_as_ints

        if sum_mode:
            output_converter = rbm.truncate_output_subpatterns
        else:
            output_converter = rbm.truncate_output  # rbm.truncate_output_max

        for idx in range(HRBM_MANUAL_MAXSTEPS):
            visual_step, hidden_step, output_step = rbm.RBM_step(visual_step)
            output_truncated = output_converter(output_step)
            classified, classification = conv_class_vector_to_label(output_truncated)
            if classified:
                break
        if idx == HRBM_MANUAL_MAXSTEPS - 1:
            print("******************** Edge case unclassified: (%d) step == MAX_STEPS_CLASSIFY - 1" % dataset_idx)
            #print("\t output_step:", output_step)
            #output_special = np.zeros(10, dtype=float)
            #K = 10
            #for idx in range(10):
            #    output_special[idx] = np.sum(output_step[K*idx:K*(idx + 1)])
            #print("\t output_special:", output_special)
            #print("\t output_truncated:", output_truncated)
            #print("\t classification:", classification)

    return classification


def rbm_features_MNIST(rbm, visual_init, steps=HRBM_CLASSIFIER_STEPS, use_hidden=True, plot_visible=False, titlemod='',
                       scheduler=USE_BETA_SCHEDULER, onehot=False):
    visual_step = visual_init
    P = rbm.dim_hidden

    if use_hidden:
        if onehot:
            visual_step = extend_init_visible_state(visual_init, rbm.dim_hidden, init_mode='A')
            features = np.zeros(P * 2)
        else:
            features = np.zeros(P)
    else:
        features = np.zeros(rbm.dim_visible)

    # build temperature schedule  TODO move out for speed
    beta_schedule = [BETA for _ in range(steps)]
    if scheduler:
        assert steps == 5
        switchpoint = 1
        for idx in range(steps):
            if idx < switchpoint:
                beta_schedule[idx] = 200.0  # 2 seems too strong, 8 too weak
            else:
                beta_schedule[idx] = 8.0 - idx

    if plot_visible:
        rbm.plot_visible(visual_init, title='%s_0' % titlemod)
    for idx in range(steps):
        visual_step, hidden_step, _ = rbm.RBM_step(visual_step, beta=beta_schedule[idx])
        if plot_visible:
            title = '%s_%d' % (titlemod, idx+1)
            rbm.plot_visible(visual_step, title=title)
    if use_hidden:
        features[0:P] = hidden_step
        if onehot:
            onehot_segment = visual_step[-P:]
            features[P:] = onehot_segment
    else:
        features[:] = visual_step
    return features


def confusion_matrix_from_pred(predictions, true_labels):
    #confusion_matrix = np.zeros((rbm.dim_hidden, rbm.dim_hidden), dtype=int)
    confusion_matrix_10 = np.zeros((10, 10), dtype=int)
    matches = [False for _ in predictions]
    for idx in range(len(predictions)):
        if true_labels[idx] == predictions[idx]:
            matches[idx] = True
        confusion_matrix_10[true_labels[idx], predictions[idx]] += 1
    return confusion_matrix_10, matches


def classifier_on_rbm_features(rbm, dataset_train, dataset_test, use_hidden=True, binarize=False, classifier=CLASSIFIER,
                               fast=None, onehot=False):
    """
    fast: None or a 4-tuple of X_train, y_train, X_test, y_test
    """
    if use_hidden:
        if onehot:
            feature_dim = rbm.dim_hidden * 2
        else:
            feature_dim = rbm.dim_hidden
    else:
        feature_dim = rbm.dim_visible

    def get_X_y_features(dataset, steps=HRBM_CLASSIFIER_STEPS, scheduler=USE_BETA_SCHEDULER):
        X = np.zeros((len(dataset), feature_dim))
        y = np.zeros(len(dataset), dtype=int)
        for idx, pair in enumerate(dataset):
            elem_arr, elem_label = pair
            if use_hidden:
                preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
                features = rbm_features_MNIST(rbm, preprocessed_input, titlemod='%d_true%d' % (idx,elem_label),
                                              steps=steps, scheduler=scheduler, onehot=onehot)
            else:
                preprocessed_input = image_data_collapse(elem_arr)
                if binarize:
                    preprocessed_input = binarize_image_data(preprocessed_input, threshold=MNIST_BINARIZATION_CUTOFF)
                features = preprocessed_input
            X[idx, :] = features
            y[idx] = elem_label
        return X, y

    def get_X_fast(X):
        X_features = np.zeros((X.shape[0], feature_dim))
        for idx in range(X.shape[0]):
            visible_input = X[idx, :]
            features = rbm_features_MNIST(rbm, visible_input, use_hidden=use_hidden, titlemod='', plot_visible=False,
                                          onehot=onehot)
            X_features[idx, :] = features
        return X_features

    print("classifier_on_rbm_features; Step 1: get features for training")
    if fast is None:
        X_train_reduced, y_train = get_X_y_features(dataset_train)
    else:
        X_train_reduced = get_X_fast(fast[0])
        y_train = fast[1]
    print("classifier_on_rbm_features; Step 2: train classifier layer")
    classifier.fit(X_train_reduced, y_train)  # fit data
    print("classifier_on_rbm_features; Step 3: get features for testing")
    if fast is None:
        X_test_reduced, y_test = get_X_y_features(dataset_test)
        #X_test_reduced, y_test = get_X_y_features(dataset_test, steps=1, scheduler=False)  # TODO try diff steps/beta rules for train vs test?
    else:
        X_test_reduced = get_X_fast(fast[2])
        y_test = fast[3]
    print("classifier_on_rbm_features; Step 4: classification metrics and confusion matrix")
    # sparsity1 = np.mean(clf1.coef_ == 0) * 100  # percentage of nonzero weights
    predictions = classifier.predict(X_test_reduced).astype(int)
    confusion_matrix, matches = confusion_matrix_from_pred(predictions, y_test)
    acc = float(matches.count(True) / len(matches))
    print("Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), acc))
    return confusion_matrix, acc


def get_X_y_dataset(dataset, dim_visible, binarize=True):
    X = np.zeros((len(dataset), dim_visible))
    y = np.zeros(len(dataset), dtype=int)
    for idx, pair in enumerate(dataset):
        elem_arr, elem_label = pair
        preprocessed_input = image_data_collapse(elem_arr)
        if binarize:
            preprocessed_input = binarize_image_data(preprocessed_input, threshold=MNIST_BINARIZATION_CUTOFF)
        features = preprocessed_input
        X[idx, :] = features
        y[idx] = elem_label
    return X, y


def classifier_on_proj(dataset_train, dataset_test, dim_visible, binarize=False, dim=10, classifier=CLASSIFIER, proj=None, fast=None):

    if proj is not None:
        assert proj.shape == (dim_visible, dim)

    print("classifier_on_proj; Step 1: get features for training")
    if fast is None:
        X_train, y_train = get_X_y_dataset(dataset_train, dim_visible, binarize=binarize)
    else:
        X_train = fast[0]
        y_train = fast[1]
    if proj is None:
        pca = PCA(n_components=dim)
        pca.fit(X_train)
        X_train_reduced = pca.transform(X_train)
    else:
        X_train_reduced = np.dot(X_train, proj)

    print("classifier_on_proj; Step 2: train classifier layer")
    classifier.fit(X_train_reduced, y_train)  # fit data

    print("classifier_on_proj; Step 3: get features for testing")
    # use PCA to reduce dim of testing set
    if fast is None:
        X_test, y_test = get_X_y_dataset(dataset_test, dim_visible, binarize=binarize)
    else:
        X_test = fast[2]
        y_test = fast[3]
    if proj is None:
        X_test_reduced = pca.transform(X_test)
    else:
        X_test_reduced = np.dot(X_test, proj)

    print("classifier_on_proj; Step 4: classification metrics and confusion matrix")
    predictions = classifier.predict(X_test_reduced).astype(int)
    confusion_matrix, matches = confusion_matrix_from_pred(predictions, y_test)
    acc = float(matches.count(True) / len(matches))
    print("Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), acc))
    return confusion_matrix, acc


def plot_confusion_matrix(confusion_matrix, classlabels=list(range(10)), title='', save=None):
    # Ref: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    import seaborn as sn
    import pandas as pd

    ylabels = classlabels
    if confusion_matrix.shape[1] == len(ylabels) + 1:
        xlabels = ylabels + ['Other']
    else:
        xlabels = ylabels
    df_cm = pd.DataFrame(confusion_matrix, index=ylabels, columns=xlabels)

    plt.figure(figsize=(11, 7))
    sn.set(font_scale=1.2)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='d')  # font size
    plt.gca().set(xlabel='Predicted', ylabel='True label')
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    return


def multiscore_save():
    # TODO refactor main blocks into this fn?
    return


if __name__ == '__main__':
    DIM_VISIBLE = 28 ** 2

    multiscore_save_rbm = False
    multiscore_save_pca = False
    multiscore_save_random = False

    # flags for single model classification block
    dont_use_rbm = False
    use_random_proj = False

    plot_wrong = False
    append_classifier_layer = True
    manual_onehot = True
    manual_hidden = 100

    if multiscore_save_rbm:
        label = 'Kvary_betaFix%.2f_steps%d' % (BETA, HRBM_CLASSIFIER_STEPS)
        params = list(range(1, 21))
        scores = [0] * len(params)
        X_train, y_train = get_X_y_dataset(TRAINING, DIM_VISIBLE, binarize=True)
        X_test, y_test = get_X_y_dataset(TESTING, DIM_VISIBLE, binarize=True)
        fast_input = (X_train, y_train, X_test, y_test)

        for idx in range(len(params)):
            npzpath = DIR_MODELS + os.sep + 'saved' + os.sep + 'hopfield_mnist_%d0.npz' % params[idx]
            rbm_hopfield = load_rbm_hopfield(npzpath=npzpath)
            confusion_matrix, acc = classifier_on_rbm_features(rbm_hopfield, TRAINING, TESTING, use_hidden=True,
                                                               binarize=True, fast=fast_input)
            plot_confusion_matrix(confusion_matrix)
            scores[idx] = acc
        out = [[params[idx], scores[idx]] for idx in range(len(params))]
        np.savetxt(DIR_OUTPUT + os.sep + 'multiscore_%s.txt' % label, out)

    elif multiscore_save_pca:
        params = list(range(1, 201))
        for binarize in [True, False]:
            label = 'Kvary_pca_binary%d' % binarize
            scores = [0] * len(params)
            X_train, y_train = get_X_y_dataset(TRAINING, DIM_VISIBLE, binarize=binarize)
            X_test, y_test = get_X_y_dataset(TESTING, DIM_VISIBLE, binarize=binarize)
            fast_input = (X_train, y_train, X_test, y_test)
            for idx in range(len(params)):
                print(idx, params[idx], binarize)
                confusion_matrix, acc = classifier_on_proj(TRAINING, TESTING, DIM_VISIBLE, proj=None,
                                                           binarize=binarize, dim=params[idx], fast=fast_input)
                scores[idx] = acc
            out = [[params[idx], scores[idx]] for idx in range(len(params))]
            np.savetxt(DIR_OUTPUT + os.sep + 'multiscore_%s.txt' % label, out)

    elif multiscore_save_random:
        params = list(range(1, 201))
        for rerun in range(5):
            # premake the random arrays for the run (shared by True/False binarize)
            rand_arrs = [0] * len(params)
            print("Generating random arrays for run %d" % rerun)
            for idx in range(len(params)):
                #proj = np.random.rand(DIM_VISIBLE, params[idx]) * 2 - 1
                proj = np.random.normal(0, scale=1.0, size=(DIM_VISIBLE, params[idx]))  # TODO also try normal?
                rand_arrs[idx] = proj
            # gather data
            for binarize in [True, False]:
                label = 'Kvary_rand_normal_binary%d_run%d' % (binarize, rerun)
                scores = [0] * len(params)
                for idx in range(len(params)):
                    print(idx, params[idx], binarize)
                    X_train, y_train = get_X_y_dataset(TRAINING, DIM_VISIBLE, binarize=binarize)
                    X_test, y_test = get_X_y_dataset(TESTING, DIM_VISIBLE, binarize=binarize)
                    fast_input = (X_train, y_train, X_test, y_test)
                    confusion_matrix, acc = classifier_on_proj(TRAINING, TESTING, DIM_VISIBLE, proj=rand_arrs[idx],
                                                               binarize=binarize, dim=params[idx], fast=fast_input)
                    scores[idx] = acc
                out = [[params[idx], scores[idx]] for idx in range(len(params))]
                np.savetxt(DIR_OUTPUT + os.sep + 'multiscore_%s.txt' % label, out)

    else:
        DATASET = TESTING  # TESTING or TRAINING
        npzpath_default = DEFAULT_HOPFIELD  # DEFAULT_HOPFIELD
        # set npzpath to None to build rbm
        if manual_onehot:
            npzpath_mod = '_onehot'  # _onehot' or '_onehotBlock'
        else:
            npzpath_mod = ''
        npzpath = DIR_MODELS + os.sep + 'saved' + os.sep + 'hopfield_mnist_%d%s.npz' % (manual_hidden, npzpath_mod)

        if npzpath is None:
            rbm_hopfield = build_rbm_hopfield(subpatterns=True, visible_field=False, save=False,
                                              k_pattern=int(manual_hidden/10), onehot_classify=manual_onehot)
        else:
            rbm_hopfield = load_rbm_hopfield(npzpath=npzpath)

        # decision here: classify on the raw data ELSE use features from the rbm
        if dont_use_rbm:
            N = rbm_hopfield.dim_visible
            P = rbm_hopfield.dim_hidden
            proj = None
            if use_random_proj:
                print('Using random projection...')
                proj = np.random.rand(N,P)*2 - 1
                print(proj.shape)
            confusion_matrix, acc = classifier_on_proj(TRAINING, TESTING, N, binarize=False, dim=P, proj=proj)
            plot_confusion_matrix(confusion_matrix)
            confusion_matrix, acc = classifier_on_proj(TRAINING, TESTING, N, binarize=True, dim=P, proj=proj)
            plot_confusion_matrix(confusion_matrix)

        else:
            if append_classifier_layer:
                confusion_matrix, acc = classifier_on_rbm_features(rbm_hopfield, TRAINING, TESTING, use_hidden=True, binarize=True, onehot=manual_onehot)
                plot_confusion_matrix(confusion_matrix)
                """
                confusion_matrix_vis_raw, acc = classifier_on_rbm_features(rbm_hopfield, TRAINING, TESTING, use_hidden=False, binarize=False)
                plot_confusion_matrix(confusion_matrix_vis_raw)
                confusion_matrix_vis_binary, acc = classifier_on_rbm_features(rbm_hopfield, TRAINING, TESTING, use_hidden=False, binarize=True)
                plot_confusion_matrix(confusion_matrix_vis_binary)
                """

            else:  # manual RBM classification
                predictions = len(DATASET) * [0]
                true_labels = [str(pair[1]) for pair in DATASET]
                MNIST_output_to_label = setup_MNIST_classification(rbm_hopfield)

                confusion_matrix = np.zeros((10, 11), dtype=int)  # last column is "unclassified"
                matches = [False for _ in DATASET]
                for idx, pair in enumerate(DATASET):
                    elem_arr, elem_label = pair
                    preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
                    predictions[idx] = classify_MNIST(rbm_hopfield, preprocessed_input, idx, MNIST_output_to_label, onehot=manual_onehot)
                    if true_labels[idx] == predictions[idx]:
                        matches[idx] = True
                        if len(predictions[idx]) == 1:
                            confusion_matrix[elem_label, int(predictions[idx])] += 1
                    else:
                        if predictions[idx] is None:
                            prediction_label = -1
                        else:
                            prediction_label = int(predictions[idx])

                        confusion_matrix[elem_label, prediction_label] += 1
                        if plot_wrong:
                            plt.imshow(preprocessed_input.reshape((28, 28)))
                            plt.savefig(DIR_OUTPUT + os.sep + 'wrong_idx%s_true%s_%s.png'
                                        % (idx, true_labels[idx], predictions[idx]))

                acc = float(matches.count(True) / len(matches))
                cm_title = "Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), acc)
                print(cm_title)
                plot_confusion_matrix(confusion_matrix, title=cm_title)
