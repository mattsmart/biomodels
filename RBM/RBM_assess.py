import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
from data_process import image_data_collapse, torch_image_to_numpy, binarize_image_data
from RBM_train import RBM, TRAINING, TESTING, build_rbm_hopfield
from settings import MNIST_BINARIZATION_CUTOFF, DIR_OUTPUT, VISIBLE_FIELD, HRBM_LOGREG_STEPS, HRBM_MANUAL_MAXSTEPS


# TODO
#   Question: how to perform classification with binary-binary or binary-gaussian RBM? append module to end?


def setup_MNIST_classification():
    label_dict = {}
    for idx in range(10):
        key_tuple = tuple([0 if a != idx else 1 for a in range(10)])
        value_str = '%d' % idx
        label_dict[key_tuple] = value_str
        neg_key_tuple = tuple([0 if a != idx else -1 for a in range(10)])
        neg_value_str = 'anti-%d' % idx
        label_dict[neg_key_tuple] = neg_value_str
    return label_dict


MNIST_output_to_label = setup_MNIST_classification()


def classify_MNIST(rbm, visual_init, dataset_idx, max_mode=False):
    visual_step = visual_init

    def conv_class_vector_to_label(output_as_ints):
        if tuple(output_as_ints) in MNIST_output_to_label.keys():
            return True, MNIST_output_to_label[tuple(output_as_ints)]
        else:
            return False, output_as_ints

    if max_mode:
        output_converter = rbm.truncate_output_max
    else:
        output_converter = rbm.truncate_output

    for idx in range(HRBM_MANUAL_MAXSTEPS):
        visual_step, hidden_step, output_step = rbm.RBM_step(visual_step)
        output_truncated = output_converter(output_step)
        classified, classification = conv_class_vector_to_label(output_truncated)
        #if len(classification) > 1:
        #    print(classification, "\n\t", hidden_step, "\n\t", output_step, "\n\t", output_truncated)
        if classified:
            #print("classified at step:", idx, classification, dataset_idx)
            break
        #print("visual at step", idx, "is", visual_step)
        #print("hidden at step", idx, "is", hidden_step)
        #print("output at step", idx, "is", output_step)
        #print("output_truncated at step", idx, "is", output_truncated)
    if idx == HRBM_MANUAL_MAXSTEPS - 1:
        print("******************** Edge case unclassified: (%d) step == MAX_STEPS_CLASSIFY - 1" % dataset_idx)
        print("\t classification:", classification)

    return classification


def rbm_features_MNIST(rbm, visual_init, steps=HRBM_LOGREG_STEPS):
    visual_step = visual_init
    for idx in range(steps):
        visual_step, hidden_step, _ = rbm.RBM_step(visual_step)
    return hidden_step


def logistic_regression_on_hidden(rbm, dataset_train, dataset_test):
    print("logistic_regression_on_hidden; Step 1: get features for training")
    X_train = np.zeros((len(dataset_train), rbm.dim_hidden))
    y_train = np.zeros(len(dataset_train), dtype=int)
    for idx, pair in enumerate(dataset_train):
        elem_arr, elem_label = pair
        preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
        X_train[idx, :] = rbm_features_MNIST(rbm, preprocessed_input)
        y_train[idx] = elem_label
    print("logistic_regression_on_hidden; Step 2: logistic regression on classes2")
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1e5, multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
    clf.fit(X_train, y_train)  # fit data
    print("logistic_regression_on_hidden; Step 3: get features for testing")
    X_test = np.zeros((len(dataset_test), rbm.dim_hidden))
    y_test = np.zeros(len(dataset_test), dtype=int)
    for idx, pair in enumerate(dataset_test):
        elem_arr, elem_label = pair
        preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
        X_test[idx, :] = rbm_features_MNIST(rbm, preprocessed_input)
        y_test[idx] = elem_label
    print("logistic_regression_on_hidden; Step 4: classification metrics and confusion matrix")
    """
    sparsity1 = np.mean(clf1.coef_ == 0) * 100  # percentage of nonzero weights """
    predictions = clf.predict(X_test).astype(int)
    confusion_matrix = np.zeros((10, 10), dtype=int)
    matches = [False for _ in dataset_test]
    for idx, pair in enumerate(dataset_test):
        if y_test[idx] == predictions[idx]:
            matches[idx] = True
        confusion_matrix[y_test[idx], predictions[idx]] += 1
    print("Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), float(matches.count(True) / len(matches))))
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix):
    # Ref: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    import seaborn as sn
    import pandas as pd

    ylabels = [str(i) for i in range(10)]
    if confusion_matrix.shape[1] == 11:
        xlabels = ylabels + ['Other']
    else:
        xlabels = ylabels
    df_cm = pd.DataFrame(confusion_matrix, index=ylabels, columns=xlabels)

    plt.figure(figsize=(11,7))
    sn.set(font_scale=1.2)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='d')  # font size
    plt.gca().set(xlabel='Predicted', ylabel='True label')
    plt.show()
    return


if __name__ == '__main__':
    plot_wrong = False

    # load variants
    """
    RBM_A = load_rbm_trained()
    RBM_B = load_rbm_trained()
    RBM_C = load_rbm_trained()
    RBM_D = load_rbm_trained()
    """
    # score each variant
    # TODO
    # plot
    # TODO
    # ROUGH WORK for hopfield RBM only
    DATASET = TESTING  # TESTING or TRAINING
    rbm_hopfield = build_rbm_hopfield(visible_field=VISIBLE_FIELD)

    logistic_regression_approach = True
    if logistic_regression_approach:
        confusion_matrix = logistic_regression_on_hidden(rbm_hopfield, TRAINING, TESTING)

    else:
        confusion_matrix = np.zeros((10, 11), dtype=int)  # last column is "unclassified"
        matches = [False for _ in DATASET]
        predictions = len(DATASET) * [0]
        true_labels = [str(pair[1]) for pair in DATASET]
        for idx, pair in enumerate(DATASET):
            elem_arr, elem_label = pair
            preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
            predictions[idx] = classify_MNIST(rbm_hopfield, preprocessed_input, idx)
            if true_labels[idx] == predictions[idx]:
                matches[idx] = True
            # update confusion matrix
            if len(predictions[idx]) == 1:
                confusion_matrix[elem_label, int(predictions[idx])] += 1
            else:
                confusion_matrix[elem_label, -1] += 1
                if plot_wrong:
                    plt.imshow(preprocessed_input.reshape((28,28)))
                    plt.savefig(DIR_OUTPUT + os.sep + 'wrong_idx%s_true%s_%s.png' % (idx, true_labels[idx], predictions[idx]))
        print("Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), float(matches.count(True) / len(matches))))
    plot_confusion_matrix(confusion_matrix)
