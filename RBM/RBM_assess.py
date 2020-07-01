import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from data_process import image_data_collapse, torch_image_to_numpy, binarize_image_data
from RBM_train import RBM, TRAINING, TESTING, build_rbm_hopfield


# TODO
#   Question: how to perform classification with binary-binary or binary-gaussian RBM? append module to end?


def setup_MNIST_classification():
    label_dict = {}
    for idx in range(10):
        key_tuple = tuple([0  if a != idx else 1 for a in range(10)])
        value_str = '%d' % idx
        label_dict[key_tuple] = value_str
        neg_key_tuple = tuple([0 if a != idx else -1 for a in range(10)])
        neg_value_str = 'anti-%d' % idx
        label_dict[neg_key_tuple] = neg_value_str
    return label_dict


MNIST_output_to_label = setup_MNIST_classification()


def classify_MNIST(rbm, visual_init):
    MAX_STEPS_CLASSIFY = 20
    visual_step = visual_init

    def conv_class_vector_to_label(output_as_ints):
        if tuple(output_as_ints) in MNIST_output_to_label.keys():
            return True, MNIST_output_to_label[tuple(output_as_ints)]
        else:
            return False, output_as_ints

    for idx in range(MAX_STEPS_CLASSIFY):
        visual_step, hidden_step, output_step = rbm.RBM_step(visual_step)
        output_truncated = rbm.truncate_output(output_step)
        classified, classification = conv_class_vector_to_label(output_truncated)
        if classified:
            #print("classified at step:", idx)
            break
        #print("visual at step", idx, "is", visual_step)
        #print("hidden at step", idx, "is", hidden_step)
        #print("output at step", idx, "is", output_step)
        #print("output_truncated at step", idx, "is", output_truncated)
    if idx == MAX_STEPS_CLASSIFY - 1:
        print("******************** Edge case unclassified")
        print("\t classification:", classification)

    return classification


def plot_confusion_matrix(confusion_matrix):
    # Ref: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    import seaborn as sn
    import pandas as pd

    ylabels = [str(i) for i in range(10)]
    xlabels = ylabels + ['Other']
    df_cm = pd.DataFrame(confusion_matrix, index=ylabels, columns=xlabels)

    plt.figure(figsize=(11,7))
    sn.set(font_scale=1.2)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='d')  # font size
    plt.gca().set(xlabel='Predicted', ylabel='True label')
    plt.show()
    return


if __name__ == '__main__':
    print("TODO")
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

    # ROUGH WORK fir hopfield RBM only
    DATASET = TESTING #TESTING
    confusion_matrix = np.zeros((10, 11), dtype=int)  # last column is "unclassified"
    rbm_hopfield = build_rbm_hopfield()
    matches = [False for _ in DATASET]
    predictions = len(DATASET) * [0]
    true_labels = [str(pair[1]) for pair in DATASET]
    for idx, pair in enumerate(DATASET):
        elem_arr, elem_label = pair
        preprocessed_input = binarize_image_data(image_data_collapse(elem_arr))
        predictions[idx] = classify_MNIST(rbm_hopfield, preprocessed_input)
        #print(true_labels[idx], predictions[idx])
        if true_labels[idx] == predictions[idx]:
            matches[idx] = True
        # update confusion matrix
        if len(predictions[idx]) == 1:
            confusion_matrix[elem_label, int(predictions[idx])] += 1
        else:
            confusion_matrix[elem_label, -1] += 1
    print("Successful test cases: %d/%d (%.3f)" % (matches.count(True), len(matches), float(matches.count(True) / len(matches))))
    plot_confusion_matrix(confusion_matrix)
