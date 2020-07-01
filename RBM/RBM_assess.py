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
    NUM_STEPS_CLASSIFY = 4
    visual_step = visual_init

    def conv_class_vector_to_label(output_arr):
        if tuple(output_arr) in MNIST_output_to_label.keys():
            return MNIST_output_to_label[tuple(output_arr)]
        else:
            return output_arr

    for idx in range(NUM_STEPS_CLASSIFY):
        visual_step, hidden_step, output_step = rbm.RBM_step(visual_step)
        print("visual at step", idx, "is", visual_step)
        print("hidden at step", idx, "is", hidden_step)
        print("output at step", idx, "is", output_step)

    classification = conv_class_vector_to_label(output_step)
    return classification


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

    # ROUGH WORK
    rbm_hopfield = build_rbm_hopfield()
    for elem_arr, elem_label in TESTING:
        preprocessed_input = binarize_image_data(image_data_collapse(elem_arr))
        prediction = classify_MNIST(rbm_hopfield, preprocessed_input)
        print(elem_label, prediction, '\n')
