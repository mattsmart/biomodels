import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from RBM_train import RBM, TRAINING, TESTING


# TODO
#   Question: how to perform classification with binary-binary or binary-gaussian RBM? append module to end?


def RBM_step(visual, weights, hidden_type='gaussian'):
    # TODO foo bar each case
    if hidden_type == 'gaussian':
        hidden_step = foo(visual, beta=BETA)
        visual_step = bar(hidden_step, beta=BETA)
    else:
        assert hidden_type ==  'boolean'
        hidden_step = foo(visual, beta=BETA)
        visual_step = bar(hidden_step, beta=BETA)
    return visual_step, hidden_step


def classify(rbm, visual_init):
    # TODO
    NUM_STEPS_CLASSIFY = 4
    visual_step = visual_init
    for idx in range(NUM_STEPS_CLASSIFY):
        visual_step, hidden_step = RBM_step(visual_init, rbm_weights, hidden_type=rbm_type)


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
    for elem_arr, elem_label in testing:
        prediction = predict(rbm_hopfield, )
        print(elem_label, prediction)