import matplotlib.pyplot as plt
import numpy as np
import torch
from data_process import data_mnist, data_synthetic_dual
from settings import DIR_DATA, DIR_MODELS, CPU_THREADS, DATA_CHOICE, SYNTHETIC_DIM


assert DATA_CHOICE in ['synthetic', 'mnist']
if DATA_CHOICE == 'mnist':
    TRAINING, TESTING = data_mnist()
else:
    assert DATA_CHOICE == 'synthetic'
    TRAINING, TESTING = data_synthetic_dual()


class RBM:

    def __init__(self, dim_visible, dim_hidden, type_hidden, name):
        self.dim_visible = dim_visible
        self.dim_hidden = dim_hidden
        self.type_hidden = type_hidden
        self.name = name
        self.weights = None

    def set_weights(self, weights=None):
        self.weights = weights

    def train_rbm(self, data=TRAINING, initialization=None):
        # TODO update weights
        # TODO save if slow?
        weights_trained = None
        self.weights = weights_trained
        return


    def save_rbm_trained(self):
        # TODO
        return fpath


    def load_rbm_trained(self):
        # TODO
        return rbm_trained


def build_rbm_hopfield(data=TRAINING):
    # TODO
    # Step 1: convert data into patterns (using a prescribed rule)
    # Step 2: specify weights using the patterns
    # Step 3: conform to pytorch class
    if DATA_CHOICE == 'mnist':
        print(data[0][0].shape)
        dim_visible = data[0][0].shape
        assert 1 == 0
        dim_hidden = 10
    else:
        assert DATA_CHOICE == 'synthetic'
        dim_visible = SYNTHETIC_DIM
        assert 1 == 0
        dim_hidden = 2

    rbm_hopfield = RBM(dim_visible, dim_hidden, type_hidden, name)
    return rbm_hopfield


if __name__ == '__main__':
    # Step 1: pick data set (MNIST or synthetic)


    # Step 2: build 4 RBM variants
    #   A: hopfield RBM
    #   B: hopfield RBM then some training
    #   C: vanilla RBM binary-gaussian
    #   D: vanilla RBM binary-binary

    # data structure: list of ~60,000 2-tuples: "image" and integer label
    data_loader = torch.utils.data.DataLoader(mnist_training, batch_size=BATCH_SIZE, shuffle=True, num_workers=CPU_THREADS)

    print('TODO')
