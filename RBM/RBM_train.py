import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
import torch
from data_process import data_mnist, data_synthetic_dual, hopfield_mnist_patterns, data_dict_mnist
from settings import DIR_DATA, DIR_MODELS, CPU_THREADS, DATA_CHOICE, SYNTHETIC_DIM, MNIST_BINARIZATION_CUTOFF


assert DATA_CHOICE in ['synthetic', 'mnist']
if DATA_CHOICE == 'mnist':
    TRAINING, TESTING = data_mnist()
else:
    assert DATA_CHOICE == 'synthetic'
    TRAINING, TESTING = data_synthetic_dual()


class RBM:

    def __init__(self, dim_visible, dim_hidden, type_hidden, name):
        assert type_hidden in ['boolean', 'gaussian']
        self.dim_visible = dim_visible
        self.dim_hidden = dim_hidden
        self.type_hidden = type_hidden
        self.name = name
        self.weights = None

    def set_weights(self, weights):
        assert weights.shape == (self.dim_visible, self.dim_hidden)
        self.weights = weights

    def train_rbm(self, data=TRAINING, initialization=None):
        # TODO update weights
        # TODO save if slow?
        weights_trained = None
        self.weights = weights_trained
        return

    def RBM_step(self, visible_init, beta=2.0):

        # TODO confirm/test distribution choices
        # hidden -> visible:  probabilistic  (from Barra 2012 -- Eq. 3)
        #     input = np.dot(weights[i, :], hidden[:])
        #     p(v_i=1) = 1/(1 + exp(-2 * beta * input)
        #
        # visible -> hidden:  probabilistic  (from Barra 2012 -- Eq. 2)
        #   h_mu ~ N(<h_mu>, var=1/beta)
        #   <h_mu> = np.dot(weights[:, mu], visible[:])

        def update_visible(state_hidden):
            input_vector = np.dot(self.weights, state_hidden)
            visible_probability_one = 1/(1 + np.exp(- 2 * beta * input_vector))
            visible_step = np.random.binomial(1, visible_probability_one, self.dim_visible)
            return visible_step * 2 - 1  # +1, -1 convention

        def update_hidden(state_visible):
            if self.type_hidden == 'gaussian':
                means = np.dot(self.weights.T, state_visible)
                std_dev = np.sqrt(1/beta)
                hidden_step = np.random.normal(means, std_dev, self.dim_hidden)
            else:
                assert self.type_hidden == 'boolean'
                # TODO investigate
                assert 1 == 2
                hidden_step = None
            return hidden_step

        def update_output(state_hidden, threshold=0.5):
            # for all hidden elements, a:
            #  if a > T set output to 1      (where T is some threshold between 0, 1)
            #  if a < -T set output to -1
            #  else set to 0
            flag_any_large_patterns = np.any(np.where(np.abs(state_hidden) > threshold, True, False))
            if flag_any_large_patterns:
                output_vector = np.where(state_hidden < -threshold, -1, state_hidden)
                output_vector = np.where(output_vector > threshold, 1, 0)
            else:
                output_vector = np.zeros(len(state_hidden), dtype=int)
            return output_vector

        hidden_step = update_hidden(visible_init)
        visible_step = update_visible(hidden_step)
        output_step = update_output(hidden_step)
        return visible_step, hidden_step, output_step

    def save_rbm_trained(self):
        # TODO
        fpath = None
        return fpath


    def load_rbm_trained(self):
        # TODO
        rbm_traine = None
        return rbm_trained


def build_rbm_hopfield(data=TRAINING):
    # TODO
    # Step 1: convert data into patterns (using a prescribed rule)
    # Step 2: specify weights using the patterns
    # Step 3: conform to pytorch class
    if DATA_CHOICE == 'mnist':
        print(data[0][0].shape)
        dim_visible = data[0][0].shape[0] * data[0][0].shape[1]
        print("dim_visible", dim_visible)
        dim_hidden = 10
    else:
        assert DATA_CHOICE == 'synthetic'
        dim_visible = SYNTHETIC_DIM
        assert 1 == 0  # TODO not supported
        dim_hidden = 2
    # prep class
    rbm_name = 'hopfield_%s' % DATA_CHOICE
    rbm_hopfield = RBM(dim_visible, dim_hidden, 'gaussian', rbm_name)
    # build weights
    data_dict, category_counts = data_dict_mnist(TRAINING)
    _, xi_collapsed = hopfield_mnist_patterns(data_dict, category_counts, pattern_threshold=MNIST_BINARIZATION_CUTOFF)
    Q, R = qr(xi_collapsed, mode='economic')
    print("Q.shape", Q.shape)
    print("R.shape", R.shape)
    rbm_hopfield.set_weights(Q)
    return rbm_hopfield


if __name__ == '__main__':
    # Step 1: pick data set (MNIST or synthetic)

    # Step 2: build 4 RBM variants
    #   A: hopfield RBM
    #   B: hopfield RBM then some training
    #   C: vanilla RBM binary-gaussian
    #   D: vanilla RBM binary-binary

    print('TODO')
