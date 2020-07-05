import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import qr
import torch
from data_process import data_mnist, data_synthetic_dual, hopfield_mnist_patterns, data_dict_mnist, binarize_image_data, image_data_collapse
from settings import DIR_DATA, DIR_MODELS, CPU_THREADS, DATA_CHOICE, SYNTHETIC_DIM, MNIST_BINARIZATION_CUTOFF, BETA, PATTERN_THRESHOLD


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
        self.internal_weights = None
        self.output_weights = None
        self.visible_field = np.zeros(dim_visible)
        self.hidden_field = np.zeros(dim_hidden)

    def set_internal_weights(self, weights):
        assert weights.shape == (self.dim_visible, self.dim_hidden)
        self.internal_weights = weights

    def set_output_weights(self, weights):
        assert weights.shape[1] == self.dim_hidden
        self.output_weights = weights

    def set_visible_field(self, visible_field):
        assert len(visible_field) == self.dim_visible
        self.visible_field = visible_field

    def train_rbm(self, data=TRAINING, initialization=None):
        # TODO update weights
        # TODO save if slow?
        internal_weights_trained = None
        output_weights_trained = None
        self.internal_weights = internal_weights_trained
        self.output_weights = output_weights_trained
        return

    def RBM_step(self, visible_init, beta=BETA):

        # TODO confirm/test distribution choices
        # hidden -> visible:  probabilistic  (from Barra 2012 -- Eq. 3)
        #     input = np.dot(weights[i, :], hidden[:])
        #     p(v_i=1) = 1/(1 + exp(-2 * beta * input)
        #
        # visible -> hidden:  probabilistic  (from Barra 2012 -- Eq. 2)
        #   h_mu ~ N(<h_mu>, var=1/beta)
        #   <h_mu> = np.dot(weights[:, mu], visible[:])

        def update_visible(state_hidden):
            input_vector = np.dot(self.internal_weights, state_hidden) + self.visible_field
            visible_probability_one = 1/(1 + np.exp(- 2 * beta * input_vector))
            visible_step = np.random.binomial(1, visible_probability_one, self.dim_visible)
            return visible_step * 2 - 1  # +1, -1 convention

        def update_hidden(state_visible):
            if self.type_hidden == 'gaussian':
                means = np.dot(self.internal_weights.T, state_visible) + self.hidden_field
                std_dev = np.sqrt(1/beta)
                hidden_step = np.random.normal(means, std_dev, self.dim_hidden)
            else:
                assert self.type_hidden == 'boolean'
                # TODO investigate
                assert 1 == 2
                hidden_step = None
            return hidden_step

        def update_output(state_hidden):
            return np.dot(self.output_weights, state_hidden)

        hidden_step = update_hidden(visible_init)
        visible_step = update_visible(hidden_step)
        output_step = update_output(hidden_step)
        return visible_step, hidden_step, output_step

    def truncate_output(self, state_output, threshold=0.8):
        output_vector = np.zeros(len(state_output), dtype=int)
        flag_any_large_patterns = np.any(np.where(np.abs(state_output) > threshold, True, False))
        if flag_any_large_patterns:
            above_T_idx = state_output > threshold
            below_negT_idx = state_output < -threshold
            output_vector[above_T_idx] = 1
            output_vector[below_negT_idx] = -1
        return output_vector

    def truncate_output_max(self, state_output, threshold=0.5):
        output_vector = np.zeros(len(state_output), dtype=int)
        max_arg = np.argmax(np.abs(state_output))
        max_val = state_output[max_arg]
        if np.abs(max_val) > threshold:
            output_vector[max_arg] = 1 * np.sign(max_val)
        return output_vector

    def save_rbm_trained(self):
        fpath = DIR_MODELS + os.sep + 'hopfield_weights.npy'
        with open(fpath, 'wb') as f:
            np.save(f, self.internal_weights)
        return fpath

    def load_rbm_trained(self, fpath):
        with open(fpath, 'rb') as f:
            rbm_internal_weights = np.load(fpath)
        return rbm_internal_weights


def linalg_hopfield_patterns(data_dict, category_counts):
    xi, xi_collapsed = hopfield_mnist_patterns(data_dict, category_counts, pattern_threshold=PATTERN_THRESHOLD)
    Q, R = qr(xi_collapsed, mode='economic')
    print("Q.shape", Q.shape)
    print("R.shape", R.shape)
    return xi, xi_collapsed, Q, R


def build_rbm_hopfield(data=TRAINING, visible_field=False):
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
    # build internal weights
    data_dict, category_counts = data_dict_mnist(TRAINING)
    xi, xi_collapsed, Q, R = linalg_hopfield_patterns(data_dict, category_counts)
    """
    metas = np.zeros((28,28))
    for idx in range(10):
        metas += xi[:,:,idx]
    plt.imshow(metas)
    plt.colorbar()
    plt.show()"""
    rbm_hopfield.set_internal_weights(Q)
    if visible_field:
        pixel_means = np.zeros(dim_visible)
        for idx, pair in enumerate(data):
            elem_arr, elem_label = pair
            #preprocessed_input = image_data_collapse(elem_arr)
            preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
            pixel_means += preprocessed_input
        pixel_means = pixel_means / len(data)
        # convert anything greater than 0 to a 1
        pixel_means[pixel_means > 0] = 0
        pixel_means[pixel_means < 0] = -1
        plt.imshow(pixel_means.reshape((28,28)))
        plt.colorbar()
        plt.show()
        rbm_hopfield.set_visible_field(pixel_means)

    # build output/projection weights
    proj_remainder = np.dot( np.linalg.inv(np.dot(R.T, R)) , R.T)
    rbm_hopfield.set_output_weights(proj_remainder)
    # save weights
    rbm_hopfield.save_rbm_trained()
    return rbm_hopfield


if __name__ == '__main__':
    # Step 1: pick data set (MNIST or synthetic)

    # Step 2: build 4 RBM variants
    #   A: hopfield RBM
    #   B: hopfield RBM then some training
    #   C: vanilla RBM binary-gaussian
    #   D: vanilla RBM binary-binary

    build_rbm_hopfield(data=TRAINING, visible_field=False)
