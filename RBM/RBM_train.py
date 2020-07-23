import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import qr

from data_process import data_mnist, data_synthetic_dual, hopfield_mnist_patterns, data_dict_mnist, binarize_image_data, image_data_collapse, data_dict_mnist_detailed
from plotting import image_fancy
from settings import DIR_DATA, DIR_OUTPUT, DIR_MODELS, CPU_THREADS, DATA_CHOICE, MNIST_BINARIZATION_CUTOFF, BETA, PATTERN_THRESHOLD, DEFAULT_HOPFIELD, K_PATTERN_DIV, DIR_CLASSIFY


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
        self.pattern_labels = None
        self.pattern_idx_to_labels = None
        self.xi_image = None

    def set_internal_weights(self, weights):
        assert weights.shape == (self.dim_visible, self.dim_hidden)
        self.internal_weights = weights

    def set_output_weights(self, weights):
        assert weights.shape[1] == self.dim_hidden
        self.output_weights = weights

    def set_visible_field(self, visible_field):
        assert len(visible_field) == self.dim_visible
        self.visible_field = visible_field

    def set_pattern_labels(self, pattern_idx_to_labels):
        self.pattern_idx_to_labels = pattern_idx_to_labels
        # list of pattern string IDs like '0_0', '0_1' etc.
        self.pattern_labels = [pattern_idx_to_labels[idx] for idx in range(self.dim_hidden)]

    def train_rbm(self, data=TRAINING, initialization=None):
        # TODO update weights
        # TODO save if slow?
        internal_weights_trained = None
        output_weights_trained = None
        self.internal_weights = internal_weights_trained
        self.output_weights = output_weights_trained
        return

    def update_visible(self, state_hidden, beta=BETA):
        input_vector = np.dot(self.internal_weights, state_hidden) + self.visible_field
        visible_probability_one = 1/(1 + np.exp(- 2 * beta * input_vector))
        visible_step = np.random.binomial(1, visible_probability_one, self.dim_visible)
        return visible_step * 2 - 1  # +1, -1 convention

    def update_hidden(self, state_visible, beta=BETA):
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

    def update_output(self, state_hidden):
        return np.dot(self.output_weights, state_hidden)

    def RBM_step(self, visible_init, beta=BETA):

        # TODO confirm/test distribution choices
        # hidden -> visible:  probabilistic  (from Barra 2012 -- Eq. 3)
        #     input = np.dot(weights[i, :], hidden[:])
        #     p(v_i=1) = 1/(1 + exp(-2 * beta * input)
        #
        # visible -> hidden:  probabilistic  (from Barra 2012 -- Eq. 2)
        #   h_mu ~ N(<h_mu>, var=1/beta)
        #   <h_mu> = np.dot(weights[:, mu], visible[:])

        hidden_step = self.update_hidden(visible_init, beta=beta)
        visible_step = self.update_visible(hidden_step, beta=beta)
        output_step = self.update_output(hidden_step)
        return visible_step, hidden_step, output_step

    def truncate_output(self, state_output, threshold=0.8):
        output_vector = np.zeros(len(state_output), dtype=int)
        flag_any_large_patterns = np.any(np.where(np.abs(state_output) > threshold, True, False))
        if flag_any_large_patterns:
            above_T_idx = np.argwhere(state_output > threshold)
            below_negT_idx = np.argwhere(state_output < -threshold)
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

    def truncate_output_subpatterns(self, state_output, threshold=0.8):
        K = 6
        assert self.dim_hidden == K * 10
        output_vector = np.zeros(len(state_output), dtype=int)
        state_simplified = np.zeros(len(state_output), dtype=float)
        flag_any_large_patterns = True #np.any(np.where(np.abs(state_output) > threshold, True, False))
        if flag_any_large_patterns:
            for idx in range(10):
                state_simplified[idx*K] = np.sum(state_output[K*idx:K*(idx + 1)])
            above_T_idx = np.argwhere(state_simplified > threshold)
            below_negT_idx = np.argwhere(state_simplified < -threshold)
            output_vector[above_T_idx] = 1
            output_vector[below_negT_idx] = -1
        return output_vector

    def set_xi_image(self, xi_image):
        self.xi_image = xi_image
        return

    def save_rbm_trained(self):
        fpath = DIR_MODELS + os.sep + self.name
        np.savez(fpath, Q=self.internal_weights, proj_remainder=self.output_weights, pattern_labels=self.pattern_labels,
                 xi_image=self.xi_image)
        return fpath

    def load_rbm_trained(self, fpath):
        with open(fpath, 'rb') as f:
            rbm_internal_weights = np.load(fpath)
        return rbm_internal_weights

    def plot_visible(self, visible_state, title='def'):
        plt.imshow(visible_state.reshape(28,28))
        plt.colorbar()
        plt.title('Visible state (%s)' % title)
        plt.savefig(DIR_CLASSIFY + os.sep + 'visible_%s' % title)
        plt.close()
        return

def linalg_hopfield_patterns(data_dict, category_counts):
    xi, xi_collapsed, pattern_idx_to_labels = hopfield_mnist_patterns(data_dict, category_counts, pattern_threshold=PATTERN_THRESHOLD)
    Q, R = qr(xi_collapsed, mode='economic')
    print("Q.shape", Q.shape)
    print("R.shape", R.shape)
    return xi, xi_collapsed, pattern_idx_to_labels, Q, R


def build_rbm_hopfield(data=TRAINING, visible_field=False, subpatterns=False, name=DATA_CHOICE, fast=None, save=True):
    """
    fast is None or a 2-tuple of (data_dict, category_counts)
    """
    # Step 1: convert data into patterns (using a prescribed rule)
    # Step 2: specify weights using the patterns
    rbm_name = 'hopfield_%s' % name
    if fast is None:
        # build internal weights
        data_dict, category_counts = data_dict_mnist(data)
        if subpatterns:
            rbm_name += '_subpatterns'
            data_dict, category_counts = data_dict_mnist_detailed(data_dict, category_counts)
    else:
        data_dict = fast[0]
        category_counts = fast[1]

    xi, xi_collapsed, pattern_idx_to_labels, Q, R = linalg_hopfield_patterns(data_dict, category_counts)
    total_data = sum(category_counts.values())
    dim_img = list(data_dict.values())[0][:, :, 0].shape
    dim_visible = dim_img[0] * dim_img[1]
    dim_hidden = xi_collapsed.shape[-1]
    print("total_data", total_data)
    print("dim_hidden", dim_hidden)

    # prep class
    rbm_hopfield = RBM(dim_visible, dim_hidden, 'gaussian', rbm_name)
    """
    metas = np.zeros((28,28))
    for idx in range(10):
        metas += xi[:,:,idx]
    plt.imshow(metas)
    plt.colorbar()
    plt.show()"""
    rbm_hopfield.set_xi_image(xi)
    rbm_hopfield.set_internal_weights(Q)
    rbm_hopfield.set_pattern_labels(pattern_idx_to_labels)
    if visible_field:
        pixel_means = np.zeros(dim_visible)
        for idx, pair in enumerate(data):
            elem_arr, elem_label = pair
            #preprocessed_input = image_data_collapse(elem_arr)
            preprocessed_input = binarize_image_data(image_data_collapse(elem_arr), threshold=MNIST_BINARIZATION_CUTOFF)
            pixel_means += preprocessed_input
        pixel_means = pixel_means / total_data
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
    if save:
        rbm_hopfield.save_rbm_trained()
    return rbm_hopfield


def load_rbm_hopfield(npzpath=DEFAULT_HOPFIELD):
    # LOAD
    print("LOADING: %s" % npzpath)
    dataobj = np.load(npzpath)
    Q = dataobj['Q']
    proj_remainder = dataobj['proj_remainder']
    pattern_labels = dataobj['pattern_labels']
    xi_image = dataobj['xi_image']
    # MINOR PROCESSING
    dim_visible = Q.shape[0]
    dim_hidden = Q.shape[1]
    pattern_idx_to_labels = {idx: str(pattern_labels[idx]) for idx in range(dim_hidden)}
    # BUILD
    rbm_name = 'hopfield_loaded_%s' % DATA_CHOICE
    rbm_hopfield = RBM(dim_visible, dim_hidden, 'gaussian', rbm_name)
    rbm_hopfield.set_internal_weights(Q)
    rbm_hopfield.set_pattern_labels(pattern_idx_to_labels)
    rbm_hopfield.set_output_weights(proj_remainder)
    rbm_hopfield.set_xi_image(xi_image)
    return rbm_hopfield


if __name__ == '__main__':
    # Step 1: pick data set (MNIST or synthetic)

    # Step 2: build 4 RBM variants
    #   A: hopfield RBM
    #   B: hopfield RBM then some training
    #   C: vanilla RBM binary-gaussian
    #   D: vanilla RBM binary-binary
    build_instead_of_load = False

    # trial building
    if build_instead_of_load:
        k_patterns = K_PATTERN_DIV
        rbm = build_rbm_hopfield(data=TRAINING, visible_field=False, subpatterns=True)

    # trial loading
    else:
        k_patterns = 1
        fname = 'hopfield_mnist_%d0.npz' % k_patterns
        rbm = load_rbm_hopfield(npzpath=DIR_MODELS + os.sep + 'saved' + os.sep + fname)

    xi_images = rbm.xi_image
    print(rbm.pattern_labels)
    plt.figure(figsize=(2 + k_patterns, 10))
    fig, ax_arr = plt.subplots(k_patterns, 10)
    for k in range(k_patterns):
        for i in range(10):
            print(i, k, k_patterns*i + k, rbm.pattern_labels[k_patterns*i + k])
            if k_patterns > 1:
                axloc = ax_arr[k, i]
            else:
                axloc = ax_arr[i]
            axloc.imshow(xi_images[:, :, k_patterns*i + k], interpolation='none')
            axloc.set_xticklabels([])
            axloc.set_yticklabels([])
            # image_fancy(xi_images[:, :, k_patterns*i + k], ax=axloc)
    plt.suptitle('Heirarchical patterns example (K=%d)' % k_patterns)
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'subpatterns_%d.pdf' % k_patterns)
