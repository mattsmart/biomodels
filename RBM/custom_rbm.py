# sample code from git repo: https://github.com/GabrielBianconi/pytorch-rbm

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from settings import BETA, GAUSSIAN_STDEV, DIR_MODELS, DIR_OUTPUT


class RBM_custom():

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, weight_decay=1e-4):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

    def sample_hidden(self, visible_state):
        hidden_activations = torch.matmul(visible_state, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        hidden_sampled = torch.bernoulli(hidden_probabilities)  # ********************************************************************* NEW
        hidden_sampled_phys = -1 + hidden_sampled * 2
        return hidden_sampled_phys

    def sample_visible(self, hidden_state):
        visible_activations = torch.matmul(hidden_state, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        visible_sampled = torch.bernoulli(visible_probabilities)  # ********************************************************************* NEW
        visible_sampled_phys = -1 + visible_sampled * 2
        return visible_sampled_phys

    def contrastive_divergence(self, input_data):
        # Positive phase (WIKI: Steps 1, 2)
        positive_hidden_sampled = self.sample_hidden(input_data)
        positive_gradient = torch.matmul(input_data.t(), positive_hidden_sampled)    # WIKI 2: v dot h

        # Negative phase (WIKI: Steps 3, 4)
        hidden_sampled = positive_hidden_sampled
        for step in range(self.k):
            visible_sampled = self.sample_visible(hidden_sampled)
            hidden_sampled = self.sample_hidden(visible_sampled)
        negative_visible_sampled = visible_sampled
        negative_hidden_sampled = hidden_sampled
        negative_gradient = torch.matmul(negative_visible_sampled.t(), negative_hidden_sampled)  # WIKI 4: v' dot h'

        # Update parameters (WIKI: Steps 5, 6) -- note no momentum OR batch size affected learning rate
        # batch_size = input_data.size(0)
        self.weights += (positive_gradient - negative_gradient) * self.learning_rate
        self.visible_bias += torch.sum(input_data - negative_visible_sampled, dim=0) * self.learning_rate
        self.hidden_bias += torch.sum(positive_hidden_sampled - negative_hidden_sampled, dim=0) * self.learning_rate

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_sampled)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        return random_probabilities


class RBM_gaussian_custom():

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, weight_decay=1e-4, load_init_weights=False, use_fields=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if load_init_weights:
            npzpath = DIR_MODELS + os.sep + 'saved' + os.sep + 'hopfield_mnist_10.npz'
            print("Loading weights from %s" % npzpath)
            arr = self.load_rbm_trained(npzpath)
            self.weights = torch.from_numpy(arr).float()
            self.visible_bias = torch.zeros(num_visible).float()
        else:
            use_normal = False
            print("Setting random weights: use_normal=%s" % use_normal)
            if use_normal:
                self.weights = 0.1 * torch.randn(num_visible, num_hidden).float()
            else:
                scale = np.sqrt(6) / np.sqrt(num_visible + num_hidden)  # gaussian-binary ref cites bengio and glorot [40] on this choice
                self.weights = 2 * scale * torch.rand(num_visible, num_hidden).float() - scale
                print(scale, torch.min(self.weights), torch.max(self.weights))
            self.visible_bias = 0.5 * torch.ones(num_visible).float()
        self.hidden_bias = torch.zeros(num_hidden).float()
        self.use_fields = use_fields

        if not self.use_fields:
            self.visible_bias = torch.zeros(num_visible).float()
            self.hidden_bias = torch.zeros(num_hidden).float()

    def sample_hidden(self, visible_state, stdev=GAUSSIAN_STDEV):
        hidden_activations = torch.matmul(visible_state, self.weights) + self.hidden_bias
        if stdev > 0:
            hidden_sampled = torch.normal(hidden_activations, stdev)  # ***************** NEW
        else:
            hidden_sampled = hidden_activations
        return hidden_sampled

    def sample_hidden_forcebinary(self, visible_state):
        visible_activations = torch.matmul(visible_state, self.weights)  # + self.hidden_bias
        hidden_probabilities = self._sigmoid(2 * BETA * visible_activations)  # self._sigmoid(visible_activations)
        hidden_sampled = torch.bernoulli(hidden_probabilities)  # **************************** NEW
        return hidden_sampled

    def sample_visible(self, hidden_state):
        visible_activations = torch.matmul(hidden_state, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(2 * BETA * visible_activations)  # self._sigmoid(visible_activations)
        visible_sampled = torch.bernoulli(visible_probabilities)  # ***************** NEW
        visible_sampled_phys = -1 + visible_sampled * 2
        return visible_sampled_phys

    def contrastive_divergence_orig(self, input_data):
        # Positive phase (WIKI: Steps 1, 2)
        positive_hidden_sampled = self.sample_hidden(input_data)
        positive_gradient = torch.matmul(input_data.t(), positive_hidden_sampled)    # WIKI 2: v dot h

        # Negative phase (WIKI: Steps 3, 4)
        hidden_sampled = positive_hidden_sampled
        for step in range(self.k):
            visible_sampled = self.sample_visible(hidden_sampled)
            hidden_sampled = self.sample_hidden(visible_sampled)
        negative_visible_sampled = visible_sampled
        negative_hidden_sampled = hidden_sampled
        negative_gradient = torch.matmul(negative_visible_sampled.t(), negative_hidden_sampled)  # WIKI 4: v' dot h'

        # Update parameters (WIKI: Steps 5, 6) -- note no momentum OR batch size affected learning rate
        # batch_size = input_data.size(0)
        self.weights += (positive_gradient - negative_gradient) * self.learning_rate  # / batch_size
        self.visible_bias += torch.sum(input_data - negative_visible_sampled, dim=0) * self.learning_rate  # / batch_size
        self.hidden_bias += torch.sum(positive_hidden_sampled - negative_hidden_sampled, dim=0) * self.learning_rate  # / batch_size
        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_sampled)**2)

        return error

    def contrastive_divergence(self, input_data):
        learning_rate_scaled = self.learning_rate / input_data.shape[0]  # larger batches mean higher learning rate

        # Positive phase (WIKI: Steps 1, 2)
        positive_hidden_sampled = self.sample_hidden(input_data, stdev=0)            # math says it should be the mean
        positive_gradient = torch.matmul(input_data.t(), positive_hidden_sampled)  # WIKI 2: v dot h

        # Negative phase (WIKI: Steps 3, 4) - use CD-k
        visible_sampled = input_data
        for step in range(self.k):
            hidden_sampled = self.sample_hidden(visible_sampled)
            visible_sampled = self.sample_visible(hidden_sampled)
        negative_hidden_sampled = hidden_sampled
        negative_visible_sampled = visible_sampled
        negative_gradient = torch.matmul(negative_visible_sampled.t(), negative_hidden_sampled)  # WIKI 4: v' dot h'

        # Update parameters (WIKI: Steps 5, 6) -- note no momentum OR batch size affected learning rate
        # batch_size = input_data.size(0)
        self.weights += (positive_gradient - negative_gradient) * learning_rate_scaled
        if self.use_fields:
            self.visible_bias += torch.sum(input_data - negative_visible_sampled, dim=0) * learning_rate_scaled
            self.hidden_bias += torch.sum(positive_hidden_sampled - negative_hidden_sampled, dim=0) * learning_rate_scaled
        #self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_sampled)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)
        return random_probabilities

    def load_rbm_trained(self, fpath):
        with open(fpath, 'rb') as f:
            rbm_internal_weights = np.load(fpath)['Q']
        self.weights = rbm_internal_weights
        return rbm_internal_weights

    def plot_model(self, title='def'):
        """
        Makes K + 2 plots
           where K = self.num_hidden
        """
        for col in range(self.num_hidden):
            plt.imshow(self.weights[:, col].view(28, 28), interpolation='none')
            plt.colorbar()
            plot_title = 'trained_weights_col%d_%s' % (col, title)
            plt.title(plot_title)
            plt.savefig(DIR_OUTPUT + os.sep + 'training' + os.sep + plot_title + '.jpg')
            plt.close()

        plt.title(title)
        plt.imshow(self.visible_bias.view(28, 28), interpolation='none')
        plt.colorbar()
        plot_title = 'trained_visibleField_col%d_%s' % (col, title)
        plt.title(plot_title)
        plt.savefig(DIR_OUTPUT + os.sep + 'training' + os.sep + plot_title + '.jpg')
        plt.close()

        plt.plot(self.hidden_bias)
        plt.hlines(0.0, 0.0, self.num_hidden, linestyles='--', colors='k')
        plot_title = 'trained_hiddenField_col%d_%s' % (col, title)
        plt.title(title)
        plt.savefig(DIR_OUTPUT + os.sep + 'training' + os.sep + plot_title + '.jpg')
        plt.close()
        return
