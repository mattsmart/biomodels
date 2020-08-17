# sample code from git repo: https://github.com/GabrielBianconi/pytorch-rbm

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from plotting import image_fancy
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
            npzpath = DIR_MODELS + os.sep + 'saved' + os.sep + 'hopfield_mnist_%d.npz' % num_hidden
            print("Loading weights from %s" % npzpath)
            self.load_rbm_trained(npzpath)
            self.visible_bias = torch.zeros(num_visible).float()
        else:
            use_normal = True
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

    def sample_visible(self, hidden_state, beta=BETA):
        visible_activations = torch.matmul(hidden_state, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(2 * beta * visible_activations)  # self._sigmoid(visible_activations)
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
            self.hidden_bias += torch.sum(positive_hidden_sampled - negative_hidden_sampled, dim=0) * 2 * learning_rate_scaled
        #self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_sampled)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)
        return random_probabilities

    def load_rbm_trained(self, fpath, weights_id='Q'):
        with open(fpath, 'rb') as f:
            arr = np.load(fpath)[weights_id]
        self.weights = torch.from_numpy(arr).float()
        return

    def plot_model(self, title='def', outdir=None):
        """
        Makes K + 2 plots
           where K = self.num_hidden
        """
        if outdir is None:
            outdir = DIR_OUTPUT + os.sep + 'training'

        for col in range(self.num_hidden):
            plt.imshow(self.weights[:, col].view(28, 28), interpolation='none')
            plt.colorbar()
            plot_title = 'trained_weights_col%d_%s' % (col, title)
            plt.title(plot_title)
            plt.savefig(outdir + os.sep + plot_title + '.jpg')
            plt.close()

        plt.title(title)
        plt.imshow(self.visible_bias.view(28, 28), interpolation='none')
        plt.colorbar()
        plot_title = 'trained_visibleField_col%d_%s' % (col, title)
        plt.title(plot_title)
        plt.savefig(outdir + os.sep + plot_title + '.jpg')
        plt.close()

        plt.plot(self.hidden_bias)
        plt.hlines(0.0, 0.0, self.num_hidden, linestyles='--', colors='k')
        plot_title = 'trained_hiddenField_col%d_%s' % (col, title)
        plt.title(title)
        plt.savefig(outdir + os.sep + plot_title + '.jpg')
        plt.close()
        return

    def get_sample_images(self, num_images, k=20):
        # do k steps of annealing on random initial state to arrive at final sampled images
        # output shape: (k steps X num images X visible dimension)
        switch_A = k/4
        switch_B = 3 * k/4

        def anneal_schedule(step):
            """
            if step > switch:
                beta_step = 20
            else:
                beta_step = 0.5
            return beta_step
            """
            if step > switch_B:
                beta_step = 20
            elif step > switch_A:
                beta_step = 2
            else:
                beta_step = 0.1
            return beta_step

        # initial states are coin flips up/down
        a = 0.5 * torch.ones(num_images, self.num_visible)
        visible_block = torch.bernoulli(a) * 2 - 1

        # track timeseries of the chain
        visible_timeseries = torch.zeros(k, num_images, self.num_visible)
        visible_timeseries[0, :, :] = visible_block

        for step in range(k):
            beta_step = anneal_schedule(step)
            hidden_block = self.sample_hidden(visible_block, stdev=1/np.sqrt(beta_step))
            visible_block = self.sample_visible(hidden_block, beta=beta_step)
            visible_timeseries[step, :, :] = visible_block

        return visible_timeseries

    def plot_sample_images(self, visible_timeseries, outdir, only_last=True):
        visible_timeseries_numpy = visible_timeseries.numpy()
        num_steps = visible_timeseries_numpy.shape[0]
        num_images = visible_timeseries_numpy.shape[1]

        steps_to_plot = range(num_steps)
        if only_last:
            steps_to_plot = [-1]

        for idx in range(num_images):
            for k in steps_to_plot:
                image = visible_timeseries_numpy[k, idx, :].reshape((28, 28))
                plt.figure()
                image_fancy(image, ax=plt.gca(), show_labels=False)
                plt.title('Trajectory: %d | Step: %d' % (num_images, k))
                plt.savefig(outdir + os.sep + 'traj%d_step%d.png' % (idx, k));
                plt.close()

        return


if __name__ == '__main__':

    sample_trained_rbm = True

    if sample_trained_rbm:

        # pick data to load
        num_hidden = 20
        total_epochs = 100
        batch = 100
        cdk = 20
        use_fields = False
        ais_steps = 200
        beta = 2
        assert beta == 2
        run = 1
        epoch_idx = [5, 99]  # [96, 97, 98]

        # load data
        bigruns = DIR_OUTPUT + os.sep + 'archive' + os.sep + 'big_runs' + os.sep + 'rbm'
        subdir = 'F_beta2duringTraining_%dbatch_%depochs_%dcdk_1.00E-04eta_%dais' % (batch, total_epochs, cdk, ais_steps)
        fname = 'weights_%dhidden_%dfields_%dcdk_%dstepsAIS_%.2fbeta.npz' % (num_hidden, use_fields, cdk, ais_steps, beta)
        dataobj = np.load(bigruns + os.sep + subdir + os.sep + 'run%d' % run + os.sep + fname)
        weights_timeseries = dataobj['weights']

        for idx in epoch_idx:

            outdir = DIR_OUTPUT + os.sep + 'samples' + os.sep + 'epoch%d' % idx
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            weights = weights_timeseries[:, :, idx]
            rbm = weights_timeseries[:, :, idx]
            rbm = RBM_gaussian_custom(28**2, num_hidden, 0, load_init_weights=False, use_fields=False, learning_rate=0)
            rbm.weights = torch.from_numpy(weights).float()

            timeseries = True
            num_images = 10
            k_steps = 40
            visible_block = rbm.get_sample_images(num_images, k=k_steps)
            rbm.plot_sample_images(visible_block, outdir, only_last=False)
