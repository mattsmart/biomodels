# sample code from git repo: https://github.com/GabrielBianconi/pytorch-rbm

import torch


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
        self.weights += (positive_gradient - negative_gradient) * self.learning_rate  # / batch_size
        self.visible_bias += torch.sum(input_data - negative_visible_sampled, dim=0) * self.learning_rate  # / batch_size
        self.hidden_bias += torch.sum(positive_hidden_sampled - negative_hidden_sampled, dim=0) * self.learning_rate  # / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_sampled)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        return random_probabilities
