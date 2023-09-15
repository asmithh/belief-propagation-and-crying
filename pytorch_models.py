import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_hidden_layers, hidden_dim, in_dim, out_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_layer = nn.Linear(self.in_dim, self.hidden_dim)

        self.hidden_layers = [nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.n_hidden_layers)]

        self.out_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, inp):
        res = self.in_layer(inp)
        for layer in self.hidden_layers:
            res = layer(res)
        pre_activation = self.out_layer(res)
        return nn.relu(pre_activation)


class GraphBeliefPropagationNN(nn.Module):
    def __init__(self, perceptron_params, bp_iterations):
        super(GraphBeliefPropagationNN, self).__init__()
        self.MultiLayerPerceptron = MultiLayerPerceptron(**perceptron_params)
        self.bp_iterations = bp_iterations

    def compute_self_potentials(self, X):
        return nn.LogSoftmax(self.MultiLayerPerceptron(X))


