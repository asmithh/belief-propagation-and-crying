import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_hidden_layers, hidden_dim, in_dim, out_dim):
        super().__init__()
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
    def __init__(self, n_nodes, y_labels_truth, perceptron_params, bp_iterations, n_cats=2):
        super().__init__()
        self.MultiLayerPerceptron = MultiLayerPerceptron(**perceptron_params)
        self.bp_iterations = bp_iterations
        self.H_matrix = nn.Parameter(torch.zeros(n_cats, n_cats))
        self.n_cats = n_cats
        self.n_nodes = n_nodes
        self.y_labels_truth = y_labels_truth

    def compute_self_potentials(self, X):
        return nn.LogSoftmax(self.MultiLayerPerceptron(X))

    def belief_propagation(self, edges, X):
        zero_log_ps = self.compute_self_potentials(X)
        log_ps = zero_log_ps.copy()
        log_ms = (1.0 / self.c) * torch.zeros((n_cats, len(edges)))
        for i in range(self.bp_iterations):
            log_ps, log_ms = self._run_bp(edges, zero_log_ps, log_ps, log_ms)

        return log_ps, log_ms

    def normalize(self, log_ps):
        log_ps_new = torch.zeros((self.n_nodes, self.n_cats))
        for row in range(self.n_nodes):
            log_ps_new[row, :] = log_ps[row] - torch.logsumexp(log_ps[row]) * torch.ones((self.n_cats))

        return log_ps_new

    def forward(self, edges, X):
        log_ps, log_ms = self.belief_propagation(edges, X)
        return self.normalize(log_ps)

    def _run_bp(edges, zero_log_ps, log_ps, log_ms):
        new_log_ms = torch.zeros((len(edges), self.n_cats))
        for edge in range(len(edges)):
            for i in range(self.n_cats):
                j_res = torch.Tensor([self.H_matrix[i, j] + log_ps[j] - log_ms[edge, j] for j in range(self.n_cats)])
                j_res = torch.logsumexp(j_res)
                new_log_ms[edge, i] = j_res

        new_log_ps = torch.zeros((self.n_nodes, self.n_cats))
        for node in range(self.n_nodes):
            neighbors = [idx for idx, e in enumerate(edges) if e[0] == node]
            for cat in range(self.n_cats):
                new_log_ps[node, cat] = torch.sum(torch.Tensor([new_log_ms[i, cat] for i in neighbors])) + zero_log_ps[node, cat]

        return new_log_ms, new_log_ps


 
