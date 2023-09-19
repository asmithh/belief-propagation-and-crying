import itertools

import networkx as nx
import numpy as np
from scipy.stats import rv_discrete
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from pytorch_models import GraphBeliefPropagationNN


class IsingDataset(Dataset):
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.H_matrix = torch.Tensor([[0.7, 0.3], [0.3, 0.7]])
        self.H_vector = torch.Tensor([0.25, 0.75])
        self.X_given_y = [
            {1: [0.9, 0.1], 0: [0.5, 0.5]},
            {1: [0.3, 0.7], 0: [0.1, 0.9]},
            {1: [0.4, 0.6], 0: [0.1, 0.9]},
            {1: [0.75, 0.25], 0: [0.5, 0.5]}
        ]
        self.possible_ys = list(itertools.product([0, 1], repeat=self.n_nodes))

    def _get_proba_of_y(self, vec, edges):
        proba = 0.0
        proba += np.sum(vec) * self.H_vector[1]
        proba += (len(vec) - np.sum(vec)) * self.H_vector[0]
        for e in edges:
            proba += self.H_matrix[vec[e[0]], vec[e[1]]]
        return np.exp(proba)

    def _draw_x(self, y):
        vec = []
        for idx, my_dict in enumerate(self.X_given_y):
            vec.append(rv_discrete([0, 1], my_dict[y[idx]]))
        return vec

    def __iter__(self):
        edges = nx.gnp_random_graph(self.n_nodes, 0.25).edges
        wts = [self._get_proba_of_y(y, edges) for y in self.possible_ys]
        wts = [wt / np.sum(wts) for wt in wts]
        ys = rv_discrete(self.possible_ys, wts)
        X = torch.Tensor([self._draw_x(y) for y in ys])

        

        # generate y labels for nodes
        # generate ties based on probas
        # generate features
        yield edges, X, ys

def train_GBPN(device):
    n_nodes = 100
    perceptron_params = {
        'n_hidden_layers': 3,
        'hidden_dim': 32,
        'in_dim': 8,
        'out_dim': 2,
    }
    bp_iterations = 5
    model_GBPN = GraphBeliefPropagationNN(n_nodes, perceptron_params, bp_iterations, n_cats=2)
    optimizers = (
        optim.Adam(model_GBPN.MultiLayerPerceptron.parameters(), lr=0.03, weight_decay=2.0e-4),
        optim.Adam([model_GBPN.H_matrix], lr=0.03, weight_decay=2.0e-4)
    )
    loss_fn = F.nll_loss
    model_GBPN.train()

    model_GBPN = model_GBPN.to(device)
    data_loader = IsingDataset(n_nodes=20)
    for edges, X, y in data_loader:
        print(y)
        for optimizer in optimizers:
            optimizer.zero_grad()
        res_probas = model_GBPN(edges, X)
        loss = loss_fn(res_probas, y)
        loss_fn.backward()
        for optimizer in optimizers:
            optimizer.step()
        print(loss)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_GBPN(device)
