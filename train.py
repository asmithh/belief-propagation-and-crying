import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from pytorch_models import GraphBeliefPropagationNN


class IsingDataset(Dataset):
    def __init__(self):
        self.H_matrix = torch.Tensor([[0.9, 0.1], [0.1, 0.9]])

    def __getitem__(self, idx):
        # generate y labels for nodes
        # generate ties based on probas
        # generate features
        return graph, X, y

def train_GPBN():
    n_nodes = 100
    perceptron_params = {
        'n_hidden_layers': 3,
        'hidden_dim': 32,
        'in_dim': 8,
        'out_dim': 2,
    }
    model_GBPN = GraphBeliefPropgationNN(n_nodes, perceptron_params, bp_iterations, n_cats=2)
    optim = MultiOptimizer(
        optim.Adam(model_GBPN.MultiLayerPerceptron.parameters(), lr=0.03, weight_decay=2.0e-4),
        optim.Adam(model_GBPN.H_matrix, lr=0.03, weight_decay=2.0e-4)
    )
    loss_fn = F.nll_loss()
    model_GBPN.train()

    model = model.to(device)

    for edges, X, y in data_loader:
        optim.zero_grad()
        res_probas = model_GBPN(edges, X)
        loss = loss_fn(res_probas, y)
        loss_fn.backward()
        optim.step()
        print(loss)


