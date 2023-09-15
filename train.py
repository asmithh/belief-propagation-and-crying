import torch
import torch.nn.functional as F

from pytorch_models import GraphBeliefPropagationNN


def train_GPBN():
    model_GBPN = GraphBeliefPropgationNN()
    optim = MultiOptimizer(
        optim.Adam(model_GBPN.MultiLayerPerceptron.parameters(), lr=0.03, weight_decay=2.0e-4),
        optim.Adam(model_GBPN.H_matrix, lr=0.03, weight_decay=2.0e-4)
    )
    loss_fn = F.nll_loss()
    model_GBPN.train()

    model = model.to(device)

    for edges, X, y in data:
        optim.zero_grad()
        res_probas = model_GBPN(edges, X)
        loss = loss_fn(res_probas, y)
        loss_fn.backward()
        optim.step()
        print(loss)


