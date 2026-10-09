import numpy as np
import torch
from tqdm import tqdm


def fit(model, data, laplace_scale, n_epochs=10, batch_size=1024, lr=1e-2, device="cpu"):
    """MAP fit of `model` with a Laplace(0, laplace_scale) prior on W.

    Maximizes  sum_cells log p(X | W) + log p(W).  We minimize this divided by
    the number of cells, so each minibatch uses the mean over its cells.

    Adam on |W| gives small but nonzero weights rather than exact zeros; a
    proximal (soft-thresholding) step on W would give exact zeros.

    Returns the mean loss of each epoch.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_cells = data.n_cells

    history = []
    for _ in tqdm(range(n_epochs)):
        losses = []
        for idx in torch.randperm(n_cells).split(batch_size):
            x, h, lib, keep = (t.to(device) for t in data.batch(idx.numpy()))
            log_lik = (model.log_prob(x, h, lib) * keep).sum(1).mean()
            log_prior = -model.weights().abs().sum() / laplace_scale
            loss = -(log_lik + log_prior / n_cells)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return history
