import anndata
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from grn_lasso import GRN, GRNData, auprc, fit, top_new_edges


def simulate(n_cells=500, n_genes=20, seed=0):
    """Gene 0 activates gene 1; cells labelled 'g0' have gene 0 knocked down."""
    rng = np.random.default_rng(seed)
    lib = rng.integers(2000, 5000, n_cells)
    kd = rng.random(n_cells) < 0.5
    rate = np.full((n_cells, n_genes), 1 / n_genes)
    rate[:, :2] = 0.01
    rate[kd, 0] /= 10
    rate[kd, 1] /= 5
    counts = rng.poisson(rate * lib[:, None])
    adata = anndata.AnnData(
        X=np.zeros((n_cells, n_genes), dtype=np.float32),
        obs=pd.DataFrame({"target": np.where(kd, "g0", "control")}),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    adata.layers["reads"] = sparse.csr_matrix(counts)
    return adata


@pytest.mark.parametrize("nonlinearity", ["exp", "sigmoid"])
def test_fit(nonlinearity):
    adata = simulate()
    regulators = np.array([0, 2, 3])
    data = GRNData.from_anndata(adata, regulators, perturbation_key="target")
    model = GRN(regulators, data.n_genes, nonlinearity=nonlinearity, init_rate=data.mean_rate())

    x, h, lib, keep = data.batch(np.arange(10))
    assert model.log_prob(x, h, lib).shape == (10, data.n_genes)
    assert (~keep[:, 0] == torch.tensor(data.kd_gene[:10] == 0)).all()

    history = fit(model, data, laplace_scale=0.1, n_epochs=30, batch_size=100, lr=0.05)
    assert history[-1] < history[0]
    assert (model.weights()[regulators, np.arange(3)] == 0).all()

    known = np.zeros((data.n_genes, len(regulators)), dtype=bool)
    known[1, 0] = True
    assert auprc(model, known) > 0.5
    names = np.array(data.gene_names)[regulators]
    assert len(top_new_edges(model, known, data.gene_names, names, k=5)) == 5
