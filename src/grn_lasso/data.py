from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse


@dataclass
class GRNData:
    """Counts of one dataset, and how to turn them into model inputs.

    counts:        (n_cells, n_genes) sparse raw counts
    lib:           (n_cells,) library size
    kd_gene:       (n_cells,) index of the knocked-down gene, -1 if none
    regulator_idx: (n_regulators,) gene index of each regulator
    h_mean, h_std: (n_regulators,) used to standardize the regulator inputs
    """

    counts: sparse.csr_matrix
    lib: np.ndarray
    kd_gene: np.ndarray
    regulator_idx: np.ndarray
    gene_names: list
    h_mean: np.ndarray
    h_std: np.ndarray

    @property
    def n_cells(self):
        return self.counts.shape[0]

    @property
    def n_genes(self):
        return self.counts.shape[1]

    def mean_rate(self):
        """Per-gene mean of X / l."""
        return (sparse.diags(1 / self.lib) @ self.counts).mean(0).A1

    def log_cp10k(self, idx):
        """Sparse log-CP10K of the regulators for the cells `idx`."""
        h = sparse.diags(1e4 / self.lib[idx]) @ self.counts[idx][:, self.regulator_idx]
        return h.log1p()

    def batch(self, idx):
        """Tensors (x, h, lib, keep) for the cells `idx`.

        keep is False for the knocked-down gene of each cell, which is left
        out of the likelihood.
        """
        x = self.counts[idx].toarray()
        h = (self.log_cp10k(idx).toarray() - self.h_mean) / self.h_std
        keep = np.ones_like(x, dtype=bool)
        kd = self.kd_gene[idx]
        rows = np.where(kd >= 0)[0]
        keep[rows, kd[rows]] = False
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(h, dtype=torch.float32),
            torch.tensor(self.lib[idx], dtype=torch.float32),
            torch.tensor(keep),
        )

    @classmethod
    def from_anndata(cls, adata, regulator_idx, perturbation_key, layer="reads", standardize=True):
        """`adata.obs[perturbation_key]` holds the knocked-down gene's name
        (as in `adata.var_names`); other labels, e.g. controls, mean no knockdown."""
        counts = sparse.csr_matrix(adata.layers[layer], dtype=np.float32)
        lib = counts.sum(1).A1
        gene_idx = {g: i for i, g in enumerate(adata.var_names)}
        kd_gene = np.array([gene_idx.get(p, -1) for p in adata.obs[perturbation_key]])
        regulator_idx = np.asarray(regulator_idx)

        data = cls(
            counts=counts,
            lib=lib,
            kd_gene=kd_gene,
            regulator_idx=regulator_idx,
            gene_names=list(adata.var_names),
            h_mean=np.zeros(len(regulator_idx), dtype=np.float32),
            h_std=np.ones(len(regulator_idx), dtype=np.float32),
        )
        if standardize:
            h = data.log_cp10k(np.arange(data.n_cells))
            mean = h.mean(0).A1
            std = np.sqrt(np.maximum(h.power(2).mean(0).A1 - mean**2, 0))
            data.h_mean = mean
            data.h_std = np.where(std > 0, std, 1.0)
        return data
