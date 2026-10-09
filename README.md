# grn_lasso

Gene regulatory network inference from Perturb-seq counts, in PyTorch.

Each gene `i` is regressed on the regulators `j` (one pass, no simulation):

```
h_j = log1p(1e4 * X_j / l)               log-CP10K of regulator j (optionally standardized)
r_i = f_i(sum_j W_ij h_j + b_i)          f_i = exp, or f_i = softplus(a_i) * sigmoid
X_i ~ NB(mean = l * r_i, inverse dispersion theta_i)   Var = mu + mu^2 / theta
W   ~ Laplace(0, lambda)
```

`W` is fit by MAP, `max sum_cells log p(X | W) + log p(W)`. Entries with a large
`|W|` that RegulonDB does not know about are candidate new edges.

## Install

```bash
mamba create -y -n grn_lasso python=3.11
conda activate grn_lasso
pip install -e ".[dev]"
pytest tests
```

On a node whose NVIDIA driver is older than the default wheel's CUDA runtime,
install the matching torch build first, or `torch.cuda` fails to initialize --
which also breaks CPU training, since `Adam.step` queries the accelerator.
Check the driver with `nvidia-smi` (this node: 570.124.04 / CUDA 12.8):

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple "torch==2.11.0+cu128"
```

## Files

- `src/grn_lasso/model.py`: the model, `GRN`. The likelihood is
  `scvi.distributions.NegativeBinomial` in its `(mu, theta)` parameterization
  (`theta` = numpyro's `concentration`, as in `tf_prediction.distributions`)
- `src/grn_lasso/train.py`: the training loop, `fit`
- `src/grn_lasso/data.py`: AnnData to model inputs, `GRNData`
- `src/grn_lasso/regulondb.py`: reads the RegulonDB downloads `NetworkRegulatorGene.tsv` and `GeneProductAllIdentifiersSet.tsv`
- `src/grn_lasso/evaluate.py`: AUPRC against RegulonDB, and a table of top new edges
- `notebooks/tutorial.ipynb`: end-to-end example
