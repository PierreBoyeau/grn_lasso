import torch
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial
from torch import nn


class GRN(nn.Module):
    """One-pass regression of every gene on the regulators.

        r_i = f_i(sum_j W_ij h_j + b_i)
        X_i ~ NB(mean = l * r_i, inverse dispersion = theta_i)

    with f_i = exp ("exp") or f_i = softplus(a_i) * sigmoid ("sigmoid"). The
    likelihood is `scvi.distributions.NegativeBinomial` in its (mu, theta)
    parameterization, so Var[X_i] = mu_i + mu_i^2 / theta_i for mu_i = l * r_i:
    theta_i -> inf is Poisson and small theta_i is strongly overdispersed.
    (`theta` is scvi's name for what numpyro's NegativeBinomial2 calls
    `concentration`, as used in tf_prediction.distributions.)

    regulator_idx: gene index of each regulator (column of W). A gene never
    regulates itself: W[i, k] is fixed to 0 when regulator k is gene i.
    init_rate: per-gene mean of X / l, used only to initialize b (and a).
    """

    def __init__(self, regulator_idx, n_genes, nonlinearity="exp", init_rate=None):
        super().__init__()
        if nonlinearity not in ("exp", "sigmoid"):
            raise ValueError(f"nonlinearity must be 'exp' or 'sigmoid', got {nonlinearity!r}")
        self.nonlinearity = nonlinearity
        regulator_idx = torch.as_tensor(regulator_idx)
        n_regulators = len(regulator_idx)

        mask = torch.ones(n_genes, n_regulators)
        mask[regulator_idx, torch.arange(n_regulators)] = 0.0
        self.register_buffer("mask", mask)

        if init_rate is None:
            init_rate = torch.full((n_genes,), 1e-4)
        init_rate = torch.as_tensor(init_rate, dtype=torch.float32).clamp(min=1e-8)

        self.W = nn.Parameter(torch.zeros(n_genes, n_regulators))
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        if nonlinearity == "exp":
            self.b = nn.Parameter(init_rate.log())
        else:
            # softplus(a) = 2 * init_rate, so that the rate starts at init_rate when b = 0
            self.a = nn.Parameter(torch.log(torch.expm1(2 * init_rate)))
            self.b = nn.Parameter(torch.zeros(n_genes))

    def weights(self):
        return self.W * self.mask

    def log_rate(self, h):
        z = h @ self.weights().T + self.b
        if self.nonlinearity == "exp":
            return z
        return torch.log(F.softplus(self.a)) + F.logsigmoid(z)

    def log_prob(self, x, h, lib):
        """Log-likelihood of each count, shape (n_cells, n_genes)."""
        mu = torch.exp(torch.log(lib)[:, None] + self.log_rate(h))
        nb = NegativeBinomial(mu=mu, theta=self.log_theta.exp())
        return nb.log_prob(x)
