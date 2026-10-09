import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def auprc(model, known_edges):
    """Area under the precision-recall curve of |W| against known edges
    (e.g. RegulonDB), over all entries of W that are not fixed to 0."""
    W = model.weights().detach().cpu().numpy()
    allowed = model.mask.cpu().numpy() > 0
    return average_precision_score(known_edges[allowed], np.abs(W[allowed]))


def top_new_edges(model, known_edges, gene_names, regulator_names, k=50):
    """The k largest |W| entries that are not known edges."""
    W = model.weights().detach().cpu().numpy()
    score = np.where(known_edges, 0.0, np.abs(W))
    flat = np.argsort(score, axis=None)[::-1][:k]
    genes, regs = np.unravel_index(flat, W.shape)
    return pd.DataFrame(
        {
            "regulator": np.asarray(regulator_names)[regs],
            "target": np.asarray(gene_names)[genes],
            "weight": W[genes, regs],
        }
    )
