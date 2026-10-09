"""Read the RegulonDB downloads (https://regulondb.ccg.unam.mx/datasets).

NetworkRegulatorGene.tsv:          regulator -> target gene edges
GeneProductAllIdentifiersSet.tsv:  gene synonyms, to match names with the data
"""

import numpy as np
import pandas as pd

REGULATOR_TYPES = {
    "RDBECOLITFC": "tf",
    "RDBECOLIPDC": "srna_protein",
    "RDBECOLIRCC": "complex",
    "RDBECOLICNC": "compound",
}


def read_tsv(path):
    """RegulonDB tables: '#' comment header, columns named like '1)regulatorId'."""
    df = pd.read_csv(path, sep="\t", comment="#")
    df.columns = df.columns.str.replace(r"^\d+\)", "", regex=True).str.strip()
    return df


def synonym_map(synonyms_tsv, gene_names):
    """Map every lowercased gene name or synonym to a name in `gene_names`."""
    df = read_tsv(synonyms_tsv)
    gene_names = set(gene_names)
    out = {}
    for name, synonyms in zip(df["geneName"].str.lower(), df["geneSynonyms"].fillna("")):
        aliases = [name] + [s.strip().lower() for s in synonyms.split(",") if s.strip()]
        matches = sorted(a for a in aliases if a in gene_names)
        if matches:
            for a in aliases:
                out.setdefault(a, matches[0])
    for g in gene_names:
        out[g] = g
    return out


def load_edges(network_tsv, synonyms_tsv, gene_names):
    """Regulator -> target edges, with names matched to `gene_names` (lowercase).

    Returns columns `regulator`, `target`, `regulator_type`. Edges whose
    regulator or target is not in `gene_names` are dropped. A regulator made
    of several genes (e.g. ihfA;ihfB) gives one edge per gene.
    """
    df = read_tsv(network_tsv)
    df = pd.DataFrame(
        {
            "regulator": df["RegulatorGeneName"].str.lower().str.split(";"),
            "target": df["regulatedGeneName"].str.lower(),
            "regulator_type": df["regulatorId"].str[:11].map(REGULATOR_TYPES),
        }
    ).explode("regulator")

    to_data = synonym_map(synonyms_tsv, gene_names)
    df["regulator"] = df["regulator"].str.strip().map(to_data)
    df["target"] = df["target"].map(to_data)
    return df.dropna().drop_duplicates().reset_index(drop=True)


def edge_matrix(edges, gene_names, regulator_names):
    """(n_genes, n_regulators) bool matrix, True where regulator -> gene is an edge."""
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    reg_idx = {r: k for k, r in enumerate(regulator_names)}
    A = np.zeros((len(gene_names), len(regulator_names)), dtype=bool)
    for r, t in zip(edges["regulator"], edges["target"]):
        if r in reg_idx and t in gene_idx:
            A[gene_idx[t], reg_idx[r]] = True
    return A
