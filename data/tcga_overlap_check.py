"""Label-free screen for ICI-cohort patients that also appear in the TCGA reference data.

Assumption 3 needs the external structure to be independent of the cohort data, so a cohort
patient whose tumour is also in the TCGA expression used to learn the structure would break it.
Same statistic, gene set rule and calibrated threshold as dedup_check.py (within-dataset ranks,
top K_PC components removed, sample correlation over top-variance shared genes, flag rho > 0.5).
Every ICI cohort is screened against every TCGA reference type. Writes results/tcga_overlap_*.csv.
"""
import itertools
import os

import numpy as np
import pandas as pd

from dedup_check import COHORTS, N_GENES, PROC, OUT, THRESH, deviations

TCGA = ["tcga_skcm", "tcga_stad", "tcga_luad", "tcga_lusc", "tcga_blca", "tcga_kirc"]


def main():
    load = lambda c: pd.read_csv(f"{PROC}/{c}_expr.csv.gz", index_col=0)
    ex = {c: load(c) for c in COHORTS + TCGA if os.path.exists(f"{PROC}/{c}_expr.csv.gz")}
    common = sorted(set.intersection(*[set(e.index) for e in ex.values()]))
    allx = pd.concat([e.loc[common] for e in ex.values()], axis=1)
    genes = allx.var(axis=1).sort_values(ascending=False).index[:N_GENES]
    devs = {c: deviations(e.loc[genes].values) for c, e in ex.items()}
    rows, summary = [], []
    for a, b in itertools.product([c for c in COHORTS if c in ex], [t for t in TCGA if t in ex]):
        c = devs[a].T @ devs[b] / len(genes)
        summary.append(dict(cohort=a, tcga=b, max_rho=c.max(), p99=np.quantile(c, 0.99)))
        for i, j in zip(*np.where(c > THRESH)):
            rows.append(dict(cohort=a, sample=ex[a].columns[i], tcga=b, tcga_sample=ex[b].columns[j], rho=c[i, j]))
    pd.DataFrame(rows, columns=["cohort", "sample", "tcga", "tcga_sample", "rho"]).to_csv(
        f"{OUT}/tcga_overlap_pairs.csv", index=False)
    s = pd.DataFrame(summary).sort_values("max_rho", ascending=False)
    s.to_csv(f"{OUT}/tcga_overlap_summary.csv", index=False)
    print(f"shared genes {len(common)}, screened {len(genes)}; flagged pairs: {len(rows)}")
    print(s.head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
