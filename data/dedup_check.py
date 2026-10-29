"""Label-free cross-cohort duplicate-patient screen.

The cross-cohort combiners assume independent cohorts, so a patient appearing in two
cohorts breaks validity. Each gene is rank-transformed and centred WITHIN its cohort
(removes pipeline / normalization offsets), samples are standardized, and pairs from
different cohorts are correlated over the top-variance shared genes.

The top K_PC principal components are removed within each cohort first, so shared biology
(immune infiltration, purity) does not make unrelated tumours look alike.

Calibration (K_PC = 5): positive control Hugo 2016 GEO vs iAtlas re-processing (same 26
patients, different pipelines) gives true-pair rho >= 0.674 and non-pairs <= 0.266;
impossible pairs (NSCLC vs ccRCC) <= 0.284. Threshold 0.5. Writes results/dedup_pairs.csv.
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROC = os.path.join(os.path.dirname(__file__), "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
COHORTS = ["hugo", "riaz", "gide_pd1", "gide_combo", "liu2019", "jung", "gse176307", "imvigor210",
           "immotion_atezo", "immotion_sunitinib", "braun_nivo", "braun_evero"]
N_GENES, THRESH, K_PC = 2000, 0.5, 5


def deviations(m):
    r = np.apply_along_axis(rankdata, 1, m)
    r = r - r.mean(1, keepdims=True)
    u, s, vt = np.linalg.svd(r, full_matrices=False)
    r = r - (u[:, :K_PC] * s[:K_PC]) @ vt[:K_PC]
    return (r - r.mean(0)) / r.std(0)


def main():
    ex = {c: pd.read_csv(f"{PROC}/{c}_expr.csv.gz", index_col=0) for c in COHORTS
          if os.path.exists(f"{PROC}/{c}_expr.csv.gz")}
    common = sorted(set.intersection(*[set(e.index) for e in ex.values()]))
    allx = pd.concat([e.loc[common] for e in ex.values()], axis=1)
    genes = allx.var(axis=1).sort_values(ascending=False).index[:N_GENES]
    devs = {c: deviations(e.loc[genes].values) for c, e in ex.items()}
    rows, summary = [], []
    for a, b in itertools.combinations(ex, 2):
        c = devs[a].T @ devs[b] / len(genes)
        summary.append(dict(a=a, b=b, max_rho=c.max(), p99=np.quantile(c, 0.99)))
        for i, j in zip(*np.where(c > THRESH)):
            rows.append(dict(a=a, sample_a=ex[a].columns[i], b=b, sample_b=ex[b].columns[j], rho=c[i, j]))
    os.makedirs(OUT, exist_ok=True)
    pairs = pd.DataFrame(rows, columns=["a", "sample_a", "b", "sample_b", "rho"])
    pairs.to_csv(f"{OUT}/dedup_pairs.csv", index=False)
    # Pre-specified, label-free rule: exclude every flagged sample (both sides of each pair).
    excl = pd.concat([pairs[["a", "sample_a"]].set_axis(["cohort", "sample"], axis=1),
                      pairs[["b", "sample_b"]].set_axis(["cohort", "sample"], axis=1)]).drop_duplicates()
    excl.to_csv(f"{OUT}/dedup_exclude.csv", index=False)
    print(f"excluding {len(excl)} samples")
    s = pd.DataFrame(summary).sort_values("max_rho", ascending=False)
    s.to_csv(f"{OUT}/dedup_summary.csv", index=False)
    print(f"shared genes {len(common)}, screened {len(genes)}; flagged pairs: {len(rows)}")
    print(s.head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
