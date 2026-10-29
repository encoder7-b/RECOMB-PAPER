"""Real-data permutation null: shuffle response within batch strata of every cohort.

Keeps the real expression covariance, real external structure and real designs, and
breaks every response association, so any discovery is false. Reports the per-test
rejection rate of module z-scores, and the family-wise behaviour of the pre-registered
procedure (BH) and of BY. Labels are only permuted, never used for inference.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import run_real as R  # noqa: E402
from icko.core import bh, by, pc_stouffer  # noqa: E402
from icko.external import fit_external  # noqa: E402

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def main():
    cohorts = R.load_cohorts(R.ALL)
    tcga = R.load_tcga()
    genes, markers, centred = R.gene_universe(tcga, cohorts)
    mods = R.make_modules(centred, genes)
    rng = np.random.default_rng(29102026)
    prep = {}
    for c in R.DISCOVERY:
        ref = pd.concat([tcga[t].loc[genes + markers] for t in R.MATCH[c]], axis=1).T.values
        ext = fit_external(ref, k=R.k_for(len(cohorts[c][1])), max_deg=R.MAX_DEG, w_span=R.W_SPAN)
        x, y, covars, base = R.cohort_inputs(*cohorts[c], genes, markers, ext, rng, False)
        prep[c] = (x, y, covars, base, ext, cohorts[c][1]["batch"].values)
    rows, zall = [], []
    for b in range(N_PERM):
        zs = []
        for c, (x, y, covars, base, ext, batch) in prep.items():
            yp = y.copy()
            for v in np.unique(batch):  # permute within batch stratum
                idx = np.flatnonzero(batch == v)
                yp[idx] = rng.permutation(y[idx])
            zs.append(R.cohort_z(x, yp, covars, base, ext, mods)[0])
        zm = np.column_stack(zs)
        zall.append(zm)
        p1 = pc_stouffer(zm, 1)
        rows.append(dict(perm=b, bh=int(bh(p1, R.ALPHA).sum()), by=int(by(p1, R.ALPHA).sum()),
                         bh_r2=int(bh(pc_stouffer(zm, 2), R.ALPHA).sum()), min_p=float(p1.min())))
        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{N_PERM}", flush=True)
    df = pd.DataFrame(rows)
    zall = np.stack(zall)  # perm x module x cohort
    pz = 2 * stats.norm.sf(np.abs(zall))
    out = os.path.join(R.ROOT, "results")
    df.to_csv(f"{out}/perm_null.csv", index=False)
    rate = np.nanmean(pz < 0.05)
    se = np.sqrt(rate * (1 - rate) / np.isfinite(pz).sum())
    summ = {"n_perm": N_PERM, "per_test_rate_0.05": float(rate), "per_test_rate_se": float(se),
            "fwer_bh_r1": float((df.bh > 0).mean()), "fwer_by_r1": float((df.by > 0).mean()),
            "fwer_bh_r2": float((df.bh_r2 > 0).mean()),
            "per_cohort_rate_0.05": {c: float(np.nanmean(pz[:, :, i] < 0.05)) for i, c in enumerate(prep)},
            "real_min_p_r1_quantile": float(np.mean(df.min_p <= pd.read_csv(f"{out}/real_real_modules.csv").p_r1.min()))}
    json.dump(summ, open(f"{out}/perm_null_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
