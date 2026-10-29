"""G3: structured factor-space misspecification (oracle graph, controlled W).

Cases (columns of the true W = Omega Lam: 0 = immune F1, 1 = F2, 2 = F3, 3 = diffuse F4):
  oracle          W
  contained       [W, 4 random directions]          true span contained (over-spanned)
  omit_unrelated  W without F2                      omitted direction ~uncorrelated with r
  omit_related    W without F1                      omitted direction correlated with r
                                                    (signal genes load on F1)
  omit_diffuse    W without F4                      weakly related (some signal modules load on F4)
Signal: AUC 0.8 (sparse, coherent). Reported: null-module rejection rate at 0.05 and FDR (r = 1).
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icko.core import bh, pc_stouffer  # noqa: E402
from sim import truth as T  # noqa: E402
from sim.g1_external import cohort_pvalues  # noqa: E402

CASES = ["oracle", "contained", "omit_unrelated", "omit_related", "omit_diffuse"]


def case_w(tr, case, rng):
    w = tr["w"]
    if case == "oracle":
        return w
    if case == "contained":
        return np.hstack([w, rng.normal(size=(T.P, 4))])
    drop = {"omit_unrelated": 1, "omit_related": 0, "omit_diffuse": 3}[case]
    return np.delete(w, drop, axis=1)


def one_rep(rep, n, amp, auc, case, k, tr):
    rng = np.random.default_rng(15_485_863 * rep + 31 * n + CASES.index(case))
    ext = dict(scale=np.ones(T.P), w=case_w(tr, case, np.random.default_rng(7)), nbrs=tr["nbrs"])
    per = [cohort_pvalues(tr, n, amp, rng, ext, tr["nbrs"]) for _ in range(k)]
    z = np.column_stack([p[0] for p in per])
    nn = np.isin(np.arange(T.N_TEST_MOD), tr["sig_mods"])
    rej = bh(pc_stouffer(z, 1), 0.1)
    pz = 2 * stats.norm.sf(np.abs(z))
    return dict(pattern=tr["pattern"], case=case, n=n, auc=auc, rep=rep,
                null_rate=np.mean(pz[~nn] < 0.05), fdp=(rej & ~nn).sum() / max(1, rej.sum()),
                power=(rej & nn).sum() / nn.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--ns", default="50,100,300")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    jobs = []
    for pat in ("sparse", "coherent"):
        tr = T.build_truth(2026, pattern=pat)
        amp = T.amp_for_auc(tr, 0.8)
        jobs += [delayed(one_rep)(rep, n, amp, 0.8, case, 6, tr)
                 for case, n, rep in itertools.product(CASES, [int(v) for v in args.ns.split(",")],
                                                       range(args.reps))]
    df = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=2)(jobs))
    out = os.path.join(os.path.dirname(__file__), "..", "results", "g3_factor_misspec")
    df.to_csv(out + ".csv", index=False)
    g = df.groupby(["pattern", "case", "n"])
    summ = g.agg(null_rate=("null_rate", "mean"), fdr=("fdp", "mean"), power=("power", "mean"),
                 reps=("rep", "count")).reset_index()
    summ["fdr_se"] = g.fdp.std().values / np.sqrt(summ.reps)
    summ.to_csv(out + "_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
    with pd.option_context("display.width", 160, "display.max_rows", 100):
        print(summ.round(3))


if __name__ == "__main__":
    main()
