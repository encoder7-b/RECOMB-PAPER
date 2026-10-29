"""Gate G0': exact conditional module/gene tests + partial conjunction, oracle graph and W."""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icko.core import bh, pc_simes, pc_stouffer, test_sets  # noqa: E402
from sim import truth as T  # noqa: E402

ALPHA = 0.1


def cohort_pvalues(tr, n, amp, rng, w, nbrs):
    x, y, batch, score_b = T.simulate_cohort(tr, n, amp, rng)
    covars = np.column_stack([np.ones(n), batch])
    base = np.column_stack([covars, score_b])
    beta, *_ = np.linalg.lstsq(base, y, rcond=None)
    r = y - base @ beta
    mods_mean = test_sets(x, T.module_sets(), nbrs, w, covars, r, stat="mean", signed=True)
    mods_wilks = test_sets(x, T.module_sets(), nbrs, w, covars, r, stat="wilks")
    genes = test_sets(x, [np.array([j]) for j in range(T.P_TEST)], nbrs, w, covars, r, signed=True)
    return mods_mean, mods_wilks, genes


def one_rep(rep, n, amp, k, tr, w, nbrs):
    rng = np.random.default_rng(1_000_003 * rep + 31 * n + int(1000 * amp))
    per = [cohort_pvalues(tr, n, amp, rng, w, nbrs) for _ in range(k)]
    mod_nn = np.isin(np.arange(T.N_TEST_MOD), tr["sig_mods"])
    nonnull = {"module_mean": mod_nn, "module_wilks": mod_nn,
               "gene": np.isin(np.arange(T.P_TEST), tr["signal"])}
    rows = []
    for li, level in enumerate(["module_mean", "module_wilks", "gene"]):
        pm = np.column_stack([c[li] for c in per])
        nn = nonnull[level]
        signed = level != "module_wilks"
        if signed:  # z-scores -> two-sided p for Simes and calibration
            zm, pm = pm, 2 * stats.norm.sf(np.abs(pm))
        for r in range(1, k + 1):
            combs = {"simes": pc_simes(pm, r)}
            if signed:
                combs["stouffer"] = pc_stouffer(zm, r)
            for comb, pc in combs.items():
                rej = bh(pc, ALPHA)
                rows.append(dict(pattern=tr["pattern"], rep=rep, n=n, amp=amp, level=level, comb=comb, r=r,
                                 R=rej.sum(), fdp=(rej & ~nn).sum() / max(1, rej.sum()),
                                 power=(rej & nn).sum() / nn.sum()))
        # null calibration: fraction of null p-values below 0.05 (should be ~0.05)
        rows.append(dict(pattern=tr["pattern"], rep=rep, n=n, amp=amp, level=level, comb="none", r=0, R=np.nan,
                         fdp=np.nanmean(pm[~nn] < 0.05), power=np.nanmean(pm[nn] < 0.05)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--cohorts", type=int, default=6)
    ap.add_argument("--ns", default="25,50,100,300")
    ap.add_argument("--aucs", default="0.65,0.70,0.75")
    ap.add_argument("--patterns", default="sparse,coherent")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    ns = [int(v) for v in args.ns.split(",")]
    targets = [float(v) for v in args.aucs.split(",")]
    t0 = time.time()
    jobs, amp_auc = [], {}
    for pat in args.patterns.split(","):
        tr = T.build_truth(2026, pattern=pat)
        for target in targets:
            amp = T.amp_for_auc(tr, target)
            amp_auc[(pat, amp)] = target
            print(f"{pat}: AUC {target} -> amp {amp:.4f}", flush=True)
            jobs += [delayed(one_rep)(rep, n, amp, args.cohorts, tr, tr["w"], tr["nbrs"])
                     for n, rep in itertools.product(ns, range(args.reps))]
    df = pd.DataFrame(sum(Parallel(n_jobs=args.jobs, verbose=2)(jobs), []))
    df["auc"] = [amp_auc[(p, a)] for p, a in zip(df.pattern, df.amp)]
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"g0prime_{args.tag}.csv")
    df.to_csv(out, index=False)
    summ = (df.groupby(["pattern", "level", "comb", "auc", "n", "r"])
              .agg(fdr=("fdp", "mean"), power=("power", "mean"), R=("R", "mean"),
                   reps=("rep", "count")).reset_index())
    summ.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")
    with pd.option_context("display.max_rows", 500, "display.width", 160):
        print(summ[summ.r.isin([0, 1, 2, 3])].round(3))


if __name__ == "__main__":
    main()
