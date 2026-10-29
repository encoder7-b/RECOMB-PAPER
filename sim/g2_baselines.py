"""G2: our method vs baselines on the same cohorts. Module level, 1-df mean statistic.

Methods (all produce per-cohort signed z or a pooled p, then BH at ALPHA):
  external  : ours. Graph + W from an unlabeled external sample; Stouffer PC.
  insample  : same test, but graph + W fit on the cohort itself (double use of X_G).
  marginal  : per-cohort z for module mean ~ residual (batch + infiltration adjusted),
              no conditioning on other genes; Stouffer PC. Targets a marginal null.
  pooled_labelaware : stack cohorts, remove cohort effects per module estimated with response
              as a protected covariate (ComBat-style), pooled marginal test (batch + infiltration adjusted).
  pooled_labelblind : same, but plain per-cohort centering (no labels in the correction).
Truth = conditional non-null modules (the estimand of the paper).
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
from icko.core import bh, pc_stouffer, test_sets  # noqa: E402
from icko.external import fit_external  # noqa: E402
from sim import truth as T  # noqa: E402
from sim.g1_external import external_sample  # noqa: E402

ALPHA = 0.1


def marginal_z(xm, covars, r):
    d = covars
    e = xm - d @ np.linalg.lstsq(d, xm, rcond=None)[0]
    rr = r - d @ np.linalg.lstsq(d, r, rcond=None)[0]
    df = len(r) - d.shape[1] - 1
    out = np.empty(xm.shape[1])
    for j in range(xm.shape[1]):
        c = np.corrcoef(e[:, j], rr)[0, 1]
        t = c * np.sqrt(df / max(1e-12, 1 - c ** 2))
        out[j] = stats.norm.ppf(stats.t.cdf(t, df))
    return out


def module_means(x):
    return np.column_stack([x[:, s].mean(1) for s in T.module_sets()])


def run_rep(rep, n, amp, auc, k, tr, ext):
    rng = np.random.default_rng(104_729 * rep + 31 * n + int(1e4 * amp))
    nn = np.isin(np.arange(T.N_TEST_MOD), tr["sig_mods"]) if amp > 0 else np.zeros(T.N_TEST_MOD, bool)
    z = {m: [] for m in ("external", "insample", "marginal")}
    pool_x, pool_y, pool_c, pool_b = [], [], [], []
    for c in range(k):
        # cohort-specific response rate (ORR differs by tumour type), drives label-aware leakage
        x, y, batch, score_b = T.simulate_cohort(tr, n, amp, rng, b0=rng.uniform(-2.0, 0.5))
        x = x / ext["scale"]
        covars = np.column_stack([np.ones(n), batch])
        base = np.column_stack([covars, score_b])
        r = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
        z["external"].append(test_sets(x, T.module_sets(), ext["nbrs"], ext["w"], covars, r, "mean", True))
        ins = fit_external(x, k=4, max_deg=min(6, max(1, n // 20)))
        z["insample"].append(test_sets(x / ins["scale"], T.module_sets(), ins["nbrs"], ins["w"], covars, r,
                                       "mean", True))
        z["marginal"].append(marginal_z(module_means(x), base, y))
        pool_x.append(module_means(x)); pool_y.append(y); pool_c.append(np.full(n, c))
        pool_b.append(np.column_stack([batch, score_b]))
    rows = []
    for m, zl in z.items():
        rej = bh(pc_stouffer(np.column_stack(zl), 1), ALPHA)
        rows.append(dict(method=m, rej=rej))
    # pooled, label-aware cohort correction: x_gj ~ cohort dummies + y, remove cohort part
    px, py, pc = np.vstack(pool_x), np.concatenate(pool_y), np.concatenate(pool_c)
    adj = np.column_stack([np.ones(len(py)), np.vstack(pool_b)])
    dmy = (pc[:, None] == np.arange(k)).astype(float)
    # label-aware: cohort effects estimated with y as protected covariate
    beta = np.linalg.lstsq(np.column_stack([dmy, py]), px, rcond=None)[0]
    zp = marginal_z(px - dmy @ beta[:k], adj, py)
    rows.append(dict(method="pooled_labelaware", rej=bh(2 * stats.norm.sf(np.abs(zp)), ALPHA)))
    # label-blind: plain per-cohort centering
    zb = marginal_z(px - dmy @ np.linalg.lstsq(dmy, px, rcond=None)[0], adj, py)
    rows.append(dict(method="pooled_labelblind", rej=bh(2 * stats.norm.sf(np.abs(zb)), ALPHA)))
    return [dict(pattern=tr["pattern"], rep=rep, n=n, auc=auc, method=r_["method"], R=r_["rej"].sum(),
                 fdp=(r_["rej"] & ~nn).sum() / max(1, r_["rej"].sum()),
                 power=(r_["rej"] & nn).sum() / max(1, nn.sum())) for r_ in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--cohorts", type=int, default=6)
    ap.add_argument("--ns", default="50,100,300")
    ap.add_argument("--aucs", default="0,0.75")
    ap.add_argument("--patterns", default="sparse,coherent")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    t0 = time.time()
    jobs = []
    for pat in args.patterns.split(","):
        tr = T.build_truth(2026, pattern=pat)
        ext = fit_external(external_sample(tr, 2000, np.random.default_rng(99)))
        for auc in [float(v) for v in args.aucs.split(",")]:
            amp = 0.0 if auc == 0 else T.amp_for_auc(tr, auc)
            jobs += [delayed(run_rep)(rep, n, amp, auc, args.cohorts, tr, ext)
                     for n, rep in itertools.product([int(v) for v in args.ns.split(",")], range(args.reps))]
    df = pd.DataFrame(sum(Parallel(n_jobs=args.jobs, verbose=2)(jobs), []))
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"g2_{args.tag}.csv")
    df.to_csv(out, index=False)
    summ = (df.groupby(["pattern", "auc", "n", "method"])
              .agg(fdr=("fdp", "mean"), power=("power", "mean"), R=("R", "mean"), reps=("rep", "count"))
              .reset_index())
    summ.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")
    with pd.option_context("display.max_rows", 500, "display.width", 160):
        print(summ.round(3))


if __name__ == "__main__":
    main()
