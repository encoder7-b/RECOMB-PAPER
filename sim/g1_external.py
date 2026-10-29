"""G1: same tests, but graph + factor weights learned from an unlabeled external sample.

Stressors:
  tau      : cohort factor loadings = external loadings * (1 + tau * N(0,1)) per gene
             (external population differs from the cohorts)
  drop     : fraction of learned graph edges deleted at random (graph error)
  amp = 0  : global null, for calibration
"""
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
from icko.external import fit_external  # noqa: E402
from sim import truth as T  # noqa: E402

ALPHA = 0.1
N_EXT = 2000
N_EXT_FITS = 4


def external_sample(tr, n, rng):
    f = rng.normal(size=(n, 4))
    return f @ tr["lam"].T + rng.normal(size=(n, T.P)) @ tr["chol_e"].T


def fit_one(seed, tr, k=4, w_span="omega_lam", n_ext=N_EXT):
    return fit_external(external_sample(tr, n_ext, np.random.default_rng(seed)), k=k, w_span=w_span)


def drop_edges(ext, frac, rng, true_adj=None):
    """Delete a fraction of learned edges; if true_adj is given, delete only TRUE edges."""
    if frac == 0:
        return ext["nbrs"]
    adj = ext["adj"].copy()
    if true_adj is not None:  # only TRUE edges between different modules (the ones module tests use)
        mod_of = np.arange(adj.shape[0]) // T.MOD
        pool = adj & true_adj & (mod_of[:, None] != mod_of[None, :])
    else:
        pool = adj
    iu = np.argwhere(np.triu(pool, 1))
    kill = iu[rng.random(len(iu)) < frac]
    adj[kill[:, 0], kill[:, 1]] = adj[kill[:, 1], kill[:, 0]] = False
    return [np.flatnonzero(adj[j]) for j in range(adj.shape[0])]


def cohort_pvalues(tr_k, n, amp, rng, ext, nbrs, distill=False):
    x, y, batch, score_b = T.simulate_cohort(tr_k, n, amp, rng)
    x = x / ext["scale"]
    covars = np.column_stack([np.ones(n), batch])
    base = np.column_stack([covars, score_b])
    beta, *_ = np.linalg.lstsq(base, y, rcond=None)
    r = y - base @ beta
    dist = (y, base) if distill else None
    mm = test_sets(x, T.module_sets(), nbrs, ext["w"], covars, r, stat="mean", signed=True, distill=dist)
    mw = test_sets(x, T.module_sets(), nbrs, ext["w"], covars, r, stat="wilks", distill=dist)
    return mm, mw


def perturbed_truth(tr, tau, rng):
    if tau == 0:
        return tr
    t2 = dict(tr)
    t2["lam"] = tr["lam"] * (1 + tau * rng.normal(size=(T.P, 1)))
    return t2


def one_rep(rep, n, amp, auc, tau, drop, k, tr, exts, drop_true=False, distill=False):
    rng = np.random.default_rng(7_919 * rep + 31 * n + int(1e4 * amp) + int(100 * tau) + int(1000 * drop))
    ext = exts[rep % len(exts)]
    true_adj = ((tr["omega"] != 0) & ~np.eye(T.P, dtype=bool)) if drop_true else None
    nbrs = drop_edges(ext, drop, rng, true_adj)
    per = [cohort_pvalues(perturbed_truth(tr, tau, rng), n, amp, rng, ext, nbrs, distill) for _ in range(k)]
    nn = np.isin(np.arange(T.N_TEST_MOD), tr["sig_mods"]) if amp > 0 else np.zeros(T.N_TEST_MOD, bool)
    rows = []
    for li, level in enumerate(["module_mean", "module_wilks"]):
        pm = np.column_stack([c[li] for c in per])
        signed = level == "module_mean"
        if signed:
            zm, pm = pm, 2 * stats.norm.sf(np.abs(pm))
        base = dict(pattern=tr["pattern"], rep=rep, n=n, auc=auc, tau=tau, drop=drop,
                    drop_true=drop_true, distill=distill, level=level)
        for r in range(1, k + 1):
            combs = {"simes": pc_simes(pm, r)}
            if signed:
                combs["stouffer"] = pc_stouffer(zm, r)
            for comb, pc in combs.items():
                rej = bh(pc, ALPHA)
                rows.append(dict(base, comb=comb, r=r, R=rej.sum(), fdp=(rej & ~nn).sum() / max(1, rej.sum()),
                                 power=(rej & nn).sum() / max(1, nn.sum())))
        rows.append(dict(base, comb="none", r=0, R=np.nan, fdp=np.nanmean(pm[~nn] < 0.05),
                         power=np.nanmean(pm[nn] < 0.05) if nn.any() else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--cohorts", type=int, default=6)
    ap.add_argument("--ns", default="50,100,300")
    ap.add_argument("--aucs", default="0,0.75")
    ap.add_argument("--taus", default="0,0.2")
    ap.add_argument("--drops", default="0,0.1,0.2")
    ap.add_argument("--patterns", default="sparse,coherent")
    ap.add_argument("--drop-true", action="store_true", help="delete true edges only")
    ap.add_argument("--distill", action="store_true", help="lasso-distilled outcome residual")
    ap.add_argument("--k", type=int, default=4, help="external factors")
    ap.add_argument("--n-ext", type=int, default=N_EXT, help="external (unlabeled) sample size")
    ap.add_argument("--w-span", default="omega_lam", choices=["omega_lam", "both"])
    ap.add_argument("--cross-edges", type=int, default=0, help="true edges between modules")
    ap.add_argument("--cross-pc", type=float, default=0.15, help="partial correlation of cross edges")
    ap.add_argument("--adversarial", action="store_true", help="cross edges join signal genes to null modules")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    fl = lambda s: [float(v) for v in s.split(",")]  # noqa: E731

    t0 = time.time()
    jobs = []
    for pat in args.patterns.split(","):
        tr = T.build_truth(2026, pattern=pat, cross_edges=args.cross_edges,
                           cross_pc=args.cross_pc, adversarial=args.adversarial)
        exts = Parallel(n_jobs=args.jobs)(delayed(fit_one)(s, tr, args.k, args.w_span, args.n_ext)
                                          for s in range(N_EXT_FITS))
        deg = np.mean([len(v) for v in exts[0]["nbrs"]])
        true_adj = (tr["omega"] != 0) & ~np.eye(T.P, dtype=bool)
        recall = (exts[0]["adj"] & true_adj).sum() / true_adj.sum()
        print(f"{pat}: external graph mean degree {deg:.2f}, true-edge recall {recall:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        for auc in fl(args.aucs):
            amp = 0.0 if auc == 0 else T.amp_for_auc(tr, auc)
            for n, tau, drop, rep in itertools.product(
                    [int(v) for v in args.ns.split(",")], fl(args.taus), fl(args.drops), range(args.reps)):
                jobs.append(delayed(one_rep)(rep, n, amp, auc, tau, drop, args.cohorts, tr, exts,
                                             args.drop_true, args.distill))
    df = pd.DataFrame(sum(Parallel(n_jobs=args.jobs, verbose=2)(jobs), []))
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"g1_{args.tag}.csv")
    df.to_csv(out, index=False)
    df["cross_edges"] = args.cross_edges
    df["k"], df["w_span"], df["n_ext"] = args.k, args.w_span, args.n_ext
    summ = (df.groupby(["pattern", "level", "comb", "auc", "tau", "drop", "n", "r"])
              .agg(fdr=("fdp", "mean"), power=("power", "mean"), R=("R", "mean"),
                   reps=("rep", "count")).reset_index())
    summ.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")
    with pd.option_context("display.max_rows", 500, "display.width", 160):
        print(summ[summ.r.isin([0, 1, 2])].round(3))


if __name__ == "__main__":
    main()
