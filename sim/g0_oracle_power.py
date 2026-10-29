"""Gate G0: oracle-knockoff power for multi-cohort partial-conjunction discovery.

Knockoffs here use the TRUE conditional model X | (F, batch), so this is an
upper bound on what any conditional (graph-only) construction can achieve.
If power is near zero here, no external-graph method will rescue it.

Truth per cohort:
  X = F @ Lam.T + batch * delta + E,  E ~ N(0, Sigma_E)
  Sigma_E = block AR(1) modules + one weak unmodelled factor (dense part)
  logit P(Y=1) = b0 + 0.8*F1 + 0.5*batch + sum_{j in S} a_k * s_j * X_j
F (k=3 factors) and batch are treated as known conditioning covariates.

Outputs results/g0_*.csv with FDR / power by (n, amplitude, level, r).
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cholesky, eigh
from sklearn.linear_model import lars_path
from sklearn.metrics import roc_auc_score

P = 500
MOD = 10
N_MOD = P // MOD
K_FAC = 3
RHO = 0.6
ALPHA = 0.1
AKNS = (0.05, 0.1, 0.2, 0.3)  # per-draw knockoff level; any fixed value gives valid e-values


def build_truth(seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(MOD)
    ar = RHO ** np.abs(idx[:, None] - idx[None, :])
    sig = np.kron(np.eye(N_MOD), ar)
    v = rng.normal(0, 0.3, P) * (rng.random(P) < 0.5)  # unmodelled weak factor
    sig = sig + np.outer(v, v)
    d = np.sqrt(np.diag(sig))
    sig = sig / np.outer(d, d)

    lam = np.zeros((P, K_FAC))
    for f, (lo, hi, load) in enumerate([(0, 15, 0.6), (15, 25, 0.5), (25, 35, 0.5)]):
        lam[lo * MOD:hi * MOD, f] = load

    # 10 signal modules: 5 factor-loaded (immune-like), 5 unloaded; 2 genes each
    sig_mods = list(rng.choice(15, 5, replace=False)) + list(rng.choice(np.arange(35, 50), 5, replace=False))
    signal = np.concatenate([m * MOD + rng.choice(MOD, 2, replace=False) for m in sig_mods])
    signs = rng.choice([-1.0, 1.0], len(signal))
    return dict(sig=sig, lam=lam, signal=signal, signs=signs, sig_mods=np.array(sig_mods))


def _max_gamma(sig, s_mat):
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if eigh(2 * sig - mid * s_mat, eigvals_only=True, subset_by_index=[0, 0])[0] > 1e-6:
            lo = mid
        else:
            hi = mid
    return lo


def knockoff_params(sig, level):
    """Return (A, C) with Xk = mu + (X-mu) @ A + Z @ C."""
    if level == "gene":
        # per-module equicorrelated diag, then global scaling (ASDP-style)
        s0 = np.empty(P)
        for m in range(N_MOD):
            b = slice(m * MOD, (m + 1) * MOD)
            s0[b] = min(1.0, 2 * eigh(sig[b, b], eigvals_only=True)[0])
        s_mat = np.diag(s0)
    else:
        s_mat = np.zeros_like(sig)
        for m in range(N_MOD):
            b = slice(m * MOD, (m + 1) * MOD)
            s_mat[b, b] = sig[b, b]
    s_mat = _max_gamma(sig, s_mat) * s_mat * 0.999
    sinv_s = np.linalg.solve(sig, s_mat)
    cov = 2 * s_mat - s_mat @ sinv_s
    cov = (cov + cov.T) / 2 + 1e-8 * np.eye(P)
    return np.eye(P) - sinv_s, cholesky(cov, lower=False)


def simulate_cohort(truth, n, amp, rng):
    f = rng.normal(size=(n, K_FAC))
    batch = (rng.random(n) < 0.5).astype(float)
    delta = rng.normal(0, 0.5, P)
    e = rng.multivariate_normal(np.zeros(P), truth["sig"], size=n, method="cholesky")
    mu = f @ truth["lam"].T + batch[:, None] * delta
    x = mu + e
    a_k = amp * (1 + 0.25 * rng.normal())
    eta = -0.5 + 0.8 * f[:, 0] + 0.5 * batch + a_k * x[:, truth["signal"]] @ truth["signs"]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    cov = np.column_stack([np.ones(n), f, batch])
    return x, mu, y, cov


def entry_lambdas(x, y):
    alphas, _, coefs = lars_path(x, y, method="lasso", max_iter=x.shape[0])
    nz = coefs != 0
    first = np.where(nz.any(1), nz.argmax(1), -1)
    return np.where(first >= 0, alphas[np.clip(first, 0, None)], 0.0)


def residualize(m, cov):
    beta, *_ = np.linalg.lstsq(cov, m, rcond=None)
    return m - cov @ beta


def knockoff_evalues(w, n_tests, akn):
    ts = np.sort(np.abs(w[w != 0]))
    for t in ts:
        if (1 + np.sum(w <= -t)) / max(1, np.sum(w >= t)) <= akn:
            return n_tests * (w >= t) / (1 + np.sum(w <= -t))
    return np.zeros_like(w)


def cohort_evalues(truth, kparams, n, amp, m_draws, rng):
    x, mu, y, cov = simulate_cohort(truth, n, amp, rng)
    out, any_sel = {}, {}
    for level, (a_mat, c_mat) in kparams.items():
        ev = {akn: np.zeros(P if level == "gene" else N_MOD) for akn in AKNS}
        hits = {akn: 0 for akn in AKNS}
        for _ in range(m_draws):
            xk = mu + (x - mu) @ a_mat + rng.normal(size=(n, P)) @ c_mat
            design = residualize(np.hstack([x, xk]), cov)
            design /= np.linalg.norm(design, axis=0) + 1e-12
            z = entry_lambdas(design, residualize(y, cov))
            z, zk = z[:P], z[P:]
            if level == "module":
                z = z.reshape(N_MOD, MOD).max(1)
                zk = zk.reshape(N_MOD, MOD).max(1)
            w = np.maximum(z, zk) * np.sign(z - zk)
            for akn in AKNS:
                e = knockoff_evalues(w, len(w), akn)
                hits[akn] += e.any()
                ev[akn] += e / m_draws
        for akn in AKNS:
            out[(level, akn)] = ev[akn]
            any_sel[(level, akn)] = hits[akn] / m_draws
    return out, any_sel


def ebh(e, alpha):
    p = len(e)
    order = np.argsort(-e)
    ks = np.arange(1, p + 1)
    ok = np.where(e[order] >= p / (alpha * ks))[0]
    rej = np.zeros(p, bool)
    if len(ok):
        rej[order[: ok.max() + 1]] = True
    return rej


def one_rep(rep, n, amp, n_cohorts, m_draws, truth, kparams):
    rng = np.random.default_rng(10_000 * rep + 17 * n + int(100 * amp))
    per = [cohort_evalues(truth, kparams, n, amp, m_draws, rng) for _ in range(n_cohorts)]
    rows = []
    nonnull = {"gene": np.isin(np.arange(P), truth["signal"]),
               "module": np.isin(np.arange(N_MOD), truth["sig_mods"])}
    for level, akn in itertools.product(kparams, AKNS):
        e_mat = np.column_stack([c[0][(level, akn)] for c in per])
        sel_rate = np.mean([c[1][(level, akn)] for c in per])
        for r in range(1, n_cohorts + 1):
            m = n_cohorts - r + 1
            comb = np.sort(e_mat, axis=1)[:, :m].mean(1)
            rej = ebh(comb, ALPHA * m / n_cohorts)
            nn = nonnull[level]
            rows.append(dict(rep=rep, n=n, amp=amp, level=level, akn=akn, r=r, R=rej.sum(),
                             fdp=(rej & ~nn).sum() / max(1, rej.sum()),
                             power=(rej & nn).sum() / nn.sum(), draw_any_sel=sel_rate))
    return rows


def oracle_auc(truth, amp, rng):
    x, _, y, _ = simulate_cohort(truth, 20_000, amp, rng)
    f_eta = x[:, truth["signal"]] @ truth["signs"]
    return roc_auc_score(y, f_eta)  # AUC of the gene part only (no F, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--cohorts", type=int, default=6)
    ap.add_argument("--ns", default="25,50,100,300")
    ap.add_argument("--amps", default="0.15,0.3,0.5")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    truth = build_truth(2026)
    kparams = {lv: knockoff_params(truth["sig"], lv) for lv in ("gene", "module")}
    ns = [int(v) for v in args.ns.split(",")]
    amps = [float(v) for v in args.amps.split(",")]

    rng = np.random.default_rng(1)
    aucs = {a: oracle_auc(truth, a, rng) for a in amps}
    print("gene-part AUC by amplitude:", {a: round(v, 3) for a, v in aucs.items()}, flush=True)

    t0 = time.time()
    jobs = [delayed(one_rep)(rep, n, a, args.cohorts, args.draws, truth, kparams)
            for n, a, rep in itertools.product(ns, amps, range(args.reps))]
    rows = sum(Parallel(n_jobs=args.jobs, verbose=5)(jobs), [])
    df = pd.DataFrame(rows)
    df["gene_auc"] = df["amp"].map(aucs)
    out = os.path.join(os.path.dirname(__file__), "..", "results", f"g0_{args.tag}.csv")
    df.to_csv(out, index=False)
    summ = (df.groupby(["level", "akn", "n", "amp", "r"])
              .agg(fdr=("fdp", "mean"), power=("power", "mean"), R=("R", "mean"),
                   draw_any_sel=("draw_any_sel", "mean")).reset_index())
    summ.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")
    with pd.option_context("display.max_rows", 500, "display.width", 160):
        print(summ[summ.r.isin([1, 2, 3])])


if __name__ == "__main__":
    main()
