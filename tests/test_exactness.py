"""Independent checks of the core conditional test (icko.core.wilks_pvalue).

1. Agreement with independent implementations:
   g = 1 : statsmodels OLS t-test of the coefficient on r.
   g > 1 : statsmodels MANOVA/multivariate OLS (mv_test, Wilks' lambda) on the same design.
2. Monte Carlo exactness of the reference distribution under the conditional null,
   for g in {1, 5, 10} and small n. Y is binary and depends NONLINEARLY on X_{-G},
   and r is built from Y and X_{-G}, as in the method.

Run:  python tests/test_exactness.py  [--reps 20000]
"""
import argparse
import os
import sys

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icko.core import wilks_pvalue  # noqa: E402


def make_null_data(rng, n, g, q_extra=4):
    """X_G = D beta + E with correlated rows of E; Y depends on X_{-G} only."""
    xrest = rng.normal(size=(n, 12))
    batch = (rng.random(n) < 0.5).astype(float)
    d = np.column_stack([np.ones(n), batch, xrest[:, :q_extra]])
    a = rng.normal(size=(g, g))
    e = rng.normal(size=(n, g)) @ (a @ a.T + np.eye(g))
    xg = d @ rng.normal(size=(d.shape[1], g)) + e
    eta = np.sin(xrest[:, 0]) + xrest[:, 5] ** 2 - 1 + 0.8 * batch  # nonlinear, uses genes outside D too
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    base = np.column_stack([np.ones(n), batch, xrest[:, 5]])
    r = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
    return xg, d, r


def check_agreement(rng, trials=200):
    worst = {1: 0.0, 5: 0.0}
    for g in worst:
        for _ in range(trials):
            xg, d, r = make_null_data(rng, 40, g)
            ours = wilks_pvalue(xg, d, r)
            design = np.column_stack([d, r])
            if g == 1:
                ref = sm.OLS(xg[:, 0], design).fit().pvalues[-1]
            else:
                names = [f"x{i}" for i in range(design.shape[1])]
                res = _MultivariateOLS(xg, design).fit()
                contrast = np.zeros((1, design.shape[1]))
                contrast[0, -1] = 1
                ref = res.mv_test([("r", contrast)]).results["r"]["stat"].loc["Wilks' lambda", "Pr > F"]
                del names
            worst[g] = max(worst[g], abs(np.log(ours) - np.log(ref)))
    return worst


def check_exactness(rng, reps):
    out = []
    for n, g in [(15, 1), (25, 1), (25, 5), (50, 10), (100, 10)]:
        p = np.array([wilks_pvalue(*make_null_data(rng, n, g)) for _ in range(reps)])
        rej = {a: (p < a).mean() for a in (0.05, 0.01)}
        ci = {a: 1.96 * np.sqrt(a * (1 - a) / reps) for a in rej}
        ks = stats.kstest(p, "uniform").pvalue
        out.append((n, g, rej, ci, ks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261029)
    w = check_agreement(rng)
    print("max |log p_ours - log p_statsmodels|:", {g: f"{v:.2e}" for g, v in w.items()})
    assert max(w.values()) < 1e-6, "disagreement with independent implementation"
    print(f"\nexactness under the conditional null ({args.reps} reps per row)")
    print("   n   g   rej@0.05 (±95%)     rej@0.01 (±95%)     KS p")
    rows = []
    for n, g, rej, ci, ks in check_exactness(rng, args.reps):
        print(f"{n:4d} {g:3d}   {rej[0.05]:.4f} (±{ci[0.05]:.4f})   {rej[0.01]:.4f} (±{ci[0.01]:.4f})   {ks:.3f}")
        rows.append(dict(n=n, g=g, rej05=rej[0.05], ci05=ci[0.05], rej01=rej[0.01], ci01=ci[0.01], ks_p=ks,
                         reps=args.reps))
    import pandas as pd
    out = os.path.join(os.path.dirname(__file__), "..", "results", "exactness")
    pd.DataFrame(rows).to_csv(out + "_summary.csv", index=False)
    pd.DataFrame([dict(g=g, max_abs_log_p_diff=v) for g, v in w.items()]).to_csv(out + "_agreement.csv", index=False)


if __name__ == "__main__":
    main()
