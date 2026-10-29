"""Exact per-hypothesis conditional tests with an external graph and factor directions.

Model (within one cohort, rows i.i.d.):
    X = F Lam^T + batch effects + E,   E ~ N(0, Omega^{-1}),  Omega sparse on graph.
Then for any gene set G,
    E[X_G | X_{-G}] is linear in  [1, batch, X_{N(G)}, X_{-G} W_{-G}],   W = Omega Lam,
with homoscedastic Gaussian error. So conditional on (X_{-G}, Y), X_G follows a
multivariate linear model in a design D that is a function of X_{-G}, batch, and
external quantities (graph N, weights W) only. Under H_G: Y _||_ X_G | X_{-G}, batch,
any residual r = r(Y, X_{-G}, batch) has zero coefficient when added to that
model, and the one-regressor Wilks test is exactly F(g, n - q - g) for every
outcome distribution. Unknown cohort-specific means, coefficients and residual
covariance never need to be estimated.
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LassoLarsIC


def _resid(m, d):
    beta, *_ = np.linalg.lstsq(d, m, rcond=None)
    return m - d @ beta


def conditional_design(x, cols, neighbors, w, covars):
    """Design D for testing gene set `cols`.

    neighbors: gene indices outside `cols` that are graph neighbors of `cols`.
    w: (p, k) external factor weights (Omega @ Lam); rows of `cols` are excluded.
    covars: (n, c) cohort covariates (intercept, batch indicators).
    """
    rest = np.setdiff1d(np.arange(x.shape[1]), cols)
    parts = [covars]
    if len(neighbors):
        parts.append(x[:, neighbors])
    if w is not None and w.shape[1]:
        parts.append(x[:, rest] @ w[rest])
    return np.column_stack(parts)


def wilks_pvalue(xg, d, r, signed=False):
    """Exact test that r has zero coefficient in xg ~ d + r (multivariate OLS).

    signed=True (single column only): return the exact z-score Phi^{-1}(F_t(t)) of the
    coefficient's t statistic instead of a two-sided p-value."""
    n, g = xg.shape
    q = np.linalg.matrix_rank(d)
    df2 = n - q - g
    if df2 < 1:
        return np.nan
    e = _resid(xg, d)
    rr = _resid(r[:, None], d)[:, 0]
    rss = rr @ rr
    if rss < 1e-12:
        return 1.0
    b = rr @ e / rss
    s_full = e.T @ e - rss * np.outer(b, b)
    if np.linalg.eigvalsh(s_full).min() <= 1e-10 * max(1.0, np.trace(e.T @ e)):
        return np.nan  # x_G (numerically) in span of design + r: test undefined
    # 1 - Lambda = rss * b^T S_red^{-1} b ; F = (1-L)/L * df2/g
    t = rss * b @ np.linalg.solve(s_full, b)
    if signed:
        tstat = np.sign(b[0]) * np.sqrt(t * df2)
        return stats.norm.ppf(stats.t.cdf(tstat, df2))
    return stats.f.sf(t * df2 / g, g, df2)


def distilled_residual(y, x, cols, base, screen=100):
    """r = Y minus a lasso (BIC) fit on X_{-G}, after partialling out `base`.

    A function of (Y, X_{-G}, base) only, so the conditional test stays exact; it removes
    signal carried by other genes, which is what leaks into a null module through a
    graph edge that the design misses."""
    rest = np.setdiff1d(np.arange(x.shape[1]), cols)
    yb = _resid(y[:, None], base)[:, 0]
    xb = _resid(x[:, rest], base)
    xb /= xb.std(0) + 1e-12
    keep = np.argsort(-np.abs(xb.T @ yb))[:min(screen, len(rest), max(2, len(y) // 3))]
    fit = LassoLarsIC(criterion="bic", fit_intercept=False).fit(xb[:, keep], yb)
    return yb - xb[:, keep] @ fit.coef_


def test_sets(x, sets, graph_nbrs, w, covars, r, stat="wilks", signed=False, distill=None):
    """p-value for each gene set (list of index arrays).

    stat="wilks": g-df test on all genes of the set.
    stat="mean":  1-df test on the set mean (a fixed projection, so still exact).
    distill=(y, base): replace r by distilled_residual(y, x, G, base) for each set G.
    """
    out = np.empty(len(sets))
    for i, cols in enumerate(sets):
        cols = np.asarray(cols)
        if distill is not None:
            r = distilled_residual(distill[0], x, cols, distill[1])
        nb = np.setdiff1d(np.unique(np.concatenate([graph_nbrs[j] for j in cols])), cols)
        d = conditional_design(x, cols, nb, w, covars)
        xg = x[:, cols] if stat == "wilks" else x[:, cols].mean(1, keepdims=True)
        out[i] = wilks_pvalue(xg, d, r, signed=signed and xg.shape[1] == 1)
    return out


def pc_simes(pmat, r):
    """Partial-conjunction p-value for 'non-null in >= r of K' (Benjamini-Heller, Simes).
    pmat: (m_hyp, K). Valid when cohorts are independent."""
    k = pmat.shape[1]
    m = k - r + 1
    ps = np.sort(np.nan_to_num(pmat, nan=1.0), axis=1)[:, r - 1:]
    return np.min(ps * m / np.arange(1, m + 1), axis=1).clip(max=1.0)


def pc_stouffer(zmat, r):
    """Partial-conjunction p-value with a direction-consistent Stouffer combiner.

    For each sign s, one-sided p_k = 1 - Phi(s z_k); combine the m = K-r+1 LARGEST
    one-sided p-values by Stouffer (valid PC p-value under independent cohorts,
    Benjamini-Heller); two-sided by Bonferroni over the two signs."""
    k = zmat.shape[1]
    m = k - r + 1
    z = np.nan_to_num(zmat, nan=0.0)
    out = []
    for s in (1.0, -1.0):
        p1 = np.sort(stats.norm.sf(s * z), axis=1)[:, r - 1:]
        out.append(stats.norm.sf(stats.norm.isf(p1).sum(1) / np.sqrt(m)))
    return np.minimum(1.0, 2 * np.minimum(*out))


def by(p, alpha):
    """Benjamini-Yekutieli: BH at alpha / H_m, valid under arbitrary dependence."""
    return bh(p, alpha / np.sum(1.0 / np.arange(1, len(p) + 1)))


def bh(p, alpha):
    m = len(p)
    order = np.argsort(p)
    ok = np.where(p[order] <= alpha * np.arange(1, m + 1) / m)[0]
    rej = np.zeros(m, bool)
    if len(ok):
        rej[order[: ok.max() + 1]] = True
    return rej
