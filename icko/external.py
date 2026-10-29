"""Learn dependence structure from unlabeled external expression (e.g. TCGA).

Outputs, all fixed before any cohort is touched:
  scale : per-gene SD in the external data (used to put cohort genes on a common scale)
  nbrs  : inflated, degree-capped graph neighbourhoods of the residual GGM
  w     : (p, k) factor weights  W = Omega_hat @ Lam_hat
"""
import numpy as np
from sklearn.linear_model import LassoLarsIC


def fit_factors(z, k):
    """PCA-based factor loadings on standardized data (principal-factor style)."""
    u, s, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    ev = s ** 2 / (len(z) - 1)
    noise = ev[k:].mean()
    lam = vt[:k].T * np.sqrt(np.clip(ev[:k] - noise, 1e-8, None))
    scores = (z - z.mean(0)) @ vt[:k].T
    resid = (z - z.mean(0)) - scores @ vt[:k]
    return lam, resid


def neighborhood_graph(e, max_deg=6, screen=30, knn=2):
    """Neighbourhood selection (lasso + BIC) on screened candidates, OR-symmetrized,
    unioned with each gene's top-`knn` absolute-correlation partners, then degree-capped
    by keeping the largest |partial-correlation proxy| edges."""
    n, p = e.shape
    c = np.corrcoef(e, rowvar=False)
    np.fill_diagonal(c, 0)
    score = np.zeros((p, p))
    for j in range(p):
        cand = np.argsort(-np.abs(c[j]))[:screen]
        fit = LassoLarsIC(criterion="bic").fit(e[:, cand], e[:, j])
        score[j, cand] = np.abs(fit.coef_)
        top = np.argsort(-np.abs(c[j]))[:knn]
        score[j, top] = np.maximum(score[j, top], 1e-6)
    score = np.maximum(score, score.T)
    adj = score > 0
    for j in range(p):  # degree cap
        nb = np.flatnonzero(adj[j])
        if len(nb) > max_deg:
            drop = nb[np.argsort(-score[j, nb])[max_deg:]]
            adj[j, drop] = False
    adj = adj & adj.T  # keep an edge only if both ends retain it under the cap
    return [np.flatnonzero(adj[j]) for j in range(p)], adj


def precision_on_graph(e, adj):
    """Regression-based precision estimate restricted to the graph."""
    p = e.shape[1]
    omega = np.zeros((p, p))
    for j in range(p):
        nb = np.flatnonzero(adj[j])
        if len(nb):
            beta, *_ = np.linalg.lstsq(e[:, nb], e[:, j], rcond=None)
            res = e[:, j] - e[:, nb] @ beta
        else:
            beta, res = np.zeros(0), e[:, j]
        s2 = res.var()
        omega[j, j] = 1 / s2
        omega[j, nb] = -beta / s2
    return (omega + omega.T) / 2


def fit_external(x_ext, k=4, max_deg=6, w_span="omega_lam"):
    """w_span="omega_lam": W = Omega_hat Lam_hat (k columns).
    w_span="both": W = [Omega_hat Lam_hat, Lam_hat] (2k columns). Over-spanning is the factor
    analogue of graph inflation: exactness needs span(W_hat) to contain span(W)."""
    scale = x_ext.std(0)
    z = (x_ext - x_ext.mean(0)) / scale
    lam, resid = fit_factors(z, k)
    nbrs, adj = neighborhood_graph(resid, max_deg=max_deg)
    omega = precision_on_graph(resid, adj)
    w = omega @ lam if w_span == "omega_lam" else np.hstack([omega @ lam, lam])
    return dict(scale=scale, nbrs=nbrs, adj=adj, w=w, lam=lam)
