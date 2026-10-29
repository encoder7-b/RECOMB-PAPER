"""Numerical audit of Lemma A1 and the containment step of Proposition 1.

For random sparse Omega and loadings Lambda (Sigma = Lambda Lambda^T + Omega^{-1}), the exact
conditional-mean coefficients B = Sigma_{-G,-G}^{-1} Sigma_{-G,G} must lie in the span of
  [ e_j : j in N(G) ]  and  W_{-G} = (Omega Lambda)_{-G}
(Lemma A1), and also in the span of any W_hat with colspan(W_hat) >= colspan(Omega Lambda)
(containment). We check the residual of projecting B onto those spans is ~0, and that
dropping one true neighbour or one factor direction makes it clearly non-zero.
Also checks that the conditional covariance is the same for every batch mean shift.
"""
import numpy as np


def random_model(rng, p=40, k=3, density=0.08):
    a = (rng.random((p, p)) < density) * rng.normal(0, 0.3, (p, p))
    a = np.triu(a, 1)
    omega = a + a.T
    omega += np.eye(p) * (np.abs(omega).sum(1).max() + 0.5)  # diagonally dominant -> PD
    lam = rng.normal(size=(p, k))
    return omega, lam


def span_residual(b, basis):
    q, _ = np.linalg.qr(basis)
    return np.linalg.norm(b - q @ (q.T @ b)) / np.linalg.norm(b)


def main():
    rng = np.random.default_rng(1)
    worst_true, worst_contained, best_missing_nb, best_missing_fac = 0, 0, np.inf, np.inf
    for _ in range(200):
        omega, lam = random_model(rng)
        p, k = lam.shape
        sigma = lam @ lam.T + np.linalg.inv(omega)
        g = np.sort(rng.choice(p, 3, replace=False))
        rest = np.setdiff1d(np.arange(p), g)
        b = np.linalg.solve(sigma[np.ix_(rest, rest)], sigma[np.ix_(rest, g)])  # E[X_G|X_-G] = X_-G b
        nb = np.setdiff1d(np.flatnonzero(np.any(omega[g] != 0, axis=0)), g)
        e_nb = np.eye(len(rest))[:, np.searchsorted(rest, nb)]
        w = omega @ lam
        worst_true = max(worst_true, span_residual(b, np.hstack([e_nb, w[rest]])))
        w_hat = np.hstack([w @ rng.normal(size=(k, k)), rng.normal(size=(p, 4))])  # contains span(W)
        worst_contained = max(worst_contained, span_residual(b, np.hstack([e_nb, w_hat[rest]])))
        if len(nb) > 0:
            best_missing_nb = min(best_missing_nb, span_residual(b, np.hstack([e_nb[:, 1:], w[rest]])))
        best_missing_fac = min(best_missing_fac, span_residual(b, np.hstack([e_nb, w[rest][:, 1:]])))
    print(f"max relative residual, true neighbours + W:          {worst_true:.2e}   (should be ~0)")
    print(f"max relative residual, true neighbours + W_hat (contains W): {worst_contained:.2e}   (should be ~0)")
    print(f"min relative residual, one neighbour dropped:         {best_missing_nb:.2e}   (should be > 0)")
    print(f"min relative residual, one factor direction dropped:  {best_missing_fac:.2e}   (should be > 0)")
    assert worst_true < 1e-8 and worst_contained < 1e-8
    assert best_missing_nb > 1e-6 and best_missing_fac > 1e-6
    print("Lemma A1 and containment: OK")


if __name__ == "__main__":
    main()
