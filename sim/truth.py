"""Shared simulation truth: factor + sparse-GGM expression, batch, binary response.

Genes 0..499 are the testing set A (50 modules of 10). Genes 500..519 are the
marker set B (2 modules), whose mean is an observed infiltration score that
drives response. B genes are never tested.
"""
import numpy as np

MOD = 10
N_TEST_MOD = 50
N_MARK_MOD = 2
P_TEST = MOD * N_TEST_MOD
P = MOD * (N_TEST_MOD + N_MARK_MOD)
B_IDX = np.arange(P_TEST, P)
RHO = 0.6


def module_sets():
    return [np.arange(m * MOD, (m + 1) * MOD) for m in range(N_TEST_MOD)]


def _add_cross_edges(omega, rng, count, pc, pick):
    """Add `count` cross-module edges with partial correlation ~pc, keeping omega PD."""
    mod_of = np.arange(P) // MOD
    added = 0
    while added < count:
        i, j = pick()
        if mod_of[i] == mod_of[j] or omega[i, j] != 0:
            continue
        omega[i, j] = omega[j, i] = -pc * np.sqrt(omega[i, i] * omega[j, j])
        if np.linalg.eigvalsh(omega).min() <= 0.02:
            omega[i, j] = omega[j, i] = 0
            continue
        added += 1


def build_truth(seed, n_signal_mods=10, pattern="sparse", cross_edges=0, cross_pc=0.15, adversarial=False):
    """pattern="sparse": 2 genes per signal module, random signs.
    pattern="coherent": all genes of a signal module, one sign per module.
    cross_edges: true edges between modules. Random pairs by default (drawn before the
    loadings, as in the original runs); adversarial=True instead joins a SIGNAL gene to a
    gene in a NULL module, drawn after the signal (worst case for a missing edge)."""
    rng = np.random.default_rng(seed)
    n_mod = N_TEST_MOD + N_MARK_MOD
    # AR(1) residual within module: tridiagonal precision
    prec_blk = np.zeros((MOD, MOD))
    np.fill_diagonal(prec_blk, (1 + RHO ** 2) / (1 - RHO ** 2))
    prec_blk[0, 0] = prec_blk[-1, -1] = 1 / (1 - RHO ** 2)
    i = np.arange(MOD - 1)
    prec_blk[i, i + 1] = prec_blk[i + 1, i] = -RHO / (1 - RHO ** 2)
    omega = np.kron(np.eye(n_mod), prec_blk)
    if cross_edges and not adversarial:
        _add_cross_edges(omega, rng, cross_edges, cross_pc, lambda: rng.choice(P, 2, replace=False))

    lam = np.zeros((P, 4))
    immune = np.r_[np.arange(0, 15 * MOD), B_IDX]
    lam[immune, 0] = 0.6
    lam[15 * MOD:25 * MOD, 1] = 0.5
    lam[25 * MOD:35 * MOD, 2] = 0.5
    lam[rng.random(P) < 0.5, 3] = 0.3  # weak diffuse factor

    sig_mods = np.r_[rng.choice(15, n_signal_mods // 2, replace=False),
                     rng.choice(np.arange(35, 50), n_signal_mods - n_signal_mods // 2, replace=False)]
    if pattern == "sparse":
        signal = np.concatenate([m * MOD + rng.choice(MOD, 2, replace=False) for m in sig_mods])
        signs = rng.choice([-1.0, 1.0], len(signal))
    else:
        signal = np.concatenate([np.arange(m * MOD, (m + 1) * MOD) for m in sig_mods])
        signs = np.repeat(rng.choice([-1.0, 1.0], len(sig_mods)), MOD)
    if cross_edges and adversarial:
        null_genes = np.setdiff1d(np.arange(P_TEST),
                                  np.concatenate([np.arange(m * MOD, (m + 1) * MOD) for m in sig_mods]))
        _add_cross_edges(omega, rng, cross_edges, cross_pc,
                         lambda: (rng.choice(signal), rng.choice(null_genes)))

    nbrs = [np.flatnonzero((omega[j] != 0) & (np.arange(P) != j)) for j in range(P)]
    sig_e = np.linalg.inv(omega)
    return dict(pattern=pattern, omega=omega, sig_e=sig_e, chol_e=np.linalg.cholesky(sig_e), lam=lam,
                w=omega @ lam, nbrs=nbrs, signal=signal, signs=signs, sig_mods=np.sort(sig_mods))


def simulate_cohort(truth, n, amp, rng, infil=1.0, batch_shift=0.5, batch_resp=0.5, het=0.25, b0=-0.5):
    f = rng.normal(size=(n, 4))
    batch = (rng.random(n) < 0.5).astype(float)
    delta = rng.normal(0, batch_shift, P)
    x = f @ truth["lam"].T + batch[:, None] * delta + rng.normal(size=(n, P)) @ truth["chol_e"].T
    score_b = x[:, B_IDX].mean(1) / 0.6  # roughly unit-scale infiltration score
    a_k = amp * (1 + het * rng.normal())
    eta = b0 + infil * score_b + batch_resp * batch + a_k * x[:, truth["signal"]] @ truth["signs"]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return x, y, batch, score_b


def gene_part_auc(truth, amp, seed=1):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    x, y, _, _ = simulate_cohort(truth, 20_000, amp, rng)
    return roc_auc_score(y, x[:, truth["signal"]] @ truth["signs"])


def amp_for_auc(truth, target, lo=0.0, hi=1.0, iters=14):
    """Bisection for the effect amplitude giving a target gene-part AUC."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if gene_part_auc(truth, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
