"""G4: dependent cohorts. A fraction f of each cohort's patients are copies (X and Y) of
distinct patients in the previous cohort (adjacent cohorts share f*n patients; no chaining),
which violates the cohort-independence assumption of Lemma 1. Batch shifts are set to zero so
the copied rows match their new cohort's model: this isolates dependence from batch mismatch.
Oracle structure, K = 6, BH at 0.1, r = 1 and r = 2. Global null and signal (AUC 0.8).
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icko.core import bh, pc_stouffer, test_sets  # noqa: E402
from sim import truth as T  # noqa: E402

K, N = 6, 100


def cohort_z(tr, x, y, batch, score_b):
    covars = np.column_stack([np.ones(len(y)), batch])
    base = np.column_stack([covars, score_b])
    r = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
    return test_sets(x, T.module_sets(), tr["nbrs"], tr["w"], covars, r, stat="mean", signed=True)


def one_rep(rep, frac, amp, auc, tr):
    rng = np.random.default_rng(32_452_843 * rep + int(1000 * frac) + int(1e4 * amp))
    own = [T.simulate_cohort(tr, N, amp, rng, batch_shift=0.0) for _ in range(K)]
    n_dup = int(round(frac * N))
    cohorts = [own[0]]
    for k in range(1, K):  # rows [0, n_dup) of cohort k are copies of rows [n_dup, 2 n_dup) of cohort k-1
        cohorts.append(tuple(np.concatenate([p[n_dup:2 * n_dup], c[n_dup:]]) for p, c in zip(own[k - 1], own[k])))
    z = np.column_stack([cohort_z(tr, *c) for c in cohorts])
    nn = np.isin(np.arange(T.N_TEST_MOD), tr["sig_mods"]) if amp > 0 else np.zeros(T.N_TEST_MOD, bool)
    out = []
    for r in (1, 2):
        rej = bh(pc_stouffer(z, r), 0.1)
        out.append(dict(rep=rep, frac=frac, auc=auc, r=r, fdp=(rej & ~nn).sum() / max(1, rej.sum()),
                        power=(rej & nn).sum() / max(1, nn.sum())))
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    tr = T.build_truth(2026, pattern="coherent")
    jobs = []
    for auc in (0.0, 0.8):
        amp = 0.0 if auc == 0 else T.amp_for_auc(tr, auc)
        jobs += [delayed(one_rep)(rep, f, amp, auc, tr)
                 for f, rep in itertools.product([0.0, 0.1, 0.25, 0.5], range(reps))]
    df = pd.DataFrame(sum(Parallel(n_jobs=10, verbose=2)(jobs), []))
    g = df.groupby(["auc", "r", "frac"])
    s = g.agg(fdr=("fdp", "mean"), power=("power", "mean"), reps=("rep", "count")).reset_index()
    s["fdr_se"] = g.fdp.std().values / np.sqrt(s.reps)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "g4_shared_patients")
    df.to_csv(out + ".csv", index=False)
    s.to_csv(out + "_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
