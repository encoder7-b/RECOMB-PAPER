"""Calibration among FEASIBLE tests at small n (infeasible modules return NaN and are
counted as non-rejections in the G1 per-test rate). External structure as in the protocol."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim import g1_external as G  # noqa: E402
from sim import truth as T  # noqa: E402

out = {}
for n, k in [(25, 2), (50, 2)]:
    tr = T.build_truth(2026)
    ext = G.fit_one(0, tr, k=k, w_span="both", n_ext=500)
    rng = np.random.default_rng(5 + n)
    z = np.array([G.cohort_pvalues(tr, n, 0.0, rng, ext, ext["nbrs"])[0] for _ in range(400)])
    feas = np.isfinite(z)
    p = 2 * stats.norm.sf(np.abs(z[feas]))
    out[str(n)] = dict(feasible=float(feas.mean()), rate05=float((p < 0.05).mean()), n_tests=int(feas.sum()),
                       ks_p=float(stats.kstest(p, "uniform").pvalue))
    print(n, out[str(n)], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "feasible_calib.json"), "w"), indent=1)
