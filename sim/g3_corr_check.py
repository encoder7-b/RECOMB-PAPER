"""Measure Corr(r, U | D) for each omitted factor direction in G3 (Proposition 4's driver)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icko.core import _resid, conditional_design  # noqa: E402
from sim import truth as T  # noqa: E402

rows = []
for pat in ("sparse", "coherent"):
    tr = T.build_truth(2026, pattern=pat)
    amp = T.amp_for_auc(tr, 0.8)
    rng = np.random.default_rng(3)
    null_mods = [m for i, m in enumerate(T.module_sets()) if i not in set(tr["sig_mods"])]
    for name, col in [("omit_related (F1)", 0), ("omit_unrelated (F2)", 1), ("omit_diffuse (F4)", 3)]:
        w_kept = np.delete(tr["w"], col, axis=1)
        cs = []
        for _ in range(20):
            x, y, batch, score_b = T.simulate_cohort(tr, 300, amp, rng)
            covars = np.column_stack([np.ones(300), batch])
            base = np.column_stack([covars, score_b])
            r = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
            for cols in null_mods:
                rest = np.setdiff1d(np.arange(T.P), cols)
                nb = np.setdiff1d(np.unique(np.concatenate([tr["nbrs"][j] for j in cols])), cols)
                d = conditional_design(x, cols, nb, w_kept, covars)
                u = x[:, rest] @ tr["w"][rest, col]
                cs.append(abs(np.corrcoef(_resid(r[:, None], d)[:, 0], _resid(u[:, None], d)[:, 0])[0, 1]))
        rows.append(dict(pattern=pat, case=name, mean_abs_corr_r_U_given_D=np.mean(cs)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "g3_corr_check.csv"), index=False)
print(df.round(4).to_string(index=False))
print("null reference (|corr| of independent n=300 vectors) ~", round(np.sqrt(2 / np.pi / 300), 4))
