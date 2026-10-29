"""Sensitivity analysis: untested cohorts dropped from each module's combination
(pc_stouffer_feasible) instead of z = 0 (pre-registered). Uses the saved real-data z-scores."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from icko.core import bh, by, pc_stouffer, pc_stouffer_feasible  # noqa: E402

DISC = ["hugo", "riaz", "gide_pd1", "gide_combo", "liu2019", "jung", "gse176307", "immotion_atezo"]
m = pd.read_csv(os.path.join(ROOT, "results", "real_real_modules.csv"))
z = m[[f"z_{c}" for c in DISC]].values
out = {"n_untested_cells": int(np.isnan(z).sum()), "n_cells": int(z.size)}
for r in (1, 2):
    p0, p1 = pc_stouffer(z, r), pc_stouffer_feasible(z, r)
    out[f"r{r}"] = {"prereg_zero": {"bh": int(bh(p0, 0.1).sum()), "by": int(by(p0, 0.1).sum()), "min_p": float(p0.min())},
                    "drop_untested": {"bh": int(bh(p1, 0.1).sum()), "by": int(by(p1, 0.1).sum()), "min_p": float(p1.min())}}
json.dump(out, open(os.path.join(ROOT, "results", "sensitivity_feasible.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
