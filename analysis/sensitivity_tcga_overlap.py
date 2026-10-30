"""Sensitivity analysis: rerun the pre-registered analysis with the one cohort sample flagged
by data/tcga_overlap_check.py (immotion_atezo p25799 vs TCGA-AS-3777-01, rho = 0.50) removed.
Writes results/real_no_tcga_overlap_*."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import run_real  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
flag = pd.read_csv(os.path.join(ROOT, "results", "tcga_overlap_pairs.csv"), dtype=str)
_load = run_real.load_cohorts


def load_without_flagged(names):
    cohorts = _load(names)
    for c in names:
        e, lab = cohorts[c]
        drop = set(flag.loc[flag.cohort == c, "sample"])
        cohorts[c] = (e, lab[~lab["sample"].isin(drop)].reset_index(drop=True))
    return cohorts


run_real.load_cohorts = load_without_flagged
sys.argv = [sys.argv[0], "--real", "--tag", "no_tcga_overlap"]
run_real.main()
