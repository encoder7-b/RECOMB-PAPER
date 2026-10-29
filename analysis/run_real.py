"""Real-data analysis, following docs/PROTOCOL.md. Everything before the tests is label-free.

  1. gene universe: top-variance genes in TCGA (per-type centred), present in every cohort
  2. marker set B : pre-specified T-effector / IFN-gamma genes (never tested)
  3. modules      : clusters of the pooled-TCGA factor-residual correlation (external only)
  4. structure    : per-cohort external fit on the matched TCGA tumour type
  5. feasibility  : a module is tested only if residual df >= MIN_DF and at least half
                    of its genes vary in that cohort (untested -> z = 0 in Stouffer)
  6. discovery    : exact conditional module-mean z per discovery cohort -> Stouffer PC -> BH
  7. audit        : same z in the sealed cohorts; replication vs random module sets and baselines
  8. controls     : same z in the non-ICI arms (descriptive)

--permute shuffles labels within every cohort (dry run).
--real runs on true labels; it refuses unless docs/PROTOCOL.md is committed and every
label file still matches the SHA-256 recorded in it.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from icko.core import bh, conditional_design, pc_simes, pc_stouffer, wilks_pvalue  # noqa: E402
from icko.external import fit_external  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
TCGA_TYPES = ("skcm", "stad", "luad", "lusc", "blca", "kirc")
MATCH = {"hugo": ["skcm"], "riaz": ["skcm"], "jung": ["luad", "lusc"], "gse176307": ["blca"],
         "imvigor210": ["blca"], "gide_pd1": ["skcm"], "gide_combo": ["skcm"], "liu2019": ["skcm"],
         "immotion_atezo": ["kirc"], "braun_nivo": ["kirc"],
         "immotion_sunitinib": ["kirc"], "braun_evero": ["kirc"]}
DISCOVERY = ["hugo", "riaz", "gide_pd1", "gide_combo", "liu2019", "jung", "gse176307", "immotion_atezo"]
SEALED = ["imvigor210", "braun_nivo"]
CONTROLS = ["immotion_sunitinib", "braun_evero"]  # non-ICI arms: prognostic-vs-predictive check only
ALL = DISCOVERY + SEALED + CONTROLS
MARKERS = ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "CXCL9", "CXCL10", "IFNG", "IDO1", "STAT1",
           "HLA-DRA", "CD274", "LAG3", "CD3E", "CD2"]
N_GENES, K_FAC, MAX_DEG, MIN_DF, MOD_MAX, MOD_MIN = 1500, 5, 3, 8, 12, 4
W_SPAN = "both"  # over-span the factor directions (G1-adversarial: removes estimation-error inflation)
ALPHA, N_RANDOM_SETS = 0.1, 1000


def k_for(n):
    """Pre-specified n-adaptive factor count: more over-factoring where df allows."""
    return int(np.clip(n // 25, 2, 8))


def load_tcga():
    return {t: pd.read_csv(f"{PROC}/tcga_{t}_expr.csv.gz", index_col=0) for t in TCGA_TYPES}


def gene_universe(tcga, cohorts):
    common = set.intersection(*[set(x.index) for x in tcga.values()],
                              *[set(e.index) for e, _ in cohorts.values()])
    common = sorted(common)
    centred = pd.concat([x.loc[common].sub(x.loc[common].mean(axis=1), axis=0) for x in tcga.values()], axis=1)
    var = centred.var(axis=1)
    markers = [g for g in MARKERS if g in common]
    test = [g for g in var.sort_values(ascending=False).index if g not in markers][:N_GENES]
    return test, markers, centred


def make_modules(centred, genes):
    z = centred.loc[genes].T.values
    z = (z - z.mean(0)) / z.std(0)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    resid = z - (u[:, :K_FAC] * s[:K_FAC]) @ vt[:K_FAC]
    c = np.corrcoef(resid, rowvar=False)
    dist = squareform(1 - np.abs(c), checks=False)
    lab = fcluster(linkage(dist, "average"), t=0.7, criterion="distance")
    mods = []
    for k in np.unique(lab):
        idx = np.flatnonzero(lab == k)
        for i in range(0, len(idx), MOD_MAX):  # split large clusters into fixed-size chunks
            chunk = idx[i:i + MOD_MAX]
            if len(chunk) >= MOD_MIN:
                mods.append(chunk)
    return mods


def cohort_inputs(expr, lab, genes, markers, ext, rng, permute):
    x = expr.loc[genes + markers, lab["sample"]].T.values / ext["scale"]
    y = lab["y"].values.astype(float)
    if permute:
        y = rng.permutation(y)
    n = len(y)
    levels = np.unique(lab["batch"].values)
    dummies = [(lab["batch"].values == v).astype(float) for v in levels[1:]]
    covars = np.column_stack([np.ones(n), *dummies])
    score_b = x[:, len(genes):].mean(1)
    base = np.column_stack([covars, score_b])
    return x, y, covars, base


def cohort_z(x, y, covars, base, ext, mods):
    """Exact conditional z (primary) and marginal z (baseline) per module."""
    n = len(y)
    r = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
    z, zm = np.full(len(mods), np.nan), np.full(len(mods), np.nan)
    for i, cols in enumerate(mods):
        xm = x[:, cols].mean(1, keepdims=True)
        if np.mean(x[:, cols].std(0) < 1e-8) > 0.5:  # module not expressed in this cohort
            continue
        zm[i] = wilks_pvalue(xm, base, r, signed=True)
        nb = np.setdiff1d(np.unique(np.concatenate([ext["nbrs"][j] for j in cols])), cols)
        d = conditional_design(x, cols, nb, ext["w"], covars)
        if n - np.linalg.matrix_rank(d) - 1 < MIN_DF:
            continue
        z[i] = wilks_pvalue(xm, d, r, signed=True)
    return z, zm


def pooled_labelaware_z(inputs, mods):
    """Baseline: stack discovery cohorts, remove cohort effects estimated with y protected."""
    xs, ys, cs, bs = [], [], [], []
    for k, (x, y, covars, base) in enumerate(inputs):
        xs.append(np.column_stack([x[:, c].mean(1) for c in mods]))
        ys.append(y); cs.append(np.full(len(y), k)); bs.append(base[:, -1])
    px, py, pc, pb = np.vstack(xs), np.concatenate(ys), np.concatenate(cs), np.concatenate(bs)
    dmy = (pc[:, None] == np.arange(len(inputs))).astype(float)
    beta = np.linalg.lstsq(np.column_stack([dmy, py]), px, rcond=None)[0]
    pxc = px - dmy @ beta[:len(inputs)]
    adj = np.column_stack([dmy, pb])
    r = py - adj @ np.linalg.lstsq(adj, py, rcond=None)[0]
    return np.array([wilks_pvalue(pxc[:, [i]], adj, r, signed=True) for i in range(len(mods))])


def direction(zcols):
    """Sign of the discovery evidence (sign of the Stouffer sum)."""
    return np.sign(np.nansum(zcols, axis=1))


def replicated(zs_sealed, sign):
    """Same sign and one-sided p < 0.05 in >= 1 sealed cohort."""
    p1 = stats.norm.sf(sign[:, None] * zs_sealed)
    return np.any(np.nan_to_num(p1, nan=1.0) < 0.05, axis=1)


def verify_seal():
    """Refuse real labels unless PROTOCOL.md is committed and every label hash matches."""
    proto = os.path.join(ROOT, "docs", "PROTOCOL.md")
    tracked = subprocess.run(["git", "-C", ROOT, "log", "--oneline", "--", "docs/PROTOCOL.md"],
                             capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "docs/PROTOCOL.md"],
                           capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        sys.exit("docs/PROTOCOL.md is not committed (or has uncommitted changes); labels stay sealed.")
    want = dict((f, h) for h, f in re.findall(r"^([0-9a-f]{64})\s+(\S+_labels\.csv)$",
                                              open(proto).read(), flags=re.M))
    for f, h in want.items():
        got = hashlib.sha256(open(os.path.join(PROC, f), "rb").read()).hexdigest()
        if got != h:
            sys.exit(f"label hash mismatch for {f}; refusing to unseal.")
    print(f"seal verified: protocol commit {tracked.splitlines()[0]}; {len(want)} label hashes match")


def load_cohorts(names):
    cohorts = {c: (pd.read_csv(f"{PROC}/{c}_expr.csv.gz", index_col=0),
                   pd.read_csv(f"{PROC}/{c}_labels.csv", dtype={"sample": str, "patient": str}))
               for c in names}
    excl_path = os.path.join(ROOT, "results", "dedup_exclude.csv")
    if os.path.exists(excl_path):  # label-free duplicate screen (data/dedup_check.py)
        excl = pd.read_csv(excl_path, dtype=str)
        for c in names:
            e, lab = cohorts[c]
            drop = set(excl.loc[excl.cohort == c, "sample"])
            cohorts[c] = (e, lab[~lab["sample"].isin(drop)].reset_index(drop=True))
    return cohorts


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--permute", action="store_true")
    mode.add_argument("--real", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.real:
        verify_seal()
    tag = args.tag or ("real" if args.real else f"perm{args.seed}")

    cohorts = load_cohorts(ALL)
    tcga = load_tcga()
    genes, markers, centred = gene_universe(tcga, cohorts)  # universe fixed over ALL cohorts
    mods = make_modules(centred, genes)
    print(f"genes={len(genes)} markers={len(markers)} modules={len(mods)}", flush=True)

    rng = np.random.default_rng(args.seed)
    z, zmarg, inputs = {}, {}, {}
    for c in ALL:
        ref = pd.concat([tcga[t].loc[genes + markers] for t in MATCH[c]], axis=1).T.values
        ext = fit_external(ref, k=k_for(len(cohorts[c][1])), max_deg=MAX_DEG, w_span=W_SPAN)
        inputs[c] = cohort_inputs(*cohorts[c], genes, markers, ext, rng, args.permute)
        z[c], zmarg[c] = cohort_z(*inputs[c], ext, mods)
        print(f"  {c:20s} n={len(cohorts[c][1]):3d} feasible {np.mean(~np.isnan(z[c])):.2f}", flush=True)

    zd = np.column_stack([z[c] for c in DISCOVERY])
    zs = np.column_stack([z[c] for c in SEALED])
    res = pd.DataFrame({"module": np.arange(len(mods)), "size": [len(m) for m in mods],
                        "genes": [",".join(genes[j] for j in m) for m in mods]})
    for c in ALL:
        res[f"z_{c}"] = z[c]
    for r in (1, 2):
        res[f"p_r{r}"] = pc_stouffer(zd, r)
        res[f"disc_r{r}"] = bh(res[f"p_r{r}"].values, ALPHA)
    pd_simes = pc_simes(2 * stats.norm.sf(np.abs(np.nan_to_num(zd))), 1)
    res["disc_r1_simes"] = bh(pd_simes, ALPHA)
    res["direction"] = direction(zd)
    res["replicated"] = replicated(zs, res["direction"].values)

    # baselines on the same discovery cohorts
    zmd = np.column_stack([zmarg[c] for c in DISCOVERY])
    zms = np.column_stack([zmarg[c] for c in SEALED])
    res["disc_marginal"] = bh(pc_stouffer(zmd, 1), ALPHA)
    res["rep_marginal_own"] = replicated(zms, direction(zmd))
    zp = pooled_labelaware_z([inputs[c] for c in DISCOVERY], mods)
    res["disc_pooled_labelaware"] = bh(2 * stats.norm.sf(np.abs(zp)), ALPHA)
    res["rep_pooled_own"] = replicated(zms, np.sign(zp))

    def rate(mask, rep):
        return float(rep[mask].mean()) if mask.any() else None

    disc = res["disc_r1"].values
    rand = []
    if disc.sum():
        for _ in range(N_RANDOM_SETS):
            idx = rng.choice(len(mods), disc.sum(), replace=False)
            rand.append(res["replicated"].values[idx].mean())
    summary = {
        "mode": "real" if args.real else f"permuted(seed={args.seed})",
        "n_modules": len(mods),
        "discovered_r1": int(disc.sum()), "discovered_r2": int(res["disc_r2"].sum()),
        "discovered_r1_simes": int(res["disc_r1_simes"].sum()),
        "replication_rate_r1": rate(disc, res["replicated"].values),
        "random_set_rate_mean": float(np.mean(rand)) if rand else None,
        "random_set_p": float(np.mean(np.array(rand) >= rate(disc, res["replicated"].values))) if rand else None,
        "baseline_marginal": {"discovered": int(res["disc_marginal"].sum()),
                              "rep_own_stat": rate(res["disc_marginal"].values, res["rep_marginal_own"].values),
                              "rep_conditional_stat": rate(res["disc_marginal"].values, res["replicated"].values)},
        "baseline_pooled_labelaware": {"discovered": int(res["disc_pooled_labelaware"].sum()),
                                       "rep_own_stat": rate(res["disc_pooled_labelaware"].values,
                                                            res["rep_pooled_own"].values),
                                       "rep_conditional_stat": rate(res["disc_pooled_labelaware"].values,
                                                                    res["replicated"].values)},
        "feasible_frac": {c: float(np.mean(~np.isnan(z[c]))) for c in ALL},
    }
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    res.to_csv(f"{out}/real_{tag}_modules.csv", index=False)
    with open(f"{out}/real_{tag}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
