"""EXPLORATORY (post-unsealing; logged in docs/PROTOCOL.md section 6).

Incremental predictive information beyond the T-effector score, evaluated on held-out cohorts:
  A  T-effector score only
  B  T-effector score + one module mean (each of the pre-registered modules)
  C  T-effector score + elastic-net over the 1500 tested genes
Features are standardized within cohort (label-free). Models are trained on the pooled
discovery cohorts and scored by AUC (95% bootstrap CI) on the sealed ICI cohorts and on the
non-ICI control arms (treatment specificity: a predictive marker should not work there).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import run_real as R  # noqa: E402

B_BOOT = 2000


def zstd(m):
    return (m - m.mean(0)) / (m.std(0) + 1e-9)


def features(cohorts, genes, markers, mods):
    out = {}
    for c in R.ALL:
        e, lab = cohorts[c]
        g = zstd(e.loc[genes, lab["sample"]].T.values)
        mk = zstd(e.loc[markers, lab["sample"]].T.values).mean(1, keepdims=True)
        mm = np.column_stack([g[:, m].mean(1) for m in mods])
        out[c] = dict(gene=g, teff=zstd(mk), mod=zstd(mm), y=lab.y.values.astype(float))
    return out


def auc_ci(y, s, rng):
    a = roc_auc_score(y, s)
    bs = []
    for _ in range(B_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], s[i]))
    return a, np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def main():
    cohorts = R.load_cohorts(R.ALL)
    tcga = R.load_tcga()
    genes, markers, centred = R.gene_universe(tcga, cohorts)
    mods = R.make_modules(centred, genes)
    f = features(cohorts, genes, markers, mods)
    tr = {k: np.vstack([f[c][k] for c in R.DISCOVERY]) for k in ("gene", "teff", "mod")}
    ytr = np.concatenate([f[c]["y"] for c in R.DISCOVERY])
    rng = np.random.default_rng(7)
    test = R.SEALED + R.CONTROLS

    mA = LogisticRegression().fit(tr["teff"], ytr)
    mC = LogisticRegressionCV(Cs=10, penalty="elasticnet", solver="saga", l1_ratios=[0.5], cv=5,
                              scoring="roc_auc", max_iter=5000).fit(np.hstack([tr["teff"], tr["gene"]]), ytr)
    rows = []
    for c in test:
        y = f[c]["y"]
        a = auc_ci(y, mA.decision_function(f[c]["teff"]), rng)
        cc = auc_ci(y, mC.decision_function(np.hstack([f[c]["teff"], f[c]["gene"]])), rng)
        rows.append(dict(cohort=c, model="A_teff", auc=a[0], lo=a[1], hi=a[2]))
        rows.append(dict(cohort=c, model="C_teff+enet1500", auc=cc[0], lo=cc[1], hi=cc[2]))
    best_b = []
    for j in range(len(mods)):
        xb = np.hstack([tr["teff"], tr["mod"][:, [j]]])
        mB = LogisticRegression().fit(xb, ytr)
        for c in R.SEALED:
            s = mB.decision_function(np.hstack([f[c]["teff"], f[c]["mod"][:, [j]]]))
            best_b.append(dict(module=j, cohort=c, auc=roc_auc_score(f[c]["y"], s)))
    bdf = pd.DataFrame(best_b)
    base = {c: [r["auc"] for r in rows if r["cohort"] == c and r["model"] == "A_teff"][0] for c in R.SEALED}
    bdf["delta_vs_A"] = bdf.apply(lambda r: r.auc - base[r.cohort], axis=1)
    res = pd.DataFrame(rows)
    out = os.path.join(R.ROOT, "results")
    res.to_csv(f"{out}/exploratory_incremental_auc.csv", index=False)
    bdf.to_csv(f"{out}/exploratory_incremental_modules.csv", index=False)
    summ = {"n_nonzero_enet_genes": int((mC.coef_[0][1:] != 0).sum()),
            "module_delta_auc": {c: {"median": float(bdf[bdf.cohort == c].delta_vs_A.median()),
                                     "max": float(bdf[bdf.cohort == c].delta_vs_A.max()),
                                     "min": float(bdf[bdf.cohort == c].delta_vs_A.min())} for c in R.SEALED}}
    json.dump(summ, open(f"{out}/exploratory_incremental_summary.json", "w"), indent=1)
    print(res.round(3).to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
