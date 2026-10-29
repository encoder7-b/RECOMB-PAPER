"""Build every figure and table in the manuscript directly from results/.

python paper/make_assets.py   ->  paper/figs/*.pdf, paper/tables/*.tex, paper/numbers.tex
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
FIG, TAB = os.path.join(HERE, "figs"), os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# validated categorical palette (slots 1-5, fixed order); contrast WARN -> markers + direct labels + tables
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
MK = ["o", "s", "^", "D", "v"]
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
                     "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42})
W1 = 3.35  # single Bioinformatics column, inches


def alpha_line(ax):
    ax.axhline(0.1, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.annotate("α = 0.1", xy=(1, 0.1), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=6)


def log_n_axis(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_baselines():
    s = pd.read_csv(f"{RES}/g2_main_summary.csv")
    s = s[(s.pattern == "coherent") & (s.auc == 0.75)]
    meth = [("external", "Ours (external)"), ("insample", "In-sample structure"), ("marginal", "Marginal"),
            ("pooled_labelaware", "Pooled, label-aware"), ("pooled_labelblind", "Pooled, label-blind")]
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 2, 2.1))
    for ax, col, lab in zip(axes, ["fdr", "power"], ["FDR", "Power"]):
        for i, (m, name) in enumerate(meth):
            d = s[s.method == m].sort_values("n")
            ax.plot(d.n, d[col], color=C[i], marker=MK[i], ls=LS[i], label=name,
                    lw=2 if m == "external" else 1.2, zorder=3 if m == "external" else 2)
        log_n_axis(ax, [50, 100, 300])
        ax.set_xlabel("patients per cohort (n)"); ax.set_ylabel(lab)
        if col == "fdr":
            ax.set_ylim(0, 0.5); alpha_line(ax)
        else:
            ax.set_ylim(0, 1.02)
    axes[0].legend(loc="upper left", fontsize=6)
    for a, t in zip(axes, "ab"):
        a.set_title(t, loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_baselines.pdf"); plt.close(fig)


def fig_adversarial():
    old = pd.read_csv(f"{RES}/g1_adversarial_summary.csv")
    new = pd.read_csv(f"{RES}/g1_adversarial_overfac_summary.csv")
    sel = lambda d: d[(d.level == "module_mean") & (d.comb == "stouffer") & (d.r == 1) & (d.auc == 0.8)]  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 2, 2.0), sharey=True)
    for ax, pat in zip(axes, ["sparse", "coherent"]):
        for j, (d, name) in enumerate([(sel(old), "k=4, W=ΩΛ"), (sel(new), "k=8, W=[ΩΛ, Λ]")]):
            for n, ls in [(100, "--"), (300, "-")]:
                q = d[(d.pattern == pat) & (d.n == n)].sort_values("drop")
                ax.plot(100 * q["drop"], q.fdr, color=C[1 if j == 0 else 0], marker=MK[j], ls=ls,
                        label=f"{name}, n={n}")
        ax.set_xlabel("true cross-module edges missing from graph (%)"); ax.set_xticks([0, 50, 100])
        ax.set_title(f"{'a' if pat == 'sparse' else 'b'}  {pat} signal", loc="left", fontweight="bold")
        ax.set_ylim(0, 0.6); alpha_line(ax)
    axes[0].set_ylabel("FDR"); axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_adversarial.pdf"); plt.close(fig)


def fig_power():
    o = pd.read_csv(f"{RES}/g0prime_main2_summary.csv")
    o = o[(o.pattern == "coherent") & (o.level == "module_mean") & (o.comb == "stouffer") & (o.r == 1)]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ends = []
    for i, auc in enumerate([0.65, 0.70, 0.75, 0.80]):
        d = o[o.auc == auc].sort_values("n")
        ax.plot(d.n, d.power, color=C[i], marker=MK[i], ls=LS[i])
        ends.append([d.power.iloc[-1], auc, d.n.iloc[-1]])
    ends.sort(reverse=True)  # de-collide direct labels from the top down
    for k in range(1, len(ends)):
        ends[k][0] = min(ends[k][0], ends[k - 1][0] - 0.08)
    for y, auc, x in ends:
        ax.text(x * 1.06, y, f"AUC {auc:.2f}", color=INK, fontsize=6, va="center")
    log_n_axis(ax, [25, 50, 100, 300])
    ax.set_xlim(20, 520); ax.set_ylim(0, 1.02)
    ax.set_xlabel("patients per cohort (n), K = 6 cohorts"); ax.set_ylabel("power (modules, r = 1)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_power.pdf"); plt.close(fig)


def teff_auc():
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import run_real as R
    from sklearn.metrics import roc_auc_score
    co = R.load_cohorts(R.ALL)
    rows = []
    for c in R.ALL:
        e, lab = co[c]
        g = [m for m in R.MARKERS if m in e.index]
        x = e.loc[g, lab["sample"]].T.values
        s = ((x - x.mean(0)) / (x.std(0) + 1e-9)).mean(1)
        y = lab.y.values.astype(float)
        rows.append(dict(cohort=c, n=len(y), resp=int(y.sum()), auc=roc_auc_score(y, s),
                         p=stats.mannwhitneyu(s[y == 1], s[y == 0]).pvalue,
                         role="discovery" if c in R.DISCOVERY else "sealed" if c in R.SEALED else "control"))
    return pd.DataFrame(rows)


def fig_real(teff):
    m = pd.read_csv(f"{RES}/real_real_modules.csv")
    p = np.sort(m.p_r1.dropna().values)
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 2, 2.2), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    exp = (np.arange(1, len(p) + 1) - 0.5) / len(p)
    lo = stats.beta.ppf(0.025, np.arange(1, len(p) + 1), len(p) - np.arange(len(p)))
    hi = stats.beta.ppf(0.975, np.arange(1, len(p) + 1), len(p) - np.arange(len(p)))
    ax.fill_between(-np.log10(exp), -np.log10(lo), -np.log10(hi), color=GRID, lw=0)
    ax.plot([0, 2], [0, 2], color=MUTED, lw=0.8)
    ax.plot(-np.log10(exp), -np.log10(p), "o", color=C[0], ms=3.5)
    bh = 0.1 / len(p)
    ax.axhline(-np.log10(bh), color=C[1], lw=0.9, ls="--")
    ax.text(0.05, -np.log10(bh) + 0.05, "BH threshold for 1 discovery", color=INK, fontsize=6)
    ax.set_xlabel("expected −log10 p"); ax.set_ylabel("observed −log10 p (r = 1)")
    ax.set_title("a", loc="left", fontweight="bold")
    ax = axes[1]
    names = {"hugo": "Hugo", "riaz": "Riaz", "gide_pd1": "Gide PD-1", "gide_combo": "Gide ipi+PD-1",
             "liu2019": "Liu", "jung": "Jung", "gse176307": "GSE176307", "immotion_atezo": "IMmotion150 atezo",
             "imvigor210": "IMvigor210*", "braun_nivo": "Braun nivo*", "immotion_sunitinib": "IMmotion150 sunitinib†",
             "braun_evero": "Braun everolimus†"}
    t = teff.iloc[::-1].reset_index(drop=True)
    col = {"discovery": C[0], "sealed": C[2], "control": MUTED}
    for i, r in t.iterrows():
        ax.plot([0.5, r.auc], [i, i], color=GRID, lw=1.5, zorder=1)
        ax.plot(r.auc, i, MK[0 if r.role == "discovery" else 1 if r.role == "sealed" else 2],
                color=col[r.role], ms=4.5, zorder=3)
        ax.text(max(r.auc, 0.5) + 0.012, i, f"{r.auc:.2f}" + ("  (p<0.05)" if r.p < 0.05 else ""),
                va="center", fontsize=5.5, color=INK)
    ax.set_yticks(range(len(t))); ax.set_yticklabels([names[c] for c in t.cohort], fontsize=6)
    ax.axvline(0.5, color=MUTED, lw=0.8); ax.set_xlim(0.4, 1.0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC of T-effector score (exploratory)")
    ax.set_title("b", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_real.pdf"); plt.close(fig)


def fmt(v, d=3):
    return "--" if pd.isna(v) else f"{v:.{d}f}"


def tables(teff):
    s = pd.read_csv(f"{RES}/g2_main_summary.csv")
    meth = [("external", "Ours (external structure)"), ("insample", "In-sample structure"),
            ("marginal", "Marginal per cohort"), ("pooled_labelaware", "Pooled, label-aware correction"),
            ("pooled_labelblind", "Pooled, label-blind correction")]
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{FDR (coherent, AUC 0.75)} & FDR & Power \\",
             r"Method & $n{=}50$ & $n{=}100$ & $n{=}300$ & global null$^a$ & $n{=}300$ \\", r"\midrule"]
    for m, name in meth:
        d = s[(s.method == m) & (s.pattern == "coherent")]
        f = [d[(d.auc == 0.75) & (d.n == n)].fdr.iloc[0] for n in (50, 100, 300)]
        g = d[d.auc == 0].fdr.mean()
        pw = d[(d.auc == 0.75) & (d.n == 300)].power.iloc[0]
        b = (lambda x: rf"\textbf{{{x}}}") if m == "external" else (lambda x: x)
        lines.append(f"{b(name)} & " + " & ".join(fmt(v) for v in f) + f" & {fmt(g)} & {fmt(pw, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{TAB}/tab_baselines.tex", "w").write("\n".join(lines) + "\n")

    rows = [r"\begin{tabular}{llrrcc}", r"\toprule", r"Cohort & Tumour & $n$ & Resp. & Role & Batch \\", r"\midrule"]
    tum = {"hugo": "melanoma", "riaz": "melanoma", "gide_pd1": "melanoma", "gide_combo": "melanoma",
           "liu2019": "melanoma", "jung": "NSCLC", "gse176307": "urothelial", "immotion_atezo": "ccRCC",
           "imvigor210": "urothelial", "braun_nivo": "ccRCC", "immotion_sunitinib": "ccRCC", "braun_evero": "ccRCC"}
    lab = {"hugo": "Hugo 2016", "riaz": "Riaz 2017", "gide_pd1": "Gide 2019, PD-1",
           "gide_combo": "Gide 2019, ipi+PD-1", "liu2019": "Liu 2019", "jung": "Jung 2019",
           "gse176307": "GSE176307", "immotion_atezo": "IMmotion150, atezo $\\pm$ bev",
           "imvigor210": "IMvigor210", "braun_nivo": "Braun 2020, nivolumab",
           "immotion_sunitinib": "IMmotion150, sunitinib", "braun_evero": "Braun 2020, everolimus"}
    bat = {"gse176307": "IO line", "imvigor210": "biopsy site", "liu2019": "prior anti-CTLA-4",
           "immotion_atezo": "bevacizumab", "braun_nivo": "trial"}
    for _, r in teff.iterrows():
        rows.append(f"{lab[r.cohort]} & {tum[r.cohort]} & {r.n} & {r.resp} & {r.role} & {bat.get(r.cohort, '--')} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    open(f"{TAB}/tab_cohorts.tex", "w").write("\n".join(rows) + "\n")


def numbers(teff):
    """Macros for every number quoted in the text."""
    j = json.load(open(f"{RES}/real_real_summary.json"))
    m = pd.read_csv(f"{RES}/real_real_modules.csv")
    o = pd.read_csv(f"{RES}/g0prime_main2_summary.csv")
    calib = o[o.r == 0].fdr
    nu = pd.read_csv(f"{RES}/g1_nullcheck_summary.csv")
    nu = nu[(nu.level == "module_mean") & (nu.comb == "stouffer") & (nu.r == 1)].fdr.iloc[0]
    adv_old = pd.read_csv(f"{RES}/g1_adversarial_summary.csv")
    adv_new = pd.read_csv(f"{RES}/g1_adversarial_overfac_summary.csv")
    n500 = pd.read_csv(f"{RES}/g1_adversarial_overfac_n500_summary.csv")
    sel = lambda d: d[(d.level == "module_mean") & (d.comb == "stouffer") & (d.r == 1)]  # noqa: E731
    best = m.sort_values("p_r1").iloc[0]
    disc_n = int(teff[teff.role == "discovery"].n.sum())
    seal_n = int(teff[teff.role == "sealed"].n.sum())
    mac = {
        "calibLo": f"{calib.min():.3f}", "calibHi": f"{calib.max():.3f}",
        "nullFDR": f"{nu:.3f}",
        "advOldMax": f"{sel(adv_old).fdr.max():.2f}",
        "advOldNoDropHi": f"{sel(adv_old)[sel(adv_old)['drop'] == 0].fdr.max():.2f}",
        "advNewNoDropLo": f"{sel(adv_new)[sel(adv_new)['drop'] == 0].fdr.min():.3f}",
        "advNewNoDropHi": f"{sel(adv_new)[sel(adv_new)['drop'] == 0].fdr.max():.3f}",
        "advNewDropHi": f"{sel(adv_new).fdr.max():.2f}",
        "nExtCalibLo": f"{n500[n500.r == 0].fdr.min():.3f}", "nExtCalibHi": f"{n500[n500.r == 0].fdr.max():.3f}",
        "nModules": str(j["n_modules"]), "nDiscovered": str(j["discovered_r1"]),
        "bestP": f"{best.p_r1:.3f}", "bhThresh": f"{0.1 / j['n_modules']:.4f}",
        "nBelow": str(int((m.p_r1 < 0.05).sum())),
        "ksP": f"{stats.kstest(m.p_r1.dropna(), 'uniform').pvalue:.2f}",
        "nReplicated": str(int(m.replicated.sum())),
        "expReplicated": f"{len(m) * (1 - 0.95 ** 2):.1f}",
        "nDisc": str(disc_n), "nSeal": str(seal_n), "nICI": str(disc_n + seal_n),
        "nCtrl": str(int(teff[teff.role == "control"].n.sum())),
    }
    with open(f"{HERE}/numbers.tex", "w") as fh:
        for k, v in mac.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return mac


def tables_v2():
    """Tables added in revision: exactness, null calibration, factor misspecification, exploratory."""
    e = pd.read_csv(f"{RES}/exactness_summary.csv")
    L = [r"\begin{tabular}{rrccc}", r"\toprule", r"$n$ & $g$ & reject @0.05 & reject @0.01 & KS $p$ \\", r"\midrule"]
    for _, r in e.iterrows():
        L.append(f"{int(r.n)} & {int(r.g)} & {r.rej05:.4f} $\\pm$ {r.ci05:.4f} & {r.rej01:.4f} $\\pm$ {r.ci01:.4f} & {r.ks_p:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{TAB}/tab_exactness.tex", "w").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{rcccc}", r"\toprule",
         r"$n$ & $k$ & FDR, $r{=}1$ & FDR, $r{=}2$ & feasible tests: reject @0.05 \\", r"\midrule"]
    fc = json.load(open(f"{RES}/feasible_calib.json")) if os.path.exists(f"{RES}/feasible_calib.json") else {}
    for n, k in [(25, 2), (50, 2), (100, 4), (300, 8)]:
        d = pd.read_csv(f"{RES}/g1_null2000_n{n}.csv")
        d = d[(d.level == "module_mean")]
        ci = lambda x: f"{x.mean():.3f} [{max(0, x.mean() - 1.96 * x.std() / np.sqrt(len(x))):.3f}, {x.mean() + 1.96 * x.std() / np.sqrt(len(x)):.3f}]"  # noqa: E731
        f1 = ci(d[(d.comb == "stouffer") & (d.r == 1)].fdp)
        f2 = ci(d[(d.comb == "stouffer") & (d.r == 2)].fdp)
        if str(n) in fc:
            pr = f"{fc[str(n)]['rate05']:.4f} ({100 * fc[str(n)]['feasible']:.0f}\\% feasible)"
        else:
            pr = f"{d[d.r == 0].fdp.mean():.4f}"
        L.append(f"{n} & {k} & {f1} & {f2} & {pr} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{TAB}/tab_null2000.tex", "w").write("\n".join(L) + "\n")

    g = pd.read_csv(f"{RES}/g3_factor_misspec_summary.csv")
    cc = pd.read_csv(f"{RES}/g3_corr_check.csv")
    corr = lambda tag: "/".join(f"{v:.3f}" for v in cc[cc.case.str.contains(tag)].sort_values("pattern", ascending=False)  # noqa: E731
                                .mean_abs_corr_r_U_given_D)
    cases = [("oracle", "true span"), ("contained", "true span + 4 random"),
             ("omit_related", f"omit $F_1$ (immune), $|\\rho|$={corr('F1')}"),
             ("omit_unrelated", f"omit $F_2$, $|\\rho|$={corr('F2')}"),
             ("omit_diffuse", f"omit $F_4$ (diffuse), $|\\rho|$={corr('F4')}")]
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{sparse signal} & \multicolumn{3}{c}{coherent signal} \\",
         r"Factor span & $n{=}50$ & $n{=}100$ & $n{=}300$ & $n{=}50$ & $n{=}100$ & $n{=}300$ \\", r"\midrule"]
    for c, name in cases:
        v = []
        for pat in ("sparse", "coherent"):
            for n in (50, 100, 300):
                q = g[(g.pattern == pat) & (g.case == c) & (g.n == n)].iloc[0]
                v.append(f"{q.fdr:.3f}")
        L.append(name + " & " + " & ".join(v) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{TAB}/tab_factor_misspec.tex", "w").write("\n".join(L) + "\n")

    a = pd.read_csv(f"{RES}/exploratory_incremental_auc.csv")
    nm = {"imvigor210": "IMvigor210 (sealed)", "braun_nivo": "Braun nivolumab (sealed)",
          "immotion_sunitinib": "IMmotion150 sunitinib (control)", "braun_evero": "Braun everolimus (control)"}
    L = [r"\begin{tabular}{lcc}", r"\toprule", r"Test cohort & T-effector only & + elastic net (1500 genes) \\", r"\midrule"]
    for c in nm:
        q = a[a.cohort == c].set_index("model")
        f = lambda m: f"{q.loc[m].auc:.2f} [{q.loc[m].lo:.2f}, {q.loc[m].hi:.2f}]"  # noqa: E731
        L.append(f"{nm[c]} & {f('A_teff')} & {f('C_teff+enet1500')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{TAB}/tab_exploratory.tex", "w").write("\n".join(L) + "\n")


def numbers_v2():
    p = json.load(open(f"{RES}/perm_null_summary.json"))
    x = json.load(open(f"{RES}/exploratory_incremental_summary.json"))
    ag = pd.read_csv(f"{RES}/exactness_agreement.csv")
    mac = {"permN": str(p["n_perm"]), "permRate": f"{p['per_test_rate_0.05']:.3f}",
           "permBH": f"{p['fwer_bh_r1']:.3f}", "permBY": f"{p['fwer_by_r1']:.3f}",
           "permBHtwo": f"{p['fwer_bh_r2']:.3f}", "permMinQ": f"{p['real_min_p_r1_quantile']:.2f}",
           "enetGenes": str(x["n_nonzero_enet_genes"]),
           "modDeltaMax": f"{max(v['max'] for v in x['module_delta_auc'].values()):.3f}",
           "agreeMax": f"{ag.max_abs_log_p_diff.max():.0e}"}
    with open(f"{HERE}/numbers.tex", "a") as fh:
        for k, v in mac.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return mac


if __name__ == "__main__":
    teff = teff_auc()
    teff.to_csv(f"{RES}/real_teff_auc_exploratory.csv", index=False)
    fig_baselines(); fig_adversarial(); fig_power(); fig_real(teff)
    tables(teff)
    print(json.dumps(numbers(teff), indent=1))
    tables_v2()
    print(json.dumps(numbers_v2(), indent=1))
