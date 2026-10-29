# Research plan: conditional knockoffs for replicable gene discovery in small ICI cohorts

**Standard:** RECOMB, meaning a new statistical method with proven guarantees, not an application of existing methods.
**Venue:** ISMB/ECCB 2027 proceedings track (Copenhagen, July 18–22, 2027).
**Hard deadline:** Jan 14, 2027, 23:59 AoE. Up to 9 pages in the OUP *Bioinformatics* template. No extensions. Internal target: Jan 11.
**Plan written:** 2026-10-28.

---

## 1. One-sentence contribution

A knockoff procedure for cohorts of 25–300 patients that is exact under a Gaussian graphical model. It gets the gene-dependence graph and the factor directions from external unlabeled data, and conditions away everything cohort-specific. Results are combined across cohorts with an FDR-valid "non-null in at least r of K" rule for relaxed e-values, plus a proven bound on FDR inflation when the external graph is wrong.

## 2. Contributions we will claim (each needs proof or evidence)

| # | Claim | Type | Status | Owner of check |
|---|---|---|---|---|
| C1′ | **(Adopted after G0)** Exact conditional module test using an external graph and factor weights W = ΩΛ, valid for any outcome distribution. Combined across cohorts with a Stouffer partial-conjunction p-value and BH. Replaces C1. | Theorem (main) | Proof sketched in `icko/core.py` docstring | Statistician |
| C1 | *(Superseded: knockoffs had zero power at G0.)* **Conditional knockoffs given anchored latent factors.** Genes are split into a scoring set B and a testing set A. Factor scores F = X_B·L_B come from B only, with L_B learned externally. X_A given F is a GGM on the external graph. Knockoffs condition on (1ᵀX_A by batch, Fᵀ X_A, and XᵀX on the graph's edges and diagonal). | Theorem (main) | To prove | Statistician |
| C2 | **Wrong-graph FDR bound.** If a cohort's true graph has edges missing from the external graph, the procedure's FDR is at most α·e^ε plus P(some gene's misspecification KL exceeds ε). It needs no cross-cohort independence. Target a sharper version that depends on the missing-edge mass and n. | Theorem | To state and prove | Statistician |
| C3 | **Valid cross-cohort combination of relaxed e-values.** The mean of the m = K−r+1 smallest e-values, then e-BH at α·m/K, controls the partial-conjunction FDR under arbitrary dependence. | Lemma | Proof sketched (see §4) | Us, then statistician |
| C4 | **Which preprocessing is safe.** Any step that is row-wise with fixed or external parameters, or that is a function of the conditioning statistic T(X), preserves validity. This covers within-batch centering and variance filtering. Cross-sample label-blind steps (DESeq2 size factors, quantile normalization, label-free ComBat) can break validity, and label-aware ComBat does. | Proposition plus counterexample | To prove | Us |
| C5 | **Feasibility region.** For each (n, graph degree, number of batch groups, k factors), whether the exact construction exists and what power it costs. | Empirical and analytic | To build | Us |

**We will not claim:** the first multi-cohort or replicable knockoffs, the first use of external data in knockoffs (GhostKnockoffs and reference-panel HMM knockoffs came first), the first batch-aware knockoffs, any new product-rule result, causality, a clinical biomarker, or better AUC than deep predictors.

## 3. Go/no-go gates

| Gate | Date | Test | GO if | If NO-GO |
|---|---|---|---|---|
| **G0: oracle power** | Nov 10 | Knockoffs with the true model (an upper bound on any conditional construction), K = 6, n ∈ {25, 50, 100, 300}, factor-plus-sparse truth, gene-level and module-level, realistic signal (oracle AUC about 0.65–0.75). | Module-level partial-conjunction power ≥ 0.5 at n = 100 and ≥ 0.2 at n = 50, with FDR ≤ α. | Restrict to modules and n ≥ 100 (IMvigor210, Braun, Riaz, Gide), or stop. |
| **G1: theory check** | Nov 24 | Statistician signs off on C1 and C3 and on the statement of C2. | No fatal hole. | Fall back to C2 and C3 plus shrinkage knockoffs, labeled as an approximate guarantee. That is weaker, possibly not RECOMB level. Re-decide. |
| **G2: beats baselines** | Dec 15 | Full simulation grid. | Our method is the only one that controls FDR at n ≤ 100 under graph error up to 10%, with power within 2× of the oracle. | Write the honest negative or feasibility paper for a stats venue, not ISMB. |
| **G3: real data** | Dec 29 | Sealed-cohort replication. | Discoveries replicate in sealed cohorts at a higher rate than the baselines. | Submit with real data as a secondary demonstration only if G2 is strong. |

### G0 result (2026-10-28): NO-GO for the knockoff → e-value → partial-conjunction pipeline

`sim/g0_oracle_power.py`, results in `results/g0_main_summary.csv`. Setup: oracle knockoffs, K = 6, 30 reps, 20 draws, p = 500, 20 signal genes in 10 of 50 modules.

- **Power = 0 in every cell**: n ∈ {25, 50, 100, 300}, signal AUC ∈ {0.72, 0.80, 0.92}, α_kn ∈ {0.05, 0.1, 0.2, 0.3}, r ∈ 1..6, at both gene and module level. FDR was 0 in all cells, so the method is valid but powerless.
- **Even a single draw in a single cohort rarely selects anything.** At α_kn = 0.1, a draw selects in at most 4% of cases, and that's at n = 300 with AUC 0.92. At α_kn = 0.3 it reaches 57%, but those e-values are too small to pass e-BH. A selected gene's e-value is about p/(α_kn·R), and e-BH needs at least p/(α′·R), so α_kn must be ≤ α·m/K. That pushes back toward the setting where nothing is selected.
- **Cause:** knockoff+ needs about 1/α_kn clean selections, and with 10–20 true signals the procedure can't produce them. This happens with the oracle model, so no graph-based construction can do better.
- **Implication:** the knockoff-based design can't work at realistic ICI signal levels. Pivot candidate: test per hypothesis with a conditional randomization test (CRT or distilled CRT) on about 50 pre-specified modules, using the same external-graph and factor conditioning, and combine across cohorts with partial-conjunction p-values. This has no minimum-rejection problem. It needs its own G0′ simulation.

### Pivot adopted (2026-10-28): exact conditional module tests

**Method** (`icko/core.py`). Write X = FΛᵀ + E, with E a Gaussian graphical model of sparse precision Ω. Then E[X_G | X_{−G}] is linear in [1, batch, X_{N(G)}, X_{−G}·W_{−G}], where W = ΩΛ, and the error is homoscedastic Gaussian.
- Regress the module statistic on that design plus an outcome residual r(Y, X_{−G}, batch). The t or Wilks test on r is then **exact for any outcome distribution**: no cohort-specific mean, coefficient or covariance enters.
- Only N(G) and W come from external data (`icko/external.py`).
- Across cohorts, the partial-conjunction p-value uses a direction-consistent Stouffer combination of the K−r+1 largest one-sided p-values, valid when cohorts are independent. Then BH. Simes is reported as secondary.

**Pre-specified primary analysis:** the 1-df module-mean statistic with the Stouffer partial-conjunction combiner. Secondary: Wilks with g degrees of freedom, gene-level tests, and Simes.

**G0′, oracle structure** (`results/g0prime_main2_summary.csv`, 50 reps per cell)
- **Calibration:** null rejection rate at 0.05 is 0.048–0.052 in every cell.
- **FDR:** mean 0.078 at r = 1 and 0.005 at r = 2. The worst cell (0.23 at 50 reps) reran at 0.085 with 500 reps, so it was noise.
- **Module power, r = 1, coherent signal:**

  | AUC | n = 50 | n = 100 | n = 300 |
  |---|---|---|---|
  | 0.70 | 0.05 | 0.16 | 0.74 |
  | 0.75 | 0.14 | 0.51 | 0.99 |
  | 0.80 | 0.31 | 0.78 | 1.00 |

- **Module power, r = 2, AUC 0.75:** 0.10 at n = 100 and 0.90 at n = 300.
- **Gene level:** near zero except with sparse signal at n = 300 (0.35 at AUC 0.75). The paper's unit is the module.

**G1, external structure learned from N = 2000 unlabeled samples** (`results/g1_main_summary.csv`)
- The learned graph has mean degree 3.4 (inflated) and true-edge recall 1.00.
- **Calibration:** 0.046–0.053 in all 36 cells, including a 20% shift in cohort loadings (τ = 0.2) and 20% of edges deleted.
- **FDR, AUC 0.75:** 0.03–0.12 per cell. Under the global null, a 500-rep check gives 0.096.
- **Power, r = 1, coherent, AUC 0.75:** 0.40–0.50 at n = 100 and 0.96–0.98 at n = 300. That is close to the oracle, and it doesn't move with τ or edge deletion.
- **Caveat:** edge deletion here removes learned edges, most of which are inflation. A stress test that deletes *true* edges, plus non-Gaussian marginals, is still needed for Theorem 2.

**G2, baselines** (`results/g2_summary`, 50 reps, K = 6, cohort response rates vary as ORR does across tumor types)

| Method | FDR, coherent, AUC 0.75, n = 300 | FDR, global null | Power, coherent, n = 300 |
|---|---|---|---|
| **Ours (external structure)** | **0.086** | 0.08–0.16 (500-rep check: 0.096) | **0.96** |
| In-sample structure | 0.089 | 0.04–0.14 | 0.94 (0.21 vs our 0.34 at n = 100) |
| Marginal per-cohort | 0.281 | 0.06–0.12 | 0.88 |
| Pooled, label-aware correction | **0.448** | **0.20–0.22** | 0.83 |
| Pooled, label-blind correction | 0.278 | 0.04–0.10 | 0.87 |

- Label-aware correction inflates FDR even when nothing is real; label-blind correction doesn't. This supports Proposition 4.
- The marginal methods' FDR grows with n, because they absorb shared infiltration.
- At n = 50 the baselines find more, but only in cells where their FDR is already inflated.

**G1-cross, random missing true edges** (60 true between-module edges, 0–100% deleted from the learned graph)
- Calibration stays at 0.048–0.052 and power is unchanged.
- *This stress is weak:* random edges rarely touch signal genes. The adversarial version (edges joining signal genes to null modules) is `--adversarial`; results below.

**G1-adversarial** (40 true edges joining signal genes to null modules, partial correlation 0.25; 100 reps)
- **Old default (k = 4, W = ΩΛ):**
  - FDR 0.14–0.20 at n = 300 even with nothing deleted.
  - FDR up to 0.53 with the true cross edges deleted.
- **Diagnosis:** with the oracle graph and W, FDR is 0.098–0.101. So the test is exact, and the inflation comes from **error in the externally estimated W**. Since bias is fixed and SE shrinks, FDR grows with n.
- **Distilling the outcome residual** (lasso on X_{−G}) did not help: FDR stayed about 0.52, because the lasso recovers weak binary-outcome signal poorly.
- **Fix, over-factoring:** use k = 8 and W = [ΩΛ, Λ]. This is the factor analogue of graph inflation, since exactness needs the estimated span to contain the true one.
  - FDR with nothing deleted: 0.058–0.112 in every cell.
  - Power: unchanged at n = 300; at n = 100, 0.12 vs 0.20 (coherent, AUC 0.75).
- **Still inflated when true edges are deleted:** 0.09–0.17 (coherent) and 0.14–0.38 (sparse). That is the unrecoverable case, and Theorem 2 must bound it.
- **Adopted for real data:** W = [ΩΛ, Λ] with n-adaptive k = clip(n / 25, 2, 8). All 10 cohorts stay feasible (82–100% of modules).
- **N_ext = 500** (matching TCGA per-type sizes; k = 8, W = [ΩΛ, Λ]):
  - Calibration 0.048–0.053 in all 16 cells, including τ = 0.2.
  - FDR with signal 0.05–0.11. Global null 0.09–0.12 at 100 reps.
  - Power matches N_ext = 2000: 0.97 at n = 300 (coherent, AUC 0.8).

**Simulation evidence complete for G0′, G1 and G2.** Next: commit `docs/PROTOCOL.md`, then unseal.

**Real data**

| Cohort | Tumor | n (after de-duplication) | Responders | Source |
|---|---|---|---|---|
| Hugo | melanoma | 26 | 14 | GEO |
| Riaz | melanoma | 49 | 10 | GEO |
| Gide, PD-1 | melanoma | 41 | 19 | cBioPortal iAtlas |
| Gide, ipilimumab + PD-1 | melanoma | 32 | 21 | cBioPortal iAtlas |
| Liu 2019 | melanoma | 121 | 47 | cBioPortal iAtlas |
| Jung | NSCLC | 27 | 7 (durable benefit) | GEO |
| GSE176307 | urothelial | 87 | 16 | GEO |
| IMvigor210 | urothelial | 298 | 68 | cBioPortal iAtlas |
| IMmotion150, atezolizumab ± bevacizumab | ccRCC | 160 | 48 | cBioPortal iAtlas |
| Braun, nivolumab | ccRCC | 168 | 38 | Nat Med supplement |
| *Controls:* IMmotion150 sunitinib, Braun everolimus | ccRCC | 81, 106 | 24, 5 | descriptive only |

- **10 ICI cohorts, 1009 patients (543 discovery, 466 sealed)**, plus 2 non-ICI controls (187 patients). TCGA references cover 6 tumor types.
- **Braun caveat:** its expression is the authors' released, already-normalized matrix (median 23.7, max 68). Check in their methods whether label-aware batch correction was used, since that would violate Proposition 4.
- **Kim** (gastric) is still raw FASTQ only.

**Duplicate screen** (`data/dedup_check.py`, label-free)
- Each gene is ranked within its cohort and the top 5 PCs are removed, then samples are correlated across cohorts.
- **Positive control:** Hugo from GEO vs the iAtlas re-processing (same 26 patients) gives true pairs ρ ≥ 0.674 and non-pairs ≤ 0.266.
- **Negative control:** NSCLC vs ccRCC (impossible pairs) gives ≤ 0.284.
- With the threshold at 0.5, 10 pairs were flagged. All are implausible as shared patients (different tumor types, randomized arms, or one hub sample matching several others).
- **Pre-specified rule:** exclude every flagged sample (14).

**Dry run:** all 10 cohorts on permuted labels gave 0 discoveries. Testable modules per cohort: 79–100%.

## 4. Theory work

**C3 proof (already checked in outline).** Under H_j (null in at least m cohorts), take any m of the cohorts where j is null and call that set S_j. Then E_j ≤ (1/m)·Σ_{k∈S_j} e_{jk}. Summing over partial-conjunction-null genes gives null-sum ≤ (1/m)·Σ_k Σ_{j null in k} E[e_{jk}] ≤ Kp/m. e-BH gives FDR ≤ α′·(null-sum)/p, so α′ = α·m/K gives FDR ≤ α. No independence is needed anywhere.

**C1 open questions, in order:**
1. Validity: X_A given F remains Gaussian with sparse conditional precision if X_B is excluded from testing. Conditioning on Fᵀ X_A is a linear constraint on each knockoff column on its own, so it should leave Huang–Janson blocking intact. This needs proof.
2. Degrees-of-freedom cost: n − G (batch groups) − k (factors) replaces n in the feasibility condition.
3. Whether F may be estimated from X_B in the same cohort (it is a function of X_B, not of X_A, so this should be fine), and how errors in L_B affect validity (they shouldn't, since L_B is fixed externally).
4. Estimand: "gene j is associated with response beyond the other tested genes, infiltration, purity and batch." State this plainly in the paper.

**C2 plan:** start from Barber, Candès and Samworth (robust knockoffs). Make it per-gene with indicator events so that it passes through e-BH. Then bound the KL for conditional knockoffs when the true precision has extra edges, ideally as roughly n × (missing-edge mass). This is the theory piece that would make the paper stand out.

**C4:** a two-line proof for functions of T(X), plus a small worked counterexample in which quantile normalization or label-free ComBat inflates FDR at small n.

## 5. Simulation design (G2)

- **Truth:** X = FΛᵀ + E. F has k = 3 factors. E is a sparse GGM with module structure (modules of size 5–20, AR or sparse within module). There are 2 batch groups per cohort, with gene-specific mean shifts and response rates that differ by batch. Marginals are Gaussian, plus a non-Gaussian stress case (log-transformed negative binomial counts).
- **Grid:** K ∈ {4, 6, 8}; n ∈ {25, 50, 100, 300}; p_A ∈ {200, 500, 1000}; fraction of true edges missing from the external graph ∈ {0, 5, 10, 20}%; signal heterogeneity across cohorts (all non-null, or non-null in only r cohorts).
- **External graph:** estimated from a separate N = 2000 unlabeled sample, inflated and degree-capped, matching the real pipeline.
- **Baselines:** multi-environment knockoffs (Li et al.) plain and derandomized, KeLP-style partial conjunction, in-sample shrinkage knockoffs, pooled lasso, per-cohort BH plus a vote, label-aware ComBat plus pooled knockoffs, and an oracle using known-Σ knockoffs.
- **Headline figures:**
  1. FDR vs n, by method.
  2. FDR vs fraction of missing edges.
  3. Power relative to the oracle.
  4. Label-aware ComBat inflation.
  5. The feasibility region map.
- **Reps:** at least 200 per cell for FDR estimates (standard error about 0.02 at α = 0.1).

## 6. Real data

| Cohort | Tumor | Drug | n (approx.) | Role |
|---|---|---|---|---|
| Hugo 2016 | melanoma | anti-PD-1 | ~27 | discovery |
| Riaz 2017 | melanoma | nivolumab | ~50 | discovery |
| Gide 2019 (split by regimen) | melanoma | PD-1 / PD-1+CTLA-4 | ~40 + ~30 | discovery |
| Kim 2018 | gastric | pembrolizumab | ~45 | discovery or sealed |
| Jung 2019 | NSCLC | PD-1/PD-L1 | ~27 | discovery |
| GSE176307 | urothelial | PD-1/PD-L1 | ~90 | discovery or sealed |
| IMvigor210 | urothelial | atezolizumab | ~300 | sealed (largest, best audit power) |
| Braun 2020 nivolumab | ccRCC | nivolumab | ~180 | discovery |
| Braun 2020 everolimus | ccRCC | everolimus | ~150 | descriptive control only |

Sample sizes are approximate; confirm them after de-duplication.

- **External data:** TCGA by matching tumor type (SKCM, STAD, LUAD/LUSC, BLCA, KIRC) and ARCHS4, used only for the graph and the factor loadings L_B. Check patient overlap with every cohort.
- **Protocol (pre-registered in this repo before anyone looks at the labels):** the gene list, the B/A split, preprocessing, which cohorts are sealed, the r values, α = 0.1, M = 30 draws, and a hash of the sealed label files. Commit the protocol, then unseal.
- **Biology:** cell-type expression of discoveries in independent single-cell ICI data, overlap with known immune signatures, and a descriptive comparison against the everolimus arm.

## 7. Software

A Python package `ici-knockoffs`: construction, e-values, combination and feasibility checker, with tests that verify FDR on null simulations in CI. Every figure can be regenerated from one command. Code goes public on submission (ISMB reviewers check).

## 8. Timeline (11 weeks)

| Week of | Work | Gate |
|---|---|---|
| Oct 28 | G0 simulation; targeted literature search (GhostKnockoffs, multi-environment knockoffs, robust knockoffs, e-value merging, side-information knockoffs) | |
| Nov 4 | G0 analysis; write C1 and C3 proofs; start the cohort download and de-duplication | **G0 Nov 10** |
| Nov 11 | C1 construction code; feasibility checker; C4 proof and counterexample | |
| Nov 18 | Statistician review; C2 statement; external graphs from TCGA | **G1 Nov 24** |
| Nov 25 | Baselines implemented; full simulation grid running | |
| Dec 2 | C2 proof; simulation analysis | |
| Dec 9 | Figures 1–5; pre-register the real-data protocol | **G2 Dec 15** |
| Dec 16 | Real-data discovery runs; unseal | |
| Dec 23 | Replication audit; biology | **G3 Dec 29** |
| Dec 30 | Full draft | |
| Jan 6 | Internal review, revisions, supplement, code release | |
| Jan 11 | Submit (3-day buffer) | Deadline Jan 14 |

## 9. Risks

1. **Power at n = 25–50.** Knockoff+ needs about 1/α_kn selections per draw, and the lasso selects at most n − G − k variables. G0 tests this first.
   - *Found 2026-10-28:* with about 20 true signals, Ren and Barber's default α_kn = α/2 = 0.05 gives zero discoveries **even with continuous Y at n = 300**. In that case 12 positive W and 0 negative W still can't reach (1+0)/12 ≤ 0.05. Any fixed α_kn gives valid e-values, so α_kn is a design parameter: either pre-specify it, or set it from a calibration on simulated data only.
   - A larger α_kn gives more nonzero e-values, each smaller. The two must balance against the e-BH level α·m/K. Choosing α_kn for the small-signal regime is a candidate secondary contribution.
2. **Dense true precision** because of hidden factors. C1 addresses this, and simulations use factor-driven truth.
3. **Graph containment can't be tested** in small cohorts. Rely on C2 plus the missing-edge simulations.
4. **Prior work under another name.** Do the literature search in week 1, before investing in proofs.
5. **Heterogeneity** in tumor type, drug and endpoint across cohorts. The target is "at least r of K," with r reported as a curve.
6. **Schedule.** The 9-page limit forces proofs into the supplement, so plan the supplement from week 6.
