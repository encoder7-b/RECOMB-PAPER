# icko: exact conditional module tests with external dependence structure

This is the code for *Exact conditional module tests with external dependence structure for replicable response biology in small immunotherapy cohorts* (RECOMB 2027 submission).

## The question

Does a gene module carry response information beyond all other measured genes, in cohorts too small to estimate their own gene–gene covariance?

The method answers this in three steps:
1. **Structure from external data.** A sparse graph and latent-factor directions are learned from unlabelled external expression, such as TCGA.
2. **Exact per-cohort test.** Everything cohort-specific is conditioned away (means, coefficients, residual covariance, batch). The resulting test is exact in finite samples for any outcome distribution, provided the external structure contains the true structure. See Proposition 1 in the paper.
3. **Combination across cohorts.** A direction-consistent Stouffer partial-conjunction test combines cohorts, followed by BH (BY as a sensitivity analysis).

## Layout

| Path | What it does |
|---|---|
| `icko/core.py` | The conditional test (`wilks_pvalue`, `test_sets`), partial-conjunction combiners, BH/BY |
| `icko/external.py` | External structure: factors, neighbourhood-lasso graph, W = [ΩΛ, Λ] over-spanning |
| `tests/test_exactness.py` | Agreement with statsmodels (t-test / MANOVA Wilks); Monte Carlo exactness under the conditional null |
| `sim/` | Simulations: G0 (knockoff gate, negative), G0′ (oracle), G1 (external structure and stress tests), G2 (baselines), G3 (factor-space misspecification) |
| `data/` | Download (`fetch.py`, `fetch_cbio.py`), label-blind preparation (`prepare.py`), duplicate-patient screens between cohorts (`dedup_check.py`) and against the TCGA reference (`tcga_overlap_check.py`) |
| `analysis/run_real.py` | The pre-registered real-data analysis. It refuses to read labels unless `docs/PROTOCOL.md` is committed and the label hashes match. |
| `analysis/sensitivity_feasible.py`, `analysis/sensitivity_tcga_overlap.py` | Sensitivity analyses: untested cohorts dropped; TCGA-flagged patient removed |
| `analysis/perm_null.py` | Real-data permutation null (response permuted within batch strata) |
| `analysis/exploratory_incremental.py` | Exploratory, post-unsealing incremental-prediction analysis |
| `docs/PROTOCOL.md` | Pre-registration: every fixed choice, plus SHA-256 hashes of the sealed labels |
| `paper/` | Manuscript. `make_assets.py` regenerates every figure, table and quoted number from `results/`. |
| `results/` | Summary tables from all simulations and analyses |

## Reproduce

Requires Python ≥ 3.10 with numpy, scipy, pandas, scikit-learn, statsmodels, joblib, matplotlib and openpyxl.

```bash
python tests/test_exactness.py                 # independent-implementation and exactness checks
python sim/g0prime.py --reps 50                # oracle calibration and power
python sim/g1_external.py --reps 50            # external structure
python sim/g2_baselines.py --reps 50           # baselines
python sim/g3_factor_misspec.py --reps 300     # factor-space misspecification
python data/fetch.py && python data/fetch_cbio.py && python data/prepare.py && python data/dedup_check.py
python analysis/run_real.py --permute          # dry run (labels permuted)
python analysis/run_real.py --real             # pre-registered analysis (verifies the protocol seal)
python paper/make_assets.py                    # figures, tables and numbers for the paper
```

The raw and processed data are public but not redistributed here. The fetch scripts download them from GEO, cBioPortal (iAtlas), UCSC Xena (TCGA) and the Braun et al. (2020) supplement. The last one is a manual download; see `data/fetch.py`.

## Status of the claims

- **Exactness** (Proposition 1) holds under the stated containment and conditional-Gaussian assumptions. It is verified numerically in `tests/test_exactness.py`.
- **Robustness to misspecification** is empirical. Over-spanning repairs factor-span errors. Missing graph edges that touch response-relevant genes are not repaired.
- **The real-data result** is a pre-registered null: no significant incremental associations among the 39 tested modules.
