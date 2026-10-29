# Pre-registered analysis protocol (draft, not yet committed)

**Status:** DRAFT. It becomes binding at the git commit that adds this file. Real response labels must not be analysed before that commit. Until then, `analysis/run_real.py` refuses to run without `--permute`.

## 1. Cohorts

**Discovery (8 cohorts, 543 patients, 182 responders after de-duplication):** hugo, riaz, gide_pd1, gide_combo, liu2019, jung, gse176307, immotion_atezo.

**Sealed replication (2 cohorts, 466 patients, 106 responders):** imvigor210 (bladder, n = 298) and braun_nivo (ccRCC, n = 168).
- These are the two largest cohorts, which gives the replication audit the most power.
- Each has a same-tumor discovery partner: gse176307 for bladder, immotion_atezo for ccRCC.

**Non-ICI controls (descriptive only, never used for discovery):** immotion_sunitinib and braun_evero.

## 2. Fixed choices (all label-free)

| Item | Setting |
|---|---|
| Expression | log2(TPM/FPKM + 1). Braun uses the authors' released matrix as is. |
| Response | 1 = CR/PR, 0 = SD/PD, NE excluded. Jung: PFS ≥ 180 days = 1, progression before 180 days = 0, censored before 180 days excluded. |
| Samples | Pre-treatment only, one per patient. Samples flagged by `data/dedup_check.py` are excluded (14). |
| Batch | GSE176307: IO line 1 vs later. IMvigor210: biopsy site (bladder / other / unknown). Liu: prior anti-CTLA-4. IMmotion150: bevacizumab. Braun: trial. Others: none. |
| Gene universe | 1500 top-variance genes across TCGA (centred per type), present in all cohorts |
| Marker set B (never tested) | CD8A, CD8B, GZMA, GZMB, PRF1, CXCL9, CXCL10, IFNG, IDO1, STAT1, HLA-DRA, CD274, LAG3, CD3E, CD2 |
| Modules | Average-linkage clusters of the pooled-TCGA residual correlation (5 factors removed, cut at 0.7), split into chunks of ≤ 12, keeping chunks of ≥ 4 |
| External structure | Matched TCGA type(s) per cohort. Neighbourhood-lasso graph, degree cap 3. W = [ΩΛ, Λ] with k = clip(n / 25, 2, 8). |
| Feasibility | A module is tested in a cohort only if residual df ≥ 8 and at least half its genes vary in that cohort. Otherwise z = 0. |
| Outcome residual | OLS residual of Y on intercept, batch and the marker-set mean |
| Primary statistic | Exact 1-df conditional test on the module mean |
| Combination | Signed Stouffer partial-conjunction p-value, then BH at α = 0.10 |
| Primary claim | r = 1 across the discovery cohorts. Secondary: r = 2 and Simes. |

## 3. Replication audit (sealed cohorts)

For each module discovered at r = 1:
- Compute the same per-cohort conditional z in imvigor210 and braun_nivo.
- **Replication:** z has the same sign as the discovery Stouffer direction, with one-sided p < 0.05 in at least one sealed cohort.
- **Report:** the replication rate, compared with (a) the rate for 1000 random same-size module sets, and (b) the replication rate of the marginal and pooled-correction baselines run on the same discovery cohorts.
- **Baseline scoring:** each baseline's discoveries are scored twice, (i) with its own statistic in the sealed cohorts and (ii) with the conditional statistic. Both are reported.
- **Module set:** the fixed recipe in §2 yields 39 modules covering about 400 of the 1500 genes. This is reported as a coverage limitation; no re-tuning.

Implementation: `analysis/run_real.py --real`. It refuses to run unless this file is committed and every label hash below matches.

## 4. Controls

- **Predictive vs prognostic:** report the same z in immotion_sunitinib and braun_evero, descriptively only. No claim of predictiveness rests on a null result there.

## 5. Label-file hashes (SHA-256 at sealing)

```
6097ed3f33b65375dfecc55ce1a5400cc35573e7ea510fc911ab6bc846a77cf4  braun_evero_labels.csv
42f8713933a9398a0c684eb44f2cdbaf47a6642580527a55d8f4e61c916f906e  braun_nivo_labels.csv
ce80edd3d89d0203597b05942bd901e7aac4434d460de9e91c5b0480dbc6ff36  gide_combo_labels.csv
7c3b5763f766fb5d96416dfb7b3d55a554de02f05355d54234ff7c4de6011ca2  gide_pd1_labels.csv
9b7bbc2dc0fdbdb09000ca216804d34d25b82ececab8073b9e5acd4eff7a84d4  gse176307_labels.csv
01ee3004f7cd950846acd1553e8ea9ab8a3f59be90243f0353099662da3722ef  hugo_labels.csv
d8139428ec1344e9212bdb6eae8796932602fae5a7a3be06605930a20b9c38d1  immotion_atezo_labels.csv
1e314b301aa20703d83962a26cdf38d6b7557d7f2b8da8272751a51cfd0530d6  immotion_sunitinib_labels.csv
f24cc51112b46126b52c9bf987bea1787d7d9251d82ff5a85315f4b731f34e18  imvigor210_labels.csv
ca7df2da701d0de45b89a3301c6c4e3e6d6ccfc5ac440cfb2d1d8d780962f805  jung_labels.csv
f3bf6069c96a10958ed178e2aab674508d2a61e8d1f74b613577dfc1ff96e78e  liu2019_labels.csv
e22070b34b1f7df4e0d222a0728c13d8b9d62ad9cf8ddd1993f057a31f824dad  riaz_labels.csv
```

## 6. Deviations

Any change after the commit gets logged here with a date and reason, and is reported in the paper.

- **2026-10-29, exploratory (not a protocol change):** after unsealing, the per-cohort AUC of the marker-set mean (T-effector score) was computed as a label and pipeline sanity check. It is reported as exploratory.
