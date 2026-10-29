#!/usr/bin/env bash
# Reproduce the paper.
#   ./reproduce.sh quick   verification tests + every table/figure/number + PDF, from tracked results (minutes)
#   ./reproduce.sh full    also re-runs all simulations and the real-data pipeline (many hours; needs downloads)
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-quick}"

python tests/test_lemma_a1.py
python tests/test_exactness.py --reps "${EXACT_REPS:-20000}"

if [ "$MODE" = "full" ]; then
  python sim/g0_oracle_power.py --reps 30 --draws 20 --tag main
  python sim/g0prime.py --reps 50 --aucs 0.65,0.7,0.75,0.8 --tag main2
  python sim/g1_external.py --reps 50 --tag main
  python sim/g1_external.py --reps 500 --ns 100 --aucs 0 --taus 0 --drops 0 --patterns sparse --tag nullcheck
  for spec in "25 2" "50 2" "100 4" "300 8"; do
    set -- $spec
    python sim/g1_external.py --reps 2000 --ns "$1" --aucs 0 --taus 0 --drops 0 --k "$2" --w-span both \
      --n-ext 500 --patterns sparse --tag "null2000_n$1"
  done
  python sim/g1_external.py --reps 100 --ns 100,300 --aucs 0.75,0.8 --taus 0 --drops 0,0.5,1.0 \
    --cross-edges 40 --cross-pc 0.25 --adversarial --drop-true --tag adversarial
  python sim/g1_external.py --reps 100 --ns 100,300 --aucs 0.75,0.8 --taus 0 --drops 0,0.5,1.0 \
    --cross-edges 40 --cross-pc 0.25 --adversarial --drop-true --k 8 --w-span both --tag adversarial_overfac
  python sim/g1_external.py --reps 100 --ns 100,300 --aucs 0,0.8 --taus 0,0.2 --drops 0 --cross-edges 40 \
    --cross-pc 0.25 --adversarial --k 8 --w-span both --n-ext 500 --tag adversarial_overfac_n500
  python sim/g2_baselines.py --reps 50 --tag main
  python sim/g3_factor_misspec.py --reps 300
  python sim/g3_corr_check.py
  python sim/g4_shared_patients.py 500
  python sim/feasible_calib.py
  python data/fetch.py && python data/fetch_cbio.py && python data/prepare.py && python data/dedup_check.py
  python analysis/run_real.py --real
  python analysis/perm_null.py 1000
  python analysis/exploratory_incremental.py
  python analysis/sensitivity_feasible.py
fi

python paper/make_assets.py
(cd paper && tectonic -X compile main.tex)
echo "done: paper/main.pdf"
