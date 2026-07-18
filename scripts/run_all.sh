#!/bin/bash
# Single-command pipeline regeneration from a clean checkout (G5).
# Stages run strictly in gate order; any failure aborts.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "== G1: physics gates =="
pytest tests/test_forward_model.py -v
pytest tests/ -m slow -s

echo "== Stage 2: benchmark generation + leakage audit =="
python src/generate_data.py

echo "== G2: training (2 seeds) =="
python src/train.py --seed 1
python src/train.py --seed 2

echo "== G4a: oracle sanity + exact posteriors =="
python src/sanity_oracle.py
python src/exact_posterior.py --splits calib test

echo "== G3: conformal =="
python src/run_conformal.py --probs model --seed 1
python src/run_conformal.py --probs model --seed 2
python src/run_conformal.py --probs exact

echo "== G4b: sharpness audit =="
python src/audit.py --seed 1
python src/audit.py --seed 2

echo "== H3: dose-response =="
python src/dose_response.py --seed 1

echo "== G5: assembly + figures =="
python src/assemble_results.py
python src/figures.py

echo "ALL STAGES COMPLETE"
