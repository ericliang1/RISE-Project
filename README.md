# Conformal Source Regions for Sparse-Sensor Pollution Localization

Computational pipeline for the paper *"Conformal Source Regions for
Sparse-Sensor Pollution Localization: A Calibration, Exact-Posterior Audit,
and Information Dose-Response Study"*. Produces every number and figure that
replaces the paper's `[0.XX]` placeholders.

## Environment (BU SCC)

```bash
source scripts/env.sh    # module load miniconda academic-ml/spring-2026; conda activate spring-2026-pyt
```

Python 3.12.11, PyTorch 2.9.1+cu128 on NVIDIA L40S. `data/` is a symlink to
`/projectnb/rise-tower/eric1/csr-data` (home quota is 10 GB).

## One config

Everything is driven by `config/default.yaml`. Documented deviations and all
gate measurements live in `GATES.md` / `results/gates.json`.

## Pipeline (gates G1-G5 are blocking; run in order)

```bash
# G1 - physics validation
pytest tests/test_forward_model.py -v          # quadrature/zero-wind/symmetry/linearity
pytest tests/ -m slow -s                       # FD cross-check (GPU, ~15 min)

# Stage 2 - benchmark generation + freeze + leakage audit
python src/generate_data.py

# G2 - training (two seeds)
python src/train.py --seed 1
python src/train.py --seed 2

# G3 - conformal calibration (H1)
python src/run_conformal.py --probs model --seed 1

# G4 - exact posteriors + sanity + sharpness audit (H2, D1)
python src/exact_posterior.py --splits calib test
python src/sanity_oracle.py
python src/run_conformal.py --probs exact
python src/audit.py --seed 1

# H3 - dose-response (inference only)
python src/dose_response.py --seed 1

# G5 - results assembly, figures F1-F5, placeholder map, tables
python src/assemble_results.py
python src/figures.py
```

`python src/assemble_results.py` regenerates all main tables from the frozen
data + checkpoints (the reproducibility rerun).

## Layout

```
config/default.yaml     the ONE config
src/                    pipeline stages (see paper sections 3-6)
tests/                  pytest gates (G1)
results/                gates.json, checksums.json, placeholder_map.json, tables/
figures/                fig1_pipeline.pdf ... fig5_doseresponse.pdf
data/ -> /projectnb/... frozen scenario npz files, checkpoints, posteriors
GATES.md                human-readable gate summary + documented decisions
```
