# SHAP analysis — map-level exact Shapley values

How much does each of the four physics feature maps contribute? Each map is
one feature ("present" = the real map, "absent" = the zero baseline the conv
head was zero-initialized against), so with 4 features all 2^4 = 16
coalitions are enumerated and the Shapley values are **exact** — no
KernelSHAP sampling. Value function: v(S) = log p(true source cell) under
the maps in S, i.e. the training objective; a Shapley value is the map's
average marginal NLL-reduction in **nats**. Computed per scenario over all
2,000 seed-1 test scenarios (`ch4t-test-000000 .. 001999`), for all three
architectures.

Exactness shortcut: logits = base(x) + conv(M(x)), so the set-network
forward runs once per scenario and only the conv head re-runs per coalition
— mathematically identical to 16 full forwards.

## Files

| File | What it is |
|---|---|
| `shap_maps.py` | the analysis (run from repo root) |
| `shap_map_contributions.csv` | 6,000 rows = 3 models × 2,000 scenarios: per-map Shapley values, v(all), v(none), efficiency residual |
| `shap_map_summary.{pdf,png,svg}` | poster figure: mean \|φ\| per map × model + DeepSets beeswarm coloured by mast count |
| `shap_map_summary.pptx` | **editable** PowerPoint version — see below |
| `export_pptx.py` | builds the pptx from `shap_map_contributions.csv` only (no model/GPU access) |
| `retrain_gnn_st_seed1.log` | training log for the GNN / Set Transformer checkpoints |

## The PowerPoint export

`shap_map_summary.pptx` rebuilds both panels as **native PowerPoint chart
objects**, not an embedded picture — every number, color, and label is
editable directly in PowerPoint (right-click a chart → *Edit Data in
Excel*):

- **Panel A**: a native clustered bar chart, one series per model (mean
  |Shapley value| per map, matching `shap_map_summary.png` to 2 decimal
  places).
- **Panel B**: a native XY scatter chart of the DeepSets per-scenario
  values. PowerPoint scatter charts have no continuous-colormap option, so
  mast count is bucketed into 3 editable series (4–6 / 7–9 / 10–12 masts)
  instead of the PNG's smooth plasma gradient; black diamonds are the
  **full 2,000-scenario mean** (not the subsample's). For editability
  (thousands of native points make PowerPoint sluggish), the scatter plots
  a fixed-seed subsample of 400/2,000 DeepSets scenarios — stated on the
  slide itself.

Two LibreOffice-specific pitfalls hit during construction, both fixed and
worth knowing if this script is extended: (1) `series.has_data_labels =
True` alone creates an empty label — `show_value` must also be set
explicitly, or the label renders with nothing in it; (2) PowerPoint's
horizontal bar chart plots the **first** category at the *bottom*, so the
canonical map order has to be reversed in the data before it reads
top-to-bottom correctly. Verified visually via headless LibreOffice
(`soffice --headless --convert-to pdf`), not just by inspecting the XML.

## Results (mean Shapley per map, nats; mean |φ| in parentheses)

| Map | DeepSets | GNN | Set Transformer |
|---|---|---|---|
| Source evidence | +1.556 (1.622) | +1.660 (1.730) | +1.618 (1.696) |
| Sensor visibility | +0.051 (0.693) | +0.017 (0.703) | +0.048 (0.734) |
| Wind-error sensitivity | +0.361 (0.684) | +0.441 (0.688) | +0.330 (0.707) |
| Fit quality | +0.287 (0.416) | +0.167 (0.311) | +0.263 (0.384) |

The full map stack contributes ≈ 2.26–2.29 nats/scenario in every
architecture. **Source evidence dominates in all three** (~1.6–1.7 nats,
4–5× the next map) — the same ordering as Table II's one-at-a-time ablation,
now as an exact attribution on the same seed-1 models and scenarios, and
demonstrably not architecture-specific. Nuance the ablation can't show:
sensor visibility's *signed* mean is ≈ 0 but its mean |φ| is ~0.7 — it
helps or hurts per scenario in roughly equal measure, acting as context
rather than evidence.

## Sanity checks (printed on every run)

- **Efficiency**: max over scenarios of |Σᵢ φᵢ − (v(all) − v(none))| =
  **1.78e-15** for every model — exact by construction.
- **Zero-baseline**: the empty coalition matches the base-network-only
  prediction to ≤ 2e-4 nats. The trained head's `phys(0)` is a learned
  constant offset map (std 0.03–0.14 logits), not exactly zero — but a
  near-constant logit offset cancels in the softmax, so the zero-map
  coalition is the base model's prediction for all practical purposes (it
  was exactly zero at initialization).

## Checkpoints (seed-1 retrains, `PW_SAVE_CKPT=1`)

- `data/checkpoints/pw_noisy_ensr_seed1.pt` (DeepSets, val_nll 5.6595) —
  region sizes identical to the frozen audit for 2,000/2,000 scenarios
  (fully deterministic retrain).
- `data/checkpoints/pw_noisy_ensr_gnn_seed1.pt` (GNN, val_nll 5.6243) —
  region sizes identical to the frozen audit for 2,000/2,000 scenarios
  (fully deterministic retrain).
- `data/checkpoints/pw_noisy_ensr_st_seed1.pt` (Set Transformer, val_nll
  5.6336) — **not** a bit-exact reproduction. It lands close to both the
  frozen audit npz (median radius 69.48 m vs 69.41 m, 44.9% of the 2,000
  region sizes identical, p95 |Δradius| 0.37 m) and the committed
  `results/paired_wind.json` entry (val_nll 5.633561767578125 vs the
  committed 5.634796142578125) but matches neither exactly — the Set
  Transformer architecture apparently has a source of non-determinism
  (e.g. a non-deterministic attention/cuDNN kernel) that DeepSets and GNN
  training does not hit. The checkpoint used for this SHAP analysis is
  therefore a near-identical, not bit-identical, stand-in for the paper's
  ST model; `results/paired_wind.json` was left untouched (the retrain's
  slightly different numbers were not committed over it).

## Regeneration

```bash
source scripts/env.sh
python "Poster Graphics v2/shap/shap_maps.py"
# checkpoints, only if missing (deterministic; ~15 min each on an L40S):
PW_SAVE_CKPT=1 python src/paired_wind.py --stage train --views noisy --maps ensr --seed 1
PW_SAVE_CKPT=1 python src/paired_wind.py --stage train --views noisy --maps ensr --arch gnn --seed 1
PW_SAVE_CKPT=1 python src/paired_wind.py --stage train --views noisy --maps ensr --arch st --seed 1

# editable pptx (needs python-pptx; reads the CSV above, no GPU/model needed)
pip install --user python-pptx
python "Poster Graphics v2/shap/export_pptx.py"
```
