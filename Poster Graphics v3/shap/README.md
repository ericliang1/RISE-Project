# SHAP analysis (v3) — map-level exact Shapley values, three maps

How much does each of the **three** physics feature maps contribute?  Each
map is one feature ("present" = the real map, "absent" = the zero baseline
the conv head was zero-initialized against), so with 3 features all
2^3 = **8 coalitions** are enumerated and the Shapley values are **exact** —
no KernelSHAP sampling.  Value function: v(S) = log p(true source cell)
under the maps in S, i.e. the training objective; a Shapley value is the
map's average marginal NLL-reduction in **nats**.  Computed per scenario
over all 2,000 seed-1 super-emitter test scenarios
(`ch4t-test-000000 .. 001999`), for all three architectures.

Exactness shortcut: logits = base(x) + conv(M(x)), so the set-network
forward runs once per scenario and only the conv head re-runs per
coalition — mathematically identical to 8 full forwards.

## v3 changes vs v2

- **Three maps** (`ens`: source evidence, sensor visibility, wind-error
  sensitivity) — the residual "fit-quality" map is no longer part of the
  method.
- **Beeswarm colour = feature value** (user request): the vertical
  colourbar now encodes each map's per-scenario **spatial mean**,
  rank-scaled to a percentile within its row — the standard SHAP
  summary-plot gradient (house blue → violet → red, low → high) — instead
  of the mast count.
- Checkpoints come straight from the benchmark training runs
  (`csr-data-se48/checkpoints/pw_noisy_ens[_gnn|_st]_seed1.pt`); no
  retraining, and no bit-exactness caveats — the DeepSets checkpoint is
  verified region-identical to the frozen audit in `../xai graphs/`.

## Files

| File | What it is |
|---|---|
| `shap_maps.py` | the analysis (run from repo root) |
| `shap_map_contributions.csv` | 6,000 rows = 3 models × 2,000 scenarios: per-map Shapley values, per-map feature values (spatial means), v(all), v(none), efficiency residual |
| `shap_map_summary.{pdf,png,svg}` | poster figure: mean \|φ\| per map × model + DeepSets beeswarm coloured by feature value |
| `shap_map_summary.pptx` | **editable** PowerPoint version — see below |
| `export_pptx.py` | builds the pptx from `shap_map_contributions.csv` only (no model/GPU access) |

## The PowerPoint export

`shap_map_summary.pptx` rebuilds both panels as **native PowerPoint chart
objects**, not an embedded picture — every number, color, and label is
editable directly in PowerPoint (right-click a chart → *Edit Data in
Excel*):

- **Panel A**: a native clustered bar chart, one series per model (mean
  |Shapley value| per map, matching `shap_map_summary.png` to 2 decimal
  places).
- **Panel B**: a native XY scatter chart of the DeepSets per-scenario
  values.  PowerPoint scatter charts have no continuous-colormap option,
  so the feature value is bucketed into 3 editable series (low / mid /
  high **per-row terciles** of the map's spatial mean, colours sampled
  from the PNG's gradient) instead of the smooth per-row percentile
  colouring; black diamonds are the **full 2,000-scenario mean**.  For
  editability (thousands of native points make PowerPoint sluggish), the
  scatter plots a fixed-seed subsample of 400/2,000 DeepSets scenarios —
  stated on the slide itself.

The two LibreOffice pitfalls found in v2 still apply and are handled:
data labels need `show_value = True` explicitly, and horizontal bar
charts plot the first category at the bottom (category order reversed in
the data).  Rendering verified via headless LibreOffice
(`soffice --headless --convert-to pdf`).  Known cosmetic limit: series
draw in order, so where terciles overlap densely the later (high-value)
series can sit on top of the earlier ones.

## Results (mean Shapley per map, nats; mean |φ| in parentheses)

| Map | DeepSets | GNN | Set Transformer |
|---|---|---|---|
| Source evidence | +0.787 (0.888) | +0.373 (0.500) | +0.596 (0.709) |
| Sensor visibility | +0.104 (0.381) | +0.009 (0.186) | +0.053 (0.304) |
| Wind-error sensitivity | −0.108 (0.349) | −0.046 (0.180) | −0.100 (0.300) |

The full map stack contributes **0.78 / 0.34 / 0.55** nats/scenario
(DeepSets / GNN / Set Transformer) — smaller than the full-population
v2 numbers (≈2.3 nats everywhere), consistent with the harder
super-emitter benchmark, the stronger warmed baselines, and the leaner
3-map stack.  **Source evidence dominates in all three architectures**
(1.8–2.7× the next map's |φ|) — the same ordering as Table II's ladder,
now as an exact attribution on the same seed-1 models and scenarios.
Nuance the ladder can't show: wind-error sensitivity's *signed* mean is
slightly **negative** (−0.05..−0.11) while its mean |φ| is 0.18–0.35 — it
acts as context that helps or hurts per scenario in roughly equal
measure, with the balance tipping marginally negative on the true-cell
objective; the beeswarm's feature-value gradient shows the high-evidence
scenarios driving the positive tail of the evidence map's row.

## Sanity checks (printed on every run)

- **Efficiency**: max over scenarios of |Σᵢ φᵢ − (v(all) − v(none))| =
  **8.9e-16** for every model — exact by construction.
- **Zero-baseline**: the empty coalition matches the base-network-only
  prediction to ≤ 0.013 nats (mean ≤ 2e-4).  The trained head's `phys(0)`
  is a learned constant offset map (std 0.01–0.04 logits), not exactly
  zero — but a near-constant logit offset cancels in the softmax, so the
  zero-map coalition is the base model's prediction for all practical
  purposes (it was exactly zero at initialization).

## Regeneration

```bash
source scripts/env.sh
python "Poster Graphics v3/shap/shap_maps.py"

# editable pptx (needs python-pptx; reads the CSV above, no GPU/model needed)
python "Poster Graphics v3/shap/export_pptx.py"
```
