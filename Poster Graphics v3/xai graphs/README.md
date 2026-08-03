# XAI graphs (v3) — real-model figures, super-emitter benchmark

Publication-quality XAI figures computed against the **actual benchmark
model** (physics-guided DeepSets: `PhysHeadNet` = `DeepSetsT` + zero-init
conv head, `views=noisy, maps=ens, seed=1`, head-warmup recipe) and the
frozen super-emitter dataset (`csr-data-se48`). Nothing is schematic;
every heatmap, region, and attribution comes from a forward or backward
pass of the checkpoint.

## The checkpoint (and why it is exactly the benchmark's model)

Unlike v2 (which needed a deterministic retrain), the super-emitter
benchmark runs **saved their checkpoints directly**
(`csr-data-se48/checkpoints/pw_noisy_ens_seed1.pt`, val_nll 6.0424).
Verification against the frozen benchmark audit, printed on every run:

- per-scenario conformal region sizes **identical for 2,000/2,000** test
  scenarios (median |Δradius| = 0.0 m);
- coverage 0.9160 = 0.9160; median radius 75.3 = 75.3 m;
  <50 m 16.3 = 16.3% — matching `results/paired_wind.json`
  (`pw_noisy_ens_seed1`).

## Files

| File | What it is |
|---|---|
| `fig1_physics_to_region.{pdf,png,svg}` | hero panel: the three input maps + the model's softmax with the calibrated 90% region, masts, true source, argmax, scale bar |
| `fig1_maps_grid.{pdf,png,svg}` | compact 1×3 strip of the three maps alone (methodology inset; no model needed) |
| `fig1_scenario.json` | the chosen example scenario (index/id), written by `xai_fig1.py` and read by the other scripts — no hand-copied constants |
| `fig2_embedding_umap.{pdf,png,svg}` | UMAP of the decoder-input embedding, two panels (radius / masts), with evidence-map thumbnails on the extremes |
| `fig2_embedding_pca.{pdf,png,svg}` | same two panels for the PCA projection |
| `fig2_embeddings.npz` | the raw 2,000×576 embedding + both projections + labels |
| `fig3_integrated_gradients.{pdf,png,svg}` | IG of the argmax cell's log-prob w.r.t. each of the three input maps |
| `xai_fig1.py`, `xai_fig1_maps_grid.py`, `xai_fig23.py` | generation scripts (run from repo root; run `xai_fig1.py` first) |
| `xai_style.py` | shared style: canonical 3-map order, colormaps, glyphs |

## Reproducibility constants

- Scenario (Figs. 1 and 3): **`ch4t-test-000982`** (test index 982),
  6 masts, 425 kg h⁻¹, region radius 75.3 m ≈ the median, covered.
  Selection rule: 6-mast covered scenario with radius nearest the
  model's overall median (6 = midpoint of the 4–8 population).
- Conformal tail threshold `t_hat = 4.641442e-02` (α = 0.1, `score_seed`
  from `config/default.yaml`).
- Embedding hook: forward pre-hook on `model.decoder` (the concatenated
  mean-pool ⊕ max-pool ⊕ context vector, dim 576).
- UMAP `n_neighbors=30, min_dist=0.1, random_state=1`; PCA
  `random_state=1`.
- IG: zero-map baseline, 50 midpoint steps, maps pathway only (sensor
  tokens held fixed); completeness |ΣIG − Δlogp| = 0.0004 nats.

## Honest read of Figure 2 — the story CHANGED from v2

The v2 (full-population, joint-training) embedding organized by mast
count and radius structure was weak.  The v3 head-warmup model's
embedding **organizes by difficulty**: Spearman of UMAP axis 1 vs log
radius = **+0.57**, and **+0.56 within a mast count** (i.e. genuinely
difficulty, not mast count showing through); axis 1 vs masts is only
−0.11.  PCA agrees (−0.48 / −0.46 / +0.17).  Silhouette of a
median-radius split is 0.18 (UMAP) / 0.17 (PCA) — a gradient, not
clusters.  The figure titles are chosen from the measured correlations at
run time, so they cannot silently contradict the data.

## Figure 3 takeaway

IG at the predicted cell for the Figure-1 scenario: sensor visibility
ΣIG = +2.39 and source evidence +2.16 push toward the predicted cell,
wind-error sensitivity −2.79 pushes away; net, the maps pathway
contributes **+1.77 nats** to the argmax cell's log-probability
(logp −5.00 → −3.23 from zero maps to full maps).  Completeness:
ΣIG = +1.7676 vs logp(full) − logp(zero) = +1.7672.  Note this is one
scenario's attribution at the *argmax* cell — the population-level
ordering (source evidence dominant) is the exact Shapley analysis in
`../shap/`, computed on the *true* cell over all 2,000 scenarios; the
two answer different questions and need not agree scenario by scenario.

## Regeneration

```bash
source scripts/env.sh
python "Poster Graphics v3/xai graphs/xai_fig1.py"        # writes fig1_scenario.json
python "Poster Graphics v3/xai graphs/xai_fig1_maps_grid.py"
python "Poster Graphics v3/xai graphs/xai_fig23.py"
```

Checkpoints are already on disk (saved by the benchmark chains); no
retraining is needed.
