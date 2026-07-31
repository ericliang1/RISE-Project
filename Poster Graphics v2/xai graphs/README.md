# XAI graphs — real-model figures

Publication-quality XAI figures computed against the **actual paper model**
(physics-guided DeepSets: `PhysHeadNet` = `DeepSetsT` + zero-init conv head,
`views=noisy, maps=ensr, seed=1`) and the frozen CH4-T dataset. Nothing is
schematic; every heatmap, region, and attribution comes from a forward or
backward pass of the checkpoint.

## The checkpoint (and why it is exactly the paper's model)

The paper's `paired_wind.py` training runs never saved weights, so the model
was **deterministically retrained** (all seeds hard-coded in `stage_train`;
frozen data views) with a minimal, env-gated addition to
`src/paired_wind.py` (`PW_SAVE_CKPT=1` → saves
`data/checkpoints/pw_noisy_ensr_seed1.pt`). Verification against the frozen
paper run:

- per-scenario conformal region sizes **identical for 2,000/2,000** test
  scenarios (median |Δradius| = 0.0 m);
- coverage 0.9000 = 0.9000; median radius 70.7 = 70.7 m; <50 m 26.6 = 26.6%;
- best val NLL matches the committed `results/paired_wind.json` entry to
  full precision (5.65950146484375).

Training log: `retrain_ensr_seed1.log`.

## Files

| File | What it is |
|---|---|
| `fig1_physics_to_region.{pdf,png,svg}` | hero panel: the four input maps + the model's softmax with the calibrated 90% region, masts, true source, argmax, scale bar |
| `fig2_embedding_umap.{pdf,png,svg}` | UMAP of the decoder-input embedding, two panels (radius / masts), with evidence-map thumbnails on the extremes |
| `fig2_embedding_pca.{pdf,png,svg}` | same two panels for the PCA projection |
| `fig2_embeddings.npz` | the raw 2,000×576 embedding + both projections + labels |
| `fig3_integrated_gradients.{pdf,png,svg}` | IG of the argmax cell's log-prob w.r.t. each input map |
| `xai_fig1.py`, `xai_fig23.py` | generation scripts (run from repo root) |

## Reproducibility constants

- Scenario (Figs. 1 and 3): **`ch4t-test-001811`** (test index 1811),
  8 masts, 100 kg h⁻¹, region radius 71.5 m (≈ the 71 m median), covered.
  Selection rule: 8-mast covered scenario with radius nearest 71 m.
- Conformal tail threshold `t_hat = 1.129459e-01` (α = 0.1, `score_seed`
  from `config/default.yaml`).
- Embedding hook: forward pre-hook on `model.decoder` (the concatenated
  mean-pool ⊕ max-pool ⊕ context vector, dim 576).
- UMAP `n_neighbors=30, min_dist=0.1, random_state=1`; PCA `random_state=1`.
- IG: zero-map baseline, 50 midpoint steps, maps pathway only (sensor
  tokens held fixed); completeness |ΣIG − Δlogp| = 0.001 nats.

## Honest read of Figure 2 (per the brief: do not oversell a blob)

The embedding organizes **strongly by mast count** (Spearman of UMAP axis 1
vs masts = −0.79; PCA axis 1 = +0.92) — unsurprising, since `n_sensors` is
an explicit input to the context MLP. Radius structure is **weak**:
silhouette of a median-radius split is 0.03 (UMAP) / 0.04 (PCA), and the
apparent radius gradient collapses to ≈ +0.06 mean Spearman *within* a mast
count — i.e., it is mostly mast count showing through, not learned
difficulty. Stability shows no visible organization. If Figure 2 is kept,
present it as "the embedding encodes network size", not "the embedding
predicts difficulty"; otherwise cut it — Figures 1 and 3 carry the XAI
story.

## Figure 3 takeaway

IG at the predicted cell: source evidence ΣIG = +1.88 > sensor visibility
+1.35 > fit quality +0.49 > wind-error sensitivity ≈ 0.00 (this scenario), with
attribution concentrated on the predicted cell and a mild negative band
upwind. The maps pathway contributes +3.72 nats to the argmax cell's
log-probability (logp −7.53 → −3.81 from zero maps to full maps). The
evidence-map dominance matches Table II's ablation ordering.

## Regeneration

```bash
source scripts/env.sh
python "Poster Graphics v2/xai graphs/xai_fig1.py"
python "Poster Graphics v2/xai graphs/xai_fig23.py"
# checkpoint (only if data/checkpoints/pw_noisy_ensr_seed1.pt is missing):
PW_SAVE_CKPT=1 python src/paired_wind.py --stage train --views noisy --maps ensr --seed 1
```
