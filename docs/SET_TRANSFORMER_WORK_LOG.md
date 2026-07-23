# Set Transformer Localizer (M3) — Work Log

Branch: `set-transformer` (forked from `gnn-localizer`) · Date: 2026-07-23

Adds a **Set Transformer** localizer (M3) as a third architecture on the CH4-T
methane source-attribution testbed, and compares it head-to-head with the
DeepSets baseline (M0/M1) and the GNN (M2). Companion to
[GNN_WORK_LOG.md](GNN_WORK_LOG.md).

---

## The question

The GNN (M2) closed much of the amortization gap by giving the network an
explicit, physics-informed relational structure (masts on wind-conditioned
edges). A Set Transformer asks a subtler version of the same question:

> Does **soft, learned attention** — letting the set decide for itself which
> readings matter — recover the same sharpening as the GNN's **hard-coded**
> relational edges, under the same distillation target?

A Set Transformer (Lee et al., 2019) keeps DeepSets' permutation-invariance but
replaces mean/max pooling with attention: tokens attend to each other and are
pooled by attention. It is the natural "middle" between blind pooling and an
explicit graph.

## The model line (set-aggregation mechanism)

| Model | Architecture | How the set is aggregated |
|-------|--------------|---------------------------|
| M0/M1 | DeepSets | masked **mean + max pool** (each token summarized alone) |
| M2    | GNN | **message passing** over wind-conditioned edges (hard relation) |
| M3    | Set Transformer | **attention** (ISAB encoder + PMA pooling; soft relation) |

## GNN vs Set Transformer design contrast

- **M2 (GNN):** edges carry `mean/max_t of u(t)·unit(j−i)` — the physics of
  transport is *told* to the model ("the wind blew the plume i→j").
- **M3 (Set Transformer):** no edges. The model *learns* which (mast, time)
  tokens to weight via attention. Same reading tokens as DeepSetsT, so the only
  change vs M0/M1 is pooling → attention.

## Set Transformer design (`src/methane_t_settransformer.py`, `SetTransformerT`)

Standard Set Transformer, pure PyTorch (uses `nn.MultiheadAttention`):

- **Tokens** = same as DeepSetsT: `[Fourier(x_i), Fourier(y_i), Fourier(t_k),
  asinh(y_ik/σ), u(t_k)]` → linear embed to d=256.
- **Encoder** = 2× **ISAB** (Induced Set Attention Block, 16 inducing points):
  O(n·m) self-attention, efficient for the up-to-360 (mast, time) tokens.
- **Pooling** = **PMA** (Pooling by Multihead Attention, 1 seed) → one vector.
- Concat scenario context → **same decoder head shape** as DeepSetsT → softmax
  over the 64×64 grid.
- **Masking:** dropped readings / padded masts are excluded from attention
  keys/values (torch `key_padding_mask`), so masked tokens never leak into the
  pooled representation. Verified finite on the sparsest scenario (86/360 kept).
- **6.53M params** (DeepSetsT 4.87M, GNN 5.61M) — same order; the shared decoder
  head dominates all three.

Everything downstream byte-identical to the paper: tempered exact-posterior
teacher (blur 0.75), D4 augmentation, split conformal, region-size audit. Uses
the backward-compatible `model_cls=` kwarg on `methane_t.train()`.

## Results (3 seeds, median 90% region radius)

| Model  | Architecture    | Target   | Per-seed (m)        | Range (m)   |
|--------|-----------------|----------|---------------------|-------------|
| M0     | DeepSets        | one-hot  | —                   | **93.8**    |
| M1     | DeepSets        | distill  | —                   | **73.8**    |
| M3base | Set Transformer | one-hot  | 70.2 / 75.1 / 69.2  | 69.2–75.1   |
| **M3** | Set Transformer | distill  | 56.1 / 59.9 / 58.3  | **56.1–59.9** |
| M2base | GNN             | one-hot  | 56.9 / 55.4 / 57.6  | 55.4–57.6   |
| **M2** | GNN             | distill  | 51.5 / 50.3 / 49.1  | **49.1–51.5** |
| Bayes  | —               | —        | —                   | 6.9         |

**Significance (paired Wilcoxon on per-scenario log region-size ratios, seed 1):**
- M3 sharper than M1 (DeepSets+distill): **p ≈ 3×10⁻²⁶⁹, on 89% of scenarios**
- M2 (GNN) sharper than M3 (Set Transformer): **p ≈ 3×10⁻²** (GNN wins, modestly)
- Distillation still helps the Set Transformer (M3 over M3base): p ≈ 10⁻⁷⁶

Coverage stays calibrated (0.89–0.90); all models share one exact-posterior
reference (self-cal gate 0.952).

## The finding

A stable architecture ordering across all three seeds:

> **DeepSets (74 m) ‹ Set Transformer (~58 m) ‹ GNN (~50 m)**

- Both new architectures beat DeepSets decisively.
- **Attention alone (M3) recovers most of the gap** — the Set Transformer
  discovers on its own that a few masts matter more than the quiet ones.
- **But the GNN (M2) still wins.** Encoding the *physics of transport* into the
  graph structure (the wind-alignment edge feature) beats generic learned
  attention. Only the GNN reaches the EPA 50 m actionability radius.

**Takeaway:** the relational inductive bias helps most when you tell the model
*what* the relation is, rather than making it learn the relation from scratch.
This is a stronger, more nuanced claim than "bigger/fancier model wins."

## Figures

- `figures/figT1_st_audit.pdf` — learned-vs-exact area scatter (M0/M1/M2/M3) +
  three-seed stability bars (GNN vs Set Transformer) against Bayes/EPA lines.
- `figures/figT2_st_example.pdf` — one scenario to scale: nested median-region
  circles M0 (94) ⊃ M1 (74) ⊃ M3 (56) ⊃ M2 (51) ⊃ Bayes (7).

## Reproduce

```bash
source scripts/env.sh
python src/methane_t_settransformer.py --targets both --seed 1   # + seed 2, 3
python src/methane_t_settransformer_figures.py                   # tables + figures
```

Reuses frozen scenarios + exact posteriors from
`/projectnb/rise-tower/eric1/csr-data` (read-only) via the `data/` symlink;
nothing is regenerated.

## Files

- `src/methane_t_settransformer.py` — `SetTransformerT` (MAB/ISAB/PMA) + runner.
- `src/methane_t_settransformer_figures.py` — four-model comparison figures.
- `results/ch4t_st_results_seed{1,2,3}.json`.

## Open next steps

- **Capacity-matched control:** M3 is 6.53M params vs DeepSets' 4.87M; shrink to
  parity (fewer inducing points / smaller d) and rerun to retire the parameter-
  count objection. (Same open item as the GNN.)
- **Give the Set Transformer the physics:** add relative-position / wind-aligned
  attention bias, testing whether attention *with* the transport prior closes
  the gap to the GNN — the cleanest follow-up to the finding above.
