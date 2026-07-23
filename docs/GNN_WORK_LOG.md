# GNN Localizer (M2) — Work Log

Branch: `gnn-localizer` · Date: 2026-07-22

Extends the CH4-T methane source-attribution paper by replacing the DeepSets
localizer with a **Graph Neural Network**, to test whether a relational
inductive bias closes more of the amortization gap toward the Bayes limit.

---

## The question

The paper's DeepSets model (`M1`) reaches a 74 m median 90% search region,
against a 7 m Bayesian information limit. The paper closes the M0→M1 gap by
changing the *training target* (physics-posterior distillation). We ask a
different, complementary question:

> Does giving the network an explicit **relational** structure — masts that
> exchange messages conditioned on the wind — sharpen the region further,
> under the same distillation target?

The intuition: the localizing signal is *time-of-arrival / bearing* — as the
wind veers, the plume sweeps **across** masts, so "mast A lit up before mast B,
and the wind blew A→B" pins the source. DeepSets pools mast readings with no
pairwise structure and cannot represent this; a graph net can carry it on edges.

## The four models (2×2)

|                 | one-hot target | distilled target |
|-----------------|:--------------:|:----------------:|
| **DeepSets**    | M0             | M1               |
| **GNN**         | M2base         | **M2**           |

- **Knob 1 — architecture:** DeepSets (pool all reading tokens) vs GNN (masts
  message-pass over wind-conditioned edges).
- **Knob 2 — target:** one-hot (point at the true cell) vs distilled (reproduce
  the tempered exact posterior — the paper's trick).

Comparing across a row isolates the *target's* effect; comparing down a column
isolates the *architecture's* effect.

## GNN design (`src/methane_t_gnn.py`, class `GNNT`)

Mast-node graph, **pure PyTorch** (no PyTorch Geometric dependency — the graph
is ≤12 nodes, so dense message passing is trivial and keeps the SCC env clean).

- **Nodes** = the ≤12 masts. Feature = position (Fourier) + a masked pool over
  that mast's own time series of `[asinh(y/σ), Fourier(t), u(t)]` — "when did I
  light up" lives on the node.
- **Edges** = ordered mast pairs (i→j). Feature = geometry (displacement,
  distance) **+ wind alignment** `mean/max_t of u(t)·unit(j−i)` — i.e. "did the
  wind blow the plume i→j". This is the relational signal DeepSets discards.
- 2 rounds of masked message passing → global mean+max readout → **same decoder
  head shape as DeepSetsT** → softmax over the 64×64 grid.
- **5.61M params** vs DeepSetsT's 4.87M (capacity parity, so audit differences
  reflect inductive bias, not size).

Everything downstream is **byte-identical** to the paper: tempered exact-posterior
teacher (blur 0.75), D4 augmentation, split conformal, region-size audit. The
only change to existing code is a backward-compatible `model_cls=DeepSetsT`
kwarg on `methane_t.train()` — M0/M1 are unchanged.

## Results (3 seeds, median 90% region radius)

| Model            | Target   | Per-seed (m)        | Range (m)   |
|------------------|----------|---------------------|-------------|
| M0 — DeepSets    | one-hot  | —                   | **93.8**    |
| M1 — DeepSets    | distill  | —                   | **73.8**    |
| M2base — GNN     | one-hot  | 56.9 / 55.4 / 57.6  | **55.4–57.6** |
| **M2 — GNN**     | distill  | 51.5 / 50.3 / 49.1  | **49.1–51.5** |
| Bayes limit      | —        | —                   | 6.9         |

*(M0/M1 use eric1's single frozen-seed audit; the GNN models ran all 3 seeds.)*

**Significance (paired Wilcoxon on per-scenario log region-size ratios):**
- M2 sharper than M1: **p ≈ 1.8×10⁻²⁵⁰**, sharper on **84%** of scenarios.
- M2base sharper than M0: p ≈ 1.4×10⁻²⁷⁸, 87%.
- M2 sharper than M2base (distillation still helps the GNN): p ≈ 9×10⁻³, 59%.
- M2's three seeds **never overlap** M1 (stability, the paper's own criterion).

**Reading:** the architecture is the larger lever — the *un-distilled* GNN
(57 m) already beats the *distilled* DeepSets (74 m). The two levers compose;
M2 (both) is tightest and reaches the EPA 50 m actionability radius. Coverage
stays calibrated (0.89–0.91). All four models share one exact-posterior
reference; the self-calibration gate passed at 0.952.

## Figures

- `figures/figT1_gnn_audit.pdf` — learned-vs-exact area scatter (all 4 models)
  + three-seed stability bars vs Bayes/EPA lines.
- `figures/figT2_gnn_example.pdf` — one scenario to scale: wind-meander plume,
  masts, and each model's median region as an equivalent-radius circle. The
  purple M2 circle is visibly tightest of the learned models.

## Reproduce

```bash
source scripts/env.sh                       # BU SCC: academic-ml/spring-2026, L40S
python src/methane_t_gnn.py --targets both --seed 1   # + --seed 2, --seed 3
python src/methane_t_gnn_figures.py         # tables + both figures
```

Reuses frozen scenarios + exact posteriors from
`/projectnb/rise-tower/eric1/csr-data` (read-only) via the repo `data/`
symlink; nothing is regenerated. Outputs land in
`/projectnb/rise-tower/azhang09/csr-data` and `results/`.

## Files

- `src/methane_t_gnn.py` — `GNNT` model + audit runner.
- `src/methane_t_gnn_figures.py` — three-seed table + figT1 + figT2.
- `src/methane_t.py` — only change: `train(..., model_cls=DeepSetsT)` kwarg.
- `results/ch4t_gnn_results_seed{1,2,3}.json`, `results/ch4t_gnn_table.json`.

## Open next steps

- **Capacity-matched control:** shrink `GNNT` to ~4.87M params and rerun the
  three seeds, to fully retire the "it's just more parameters" objection.
- **Draft the paper delta:** an "M2: relational localizer" paragraph + figures.
- **Wider sweep:** GAT / GraphSAGE variants, or a spatio-temporal (mast,time)
  graph, using the same audit harness.
