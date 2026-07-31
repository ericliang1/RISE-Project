# Methane Poster Graphics — v2

Second batch of poster-ready figures for the physics-guided calibrated-
localization paper, complementing `Poster Graphics/` (v1: dumbbell, within-50 m
bars, ablation waterfall, example scenario). This batch adds the **XAI
visuals**, the **to-scale area comparison with the 12-mast subgroup**, and
poster-scale remakes of the paper's Figs. 2–3 plus two figures the paper only
has as table columns (paired effects, coverage).

Styling continues the v1 house style: orange = without physics maps, blue =
with physics maps, dashed ink = the 50 m facility-scale reference. Figures
that separate the three networks use DeepSets **blue** `#2a78d6` / GNN **red**
`#c94040` / Set Transformer **green** `#1baf7a` — the trio passes the
colour-vision-deficiency separation checks, and every curve is direct-labelled
so identity never rests on colour alone.

## Files

| Figure | What it shows |
|---|---|
| `01_xai_feature_maps` | the four physics input maps for one held-out scenario, plus the observed scene and the exact Bayes reference (which collapses to a single 7.8 m cell under known wind) |
| `02_xai_umap_embedding` | UMAP of the 2,000 source-evidence maps: the unsupervised layout reconstructs the site (panels A/B colour by true source x/y) and region size is locally structured (panel C) |
| `03_region_shapes_12masts` | the actual conformal 90% regions for the largest-contrast 12-mast scenario (labelled as a best case), one panel per model: baseline (orange) vs physics-guided (blue), in the style of the paper's example figure |
| `04_radius_cdf` | paper Fig. 2 at poster scale: region-radius CDFs, solid = with maps, dashed = without |
| `05_masts_curves` | paper Fig. 3 at poster scale: median radius vs mast count, three seeds pooled |
| `06_paired_effects_forest` | Table I's paired-effect columns as a forest plot with 95% bootstrap CIs |
| `07_coverage_calibration` | observed coverage per model/seed against the nominal 90% line |
| `01.5_xai_wind_ensemble` | "through the model's eyes": the eight per-wind-draw evidence maps behind the ensemble channels, then consensus, disagreement, and the final conformal region |

Each figure is exported as **PDF** and **SVG** (vector) and **PNG at 300 dpi**.

## Where the numbers come from

Nothing is transcribed by hand except Fig. 06's CIs. The script reads the
frozen files under `/projectnb/rise-tower/eric1/csr-data`:

- conformal audits `pw_audit_pw_noisy_{nomaps,ensr}[_gnn|_st]_seed{1,2,3}_noisy.npz`
  (region sizes → equivalent-area radii; `covered` where stored),
- physics feature maps `ch4tu_test_maps_ens.npz`
  (channels: mean evidence z̄, wind spread 3·std z, mean log b) and the
  marginalized residual `pw_test_resid_marg.npz`,
- test scenarios `ch4t_test.npz` and the exact posterior
  `ch4t_test_posterior.npz`,
- `results/table_deltas_ci.json` for the paired-bootstrap CIs (Fig. 06).

The example scenario in Fig. 01 is chosen by the same stated rule as v1 and
the paper's map figures (8-mast subgroup, both regions cover, jointly closest
to the subgroup medians → test index 1254). Fig. 03 uses a different,
deliberately favourable rule: among 12-mast scenarios where **all six**
seed-1 regions cover the source, the one **maximizing the
baseline-to-physics contrast** (sum over models of log baseline radius minus
log physics-guided radius) → test index 777, 59 kg h⁻¹, giving 87/57/58 m
without maps vs 23/20/18 m with. It is a best case and the figure's
subtitle says so — do not quote its numbers as typical (the medians are in
this README and Figs. 04–05). The region masks come straight from the
audits' `masks` arrays, so the shapes are the actual conformal sets, not
schematics.

Key derived numbers (medians, three seeds pooled):

- with-maps all scenarios: DeepSets 71 m, GNN 69 m, Set Transformer 70 m
- with-maps 12-mast subgroup: 58 / 55 / 56 m (33–35% less area than "all")
- without-maps all scenarios: 110 / 70 / 81 m

## The UMAP figure (Fig. 02), honestly stated

The embedding is UMAP (`n_neighbors=30, min_dist=0.15, random_state=1`) of the
per-scenario **source-evidence map** (first `ens` channel), z-scored per
scenario and reduced to 50 PCA components. What it genuinely shows:

- Panels A/B: the two embedding directions recover the site's east–west and
  north–south axes — the physics input encodes the source location before any
  network is trained.
- Panel C: region size is *locally structured* but not cleanly clustered;
  the caption quotes the measured effect (Spearman rank correlation 0.33
  between a scenario's log-radius and its 15 nearest neighbours' mean,
  p ≈ 10⁻⁵¹).
- Wind direction was tested as a colouring and shows **no** visible
  organization (several UMAP variants tried); it is deliberately not claimed.

## The wind-ensemble figure (Fig. 01.5)

The eight per-draw evidence maps are **regenerated, not schematic**: the
script imports `src/methane_t_uncertain.py` and replays `stage_gen`'s exact
construction (`perturb_wind` with RNG streams `[root_entropy, 92, split_tag,
i, k]`, `root_entropy = 20260717`), then verifies the recomputed mean against
the stored `ens` channel — max abs difference 6×10⁻⁷ (float32 rounding).
Panels A/B show the stored channels themselves; panel C overlays the ensr
seed-1 conformal region (62 m) for the same scenario as Fig. 01
(test index 1254).

## Caveats to keep on the poster

- Figs. 03/04 use the prespecified seed-1 run (same convention as Table I's
  paired effects); Fig. 05 pools the three seeds.
- Fig. 06: the GNN's radius CI includes zero — the maps do not shrink the
  GNN's already-small regions, but its within-50 m share still improves.
- Fig. 07: the DeepSets seed-2/3 audits stored only region sizes, so
  DeepSets shows one dot per condition (seed 1).
- Fig. 01 panel F: the exact posterior is a genuine delta (all mass in one
  cell) for this scenario — drawn as geometry because a heatmap of it is
  blank.

## Regeneration

```bash
source scripts/env.sh
python "Poster Graphics v2/generate_poster_graphics_v2.py"
```

Requires `umap-learn` in addition to the env's numpy/matplotlib/sklearn
(installed in the spring-2026 env with `pip install --user umap-learn`).

## Suggested poster layout with v1

1. v1 `01_region_radius_dumbbell` — headline result
2. v2 `04_radius_cdf` or v1's CDF slot — distribution-level evidence
3. v2 `03_region_shapes_12masts` — the "what it means on the ground" visual
4. v2 `01.5_xai_wind_ensemble` — the mechanism in one picture; strongest XAI centrepiece
5. v2 `01_xai_feature_maps` and `02_xai_umap_embedding` — supporting XAI panels
6. v2 `06_paired_effects_forest` + `07_coverage_calibration` — rigor corner
7. v1 `03_component_ablation_waterfall`, v2 `05_masts_curves` — ablations
