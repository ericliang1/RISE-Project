# Methane Poster Graphics

Poster-ready figures for the physics-guided calibrated-localization result,
following the poster graph-suggestions brief (`methane_poster_graph_suggestions.md`,
kept outside the repo).

Styling matches the paper's house style (`scripts/paper_data_figs.py`, which
produces `figures/paper/*`): the same
palette (orange = without physics maps / the baseline network, blue = with
physics maps, dashed ink = the 50 m reference), the same off-white surface,
DejaVu Sans, recessive grid, no top/right spines — with type scaled up for
reading from several feet away.

## Files

| Figure | Priority | What it shows |
|---|---|---|
| `01_region_radius_dumbbell` | 1 | before/after 90% guaranteed-region radius, per network |
| `02_scenarios_within_50m` | 2 | share of test scenarios reaching the 50 m facility scale |
| `03_component_ablation_waterfall` | 3 | the four physics maps added one at a time (DeepSets base) |
| `04_localization_example` | 4 | one held-out scenario: baseline vs physics-guided region |

Each is exported as **PDF** and **SVG** (vector, for poster scaling) and
**PNG at 300 dpi**.

## Where the numbers come from

Nothing is transcribed by hand. The script reads the frozen measured-wind
conformal audits under `/projectnb/rise-tower/eric1/csr-data`
(`pw_audit_pw_noisy_{nomaps,ensr}[_gnn|_st]_seed{1,2,3}_noisy.npz`), converts
region cell counts to equivalent-area radii, and reports the midpoint of the
three-seed range — the same statistic the paper's tables quote. The values
therefore reproduce Table I, Table II and Figs. 2–3 exactly:

- DeepSets 108–111 m → 70–73 m, 0% → 24–27% within 50 m
- GNN 69–71 m → 68–69 m, 1–5% → 27–29% within 50 m
- Set Transformer 80–82 m → 69–70 m, 7–12% → 26% within 50 m
- DeepSets ladder: 109 m → 88 (−21) → 83 (−5) → 76 (−8) → 72 m (−4)

Note that figure 1 annotates the *difference of medians* (−38 m for
DeepSets). Table I's ∆Radius column (−39.4 m) is the paired per-scenario
effect with its bootstrap CI — a different statistic, so don't mix the two on
the poster.

## Suggested captions

**Figure 1.** Physics-derived feature maps substantially reduced the
90%-guaranteed localization radius for DeepSets and the Set Transformer while
maintaining approximately 90% coverage. All three models converged to a radius
of about 68–73 m.

**Figure 2.** Adding physics-derived features increased the percentage of test
scenarios localized within the 50 m facility scale for all three neural-network
architectures.

**Figure 3.** The source-evidence map produced the largest individual
improvement, while the remaining maps provided complementary reductions.
Together, the four maps reduced the DeepSets localization radius by
approximately 38 m.

**Figure 4.** For the same simulated methane-leak scenario, the physics-guided
model produces a smaller 90%-guaranteed search region around the true source.

## Figure 4: scenario choice and one deviation

The scenario is chosen by a stated rule rather than by eye: among held-out
scenarios with exactly 8 masts whose regions *both* cover the true source, the
one jointly closest (log scale) to that subgroup's median baseline and median
physics-guided radii. It is typical of its subgroup, not a best case. The
current pick is test index 1254 (8 masts, 92 kg h⁻¹, 104 m → 62 m).

**Deviation from the instructions:** the suggestions ask for a predicted
probability/confidence heatmap in panels B and C. The saved audits keep the
conformal region masks but not the networks' per-cell probability maps, and no
`pw_*` checkpoints were kept, so the maps cannot be recovered without
retraining. The panels shade the 90% guaranteed region itself instead — the
quantity the result is actually about. If a probability heatmap is wanted,
the `pw_noisy_nomaps` and `pw_noisy_ensr` seed-1 runs have to be re-run with
the per-scenario softmax saved.

## Regeneration

```bash
source scripts/env.sh
python "Poster Graphics/generate_poster_graphics.py"
```

## Poster figure order

1. `01_region_radius_dumbbell` — largest, most prominent result figure
2. the paper's region-radius CDF (`figures/paper/fig_data_dist`)
3. the paper's sensor-count graph (`figures/paper/fig_data_masts`)
4. `03_component_ablation_waterfall` — can replace Table II
5. `04_localization_example` — if space permits

`02_scenarios_within_50m` is useful but can be dropped to save space; the
within-50 m percentages already appear on the waterfall's two totals.