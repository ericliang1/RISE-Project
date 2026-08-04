# What changed on `super-emitter` vs `main`

Summary of the `super-emitter` branch (== `final-benchmark`, same commit) for
anyone picking this up cold. 17 commits, 57 files changed, all dated
2026-07-31 – 2026-08-01. Full dated decision log with measurements lives in
[GATES.md](GATES.md); this file is the condensed version.

## New data: the benchmark population changed

The dataset moved from the paper's full population
(q ~ LogU(10, 500) kg/h, 4–12 masts) to **super-emitters only**:

- **q ~ LogU(100, 500) kg/h** — every source is a super-emitter
  (`Q_LO` 10 → 100 in [src/methane_pipeline.py](src/methane_pipeline.py) and
  the matching draw in [src/methane_t.py](src/methane_t.py))
- **4–8 sensor masts** (was 4–12)
- New data directory `/projectnb/rise-tower/eric1/csr-data-se48` — kept
  separate from the original `csr-data` symlink so the full-population
  artifacts stay intact; old full-pop results archived to
  `paired_wind_fullpop.json`
- `K_TEACHER` 16 → 8, so the physics-only oracle baseline gets the same
  wind-marginalization budget as the maps (a fairness fix)

**The mast range was iterated four times** before landing on 4–8, and
GATES.md discloses this explicitly as a forking-paths risk:

| Range | Outcome |
|---|---|
| 6–12 masts | Maps *hurt* GNN/Set Transformer on dense networks ("equalizer" pattern). Cancelled at 16/18 runs; archived as `paired_wind_se612.json`. |
| 2–8 masts | 3 runs done, cut short; archived as `paired_wind_se28_partial.json`. |
| **4–8 masts (final)** | The range the full-population forecast supported (all-architecture map gains, +16 to +27 pp within-50 m). Used for every result below. |

All intermediate evidence (6–12 and 2–8 grids) is retained on disk for the
record, not deleted.

## Feature map removed (and one that came and went)

- **Residual map dropped** (`6926a49`, "doesn't do much and too
  complicated"): the final method is now **`ens` — 3 channels** (mean
  evidence z, wind spread, mean log b, all over K=8 shared wind draws).
  Full-population evidence backed this: the residual was the smallest
  contributor at −4.0 m [−5.6, −2.7]. Side effect: "the same K wind draws
  calculate all maps" is now literally true — the residual used a separate
  RNG stream (93) that no longer exists.
- **A 4th "ensq" channel** (rate-consistency, asinh best-fit rate) was
  added and reverted the same day (`deae3c3`, "user directive"). The
  method is frozen at 3 maps.
- **The evidence map's transform changed**: the old hard
  `clip(z, ±60) / 10` became **`asinh(z / 10)`**
  ([src/methane_t_uncertain.py](src/methane_t_uncertain.py)). Measured on
  the super-emitter population: the old clip saturated ~30% of near-source
  cells on average (train-split measurement, not test), handing the
  network a flat plateau exactly where the peak belongs. asinh matches
  `z/10` in the small-signal regime and stays log-compressing (never flat)
  for strong sources. Old-clip artifacts are archived for comparison — they
  show an 81–83 m ceiling vs. 61–65 m after the fix.

## Training recipe: head warmup

`PW_HEAD_WARMUP=100`: the physics head is frozen for the first 100 epochs,
then joins from zero-init. Adopted after the GNN's median radius
*regressed* under joint training; validated by a pre-specified pilot
(GNN ens seed 4: 61.7 m / 28.4% <50 m vs. 72–74 m joint). Applied
identically to every map configuration; the joint-trained grid is archived
as the "before" (`paired_wind_se48_joint.json`). Baselines are unaffected
(no physics head to warm up).

## Infrastructure hardening

An adversarial review (4 lenses, 10 agents) before the final launch drove
several fixes in [src/common.py](src/common.py) and
[src/paired_wind.py](src/paired_wind.py):

- `savez_atomic` — tmp file + `os.replace`, so a wall-clock kill can't
  leave a truncated `.npz` that passes a resume guard
- `CSR_DATA_DIR` env override — lets a new benchmark generate data while
  other chains still read the config'd directory
- Resume-skip guards in datagen (`stage_gen` checks for existing map/oracle
  files before regenerating)
- File locking on `paired_wind.json` (`fcntl`) for concurrent chain writes
- CUDA preflight, to catch silent-CPU-burn jobs early

## Final results (4–8 masts, q ≥ 100 kg/h, 3 seeds)

| Model | Without maps | With `ens` maps | Paired Δ radius [95% CI] | Δ within-50 m |
|---|---|---|---|---|
| Physics-only baseline | 282 m median, 0% <50 m | — | — | — |
| DeepSets | 97–100 m | 73.5–75.3 m, 16–17% <50 m | −24.6 m [−25.8, −23.0] | +16.3 pp |
| GNN | 67.6–69.3 m | 61.1–62.3 m, 27–29% <50 m | −5.2 m [−6.4, −4.2] | +20.5 pp |
| Set Transformer | 72.3–75.1 m | 63.9–65.2 m, 28–30% <50 m | −10.7 m [−12.1, −9.5] | +20.9 pp |

Coverage holds at 0.887–0.916 across every run (target 0.90). Every paired
CI excludes zero — unlike the full-population paper, where the GNN's radius
CI crossed zero.

Deliverables: `results_package/tables_se48.pdf` (Tables I & II with paired
bootstrap CIs), both data figures, and `technical_reference.pdf` (the
complete training/statistical/verification record). Everything traces to
per-scenario audit files in `csr-data-se48` and to
`results/se48_table_deltas_ci.json`.

## Downstream

`Poster Graphics v3/` rebuilds the full poster figure set against this
branch (3-map method, 4–8 masts, `ens`) — see its own README for what
changed at the figure level, including two findings that flipped from the
full-population version: the known-wind posterior is no longer always a
near-delta (only ~52% of scenarios exceed 90% concentration in one cell),
and the model's internal embedding now organizes by difficulty rather than
mast count.
