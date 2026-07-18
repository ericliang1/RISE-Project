# Gates, measurements, and documented decisions

Machine-readable values: `results/gates.json`. This file is the honest human
summary: every gate, its measured value, pass/fail, plus deviations, nulls,
and anomalies. (Being finalized as stages complete; placeholders marked TBD.)

## Environment (Stage 0)

- BU SCC, SGE scheduler (no SLURM on this cluster — the SLURM questions from
  the runbook are moot). All stages ran **inside an interactive GPU job**
  (OOD code-server session, 8 CPU slots, one dedicated NVIDIA L40S 46 GB via
  `CUDA_VISIBLE_DEVICES`), so no batch submission scripts were needed.
- Environment: `module load miniconda academic-ml/spring-2026`,
  `conda activate spring-2026-pyt` → Python 3.12.11, PyTorch 2.9.1+cu128,
  numpy 2.2.6, scipy 1.16.0. Recorded in `scripts/env.sh`.
- Storage: home quota is 10 GB, so `data/` symlinks to
  `/projectnb/rise-tower/eric1/csr-data` (persistent project storage).

## G1 — physics validation: **PASS**

Measured values in `results/gates.json → G1` (fill Appendix A):

- Quadrature 64- vs 128-node median relative error: **3.6e-11** (gate < 1e-6).
- Zero-wind closed form (E1) median relative error: **~1e-12** (max ~1e-9).
- Rotation/reflection symmetry max relative error: **< 1e-10**.
- Linearity max relative error: **< 1e-12** (float rounding).
- FD cross-check (2nd-order central + Heun, h = 1/768, CFL 0.4, enlarged box,
  Dirichlet far boundary): **max 0.50% rel L2** over the 17/20 random
  scenarios with non-negligible signal (gate < 1%); the 3 scenarios whose
  sensors see essentially no plume (reference RMS 1e-18..1e-9) agree in
  **absolute** RMS to ≤ 1.9e-10 (gate 1e-4 = 1% of the smallest noise std).
- FD self-convergence (h = 1/256, 1/384, 1/512): monotone, ~2nd order.

### Documented decisions (Stage 1)

1. **tau_max = 12** for the log-age substitution s = t·exp(−τ). Tuned
   empirically: median 64-vs-128 error 3.6e-11; truncated tail < ~1e-6
   everywhere including sensor-on-source. (tau_max = 40 wastes nodes on a
   dead tail and fails the gate at 2e-4.) Node-resolution error on rare
   sharp-plume probes reaches ~1e-3 relative in the far tail of the probe
   distribution; data generation and the oracle share the *same* 64-node
   operator, so they remain exactly self-consistent, and absolute physical
   fidelity is bounded by the FD gate.
2. **FD cross-check metric** is two-branch (relative where there is signal,
   absolute at the noise floor) because relative L2 is ill-conditioned when a
   random scenario's sensors never see the plume.
3. C_ref = 1.0 (config), noise σ is on the normalized scale.

## Stage 2 — benchmark freeze: **PASS**

- 10,000/1,000/2,000/2,000 train/val/calib/test + 50 dose-response base
  scenarios (master 12-sensor layouts + insertion orders, N=6 geometry pairs,
  7-point σ sweeps). Total stored: **17.5 MB** compressed (< 10 GB budget).
- SHA-256 checksums: `results/checksums.json`.
- **Scenario-ID leakage audit: PASS** (zero ID overlap between any pair of
  splits; zero duplicated scenario content across all 15,000+ scenarios).
- Per-scenario RNG: `SeedSequence([root_entropy=20260717, split_tag, index])`;
  params → sensors → noise → dropout drawn from that single per-scenario
  stream. Documented choice: **dose-response sets use p_drop = 0** (the
  boundary of the training range, in-support) so that knob interventions are
  not confounded by random reading loss.
4. **Latin-hypercube source locations** for the 50 dose-response base
   scenarios ("roughly spanning the domain" per the runbook).

## Adversarial code review (pre-run)

A 10-agent adversarial review + refutation pass over the full pipeline was run
before any results were produced. Two findings survived verification and were
fixed before Stage 4/5 ran:

1. **CRITICAL — the paper's "32-node Gauss–Legendre in log q" is numerically
   inadequate** for marginalizing the release rate: the likelihood in q is a
   Gaussian of width σ_q = σ/‖g_c‖ around q̂_c = ⟨y,g_c⟩/‖g_c‖², which for
   sharp in-prior scenarios is up to ~1000× narrower than the fixed rule's
   node spacing; reproduced log-posterior errors of many nats vs a converged
   reference (the pipeline's own brute-force gate fails). **Deviation:** we
   marginalize with a peak-aware composite Gauss–Legendre rule over q∈[0.5,5]
   (panels split at q̂_c ± 8σ_q; 32 peak nodes + 2×16 outer nodes), validated
   against a 10× deep rule and a 200k-node trapezoid (gate < 1e-6). The
   paper's method section should be amended accordingly.
2. Minor: model heatmap caches now carry the checkpoint SHA-256 and
   invalidate on mismatch (stale-cache hazard on retrain).

## G2 — model training: **PASS (both seeds), with one documented deviation**

**Finding (important for the paper):** the paper's training recipe as written
(DeepSets 4.87M params — within the "~5M, verify count" spec — AdamW 3e-4
cosine, batch 256, CE to smoothed target, no augmentation) **memorizes the
10,000 training scenarios with zero generalization**: train-eval NLL → 1.5
while val NLL never improves below ~7.96 (≈ uniform over 4096 cells;
log 4096 = 8.32), and val MAP error 0.44 is worse than the peak-sensor
heuristic (0.37) and no better than prior-random guessing (0.42). Verified
not caused by AMP (identical in fp32), learning rate (1e-4: same shape,
slower), or the context branch (zeroing it still memorizes via the token
stream). The frozen data itself is highly informative (exact-posterior oracle
on frozen val scenarios: median MAP error 0.012).

**Deviation (training protocol only):** each training batch gets a random D4
symmetry of the unit square (rotation k·90° + reflection) applied jointly to
sensors, wind, and source. This is an exact physics symmetry (G1 symmetry
gate) and the benchmark prior is D4-invariant, so augmented scenarios are
exact draws from the same prior; data, architecture, loss, and all specified
hyperparameters are unchanged. With it, val NLL tracks train loss with no
overfitting gap.

Results (val split): seed 1 MAP error mean/median **0.180 / 0.101**, seed 2
**0.174 / 0.103**, vs peak-sensor **0.367 / 0.346** — the model clearly beats
the baseline (gate condition). Best val NLL 6.306 (seed 1, epoch 81 of 106;
early stop) / 6.29 (seed 2). Training cost ≈ 0.01 GPU-h per seed on the L40S
(the paper's 12–20 GPU-h estimate is very conservative for this GPU).
Leakage audit re-verified before training.

## G3 — conformal calibration (H1): machinery verified; seed-dependent verdict

**Numerical finding #2 (important for the paper):** the paper's nonconformity
score s = (mass above the true cell) saturates in floating point for sharp
heatmaps: when p(true cell) ~ 1e-30, `1 - tiny` rounds to exactly 1.0
(catastrophic cancellation), creating an atom of tied scores at the top of the
score distribution. The calibration quantile lands inside the atom and the
region rule then excludes every atom member — conformal-on-exact-posterior
covered only 0.84 instead of 0.90 (and float32 posterior storage made it
worse, 0.82). **Fix (exact-arithmetic-equivalent):** all scoring, threshold,
and region logic works with the complement tail mass t = 1 − s (mass at or
below the true cell, complementary randomization), which is representable at
both ends (the exact-posterior threshold is t̂ ≈ 4e-199). In-sample coverage
is 0.9005 = 1801/2000 by construction, verifying the quantile indexing.

Results at 90% on the 2000-scenario test split:

| heatmap | coverage | Clopper–Pearson 95% CI | contains 0.90 |
|---|---|---|---|
| learned, seed 1 | 0.8750 | [0.860, 0.889] | **no** |
| learned, seed 2 | 0.8960 | [0.882, 0.909] | yes |
| exact posterior (oracle) | 0.9070 | [0.893, 0.919] | yes |

**Honest verdict on the seed-1 shortfall** (investigated exhaustively): the
implementation is exact (in-sample 0.9005; U-randomization irrelevant —
coverage spread over 20 U-seeds < 0.0005); calib and test are exchangeable
(exact-posterior scores KS p = 0.90, learned scores KS p = 0.51; the exact
oracle covers on the same splits); and swapping the roles (calibrate on test,
cover calib) OVER-covers by the mirror amount (0.9165). A permutation test
puts the probability of a test pass-rate this low under exchangeability at
p = 0.0045. Conclusion: a genuine ~1-in-220 finite-sample fluctuation of this
particular (model, calibration-split, test-split) triple, reported as such —
not an implementation defect. The formal G3 criterion (CI contains 0.90)
passes for seed 2 and for conformal-on-exact, and fails for seed 1.

## G4 — exact posterior + audit: **PASS**

- Normalization: max |1 − Σp| = 3.4e-13 (calib), 8.9e-14 (test); gate 1e-6.
- Brute-force marginalization cross-check: composite rule vs 10×-node deep
  rule max |Δ log posterior| = 5.7e-7, vs 200k-node trapezoid 5.9e-7
  (gate 1e-6) — validates the composite quadrature deviation (see review
  section above).
- Poor-layout posteriors visibly broader: mean 90% HPD area 0.29 (confined)
  vs 0.12 (well-spread).
- Region area contracts as q rises and grows as σ rises: 12/12 scenarios each.
- Mode → true cell as σ → 0: **24/24**. Well-posedness required two
  documented formulation fixes: (a) the source is snapped to its cell center
  (candidates ARE cell centers; an off-center source at tiny σ legitimately
  fits a neighboring center better), and (b) scenarios are conditioned on
  identifiability (a sensor must see the plume above the smallest benchmark
  noise floor 0.01 — for all-upwind layouts adjacent cells differ by ~1e-30
  and no numerically reachable σ separates them). σ = 1e-6 (1e-4 is not yet
  asymptotic for weak-signal scenarios).
- Area dose behavior (24 cell-centered scenarios, frozen noise realization):
  contracts with q for 87.5% of scenarios (mean Δarea = −0.0047), grows with
  σ for 100% (mean Δarea = +0.0238). Per-realization q-monotonicity is not
  guaranteed at the rate-prior edges (q_true = 0.5 or 5 truncates the
  location–rate confusion set asymmetrically) — the gate is mean contraction
  plus a ≥75% per-scenario fraction.
- Oracle MAP error on frozen data: mean/median 0.151/0.020 (calib),
  0.160/0.023 (test) — the frozen benchmark is highly informative.

### H2 (sharpness audit, seed 1, matched verified-90% conformal coverage)

- Per-scenario Spearman(learned area, exact area) = **0.848**.
- Inefficiency factor (learned/exact area): median **5.57**, mean 47.6
  (heavy right tail: scenarios where the exact posterior is razor-sharp but
  the learned heatmap saturates at ~0.4 of the domain).
- **Honest interpretation:** the learned regions strongly track the physics
  ordering but are far from the information-theoretic sharpness limit — the
  network, not the data, is the bottleneck (the paper's Sec. 6.2 language for
  a factor ≫ 1 applies).

### H3 (dose–response, seed 1; region-size language only, constructed sets)

- Knob 1 (sensor count): per-scenario Spearman(learned, exact curves) median
  **0.81** (mean 0.65); contraction-slope ratio (learned/exact) **0.47**
  (mean curves), per-scenario median 0.46.
- Knob 2 (geometry, N = 6): confined/spread area ratio — learned median 1.20
  (mean 2.44), exact median 1.58 (mean 177.7, blown up by a few scenarios
  whose confined exact region explodes); Spearman agreement of per-scenario
  ratios **0.88**; both methods widen under the confined layout in exactly
  64% of the 50 pairs.
- Knob 3 (noise): per-scenario Spearman median **0.75** (mean 0.52); slope
  ratio (area vs log10 σ) **0.49** (mean curves), per-scenario median 0.61.
- **Honest interpretation:** the learned tool responds in the correct
  direction on every knob and its ordering agrees strongly with the oracle,
  but it reacts with roughly HALF the oracle's magnitude — consistent with
  the H2 finding that the learned regions are over-wide where the data are
  most informative.

### Appendix E — 96×96 grid refinement (full 2000-scenario test split)

Coverage changes by **−0.0065** (0.9005 at 96² vs 0.9070 at 64²) and median
exact region area by **−0.0062** (0.0066 vs 0.0128 as a domain fraction —
finer cells resolve sharper regions). Coverage is stable at nominal; the
64×64 grid is adequate. (The paper's 100-scenario version of this check is
too noisy to be meaningful — 100-window coverage varies 0.79–0.97 across the
test split — so we ran the full split; norm error at 96²: 1.4e-13.)

## G5 — assembly + reproducibility: **PASS**

- `results/placeholder_map.json`: 47 keys covering every paper placeholder
  (headline Table 3, H2/H3, appendix Table 4, gates, seed-2 spread).
- Tables: `results/tables/table3_headline.{csv,tex}`,
  `table4_auxiliary.csv`, `coverage.csv` (fig3 source).
- Figures with the exact filenames the LaTeX expects: `figures/fig1_pipeline.pdf`,
  `fig2_easy_vs_ambiguous.pdf`, `fig3_coverage.pdf`, `fig4_audit.pdf`,
  `fig5_doseresponse.pdf` — rendered and visually verified.
- **Reproducibility rerun:** regenerating the test split from the config
  yields a bit-identical npz (SHA-256 match: a2f8d103…); re-running
  `assemble_results.py` from the frozen artifacts reproduces the main tables
  byte-for-byte. G1 fast gates re-pass after all changes (4/4).
  `scripts/run_all.sh` is the single-command clean-checkout pipeline.
- Note: training itself is seeded but GPU kernels are not bitwise
  deterministic; the frozen checkpoints are part of the released artifact and
  all downstream numbers regenerate deterministically from them.

## Budget

- Storage: **237 MB** on /projectnb (data + posteriors + heatmaps +
  checkpoints) — far under the 10 GB budget.
- Compute: training 0.011 GPU-h per seed (4,869,568 params — "~5M" verified);
  the entire pipeline including physics gates, posteriors (64² and 96² on
  4000+ scenarios), audits, and dose–response used **≈ 1.5 GPU-h** on one
  L40S — far under the paper's 12–20 GPU-h estimate.

## Post-hoc physics-distilled variant (non-preregistered; added after the audit)

Clearly-labeled post-hoc variant: IDENTICAL DeepSets architecture (4.87M
params), IDENTICAL 10k train split, optimizer, schedule, and D4 augmentation
— only the loss changes: CE to the scenario's EXACT Bayes posterior as a
soft target ("physics-likelihood distillation"; under D4 augmentation the
target grid is permuted by the exact same symmetry, verified). Early stop on
validation teacher-CE (matches the objective; the preregistered true-cell
val-NLL criterion actively fights distillation — the sharp teacher has poor
true-cell NLL by construction, and using it truncated training at high LR
and made H2 WORSE, 10.8x).

Five loss variants were compared by a pre-stated validation-only rule
(min median val 90%-HPD area); test evaluated once for the winner:

| variant                         | val HPD-90 area (median) |
|---|---|
| raw teacher, lambda=1.0         | 0.197  (near-delta targets are an optimization trap) |
| raw teacher, lambda=0.75        | 0.106 |
| raw teacher, lambda=0.5         | 0.064 |
| **tempered teacher (0.75-cell blur), lambda=1.0** | **0.055  (winner, epoch 193)** |
| tempered teacher, lambda=0.75   | 0.056 |

Final test results (single evaluation): coverage 0.8985 [0.884, 0.911]
(contains 0.90); region area median 0.068 -> 0.053; H2 inefficiency median
5.6x -> **3.9x** (mean 47.6 -> 28.9); H2 Spearman 0.848 -> 0.893; JSD 0.56
-> 0.50; MAP median 0.103 -> 0.094; H3 sensor-count slope ratio 0.47 ->
**0.87**; geometry agreement 0.88 -> 0.90; noise slope ratio 0.49 -> 0.54.

Purpose in the paper: (1) attribute the H2 gap — same model, same data, new
training signal closes ~1/3 of it and most of the sensor-response gap;
(2) the audit certifies improvement, not only failure; (3) the benchmark's
shipped posteriors double as a training resource. Key methodological finding:
RAW exact-posterior targets fail; TEMPERING (0.75-cell Gaussian, matching the
baseline target smoothing) is what makes the physics signal learnable.
Caveat: requires a tractable per-scenario posterior at training time
(available by construction in simulation-based training). Note: gates.json's
G2 model2 entry reflects the last-trained variant; the winner is
model2_lam10b (see results/distill_variant_selection.json). An earlier
set-transformer variant was implemented and abandoned before any results
were produced.

## Summary of findings the paper text must absorb

1. The 32-node GL-in-log-q rate marginalization (Sec. 4.3) is numerically
   inadequate; a peak-aware composite rule is required (validated < 1e-6).
2. The nonconformity score (Eq. 5) must be computed in tail space; the naive
   mass-above form saturates in floating point and silently destroys the
   coverage guarantee for sharp posteriors.
3. The training recipe (Sec. 4.1) requires D4-symmetry augmentation; as
   written it memorizes with zero generalization at this data scale.
4. H1: seed 2 and conformal-on-exact attain nominal coverage; seed 1 lands at
   0.875 [0.860, 0.889] — a quantified finite-sample fluctuation
   (permutation p = 0.0045), not an implementation defect.
5. H2 is a partial null: calibrated and strongly rank-correlated (ρ = 0.85)
   but ~5.6× the oracle area at the median.
6. H3: correct direction on all three knobs, ~half the oracle's magnitude.
