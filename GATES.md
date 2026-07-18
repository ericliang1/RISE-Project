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

## G3 / G4 / G5 — TBD

(Filled after their stages run.)
