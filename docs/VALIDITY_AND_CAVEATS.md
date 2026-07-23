# Validity & Caveats — GNN (M2) and Set Transformer (M3) Results

Branch: `set-transformer` · Date: 2026-07-23

An honest assessment of how far the M2 (GNN) and M3 (Set Transformer) results
on CH4-T can be trusted, and where the bounds are. Companion to
[GNN_WORK_LOG.md](GNN_WORK_LOG.md) and
[SET_TRANSFORMER_WORK_LOG.md](SET_TRANSFORMER_WORK_LOG.md).

---

## What is solidly valid (with evidence)

Verified across all 14 saved audit files (M0, M1, and M2/M2base/M3/M3base ×3 seeds):

1. **The comparison is apples-to-apples.** All 14 audits share *one identical*
   exact-posterior reference (verified byte-equal across 2000 test scenarios).
   Every model was evaluated on the same frozen data, the same conformal
   wrapper, and the same region-size audit — the paper's downstream code was
   *reused*, not reimplemented. Only the network class changed.

2. **The harness reproduces the published paper.** Reconstructed baselines land
   on M0 = 93.8 m (paper 94), M1 = 73.8 m (paper 74), Bayes = 6.9 m (paper 7).
   A broken data loader, conformal step, or audit would not reproduce these.
   This is the strongest single evidence the pipeline is intact.

3. **The paper's blocking sanity gate passed.** Conformal-on-exact-posterior
   self-calibration = 0.952 (required ≈ 0.95) on every run — the same check
   that caught a real argument-order bug in the original paper. It validates
   that the exact-posterior teacher/reference is physically correct.

4. **Every model is conformally calibrated** (coverage 0.89–0.91 ≈ 0.90 target).
   So region-size differences reflect genuine *sharpness*, not a model trading
   coverage for a smaller region.

5. **Not a capacity artifact.** The *largest* model (Set Transformer, 6.53M) is
   *not* the sharpest — the smaller GNN (5.61M) beats it. Direct internal
   evidence that the ordering reflects inductive bias, not parameter count.

6. **Stable and significance-tested.** Three seeds, non-overlapping ranges,
   paired Wilcoxon on per-scenario log region-size ratios.

**Conclusion:** as CH4-T benchmark numbers, the GNN and Set Transformer results
are internally valid and trustworthy, and the original data is the same frozen
data that produced the published paper.

### Evidence snapshot

| Check | Result |
|-------|--------|
| All 14 audits share one exact reference | **True** (2000 scenarios, byte-equal) |
| M0 / M1 / Bayes vs paper (94 / 74 / 7) | 93.8 / 73.8 / 6.9 m |
| Self-calibration gate (need ≈0.95) | 0.952 (all runs) |
| Model coverage (need ≈0.90) | M2 0.910, M2base 0.896, M3 0.891, M3base 0.902 |
| Largest model sharpest? | No — SetTransformer 6.53M loses to GNN 5.61M |

## Caveats (these bound the claims; they do not invalidate the results)

- **Capacity is not formally matched** (DeepSets 4.87M / GNN 5.61M /
  SetTransformer 6.53M). Point 5 argues it doesn't drive the result, but the
  clean control — retrain at equal parameter count — has **not** been run. A
  reviewer will raise this; it is the main open item.
- **The M0/M1 baselines in the head-to-head are eric1's single-seed saved
  audits.** They sit inside the paper's 3-seed ranges (94–98 / 73–74), and the
  GNN/ST are full 3-seed, but the baseline side is strictly one seed. The
  per-scenario Wilcoxon tests were computed on seed-1 audits (the paper's own
  convention).
- **All of the paper's limitations are inherited.** CH4-T is idealized: steady
  release, known stability class, *measured* (known) wind, flat terrain, 2-D
  sensing of a 3-D plume. Coverage is marginal and in-distribution. These are
  **not** field-validated numbers — they say "on this testbed, a better
  architecture sharpens the region," nothing about real deployment.
- **Single implementation, not independently reproduced or peer-reviewed.**
  Validated by smoke tests plus baseline reproduction, but one codebase, run
  once per seed, on one machine.
- **The physics gates (G1: forward-model / finite-difference tests) were not
  re-run here.** The frozen data's validity is taken on the self-calibration
  gate passing and the baseline reproduction — strong but indirect.

## Bottom line

The results are **valid as an internally-consistent, reproducible extension of
the CH4-T benchmark** — the architecture ordering (DeepSets ‹ Set Transformer ‹
GNN) is real and defensible. They are **not yet "airtight for publication"** in
two specific senses: the capacity-matched control is unrun, and everything
inherits the testbed's idealized, in-distribution scope.

To move from "valid benchmark result" to "airtight claim," the single most
useful next step is the **capacity-matched rerun** (shrink GNN and Set
Transformer to ≈4.87M params and repeat the three seeds).
