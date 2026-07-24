"""Paired bootstrap for the two main comparisons, from existing per-scenario
region sizes (no retraining).  Scenarios are resampled jointly so every
bootstrap draw contains both methods on the same scenarios.

  exact wind    labels-only (M0) vs recipe (lever_suffstats_labels)
  measured wind no-physics vs full method (ensr), seed-1 audits

Reports: median-radius reduction with 95% paired-bootstrap CI, fraction of
scenarios improved, and Clopper-Pearson 95% CI for the fraction of our
regions below 50 m.  Writes results/paired_boot.json.
"""
import json

import numpy as np

from common import load_config, resolve
from conformal import clopper_pearson

N_CELLS = 64 * 64
B = 10000


def rad(sizes):
    return np.sqrt(sizes.astype(np.float64) / N_CELLS / np.pi) * 500


def analyze(base, ours, rng):
    n = len(base)
    assert len(ours) == n
    d0 = float(np.median(base) - np.median(ours))
    ds = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        ds[b] = np.median(base[i]) - np.median(ours[i])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    k50 = int((ours < 50).sum())
    c_lo, c_hi = clopper_pearson(k50, n)
    return {"n": n,
            "median_reduction_m": round(d0, 1),
            "ci95": [round(float(lo), 1), round(float(hi), 1)],
            "frac_improved": round(float((ours < base).mean()), 3),
            "frac_below_50m": round(k50 / n, 3),
            "frac_below_50m_ci95": [round(float(c_lo), 3),
                                    round(float(c_hi), 3)]}


def main():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    rng = np.random.default_rng(7)
    out = {
        "exact_wind_M0_vs_recipe": analyze(
            rad(np.load(dd / "ch4t_audit_M0.npz")["sizes"]),
            rad(np.load(dd / "ch4t_audit_lever_suffstats_labels.npz")
                ["sizes"]), rng),
        "measured_wind_nomaps_vs_ensr": analyze(
            rad(np.load(dd / "pw_audit_pw_noisy_nomaps_lam0.0_seed1_noisy"
                        ".npz")["sizes"]),
            rad(np.load(dd / "pw_audit_pw_noisy_ensr_lam0.0_seed1_noisy"
                        ".npz")["sizes"]), rng),
    }
    with open(rr / "paired_boot.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
