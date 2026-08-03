"""Paired bootstrap CIs for Table I's effect columns (final 4-8 benchmark).

For each architecture, seed 1: per-scenario paired comparison of the
no-maps and with-maps audits on the same 2000 test scenarios; 10,000
paired bootstrap resamples of (median radius difference, frac<50m
difference).  Writes results/se48_table_deltas_ci.json.

Usage: python scripts/se_paired_ci.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, "src")

from common import load_config, resolve

cfg = load_config()
dd = resolve(cfg, "data_dir")
rad = lambda s: np.sqrt(np.asarray(s, float) / 4096 / np.pi) * 500

out = {}
rng = np.random.default_rng(0)
for arch, at in [("DeepSets", ""), ("GNN", "_gnn"), ("SetTransf", "_st")]:
    a = rad(np.load(dd / f"pw_audit_pw_noisy_nomaps{at}_seed1_noisy.npz")["sizes"])
    b = rad(np.load(dd / f"pw_audit_pw_noisy_ens{at}_seed1_noisy.npz")["sizes"])
    n = len(a)
    dmed = np.median(b) - np.median(a)
    df = (b < 50).mean() - (a < 50).mean()
    dm, dfb = [], []
    for _ in range(10000):
        i = rng.integers(0, n, n)
        dm.append(np.median(b[i]) - np.median(a[i]))
        dfb.append((b[i] < 50).mean() - (a[i] < 50).mean())
    lo, hi = np.percentile(dm, [2.5, 97.5])
    flo, fhi = np.percentile(dfb, [2.5, 97.5])
    out[arch] = dict(d_radius_m=float(dmed), ci=[float(lo), float(hi)],
                     d_frac50_pp=float(100 * df),
                     frac_ci=[float(100 * flo), float(100 * fhi)],
                     frac_smaller=float((b < a).mean()))
    print("%-10s dRadius %+6.1f [%+6.1f, %+6.1f]   d<50m %+5.1f pp "
          "[%+5.1f, %+5.1f]   smaller %.1f%%"
          % (arch, dmed, lo, hi, 100 * df, 100 * flo, 100 * fhi,
             100 * out[arch]["frac_smaller"]))

with open("results/se48_table_deltas_ci.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results/se48_table_deltas_ci.json")
