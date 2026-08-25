import csv
import pathlib
import sys

import numpy as np

_R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_R / "simulator"), str(_R / "localization")]

from common import load_config, resolve

N_CELLS = 64 * 64
NETS = [("deepsets", ""), ("gnn", "_gnn"), ("st", "_st")]
CONDS = ("nomaps", "ens")

cfg = load_config()
dd = resolve(cfg, "data_dir")
rad = lambda s: np.sqrt(np.asarray(s, float) / N_CELLS / np.pi) * 500

cols = {}
for net, sfx in NETS:
    for cond in CONDS:
        z = np.load(dd / f"pw_audit_pw_noisy_{cond}{sfx}_seed1_noisy.npz")
        cols[f"{net}_{cond}_radius_m"] = rad(z["sizes"])
        cols[f"{net}_{cond}_covered"] = z["covered"].astype(int)

n = len(next(iter(cols.values())))
out = resolve(cfg, "results_dir") / "fig1_radius_cdf.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario"] + list(cols))
    for i in range(n):
        w.writerow([i] + [f"{cols[k][i]:.3f}" if k.endswith("_radius_m")
                          else int(cols[k][i]) for k in cols])
print(f"wrote {out} ({n} scenarios)")
for k in cols:
    if k.endswith("_radius_m"):
        q = np.percentile(cols[k], [10, 25, 50, 75, 90])
        print(f"  {k:28s} p10/p25/p50/p75/p90 = "
              + "/".join(f"{v:.1f}" for v in q)
              + f"  <50m {(cols[k] < 50).mean():.3f}")
