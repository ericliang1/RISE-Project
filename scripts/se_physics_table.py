"""Table II (super-emitter benchmark): physics-only inversion vs the method.

Row 1: the wind-averaged physics-only posterior (ch4tu_*_oracle_marg.npz,
K=8 draws from the anemometer error model, rate integrated over the
generating prior), conformalized with the identical wrapper and calibration
scenarios as every network.  Rows 2-4: the three base networks with the
physics input maps, read from their per-scenario audit npzs (3 seeds each).

Run after the training chains complete: python scripts/se_physics_table.py
"""
import sys

import numpy as np

sys.path.insert(0, "src")

from common import load_config, resolve
from conformal import clopper_pearson, regions, tail_scores, tail_threshold

cfg = load_config()
dd = resolve(cfg, "data_dir")
alpha = cfg["conformal"]["alpha"]
seed = cfg["conformal"]["score_seed"]
rad = lambda s: np.sqrt(np.asarray(s, float) / 4096 / np.pi) * 500


def stats(sizes, covered, n):
    r = rad(sizes)
    k = int(covered.sum())
    lo, hi = clopper_pearson(k, n)
    return dict(cov=k / n, cp=(lo, hi), med=float(np.median(r)),
                f50=float((r < 50).mean()))


def fmt(name, per_seed):
    cov = [s["cov"] for s in per_seed]
    med = [s["med"] for s in per_seed]
    f50 = [s["f50"] for s in per_seed]
    lo, hi = per_seed[int(np.argmin(cov))]["cp"]
    c = ("%.3f" % cov[0] if len(cov) == 1
         else "%.3f-%.3f" % (min(cov), max(cov)))
    r = ("%.1f" % med[0] if len(med) == 1
         else "%.1f-%.1f" % (min(med), max(med)))
    f = ("%.1f%%" % (100 * f50[0]) if len(f50) == 1
         else "%.1f-%.1f%%" % (100 * min(f50), 100 * max(f50)))
    print("%-28s %-12s %-12s %-12s  worst-seed CP [%.3f, %.3f]"
          % (name, c, r, f, lo, hi))


dc = dict(np.load(dd / "ch4t_calib.npz", allow_pickle=True))
dt = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
n = len(dt["ids"])

print("%-28s %-12s %-12s %-12s" % ("method", "coverage", "radius (m)",
                                   "below 50 m"))

# ---- row 1: wind-averaged physics only, same wrapper, same calib ----
Pc = np.load(dd / "ch4tu_calib_oracle_marg.npz")["probs"]
Pt = np.load(dd / "ch4tu_test_oracle_marg.npz")["probs"]
th = tail_threshold(tail_scores(Pc, dc["true_cell"],
                                np.random.default_rng(seed)), alpha)
reg = regions(Pt, th, dt["true_cell"])
fmt("Wind-averaged physics only", [stats(reg["sizes"], reg["covered"], n)])

# ---- rows 2-4: the three networks with the physics maps ----
for name, at in [("DeepSets + physics maps", ""),
                 ("GNN + physics maps", "_gnn"),
                 ("Set Transformer + physics maps", "_st")]:
    per_seed = []
    for s in (1, 2, 3):
        z = np.load(dd / f"pw_audit_pw_noisy_ensr{at}_seed{s}_noisy.npz")
        per_seed.append(stats(z["sizes"], z["covered"], n))
    fmt(name, per_seed)
