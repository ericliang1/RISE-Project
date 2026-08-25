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

ns = dict(np.load(dd / "ch4t_test.npz",
                  allow_pickle=True))["n_sensors"].astype(int)
rows = []
for net, sfx in NETS:
    for cond in CONDS:
        r = rad(np.load(dd / f"pw_audit_pw_noisy_{cond}{sfx}_seed1_noisy"
                        ".npz")["sizes"])
        for k in range(4, 9):
            m = ns == k
            rows.append([net, cond, k, int(m.sum()),
                         round(float(np.median(r[m])), 2)])

out = resolve(cfg, "results_dir") / "fig3_masts_medians.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["network", "maps", "n_masts", "n_scenarios",
                "median_radius_m"])
    w.writerows(rows)
print(f"wrote {out}")
print(f"{'network':10s} {'maps':7s} {'masts':>5s} {'n':>5s} {'median':>8s}")
for r in rows:
    print(f"{r[0]:10s} {r[1]:7s} {r[2]:>5d} {r[3]:>5d} {r[4]:>8.2f}")
