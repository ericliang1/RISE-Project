import json
import pathlib
import sys

import numpy as np

_R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_R / "simulator"), str(_R / "localization")]

from common import load_config, resolve

N_CELLS = 64 * 64
NETS = [("deepsets", ""), ("gnn", "_gnn"), ("st", "_st")]

cfg = load_config()
dd = resolve(cfg, "data_dir")
rad = lambda s: np.sqrt(np.asarray(s, float) / N_CELLS / np.pi) * 500

d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
ns = d["n_sensors"].astype(int)

audits, contrast, ok = {}, np.zeros(len(ns)), (ns == 8)
for net, sfx in NETS:
    for cond in ("nomaps", "ens"):
        z = np.load(dd / f"pw_audit_pw_noisy_{cond}{sfx}_seed1_noisy.npz")
        audits[net, cond] = z
        ok &= z["covered"]
        contrast += np.log(rad(z["sizes"])) * (1 if cond == "nomaps" else -1)
i = int(np.argmax(np.where(ok, contrast, -np.inf)))

tc = int(d["true_cell"][i])
rec = {
    "scenario": i,
    "n_masts": 8,
    "q_kg_per_h": round(float(d["q"][i]), 2),
    "sigma_ppm": round(float(d["sigma"][i]), 4),
    "true_cell": tc,
    "true_source_m": [round((tc % 64 + 0.5) / 64 * 500, 2),
                      round((tc // 64 + 0.5) / 64 * 500, 2)],
    "sensors_m": (d["sensors"][i, :8] * 500).round(2).tolist(),
    "localization_target_m": 50.0,
    "models": {},
}
for net, _ in NETS:
    entry = {}
    for cond in ("nomaps", "ens"):
        z = audits[net, cond]
        mask = np.unpackbits(z["masks"][i])[:N_CELLS].astype(bool)
        entry[cond] = {
            "radius_m": round(float(rad(z["sizes"][i : i + 1])[0]), 2),
            "covered": bool(z["covered"][i]),
            "region_cells": np.where(mask)[0].tolist(),
        }
    rec["models"][net] = entry

out = resolve(cfg, "results_dir") / "fig2_region_example.json"
with open(out, "w") as f:
    json.dump(rec, f, indent=1)
print(f"wrote {out} (scenario {i}, q {rec['q_kg_per_h']:.0f} kg/h)")
for net, _ in NETS:
    e = rec["models"][net]
    print(f"  {net:10s} without {e['nomaps']['radius_m']:6.1f} m "
          f"({len(e['nomaps']['region_cells'])} cells)   "
          f"with {e['ens']['radius_m']:6.1f} m "
          f"({len(e['ens']['region_cells'])} cells)")
