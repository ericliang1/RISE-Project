"""Subgroup coverage/radius diagnostics from the mask-instrumented audits
(no retraining).  Groups are prespecified: mast count, leak rate, mean wind
speed.  Reports n, empirical coverage, median radius for ours at exact wind
(pw_clean_seed1, clean condition) and ours full at measured wind
(pw_noisy_ensr_seed1, noisy condition).  Writes results/subgroups.json.
"""
import json

import numpy as np

from common import load_config, resolve

N_CELLS = 64 * 64
U_SCALE = 8.0


def rad(sizes):
    return np.sqrt(sizes.astype(np.float64) / N_CELLS / np.pi) * 500


def main():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    ns = d["n_sensors"].astype(int)
    q = d["q"]
    spd = np.linalg.norm(d["u_mean"], axis=1) * U_SCALE
    med_spd = float(np.median(spd))
    groups = [
        ("4-6 masts", ns <= 6),
        ("10-12 masts", ns >= 10),
        ("10-100 kg/h", q < 100),
        ("100-500 kg/h", q >= 100),
        (f"wind < {med_spd:.1f} m/s", spd < med_spd),
        (f"wind >= {med_spd:.1f} m/s", spd >= med_spd),
    ]
    out = {}
    for tag, f, cond in (("exact_wind_ours", "pw_audit_pw_clean_seed1_clean"
                          ".npz", "clean"),
                         ("measured_wind_ours",
                          "pw_audit_pw_noisy_ensr_seed1_noisy.npz",
                          "noisy")):
        z = np.load(dd / f)
        cov = z["covered"].astype(bool)
        r = rad(z["sizes"])
        rows = []
        for name, m in groups:
            rows.append({"group": name, "n": int(m.sum()),
                         "coverage": round(float(cov[m].mean()), 3),
                         "median_radius_m": round(float(np.median(r[m])),
                                                  1)})
        out[tag] = rows
        print(f"== {tag} ==")
        for row in rows:
            print(f"  {row['group']:16s} n={row['n']:4d}  "
                  f"cov {row['coverage']:.3f}  med {row['median_radius_m']}")
    with open(rr / "subgroups.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
