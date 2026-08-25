import pathlib
import sys

import numpy as np

_R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_R / "simulator"), str(_R / "localization")]

from common import load_config, resolve

cfg = load_config()
dd = resolve(cfg, "data_dir")
rad = lambda s: np.sqrt(np.asarray(s, float) / 4096 / np.pi) * 500


def stats(sizes, covered, n):
    r = rad(sizes)
    return dict(cov=int(covered.sum()) / n, med=float(np.median(r)),
                f50=float((r < 50).mean()))


def fmt(name, per_seed):
    cov = [s["cov"] for s in per_seed]
    med = [s["med"] for s in per_seed]
    f50 = [s["f50"] for s in per_seed]
    c = ("%.3f" % cov[0] if len(cov) == 1
         else "%.3f-%.3f" % (min(cov), max(cov)))
    r = ("%.1f" % med[0] if len(med) == 1
         else "%.1f-%.1f" % (min(med), max(med)))
    f = ("%.1f%%" % (100 * f50[0]) if len(f50) == 1
         else "%.1f-%.1f%%" % (100 * min(f50), 100 * max(f50)))
    print("%-30s %-12s %-12s %-12s" % (name, c, r, f))


dt = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
n = len(dt["ids"])

print("%-30s %-12s %-12s %-12s" % ("method", "coverage", "radius (m)",
                                   "below 50 m"))

for arch, at in [("DeepSets", ""), ("GNN", "_gnn"),
                 ("Set Transformer", "_st")]:
    for label, mt in [("without maps", "nomaps"), ("with maps", "ens")]:
        per_seed = []
        for s in (1, 2, 3):
            p = dd / f"pw_audit_pw_noisy_{mt}{at}_seed{s}_noisy.npz"
            if p.exists():
                z = np.load(p)
                per_seed.append(stats(z["sizes"], z["covered"], n))
        if per_seed:
            fmt(f"{arch} {label} [{len(per_seed)}s]", per_seed)
        else:
            print(f"{arch} {label}: no runs yet")
