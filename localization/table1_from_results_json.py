import json
import pathlib
import re
import sys
from collections import defaultdict

_R = pathlib.Path(__file__).resolve().parents[1]

res = json.load(open(_R / "results" / "paired_wind.json"))
groups = defaultdict(list)
for tag, row in res.items():
    groups[re.sub(r"_seed\d+$", "", tag)].append(row)


def rng(vals, fmt="{:.1f}"):
    lo, hi = min(vals), max(vals)
    one = fmt.format(lo)
    return one if fmt.format(hi) == one else f"{one}-{fmt.format(hi)}"


out = {}
print(f"{'config':38s} {'n':>2s}  {'radius (m)':>13s} {'coverage':>11s} "
      f"{'<50m':>9s}")
for cfg in sorted(groups):
    rows = groups[cfg]
    have = [r["noisy"] for r in rows]
    line = {"n_seeds": len(rows),
            "median_radius_m": rng([h["median_radius_m"] for h in have]),
            "coverage": rng([h["coverage"] for h in have], "{:.3f}"),
            "frac_below_50m": rng([h["frac_below_50m"] for h in have],
                                  "{:.2f}")}
    out[cfg] = line
    print(f"{cfg:38s} {len(rows):2d}  {line['median_radius_m']:>13s} "
          f"{line['coverage']:>11s} {line['frac_below_50m']:>9s}")
if "--json" in sys.argv:
    print(json.dumps(out, indent=2))
