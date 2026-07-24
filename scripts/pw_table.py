"""Summarize results/paired_wind.json: group seed runs per config and print
3-seed ranges for the noisy (primary) and clean conditions.
Usage: python scripts/pw_table.py [--json]"""
import json
import re
import sys
from collections import defaultdict

res = json.load(open("results/paired_wind.json"))
groups = defaultdict(list)
for tag, row in res.items():
    groups[re.sub(r"_seed\d+$", "", tag)].append(row)


def rng(vals, fmt="{:.1f}"):
    lo, hi = min(vals), max(vals)
    one = fmt.format(lo)
    return one if fmt.format(hi) == one else f"{one}-{fmt.format(hi)}"


out = {}
print(f"{'config':38s} {'n':>2s}  {'noisy radius':>13s} {'cov':>9s} "
      f"{'<50m':>9s}  {'clean radius':>13s}")
for cfg in sorted(groups):
    rows = groups[cfg]
    line = {"n_seeds": len(rows)}
    parts = f"{cfg:38s} {len(rows):2d}  "
    for cond in ("noisy", "clean"):
        have = [r[cond] for r in rows if cond in r]
        if not have:
            parts += f"{'-':>13s} {'-':>9s} {'-':>9s}  " if cond == "noisy" \
                else f"{'-':>13s}"
            continue
        line[cond] = {
            "median_radius_m": rng([h["median_radius_m"] for h in have]),
            "coverage": rng([h["coverage"] for h in have], "{:.3f}"),
            "frac_below_50m": rng([h["frac_below_50m"] for h in have],
                                  "{:.2f}")}
        c = line[cond]
        if cond == "noisy":
            parts += (f"{c['median_radius_m']:>13s} {c['coverage']:>9s} "
                      f"{c['frac_below_50m']:>9s}  ")
        else:
            parts += f"{c['median_radius_m']:>13s}"
    out[cfg] = line
    print(parts)
if "--json" in sys.argv:
    print(json.dumps(out, indent=2))
