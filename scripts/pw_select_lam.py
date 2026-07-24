"""Print the lam in {0.01,0.05,0.1} with the lowest seed-1 validation NLL
for one factorial arm.  Arg is either the shorthand nomaps|maps or a full
tag prefix ending in "_lam" (e.g. pw_noisy_marg_lam)."""
import json
import sys

arm = sys.argv[1]
pre = {"nomaps": "pw_noisy_nomaps_lam",
       "maps": "pw_noisy_lam"}.get(arm, arm)
res = json.load(open("results/paired_wind.json"))
cands = {}
for lam in ("0.01", "0.05", "0.1"):
    row = res.get(f"{pre}{lam}_seed1")
    if row and "val_nll" in row:
        cands[lam] = row["val_nll"]
assert cands, f"no lam-sweep rows with val_nll for arm {arm}"
print(min(cands, key=cands.get))
