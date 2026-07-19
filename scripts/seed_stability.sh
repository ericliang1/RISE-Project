#!/bin/bash
# M0/M1 seed-stability runs (headline metrics for 3 seeds each).
# (An M2 curriculum variant was explored and dropped: it did not improve on M1;
#  record in GATES.md and results/distill_variant_selection_model3_seed1.json.)
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "== M0 seed 3 =="
python src/train.py --seed 3
echo "== M1 seeds 2, 3 =="
python src/train.py --seed 2 --distill --mix-lambda 1.0 --teacher-blur 0.75
python src/train.py --seed 3 --distill --mix-lambda 1.0 --teacher-blur 0.75

echo "== evals (conformal + audit per model-seed) =="
python src/run_conformal.py --probs model --seed 3
python src/audit.py --seed 3 --arch model
for S in 2 3; do
  python src/run_conformal.py --probs model2 --seed $S
  python src/audit.py --seed $S --arch model2
done

echo "== ablation table =="
python3 - <<'EOF'
import json, csv
rows = []
for m, tag in [("M0", "model"), ("M1", "model2")]:
    for s in (1, 2, 3):
        try:
            c = json.load(open(f"results/conformal_{tag}_seed{s}.json"))["main"]
            a = json.load(open(f"results/audit_{tag}_seed{s}.json"))
        except FileNotFoundError:
            continue
        rows.append(dict(model=m, seed=s, coverage=round(c["coverage"], 4),
                         area_med=round(c["area_median"], 4),
                         ineff_med=round(a["h2"]["inefficiency_median"], 3),
                         ineff_mean=round(a["h2"]["inefficiency_mean"], 2),
                         spearman=round(a["h2"]["spearman_area"], 3),
                         jsd=round(a["jsd_mean"], 3),
                         map_med=round(a["map_error"]["learned_median"], 4)))
with open("results/tables/ablation_seeds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
EOF

echo "== figures =="
python src/figures.py
echo "SEED STABILITY DONE"
