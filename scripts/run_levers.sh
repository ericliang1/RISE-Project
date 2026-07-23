#!/bin/bash
# Serial lever chain (single GPU, Exclusive Process mode -- never parallelize).
# Order: cheap diagnostic first, then highest-bet levers; suffstats gen is the
# long pole so it runs while results from the first two are already readable.
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "=== smoke: 2-epoch end-to-end of anneal+sinkhorn $(date)"
LEVER_EPOCHS=2 python src/methane_t_levers.py --stage anneal
LEVER_EPOCHS=2 python src/methane_t_levers.py --stage sinkhorn
python - <<'EOF'
import json
r = json.load(open('results/ch4t_levers.json'))
for k in ('lever_anneal', 'lever_sinkhorn'):
    assert k in r, k
    print('smoke', k, 'cov', r[k]['coverage'], 'rad', round(r[k]['median_radius_m'], 1))
EOF
rm -f results/ch4t_levers.json data/ch4t_audit_lever_*.npz   # discard smoke audits

echo "=== anneal (full) $(date)"
python src/methane_t_levers.py --stage anneal
echo "=== sinkhorn (full) $(date)"
python src/methane_t_levers.py --stage sinkhorn
echo "=== gen_suffstats $(date)"
python src/methane_t_levers.py --stage gen_suffstats
echo "=== suffstats (full) $(date)"
python src/methane_t_levers.py --stage suffstats
echo "=== LEVERS CHAIN DONE $(date)"
