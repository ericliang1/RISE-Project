#!/bin/bash
# CH4-T-U serial chain (single GPU, Exclusive Process mode).
# gen: observed winds, det/ens maps, det/marg oracles (the long pole)
# oracles: audit both oracle rows (the theory prediction)
# train 2x2: maps {det,ens} x teacher {det,marg}, det/det first (paired ref)
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "=== gen $(date)"
python src/methane_t_uncertain.py --stage gen
echo "=== oracles $(date)"
python src/methane_t_uncertain.py --stage oracles
echo "=== train det/det $(date)"
python src/methane_t_uncertain.py --stage train --maps det --teacher det
echo "=== train det/marg $(date)"
python src/methane_t_uncertain.py --stage train --maps det --teacher marg
echo "=== train ens/det $(date)"
python src/methane_t_uncertain.py --stage train --maps ens --teacher det
echo "=== train ens/marg $(date)"
python src/methane_t_uncertain.py --stage train --maps ens --teacher marg
echo "=== UNCERTAIN CHAIN DONE $(date)"
