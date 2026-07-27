#!/bin/bash
set -e
cd /usr4/spclpgm/eric1/conformal-source-regions
source scripts/env.sh
for s in 1 2 3; do
  python src/paired_wind.py --stage train --views noisy --maps zdet --seed $s
done
echo ZDET DONE
