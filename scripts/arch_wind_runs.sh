#!/bin/bash
set -e
cd /usr4/spclpgm/eric1/conformal-source-regions
source scripts/env.sh
for a in gnn st; do
  for s in 1 2 3; do
    python src/paired_wind.py --stage train --views noisy --maps off --arch $a --seed $s
    python src/paired_wind.py --stage train --views noisy --maps ensr --arch $a --seed $s
  done
done
echo ARCH WIND DONE
