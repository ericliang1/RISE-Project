#!/bin/bash
# Reviewer package reruns: identical configs, audits now save covered flags
# and region masks; plus the random-map negative control and runtime bench.
set -e
cd /usr4/spclpgm/eric1/conformal-source-regions
source scripts/env.sh

python src/paired_wind.py --stage train --views clean --maps off --seed 1
python src/paired_wind.py --stage train --views clean --seed 1
python src/paired_wind.py --stage train --views noisy --maps off --seed 1
python src/paired_wind.py --stage train --views noisy --maps ensr --seed 1
python src/paired_wind.py --stage train --views noisy --maps rand --seed 1
python scripts/runtime_bench.py
echo REVIEW RUNS DONE
