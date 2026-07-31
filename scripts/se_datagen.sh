#!/bin/bash
# Super-emitter data pipeline (GPU 0, single CUDA process, idempotent):
#   1. splits + exact posteriors (calib/test) + self-calibration gate
#   2. measured winds, det/ens physics maps, physics-only oracles
#   3. wind-marginalized residual channel (ensr's 4th image) + verify gate
cd /usr4/spclpgm/eric1/conformal-source-regions || exit 1
source scripts/env.sh          # before set -u: env.sh expands unset PYTHONPATH
set -eu
# use the FIRST GPU of the job's SGE assignment (never hardcode an index:
# this job's set is e.g. "1,2" and GPU 0 belongs to another user)
IFS=',' read -ra SE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${SE_GPUS[0]}"
python -c "import torch; assert torch.cuda.is_available(), 'no usable CUDA device'"
python scripts/se_datagen.py
python src/methane_t_uncertain.py --stage gen
python src/paired_wind.py --stage gen-marg
python src/paired_wind.py --stage verify-marg
echo "== SE DATAGEN COMPLETE $(date '+%H:%M')"
