#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
[ -f env.sh ] && source env.sh
set -eu
IFS=',' read -ra SE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${SE_GPUS[0]}"
python -c "import torch; assert torch.cuda.is_available(), 'no usable CUDA device'"
python simulator/generate_data.py
python localization/physics_maps.py
echo "== DATAGEN COMPLETE $(date '+%H:%M')"
