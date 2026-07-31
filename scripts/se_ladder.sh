#!/bin/bash
# Table II ladder on the 4-8 benchmark: DeepSets component breakdown.
#   zdet  1ch  evidence only (measured wind)
#   det   2ch  + sensor visibility (measured wind)
# (nomaps and ens rungs come from the main chains.)
# Usage: bash scripts/se_ladder.sh <gpu_index 0|1>
cd /usr4/spclpgm/eric1/conformal-source-regions || exit 1
source scripts/env.sh          # before set -u: env.sh expands unset PYTHONPATH
set -u
IFS=',' read -ra SE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${SE_GPUS[$1]}"
export PW_SAVE_CKPT=1
export PW_EPOCHS=200   # pin: a leaked smoke-test override must never reach real runs
python -c "import torch; assert torch.cuda.is_available(), 'no usable CUDA device'" || exit 1
DD=$(python -c "import sys; sys.path.insert(0,'src')
from common import load_config, resolve
print(resolve(load_config(), 'data_dir'))")

run() {  # run <maps> <seed>
  local maps=$1 seed=$2
  local tag="pw_noisy_${maps}_seed${seed}"
  if [ -f "$DD/pw_audit_${tag}_noisy.npz" ]; then
    echo "== skip ${tag} (audit exists)"; return 0
  fi
  echo "== ${tag} start $(date '+%H:%M')"
  python src/paired_wind.py --stage train --views noisy \
    --maps "$maps" --arch deepsets --seed "$seed" \
    || { echo "== ${tag} FAILED"; exit 1; }
}

for s in 1 2 3; do
  run zdet "$s"; run det "$s"
done
echo "== ladder complete $(date '+%H:%M')"
