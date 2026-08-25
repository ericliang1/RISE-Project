#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
[ -f env.sh ] && source env.sh
set -u
IFS=',' read -ra SE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${SE_GPUS[$1]}"
export PW_SAVE_CKPT=1
export PW_EPOCHS=200
export PW_HEAD_WARMUP=100
python -c "import torch; assert torch.cuda.is_available(), 'no usable CUDA device'" || exit 1
DD=$(python -c "import sys, pathlib
sys.path.insert(0, 'simulator')
from common import load_config, resolve
print(resolve(load_config(), 'data_dir'))")

run() {
  local maps=$1 arch=$2 seed=$3 at=""
  [ "$arch" != "deepsets" ] && at="_${arch}"
  local mt="_${maps}"; [ "$maps" = "off" ] && mt="_nomaps"
  local tag="pw_noisy${mt}${at}_seed${seed}"
  if [ -f "$DD/pw_audit_${tag}_noisy.npz" ]; then
    echo "== skip ${tag} (audit exists)"; return 0
  fi
  echo "== ${tag} start $(date '+%H:%M')"
  python localization/train_eval.py --maps "$maps" --arch "$arch" --seed "$seed" \
    || { echo "== ${tag} FAILED"; exit 1; }
}

if [ "$1" = "0" ]; then
  for s in 1 2 3; do
    run off deepsets "$s"; run ens deepsets "$s"; run ens gnn "$s"
  done
else
  for s in 1 2 3; do
    run off st "$s"; run ens st "$s"; run off gnn "$s"
  done
fi
echo "== chain $1 complete $(date '+%H:%M')"
