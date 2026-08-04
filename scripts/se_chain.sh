#!/bin/bash
# Super-emitter training chain.  Usage: bash scripts/se_chain.sh <0|1>
#   0: DeepSets nomaps+ensr, GNN ensr      (9 runs, GPU 0)
#   1: SetTransformer nomaps+ensr, GNN nomaps  (9 runs, GPU 1)
# Seed-1 runs first in each chain so the new Table I story appears early.
# Idempotent: a run is skipped when its noisy audit npz already exists.
# Wall-clock safe: relaunch after a job dies and it resumes where it stopped.
cd /usr4/spclpgm/eric1/conformal-source-regions || exit 1
source scripts/env.sh          # before set -u: env.sh expands unset PYTHONPATH
set -u
# map chain index (0|1) onto the job's SGE GPU assignment (e.g. "1,2");
# never hardcode a physical index -- GPU 0 may belong to another user
IFS=',' read -ra SE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${SE_GPUS[$1]}"
export PW_SAVE_CKPT=1
export PW_EPOCHS=200   # pin: a leaked smoke-test override must never reach real runs
export PW_HEAD_WARMUP=100  # head joins at half budget: pilot showed joint
                           # training lets the head displace base features
                           # (GNN 72-74 joint vs 61.7 warmed, val-NLL 5.93->5.82)
python -c "import torch; assert torch.cuda.is_available(), 'no usable CUDA device'" || exit 1
# resolve the audit dir from the config (env override honored) -- a hardcoded
# path here once skipped an entire chain against a previous benchmark's audits
DD=$(python -c "import sys; sys.path.insert(0,'src')
from common import load_config, resolve
print(resolve(load_config(), 'data_dir'))")

run() {  # run <maps> <arch> <seed>
  local maps=$1 arch=$2 seed=$3 at=""
  [ "$arch" != "deepsets" ] && at="_${arch}"
  local mt="_${maps}"; [ "$maps" = "off" ] && mt="_nomaps"
  local tag="pw_noisy${mt}${at}_seed${seed}"
  if [ -f "$DD/pw_audit_${tag}_noisy.npz" ]; then
    echo "== skip ${tag} (audit exists)"; return 0
  fi
  echo "== ${tag} start $(date '+%H:%M')"
  python src/paired_wind.py --stage train --views noisy \
    --maps "$maps" --arch "$arch" --seed "$seed" \
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
