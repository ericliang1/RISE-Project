#!/bin/bash
# 2x2 factorial under measured (noisy) wind — noisy wind is part of the
# simulator output; every config trains AND evaluates on the observed wind.
#   config 1 baseline:   --maps off --lam 0
#   config 2 +maps:      --maps on  --lam 0
#   config 3 +phys loss: --maps off --lam L*
#   config 4 maps+loss:  --maps on  --lam L*
# L* per arm selected on validation NLL over {0.01,0.05,0.1} at seed 1,
# then seeds 2-3 at L*.  Idempotent: completed tags are skipped.
set -e
cd /usr4/spclpgm/eric1/conformal-source-regions
source scripts/env.sh

while pgrep -u "$USER" -f "python src/paired_wind.py" > /dev/null; do
    echo "waiting for GPU (previous paired_wind run still active)"
    sleep 30
done

run() {
    local tag=$1; shift
    if grep -q "\"$tag\"" results/paired_wind.json 2>/dev/null; then
        echo "skip $tag (already done)"
    else
        echo "=== $tag ==="
        python src/paired_wind.py --stage train "$@"
    fi
}

for s in 1 2 3; do
    run "pw_noisy_lam0.0_seed$s" --views noisy --seed "$s"
done
for s in 1 2 3; do
    run "pw_noisy_nomaps_lam0.0_seed$s" --views noisy --maps off --seed "$s"
done
for l in 0.01 0.05 0.1; do
    run "pw_noisy_nomaps_lam${l}_seed1" --views noisy --maps off --lam "$l"
done
for l in 0.01 0.05 0.1; do
    run "pw_noisy_lam${l}_seed1" --views noisy --lam "$l"
done

for arm in nomaps maps; do
    best=$(python scripts/pw_select_lam.py "$arm")
    echo "arm=$arm selected lam=$best"
    for s in 2 3; do
        if [ "$arm" = nomaps ]; then
            run "pw_noisy_nomaps_lam${best}_seed$s" \
                --views noisy --maps off --lam "$best" --seed "$s"
        else
            run "pw_noisy_lam${best}_seed$s" \
                --views noisy --lam "$best" --seed "$s"
        fi
    done
done
echo FACTORIAL DONE
