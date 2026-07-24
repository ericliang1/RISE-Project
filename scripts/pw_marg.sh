#!/bin/bash
# Wind-marginalized physics-loss arms of the 2x2 factorial: the residual in
# L_phys is the MEAN closed-form residual over 8 wind draws from the error
# model centered at u_obs (stage gen-marg), instead of the residual at the
# single misreported wind.  Queues behind pw_factorial.sh; idempotent.
set -e
cd /usr4/spclpgm/eric1/conformal-source-regions
source scripts/env.sh

while pgrep -u "$USER" -f "pw_factorial.sh|python src/paired_wind.py" \
        > /dev/null; do
    echo "waiting for GPU (factorial chain still active)"
    sleep 30
done

python src/paired_wind.py --stage gen-marg
python src/paired_wind.py --stage verify-marg

run() {
    local tag=$1; shift
    if grep -q "\"$tag\"" results/paired_wind.json 2>/dev/null; then
        echo "skip $tag (already done)"
    else
        echo "=== $tag ==="
        python src/paired_wind.py --stage train "$@"
    fi
}

for l in 0.01 0.05 0.1; do
    run "pw_noisy_nomaps_marg_lam${l}_seed1" \
        --views noisy --maps off --resid marg --lam "$l"
done
for l in 0.01 0.05 0.1; do
    run "pw_noisy_marg_lam${l}_seed1" --views noisy --resid marg --lam "$l"
done

for arm in pw_noisy_nomaps_marg_lam pw_noisy_marg_lam; do
    best=$(python scripts/pw_select_lam.py "$arm")
    echo "arm=$arm selected lam=$best"
    for s in 2 3; do
        if [ "$arm" = pw_noisy_nomaps_marg_lam ]; then
            run "${arm}${best}_seed$s" \
                --views noisy --maps off --resid marg --lam "$best" --seed "$s"
        else
            run "${arm}${best}_seed$s" \
                --views noisy --resid marg --lam "$best" --seed "$s"
        fi
    done
done
echo MARG DONE
