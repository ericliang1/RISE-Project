#!/bin/bash
# Below-80 designs under measured wind: uncertainty-aware maps + marg loss.
#   ens : 3ch input maps (mean z, spread z, mean log b over 8 wind draws)
#   ensr: ens + marginalized-residual channel (goodness-of-fit as input)
# Each at lam=0 (3 seeds) and with the marginalized loss (lam swept at
# seed 1, selected on val NLL, confirmed seeds 2-3).  Queues behind
# pw_marg.sh; idempotent.
set -e
cd /usr4/spclpgm/eric1/conformal-source-regions
source scripts/env.sh

while pgrep -u "$USER" \
        -f "pw_marg.sh|pw_factorial.sh|python src/paired_wind.py" \
        > /dev/null; do
    echo "waiting for GPU (marg chain still active)"
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
    run "pw_noisy_ens_lam0.0_seed$s" --views noisy --maps ens --seed "$s"
done
for l in 0.01 0.05 0.1; do
    run "pw_noisy_ens_marg_lam${l}_seed1" \
        --views noisy --maps ens --resid marg --lam "$l"
done
best=$(python scripts/pw_select_lam.py pw_noisy_ens_marg_lam)
echo "ens arm selected lam=$best"
for s in 2 3; do
    run "pw_noisy_ens_marg_lam${best}_seed$s" \
        --views noisy --maps ens --resid marg --lam "$best" --seed "$s"
done

for s in 1 2 3; do
    run "pw_noisy_ensr_lam0.0_seed$s" --views noisy --maps ensr --seed "$s"
done
for l in 0.01 0.05 0.1; do
    run "pw_noisy_ensr_marg_lam${l}_seed1" \
        --views noisy --maps ensr --resid marg --lam "$l"
done
best=$(python scripts/pw_select_lam.py pw_noisy_ensr_marg_lam)
echo "ensr arm selected lam=$best"
for s in 2 3; do
    run "pw_noisy_ensr_marg_lam${best}_seed$s" \
        --views noisy --maps ensr --resid marg --lam "$best" --seed "$s"
done
echo BELOW80 DONE
