#!/bin/bash
# Upgrade arms under measured wind, targeting a clear gain over ens:
#   smear: 2ch analytically wind-smeared maps (closed-form marginal, no MC)
#   ensp : ens + MC-marginal-posterior channel (log-mean-exp info)
#   full : smear + draw-spread + marg-residual + marg-posterior (5ch stack)
# lam=0 everywhere, 3 seeds each.  verify-smear gates the smeared physics.
set -e
cd /usr4/spclpgm/eric1/conformal-source-regions
source scripts/env.sh

while pgrep -u "$USER" -f "pw_maps.sh|python src/paired_wind.py" \
        > /dev/null; do
    echo "waiting for GPU (maps chain still active)"
    sleep 30
done

python src/paired_wind.py --stage gen-smear
python src/paired_wind.py --stage verify-smear

run() {
    local tag=$1; shift
    if grep -q "\"$tag\"" results/paired_wind.json 2>/dev/null; then
        echo "skip $tag (already done)"
    else
        echo "=== $tag ==="
        python src/paired_wind.py --stage train "$@"
    fi
}

for m in smear ensp full; do
    for s in 1 2 3; do
        run "pw_noisy_${m}_lam0.0_seed$s" --views noisy --maps "$m" \
            --seed "$s"
    done
done
echo UPGRADE DONE
