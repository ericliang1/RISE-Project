#!/bin/bash
# Maps-only accuracy track under measured wind (physics loss dropped as
# useless per factorial): det maps done earlier; here ens (3ch uncertainty-
# aware) and ensr (ens + marginalized-residual goodness-of-fit channel),
# lam=0 everywhere, 3 seeds each.  verify-marg gates the ensr channel.
set -e
cd /usr4/spclpgm/eric1/conformal-source-regions
source scripts/env.sh

while pgrep -u "$USER" -f "python src/paired_wind.py" > /dev/null; do
    echo "waiting for GPU (gen-marg still finishing)"
    sleep 30
done

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

for s in 1 2 3; do
    run "pw_noisy_ens_lam0.0_seed$s" --views noisy --maps ens --seed "$s"
done
for s in 1 2 3; do
    run "pw_noisy_ensr_lam0.0_seed$s" --views noisy --maps ensr --seed "$s"
done
echo MAPS DONE
