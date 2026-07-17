# Source this to activate the pinned environment on the BU SCC (SGE cluster).
#   source scripts/env.sh
# Environment: BU SCC academic-ml/spring-2026 conda env (PyTorch build).
#   python 3.12.11, torch 2.9.1+cu128, numpy 2.2.6, scipy 1.16.0,
#   pandas 2.3.0, matplotlib, pyyaml, pytest 9.0.2, tqdm 4.67.1
module load miniconda academic-ml/spring-2026 2>/dev/null
conda activate spring-2026-pyt
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src:$PYTHONPATH"
# Threads: our SGE job has NSLOTS=8
export OMP_NUM_THREADS=${NSLOTS:-8}
