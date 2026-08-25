module load miniconda academic-ml/spring-2026 2>/dev/null
conda activate spring-2026-pyt
export OMP_NUM_THREADS=${NSLOTS:-8}
