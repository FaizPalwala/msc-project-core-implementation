#!/bin/bash
# ==========================================
# slurm_iterative.sh — Iterative unlearning protocol
# ==========================================
#SBATCH --job-name=unlearn_iter
# Usage: sbatch scripts/slurm_iterative.sh <csv> <model> <out>

module purge
module load miniforge
module load cuda/12.6.2
conda activate data_gen
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
CSV="${1:-$DATA_DIR/metadata/dataset.csv}"
MODEL="${2:-$PROJECT_DIR/results/checkpoints/original_model_best.pt}"
OUT="${3:-$PROJECT_DIR/results/iterative}"

cd "$PROJECT_DIR"
echo "[$(date)] Iterative protocol → $OUT"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/iterative.py \
    --csv "$CSV" --model "$MODEL" --out "$OUT" \
    --n_steps 15 --mode cumulative \
    --seed 42 --scale 1.0 \
    --re_emergence 5 10 15

echo "[$(date)] Iterative complete."
