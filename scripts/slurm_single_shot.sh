#!/bin/bash
# ==========================================
# slurm_single_shot.sh — Single-shot evaluation
# ==========================================
#SBATCH --job-name=unlearn_single
# Usage: sbatch scripts/slurm_single_shot.sh <csv> <model> <out>

module purge
module load miniforge
module load cuda/12.6.2
conda activate data_gen
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
CSV="${1:-$DATA_DIR/metadata/dataset.csv}"
MODEL="${2:-$PROJECT_DIR/results/checkpoints/original_model_best.pt}"
OUT="${3:-$PROJECT_DIR/results/single_shot}"

cd "$PROJECT_DIR"
echo "[$(date)] Single-shot eval → $OUT"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/single_shot.py \
    --csv "$CSV" --model "$MODEL" --out "$OUT" \
    --seed 42 --scale 1.0

echo "[$(date)] Single-shot complete."
