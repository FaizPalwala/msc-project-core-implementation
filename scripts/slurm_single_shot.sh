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
CSV="${1:-$SCRATCH/unlearning_project/data/dataset/dataset.csv}"
MODEL="${2:-$SCRATCH/unlearning_project/results/checkpoints/original_model_best.pt}"
OUT="${3:-$SCRATCH/unlearning_project/results/single_shot}"

cd "$PROJECT_DIR"
echo "[$(date)] Single-shot eval → $OUT"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/single_shot.py \
    --csv "$CSV" --model "$MODEL" --out "$OUT" \
    --seed 42 --scale 1.0

echo "[$(date)] Single-shot complete."
