#!/bin/bash
# ==========================================
# slurm_hparam.sh — Hyperparameter search
# ==========================================
#SBATCH --job-name=unlearn_hp
# Usage: sbatch scripts/slurm_hparam.sh <csv> <model> <out>

module purge
module load miniforge
module load cuda/12.6.2
conda activate core
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
CSV="${1:-$DATA_DIR/metadata/dataset.csv}"
MODEL="${2:-$PROJECT_DIR/results/checkpoints/original_model_best.pt}"
OUT="${3:-$PROJECT_DIR/results/hparam}"

cd "$PROJECT_DIR"
echo "[$(date)] HP search → $OUT"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/hparam_search.py \
    --csv "$CSV" --model "$MODEL" --out "$OUT" \
    --method ng_plus --search grid --seed 42

echo "[$(date)] HP search complete."
