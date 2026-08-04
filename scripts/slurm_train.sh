#!/bin/bash
# ==========================================
# slurm_train.sh — Train original dual-head model
# ==========================================
#SBATCH --job-name=unlearn_train
# Usage: sbatch scripts/slurm_train.sh <csv_path> <save_dir>

# ── Environment ──────────────────────────────────────────────────────────
module purge
module load miniforge
module load cuda/12.6.2
conda activate data_gen

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
CSV="${1:-$DATA_DIR/metadata/dataset.csv}"
SAVE_DIR="${2:-$PROJECT_DIR/results/checkpoints}"

cd "$PROJECT_DIR"
echo "[$(date)] Training on $CSV → $SAVE_DIR"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/train.py \
    --csv "$CSV" \
    --save_dir "$SAVE_DIR" \
    --epochs 30 --lr 1e-3 --batch_size 64 \
    --age_weight 0.5 --seed 42

echo "[$(date)] Train complete."
