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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CSV="${1:-$SCRATCH/unlearning_project/data/dataset/dataset.csv}"
SAVE_DIR="${2:-$SCRATCH/unlearning_project/results/checkpoints}"

cd "$PROJECT_DIR"
echo "[$(date)] Training on $CSV → $SAVE_DIR"

python src/train.py \
    --csv "$CSV" \
    --save_dir "$SAVE_DIR" \
    --epochs 30 --lr 1e-3 --batch_size 64 \
    --age_weight 0.5 --seed 42

echo "[$(date)] Train complete."
