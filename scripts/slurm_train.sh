#!/bin/bash
# ==========================================
# slurm_train.sh — Train original dual-head model
# ==========================================
#SBATCH --job-name=unlearn_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Usage: sbatch scripts/slurm_train.sh <csv_path> <save_dir>

# ── Repo root ────────────────────────────────────────────────────────────
# Under sbatch, $0 points at the spool copy (/var/spool/slurmd/...), so
# resolve the repo from SLURM_SUBMIT_DIR — the dir sbatch was invoked
# from. Fall back to $0 for interactive runs. Climb one level if the
# script was submitted from scripts/ itself. Guarded by the configs/
# marker (present only in the repo root).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
[ -d "$PROJECT_DIR/configs" ] || PROJECT_DIR="$(cd "$PROJECT_DIR/.." 2>/dev/null && pwd)"
[ -d "$PROJECT_DIR/configs" ] || { echo "ERROR: repo root not found (no configs/ in $PROJECT_DIR)" >&2; exit 1; }
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"   # canonicalize (no "..")

# ── Environment ──────────────────────────────────────────────────────────
module purge
module load miniforge
module load cuda/12.6.2
conda activate core

DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
CSV="${1:-$DATA_DIR/metadata/dataset.csv}"
SAVE_DIR="${2:-$PROJECT_DIR/results/checkpoints}"

cd "$PROJECT_DIR"
echo "[$(date)] Training on $CSV → $SAVE_DIR"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/train.py \
    --csv "$CSV" \
    --subset train \
    --save_dir "$SAVE_DIR" \
    --epochs 30 --lr 1e-3 --batch_size 64 \
    --age_weight 0.5 --seed 42

echo "[$(date)] Train complete."
