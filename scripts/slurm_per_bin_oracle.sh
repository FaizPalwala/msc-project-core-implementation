#!/bin/bash
# ==========================================
# slurm_per_bin_oracle.sh — Protocol B per-bin oracles (imbalanced only)
# ==========================================
#SBATCH --job-name=unlearn_orb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
# Usage: sbatch scripts/slurm_per_bin_oracle.sh <csv> <model> <out> [epochs]
#
# Retrains THREE per-bin oracles (high/medium/low), each excluding ONLY
# that bin's forget identities from the training set — the Protocol B
# counterfactual (the oracle DOES change per bin).  Also writes the
# per-bin behavioral distance table vs single_shot_best results when the
# sibling directory exists.

# ── Repo root ────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
[ -d "$PROJECT_DIR/configs" ] || PROJECT_DIR="$(cd "$PROJECT_DIR/.." 2>/dev/null && pwd)"
[ -d "$PROJECT_DIR/configs" ] || { echo "ERROR: repo root not found (no configs/ in $PROJECT_DIR)" >&2; exit 1; }
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

# ── Environment ──────────────────────────────────────────────────────────
module purge
module load miniforge
module load cuda/12.6.2
conda activate core
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
source "$PROJECT_DIR/scripts/slurm_dataset_helper.sh"  # sets DATASET, CSV_DEFAULT, OUT_BASE
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
CSV="${1:-$CSV_DEFAULT}"
MODEL="${2:-$OUT_BASE/checkpoints/original_model_best.pt}"
OUT="${3:-$OUT_BASE/oracles}"
EPOCHS="${4:-30}"

cd "$PROJECT_DIR"
echo "[$(date)] Per-bin oracles (Protocol B, $EPOCHS epochs) → $OUT"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/per_bin_oracle.py \
    --csv "$CSV" --model "$MODEL" --out "$OUT" \
    --epochs "$EPOCHS" 2>&1

echo "[$(date)] Per-bin oracles complete."
