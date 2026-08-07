#!/bin/bash
# ==========================================
# slurm_single_shot.sh — Single-shot evaluation
# ==========================================
#SBATCH --job-name=unlearn_single
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Usage: sbatch scripts/slurm_single_shot.sh <csv> <model> <out> [best_configs_dir]
#   best_configs_dir: optional dir of hparam *_best_config.json — tuned
#   per-method values override YAML defaults (the "single-shot with best
#   params after HP search" stage).

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
source "$PROJECT_DIR/scripts/slurm_dataset_helper.sh"  # sets DATASET, CSV_DEFAULT, OUT_BASE
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
CSV="${1:-$CSV_DEFAULT}"
MODEL="${2:-$OUT_BASE/checkpoints/original_model_best.pt}"
OUT="${3:-$OUT_BASE/single_shot}"
BEST="${4:-}"

cd "$PROJECT_DIR"
echo "[$(date)] Single-shot eval → $OUT"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

if [ -n "$BEST" ]; then
    python src/single_shot.py \
        --csv "$CSV" --model "$MODEL" --out "$OUT" \
        --seed 42 --scale 1.0 --best_configs "$BEST" 2>&1
else
    python src/single_shot.py \
        --csv "$CSV" --model "$MODEL" --out "$OUT" \
        --seed 42 --scale 1.0 2>&1
fi

echo "[$(date)] Single-shot complete."
