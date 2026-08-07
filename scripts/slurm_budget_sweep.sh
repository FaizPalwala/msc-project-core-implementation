#!/bin/bash
# ==========================================
# slurm_budget_sweep.sh — Protocol C budget sweep (imbalanced only)
# ==========================================
#SBATCH --job-name=unlearn_bswp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Usage: sbatch scripts/slurm_budget_sweep.sh \
#            <csv> <model> <methods_json> <oracles_dir> <out> [best_configs_dir]
#
# Sweeps unlearning budget per Protocol-C-picked method, measuring per-bin
# distance-to-oracle (Protocol B oracles as reference).  Needs:
#   methods_json : protocol_c_methods.json (dynamic 1-best-per-category)
#   oracles_dir  : Protocol B per-bin oracles (oracle_{high,medium,low}.pt)
#   best_configs_dir : hparam dir of *_best_config.json (tuned budgets)

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
METHODS_JSON="${3:-$OUT_BASE/protocol_c_methods.json}"
ORACLES="${4:-$OUT_BASE/oracles}"
OUT="${5:-$OUT_BASE}"
BEST="${6:-$OUT_BASE/hparam}"

cd "$PROJECT_DIR"
echo "[$(date)] Budget sweep (Protocol C) → $OUT/budget_sweep.json"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/budget_sweep.py \
    --csv "$CSV" --model "$MODEL" \
    --methods "$METHODS_JSON" --oracles "$ORACLES" \
    --best_configs "$BEST" --out "$OUT" 2>&1

echo "[$(date)] Budget sweep complete."
