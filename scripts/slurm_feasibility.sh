#!/bin/bash
# ==========================================
# slurm_feasibility.sh — C2/C3 method-feasibility gate (small scale)
# ==========================================
#SBATCH --job-name=unlearn_feas
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
# Usage: sbatch scripts/slurm_feasibility.sh [dataset] [methods...]
#   dataset : balanced (default) | imbalanced — which CSV to subsample
#   methods : space-separated subset of
#             ga ng_plus adaptiformet msg msg_kd ct ft srl
#             (default: all 8)
#
# Cheap pre-flight before the expensive full re-run: trains a small model
# on ~12 real identities and sweeps each method's budget at 1x/3x/10x to
# answer C2/C3 — is the 750-id failure a CONFIG problem (method responds
# to more budget) or STRUCTURAL (no response even at 10x)?
# Results → results/feasibility_<dataset>/feasibility_results.json

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

DATASET="${1:-$DATASET}"
CSV="${DATASET}_csv"      # placeholder; replaced below
if [ "$DATASET" = "imbalanced" ]; then
    CSV="$(dirname "$CSV_DEFAULT")/dataset_imbalanced.csv"
else
    CSV="$CSV_DEFAULT"
fi
OUT="$PROJECT_DIR/results/feasibility_${DATASET}"
shift 2>/dev/null || true
METHODS=("$@")
[ ${#METHODS[@]} -eq 0 ] && METHODS=(ga ng_plus adaptiformet msg msg_kd ct ft srl)

cd "$PROJECT_DIR"
echo "[$(date)] Feasibility gate (${DATASET}) → $OUT"
echo "[$(date)] Methods: ${METHODS[*]}"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/feasibility_study.py \
    --src_csv "$CSV" \
    --out "$OUT" \
    --methods "${METHODS[@]}" \
    --multipliers 1 3 10 \
    --epochs 3 \
    --n_retain 8 --n_forget 4 \
    --imgs_per_id 16 \
    --step_scale 1.0 \
    --device cuda 2>&1

echo "[$(date)] Feasibility complete. → $OUT/feasibility_results.json"
