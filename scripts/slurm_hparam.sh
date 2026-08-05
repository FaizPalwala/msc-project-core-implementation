#!/bin/bash
# ==========================================
# slurm_hparam.sh — Hyperparameter search
# ==========================================
#SBATCH --job-name=unlearn_hp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage: sbatch scripts/slurm_hparam.sh <csv> <model> <out>

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
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
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
