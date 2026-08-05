#!/bin/bash
# ==========================================
# slurm_iterative.sh — Iterative unlearning protocol
# ==========================================
#SBATCH --job-name=unlearn_iter
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Usage: sbatch scripts/slurm_iterative.sh <csv> <model> <out>

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
OUT="${3:-$PROJECT_DIR/results/iterative}"

cd "$PROJECT_DIR"
echo "[$(date)] Iterative protocol → $OUT"

# GPU preflight
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1

python src/iterative.py \
    --csv "$CSV" --model "$MODEL" --out "$OUT" \
    --n_steps 15 --mode cumulative \
    --seed 42 --scale 1.0 \
    --re_emergence 5 10 15

echo "[$(date)] Iterative complete."
