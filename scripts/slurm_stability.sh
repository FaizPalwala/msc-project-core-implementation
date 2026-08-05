#!/bin/bash
# ==========================================
# slurm_stability.sh — Stability plots
# ==========================================
#SBATCH --job-name=unlearn_stab
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage: sbatch scripts/slurm_stability.sh <combined_csv> <out_dir>

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
conda activate core
export MPLBACKEND=Agg

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
COMBINED="${1:-$PROJECT_DIR/results/iterative/iterative_combined_aggregated.csv}"
OUT="${2:-$PROJECT_DIR/results/iterative/plots}"

cd "$PROJECT_DIR"
echo "[$(date)] Stability plots → $OUT"

python src/stability.py --combined "$COMBINED" --out "$OUT"

echo "[$(date)] Stability plots complete."
