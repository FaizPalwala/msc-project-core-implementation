#!/bin/bash
# ==========================================
# slurm_imbalanced_plots.sh — Per‑popularity‑bin equity plots
# ==========================================
#SBATCH --job-name=unlearn_imb_plots
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Usage: sbatch scripts/slurm_imbalanced_plots.sh <aggregated_json> <csv> <out_dir>

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
conda activate core

RESULTS="${1?Usage: slurm_imbalanced_plots.sh <aggregated.json> <csv> <out_dir>}"
CSV="${2?}"
OUT="${3?}"

cd "$PROJECT_DIR"
echo "[$(date)] Imbalanced equity plots → $OUT"

python src/imbalanced_plots.py \
    --results "$RESULTS" --csv "$CSV" --out "$OUT" 2>&1

echo "[$(date)] Imbalanced plots complete. $OUT/"
