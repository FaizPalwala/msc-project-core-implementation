#!/bin/bash
# ==========================================
# slurm_select_protocol_c.sh — Protocol C dynamic method selection (CPU)
# ==========================================
#SBATCH --job-name=unlearn_selc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
set -euo pipefail
# Usage: sbatch scripts/slurm_select_protocol_c.sh <aggregated_json> <out_dir>
#
# Picks ONE best method per category (baseline/SOTA/novel) by UF score
# from the single_shot_best aggregated results, writing
# protocol_c_methods.json for the Protocol C budget sweep.

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

AGG="${1?Usage: slurm_select_protocol_c.sh <aggregated.json> <out_dir>}"
OUT="${2?}"

cd "$PROJECT_DIR"
echo "[$(date)] Protocol C selection → $OUT"

python src/select_protocol_c_methods.py \
    --aggregated "$AGG" --out "$OUT" 2>&1

echo "[$(date)] Protocol C selection complete."
