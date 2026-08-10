#!/bin/bash
# ==========================================
# slurm_package.sh — zip results/ + logs/ for easy retrieval
# ==========================================
#SBATCH --job-name=unlearn_pkg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
# Usage: sbatch scripts/slurm_package.sh [dataset] [out_name]
#   dataset : balanced (default) | imbalanced | all — which results/ to zip
#   out_name: optional zip basename (default: results_<dataset>_<date>)
#
# Packages results/<dataset> (+ logs/) into a single zip at the repo root,
# ready to scp/rsync down.  CPU-only, quick — run AFTER the report stage.

# ── Repo root ────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
[ -d "$PROJECT_DIR/configs" ] || PROJECT_DIR="$(cd "$PROJECT_DIR/.." 2>/dev/null && pwd)"
[ -d "$PROJECT_DIR/configs" ] || { echo "ERROR: repo root not found (no configs/ in $PROJECT_DIR)" >&2; exit 1; }
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

DATASET="${1:-balanced}"
STAMP="$(date +%Y%m%d_%H%M)"
OUT_NAME="${2:-results_${DATASET}_${STAMP}}"
ZIP_PATH="$PROJECT_DIR/${OUT_NAME}.zip"

cd "$PROJECT_DIR"
echo "[$(date)] Packaging → $ZIP_PATH"

# ── What to include ──────────────────────────────────────────────────────
INCLUDE=("logs")
if [ "$DATASET" = "all" ]; then
    INCLUDE+=("results")
else
    INCLUDE+=("results/$DATASET")
fi

# ── Zip (deterministic-ish: no leading paths, exclude __pycache__/.DS_Store) ─
# Use zip -r with -x exclusions; fail loudly if nothing got packaged.
zip -r -q "$ZIP_PATH" "${INCLUDE[@]}" \
    -x "*/__pycache__/*" "*.DS_Store" "*.gitkeep"
zipinfo -t "$ZIP_PATH" >/dev/null 2>&1 || { echo "ERROR: zip failed or empty" >&2; exit 1; }

N_FILES="$(zipinfo -t "$ZIP_PATH" | awk '{print $1}')"
SIZE_MB="$(du -m "$ZIP_PATH" | cut -f1)"
echo "[$(date)] [OK] $ZIP_PATH — $N_FILES files, ${SIZE_MB} MB"
