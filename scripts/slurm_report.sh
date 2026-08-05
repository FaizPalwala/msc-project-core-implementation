#!/bin/bash
# ==========================================
# slurm_report.sh — Dissertation-ready report generation
# ==========================================
#SBATCH --job-name=unlearn_report
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage: sbatch scripts/slurm_report.sh <results_root>
#
# Consumes the pipeline artifacts under <results_root>/ and produces
# LaTeX tables + markdown digest + machine-readable CSVs via report.py:
#   <results_root>/report/{report.tex, report.md, single_shot_table.csv}

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

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
RESULTS="${1:-$PROJECT_DIR/results}"

cd "$PROJECT_DIR"
echo "[$(date)] Generating report from $RESULTS"

python src/report.py \
    --results "$RESULTS" \
    --out "$RESULTS/report"

echo "[$(date)] Report complete: $RESULTS/report/"
