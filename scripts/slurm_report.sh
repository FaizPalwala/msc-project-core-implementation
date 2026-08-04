#!/bin/bash
# ==========================================
# slurm_report.sh — Dissertation-ready report generation
# ==========================================
#SBATCH --job-name=unlearn_report
# Usage: sbatch scripts/slurm_report.sh <results_root>
#
# Consumes the pipeline artifacts under <results_root>/ and produces
# LaTeX tables + markdown digest + machine-readable CSVs via report.py:
#   <results_root>/report/{report.tex, report.md, single_shot_table.csv}

module purge
module load miniforge
conda activate data_gen

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="${1:-$PROJECT_DIR/results}"

cd "$PROJECT_DIR"
echo "[$(date)] Generating report from $RESULTS"

python src/report.py \
    --results "$RESULTS" \
    --out "$RESULTS/report"

echo "[$(date)] Report complete: $RESULTS/report/"
