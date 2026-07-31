#!/bin/bash
# ==========================================
# slurm_stability.sh — Stability plots
# ==========================================
#SBATCH --job-name=unlearn_stab
# Usage: sbatch scripts/slurm_stability.sh <combined_csv> <out_dir>

module purge
module load miniforge
conda activate data_gen
export MPLBACKEND=Agg

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMBINED="${1:-$SCRATCH/unlearning_project/results/iterative/iterative_combined.csv}"
OUT="${2:-$SCRATCH/unlearning_project/results/iterative/plots}"

cd "$PROJECT_DIR"
echo "[$(date)] Stability plots → $OUT"

python src/stability.py --combined "$COMBINED" --out "$OUT"

echo "[$(date)] Stability plots complete."
