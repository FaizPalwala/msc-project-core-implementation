#!/bin/bash
# ==========================================
# slurm_hparam.sh — Hyperparameter search
# ==========================================
#SBATCH --job-name=unlearn_hp
# Usage: sbatch scripts/slurm_hparam.sh <csv> <model> <out>

module purge
module load miniforge
module load cuda/12.6.2
conda activate data_gen
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CSV="${1:-$SCRATCH/unlearning_project/data/dataset/dataset.csv}"
MODEL="${2:-$SCRATCH/unlearning_project/results/checkpoints/original_model_best.pt}"
OUT="${3:-$SCRATCH/unlearning_project/results/hparam}"

cd "$PROJECT_DIR"
echo "[$(date)] HP search → $OUT"

python src/hparam_search.py \
    --csv "$CSV" --model "$MODEL" --out "$OUT" \
    --method ng_plus --search grid --seed 42

echo "[$(date)] HP search complete."
