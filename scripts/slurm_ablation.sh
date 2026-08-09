#!/bin/bash
# ==========================================
# slurm_ablation.sh — AdaptiForget component ablation (runs in parallel
#                     with the main suite, like the canary experiment)
# ==========================================
#SBATCH --job-name=unlearn_ablation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail   # any python failure → job FAILED (never phantom COMPLETED)
# Usage: sbatch scripts/slurm_ablation.sh <csv_path> <out_dir>
#
# Component ablation of AdaptiForget: runs the same single-shot
# unlearning with each component disabled in turn (adaptive λ, mask
# refresh, early stop, KL distillation, masking), on the SAME trained
# checkpoint the main suite uses.  Reuses <out_dir>/../checkpoints/
# original_model_best.pt — the train job's output — so it depends on
# train but runs parallel to single-shot/hparam/iterative.
#
# Results: <out_dir>/ablation_results.csv + .json (consumed by the
# report stage, which renders an 'Ablation' section).

# ── Repo root ────────────────────────────────────────────────────────────
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

source "$PROJECT_DIR/scripts/slurm_dataset_helper.sh"  # sets DATASET, CSV_DEFAULT, OUT_BASE
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
CSV="${1:-$CSV_DEFAULT}"
OUT="${2:-$OUT_BASE/ablation}"

MODEL="$OUT/../checkpoints/original_model_best.pt"     # main train job's checkpoint
[ -f "$MODEL" ] || { echo "ERROR: trained model not found: $MODEL" >&2; exit 1; }

mkdir -p "$OUT"
cd "$PROJECT_DIR"
echo "[$(date)] Ablation study on $CSV (model $MODEL) → $OUT"

python src/ablation_study.py \
    --csv "$CSV" \
    --model "$MODEL" \
    --out "$OUT" \
    --device auto --seed 42 --scale 1.0 2>&1

echo "[$(date)] Ablation study complete."
echo "  Results: $OUT/ablation_results.csv"
