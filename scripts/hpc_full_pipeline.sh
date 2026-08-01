#!/bin/bash
# ==========================================
# hpc_full_pipeline.sh — Full unlearning pipeline on Aire HPC
# ==========================================
# Chains: train → single_shot → iterative → stability
# HP search can be submitted in parallel (see comments below).
#
# Usage:
#   bash scripts/hpc_full_pipeline.sh
# ==========================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Config ──────────────────────────────────────────────────────────────
CSV="$SCRATCH/unlearning_project/data/dataset/dataset.csv"
MODEL="$SCRATCH/unlearning_project/results/checkpoints/original_model_best.pt"
OUT="$SCRATCH/unlearning_project/results"

# ── Stage 1: Train ──────────────────────────────────────────────────────
echo "[$(date)] Submitting train job…"
TRAIN_JOB=$(sbatch --parsable \
    --job-name=unlearn_train \
    --time=12:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    --output="$LOG_DIR/train_%j.out" \
    --error="$LOG_DIR/train_%j.err" \
    "$PROJECT_DIR/scripts/slurm_train.sh" \
    "$CSV" "$OUT/checkpoints")
echo "  Train job: $TRAIN_JOB"

# ── Stage 2: Single-shot + HP search (parallel) ─────────────────────────
echo "[$(date)] Submitting single-shot (dependency: $TRAIN_JOB)…"
SINGLE_JOB=$(sbatch --parsable \
    --job-name=unlearn_single \
    --dependency=afterok:$TRAIN_JOB \
    --time=24:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    --output="$LOG_DIR/single_%j.out" \
    --error="$LOG_DIR/single_%j.err" \
    "$PROJECT_DIR/scripts/slurm_single_shot.sh" \
    "$CSV" "$MODEL" "$OUT/single_shot")
echo "  Single-shot job: $SINGLE_JOB"

# HP search parallel to single-shot (can run concurrently)
echo "[$(date)] Submitting HP search (dependency: $TRAIN_JOB)…"
HP_JOB=$(sbatch --parsable \
    --job-name=unlearn_hp \
    --dependency=afterok:$TRAIN_JOB \
    --time=24:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    --output="$LOG_DIR/hp_%j.out" \
    --error="$LOG_DIR/hp_%j.err" \
    "$PROJECT_DIR/scripts/slurm_hparam.sh" \
    "$CSV" "$MODEL" "$OUT/hparam")
echo "  HP search job: $HP_JOB"

# ── Stage 3: Iterative (after single-shot, uses best configs) ───────────
echo "[$(date)] Submitting iterative (dependency: $SINGLE_JOB:$HP_JOB)…"
ITER_JOB=$(sbatch --parsable \
    --job-name=unlearn_iter \
    --dependency=afterok:$SINGLE_JOB:$HP_JOB \
    --time=48:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    --output="$LOG_DIR/iter_%j.out" \
    --error="$LOG_DIR/iter_%j.err" \
    "$PROJECT_DIR/scripts/slurm_iterative.sh" \
    "$CSV" "$MODEL" "$OUT/iterative")
echo "  Iterative job: $ITER_JOB"

# ── Stage 4: Stability plots (after iterative) ──────────────────────────
echo "[$(date)] Submitting stability (dependency: $ITER_JOB)…"
STAB_JOB=$(sbatch --parsable \
    --job-name=unlearn_stab \
    --dependency=afterok:$ITER_JOB \
    --time=2:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=4 --mem=16G \
    --output="$LOG_DIR/stab_%j.out" \
    --error="$LOG_DIR/stab_%j.err" \
    "$PROJECT_DIR/scripts/slurm_stability.sh" \
    "$OUT/iterative/iterative_combined.csv" \
    "$OUT/iterative/plots")
echo "  Stability job: $STAB_JOB"

# ── Stage 5: Canary experiment (independent of train — uses its own
#    canary-tagged dataset and training run) ─────────────────────────────
echo "[$(date)] Submitting canary experiment (independent)…"
CANARY_JOB=$(sbatch --parsable \
    --job-name=unlearn_canary \
    --time=12:00:00 --partition=gpu --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G \
    --output="$LOG_DIR/canary_%j.out" \
    --error="$LOG_DIR/canary_%j.err" \
    "$PROJECT_DIR/scripts/slurm_canary.sh" \
    "$CSV" "$OUT/canary")
echo "  Canary job: $CANARY_JOB"

# ── Stage 6: Report (after stability + canary) ──────────────────────────
echo "[$(date)] Submitting report (dependency: $STAB_JOB:$CANARY_JOB)…"
REPORT_JOB=$(sbatch --parsable \
    --job-name=unlearn_report \
    --dependency=afterok:$STAB_JOB:$CANARY_JOB \
    --time=1:00:00 --partition=gpu --gres=gpu:0 \
    --cpus-per-task=2 --mem=8G \
    --output="$LOG_DIR/report_%j.out" \
    --error="$LOG_DIR/report_%j.err" \
    "$PROJECT_DIR/scripts/slurm_report.sh" \
    "$OUT")
echo "  Report job: $REPORT_JOB"

echo ""
echo "[$(date)] Pipeline submitted."
echo "  Monitor: squeue -u \$USER"
echo "  Job IDs: train=$TRAIN_JOB single=$SINGLE_JOB hp=$HP_JOB iter=$ITER_JOB stab=$STAB_JOB canary=$CANARY_JOB report=$REPORT_JOB"
