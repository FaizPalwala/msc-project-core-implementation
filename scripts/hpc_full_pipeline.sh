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

# ── Repo root ────────────────────────────────────────────────────────────
# Runs interactively (bash scripts/hpc_full_pipeline.sh), so $0 works;
# the SLURM_SUBMIT_DIR branch is a safety net if this is ever sbatch'd.
# Guarded by the configs/ marker (present only in the repo root).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
[ -d "$PROJECT_DIR/configs" ] || PROJECT_DIR="$(cd "$PROJECT_DIR/.." 2>/dev/null && pwd)"
[ -d "$PROJECT_DIR/configs" ] || { echo "ERROR: repo root not found (no configs/ in $PROJECT_DIR)" >&2; exit 1; }
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"   # canonicalize (no "..")

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
DATASET="${DATASET:-balanced}"                   # "balanced" or "imbalanced"
if [ "$DATASET" = "imbalanced" ]; then
    CSV="$DATA_DIR/metadata/dataset_imbalanced.csv"
    SCALE=1.0
    TAG="imb"
else
    CSV="$DATA_DIR/metadata/dataset.csv"
    SCALE=1.0
    TAG="bal"
fi
MODEL="$PROJECT_DIR/results/checkpoints/original_model_best.pt"
OUT="$PROJECT_DIR/results"
echo "[$(date)] Pipeline — dataset=$DATASET ($CSV)"

# ── Stage 1: Train ──────────────────────────────────────────────────────
echo "[$(date)] Submitting train job…"
TRAIN_JOB=$(sbatch --parsable \
    "$PROJECT_DIR/scripts/slurm_train.sh" \
    "$CSV" "$OUT/checkpoints")
echo "  Train job: $TRAIN_JOB"

# ── Stage 2: Single-shot + HP search (parallel) ─────────────────────────
echo "[$(date)] Submitting single-shot (dependency: $TRAIN_JOB)…"
SINGLE_JOB=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB \
    "$PROJECT_DIR/scripts/slurm_single_shot.sh" \
    "$CSV" "$MODEL" "$OUT/single_shot")
echo "  Single-shot job: $SINGLE_JOB"

# HP search parallel to single-shot (can run concurrently)
echo "[$(date)] Submitting HP search (dependency: $TRAIN_JOB)…"
HP_JOB=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB \
    "$PROJECT_DIR/scripts/slurm_hparam.sh" \
    "$CSV" "$MODEL" "$OUT/hparam")
echo "  HP search job: $HP_JOB"

# ── Stage 3: Iterative (after single-shot, uses best configs) ───────────
echo "[$(date)] Submitting iterative (dependency: $SINGLE_JOB:$HP_JOB)…"
ITER_JOB=$(sbatch --parsable \
    --dependency=afterok:$SINGLE_JOB:$HP_JOB \
    "$PROJECT_DIR/scripts/slurm_iterative.sh" \
    "$CSV" "$MODEL" "$OUT/iterative")
echo "  Iterative job: $ITER_JOB"

# ── Stage 4: Stability plots (after iterative) ──────────────────────────
echo "[$(date)] Submitting stability (dependency: $ITER_JOB)…"
STAB_JOB=$(sbatch --parsable \
    --dependency=afterok:$ITER_JOB \
    "$PROJECT_DIR/scripts/slurm_stability.sh" \
    "$OUT/iterative/iterative_combined_aggregated.csv" \
    "$OUT/iterative/plots")
echo "  Stability job: $STAB_JOB"

# ── Stage 5: Canary experiment (independent of train — uses its own
#    canary-tagged dataset and training run) ─────────────────────────────
echo "[$(date)] Submitting canary experiment (independent)…"
CANARY_JOB=$(sbatch --parsable \
    "$PROJECT_DIR/scripts/slurm_canary.sh" \
    "$CSV" "$OUT/canary")
echo "  Canary job: $CANARY_JOB"

# ── Stage 6: Report (after stability + canary) ──────────────────────────
echo "[$(date)] Submitting report (dependency: $STAB_JOB:$CANARY_JOB)…"
REPORT_JOB=$(sbatch --parsable \
    --dependency=afterok:$STAB_JOB:$CANARY_JOB \
    "$PROJECT_DIR/scripts/slurm_report.sh" \
    "$OUT")
echo "  Report job: $REPORT_JOB"

echo ""
echo "[$(date)] Pipeline submitted."
echo "  Monitor: squeue -u \$USER"
echo "  Job IDs: train=$TRAIN_JOB single=$SINGLE_JOB hp=$HP_JOB iter=$ITER_JOB stab=$STAB_JOB canary=$CANARY_JOB report=$REPORT_JOB"
