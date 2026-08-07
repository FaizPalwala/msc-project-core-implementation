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
elif [ "$DATASET" = "balanced" ]; then
    CSV="$DATA_DIR/metadata/dataset.csv"
else
    echo "ERROR: DATASET must be 'balanced' or 'imbalanced' (got '$DATASET')" >&2
    echo "Usage:  DATASET=balanced bash scripts/hpc_full_pipeline.sh" >&2
    echo "        DATASET=imbalanced bash scripts/hpc_full_pipeline.sh" >&2
    exit 1
fi
OUT="$PROJECT_DIR/results/$DATASET"              # namespace by dataset
MODEL="$OUT/checkpoints/original_model_best.pt"
echo "[$(date)] Pipeline — dataset=$DATASET ($CSV)"
echo "  Output → $OUT"

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

# ── Imbalanced equity plots (only when DATASET=imbalanced) ─────────────
if [ "$DATASET" = "imbalanced" ]; then
    echo "[$(date)] Submitting imbalanced plots (dependency: $SINGLE_JOB)…"
    IMBPLOT_JOB=$(sbatch --parsable \
        --dependency=afterok:$SINGLE_JOB \
        "$PROJECT_DIR/scripts/slurm_imbalanced_plots.sh" \
        "$OUT/single_shot/single_shot_aggregated.json" \
        "$CSV" "$OUT/imbalanced")
    echo "  Imbalanced plots job: $IMBPLOT_JOB"
fi

# HP search parallel to single-shot (can run concurrently)
echo "[$(date)] Submitting HP search (dependency: $TRAIN_JOB)…"
HP_JOB=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB \
    "$PROJECT_DIR/scripts/slurm_hparam.sh" \
    "$CSV" "$MODEL" "$OUT/hparam")
echo "  HP search job: $HP_JOB"

# ── Stage 3: Iterative (after single-shot, uses best configs) ───────────
# Balanced only — imbalanced has no forget_step schedule to iterate over
# (its experimental axis is the popularity gradient, not time).
if [ "$DATASET" = "balanced" ]; then
    echo "[$(date)] Submitting iterative (dependency: $SINGLE_JOB:$HP_JOB)…"
    ITER_JOB=$(sbatch --parsable \
        --dependency=afterok:$SINGLE_JOB:$HP_JOB \
        "$PROJECT_DIR/scripts/slurm_iterative.sh" \
        "$CSV" "$MODEL" "$OUT/iterative")
    echo "  Iterative job: $ITER_JOB"

    # ── Stage 4: Stability plots (after iterative) ──────────────────────
    echo "[$(date)] Submitting stability (dependency: $ITER_JOB)…"
    STAB_JOB=$(sbatch --parsable \
        --dependency=afterok:$ITER_JOB \
        "$PROJECT_DIR/scripts/slurm_stability.sh" \
        "$OUT/iterative/iterative_combined_aggregated.csv" \
        "$OUT/iterative/plots" \
        "$OUT/single_shot/single_shot_per_identity.csv" \
        "$OUT/single_shot/single_shot_demographic.csv")
    echo "  Stability job: $STAB_JOB"
else
    echo "[$(date)] Skipping iterative + stability (imbalanced has no schedule axis)"
    ITER_JOB=""
    STAB_JOB=""
fi

# ── Stage 5: Canary experiment (independent of train — uses its own
#    canary-tagged dataset and training run) ─────────────────────────────
echo "[$(date)] Submitting canary experiment (independent)…"
CANARY_JOB=$(sbatch --parsable \
    "$PROJECT_DIR/scripts/slurm_canary.sh" \
    "$CSV" "$OUT/canary")
echo "  Canary job: $CANARY_JOB"

# ── Stage 6: Report (after stability + canary; stability only for balanced) ──
REPORT_DEPS="$CANARY_JOB"
[ -n "$STAB_JOB" ] && REPORT_DEPS="$STAB_JOB:$REPORT_DEPS"
echo "[$(date)] Submitting report (dependency: $REPORT_DEPS)…"
REPORT_JOB=$(sbatch --parsable \
    --dependency=afterok:$REPORT_DEPS \
    "$PROJECT_DIR/scripts/slurm_report.sh" \
    "$OUT")
echo "  Report job: $REPORT_JOB"

echo ""
echo "[$(date)] Pipeline submitted ($DATASET)."
echo "  Dataset: $DATASET ($CSV)"
echo "  Results: $OUT"
echo "  Logs:    logs/unlearn_*_*.out (match by job IDs below)"
echo "  Monitor: squeue -u \$USER"
echo "  Job IDs: train=$TRAIN_JOB single=$SINGLE_JOB hp=$HP_JOB iter=$ITER_JOB stab=$STAB_JOB canary=$CANARY_JOB report=$REPORT_JOB"
