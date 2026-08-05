#!/bin/bash
# ==========================================
# slurm_dataset_helper.sh — Parse --dataset flag for standalone slurm usage.
# ==========================================
# Source this in any slurm script to enable:
#
#   sbatch scripts/slurm_train.sh --dataset imbalanced
#   sbatch scripts/slurm_train.sh --dataset balanced
#
# Pipeline backward-compat: hpc_full_pipeline.sh passes CSV & output dirs
# as positional args, which override the defaults set here.
#
# Exports: DATASET, CSV_DEFAULT, OUT_BASE
# Caller uses:  CSV="${1:-$CSV_DEFAULT}"  or similar.
# ==========================================

# Default: respect DATASET env var (set by hpc_full_pipeline.sh), else "balanced"
DATASET="${DATASET:-balanced}"

# Parse --dataset from args (remove it; leave other args intact)
FILTERED=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET="$2"; shift 2 ;;
        --dataset=*)   DATASET="${1#*=}"; shift ;;
        *)             FILTERED+=("$1"); shift ;;
    esac
done
set -- "${FILTERED[@]}"

# Resolve
DATA_DIR="$(dirname "$PROJECT_DIR")/bench"
case "$DATASET" in
    balanced)   CSV_DEFAULT="$DATA_DIR/metadata/dataset.csv" ;;
    imbalanced) CSV_DEFAULT="$DATA_DIR/metadata/dataset_imbalanced.csv" ;;
    *)
        echo "ERROR: --dataset must be 'balanced' or 'imbalanced' (got '$DATASET')" >&2
        exit 1
        ;;
esac

OUT_BASE="$PROJECT_DIR/results/$DATASET"
