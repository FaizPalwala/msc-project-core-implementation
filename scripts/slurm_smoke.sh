#!/bin/bash
# ==========================================
# slurm_smoke.sh — End-to-end smoke test (v1.1 schema)
# ==========================================
#SBATCH --job-name=unlearn_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail
# Usage: sbatch scripts/slurm_smoke.sh
#
# Runs tests/smoke_test.py end-to-end: builds a 6-identity synthetic
# dataset (3 retain / 3 forget, v1.1 schema with forget_step +
# forget_step_poisson), trains 2 epochs, runs single-shot GA through
# schedule-aware splits, and asserts MIA AUC + retain acc.  CPU-only.

# ── Repo root ────────────────────────────────────────────────────────────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
[ -d "$PROJECT_DIR/configs" ] || PROJECT_DIR="$(cd "$PROJECT_DIR/.." 2>/dev/null && pwd)"
[ -d "$PROJECT_DIR/configs" ] || { echo "ERROR: repo root not found (no configs/ in $PROJECT_DIR)" >&2; exit 1; }
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"

# ── Environment ──────────────────────────────────────────────────────────
module purge
module load miniforge
conda activate core
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$PROJECT_DIR"
echo "[$(date)] Smoke test (v1.1 schema, CPU)…"

python tests/smoke_test.py 2>&1

echo "[$(date)] Smoke test complete."
