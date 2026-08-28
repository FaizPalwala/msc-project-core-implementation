#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# rerun_canary_ft_20260828.sh — add FT to the canary protocol (one-time)
#
# The canonical run (f3) verified the canary with 2 methods (ga,
# adaptiforget).  The 3-method contrast (adaptiforget + ft + ga) was
# decided for the dissertation: FT is the near-oracle positive-erasure
# control (its gap should also be ≈ 0), GA the negative control (gap ≫ 0).
#
# This script re-runs ONLY the canary lanes on both datasets with the
# updated slurm_canary.sh (3 methods), regenerates both reports so the
# canary tables pick up verify_ft.json, and packages everything into
# results_fplus_<stamp>.zip.
#
# Requires: slurm_canary.sh already updated to --methods ga adaptiforget ft
#           (commit ...: "feat(canary): 3-method contrast with FT control")
#
# Usage:   bash scripts/rerun_canary_ft_20260828.sh
# Then:    scp aire:core/code/results_fplus_<stamp>.zip → local, unpack,
#          and regenerate the Zenodo artefact.
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── 0. Repo + home-dir health (jobs died here before: Errno 5 on I/O) ──
if [ ! -d "$HOME/.cache/huggingface" ]; then
    echo "WARN: no $HOME/.cache/huggingface — train may re-download weights" >&2
fi
if [ ! -w "$HOME" ]; then
    echo "ERROR: home dir not writable — canary train will fail (Errno 5)." >&2
    exit 1
fi
echo "── 0. Home-dir health ──"
echo "  home OK"

# ── 1. Canary (both datasets, parallel) — 3 methods now ────────────────
# slurm_canary.sh: $1 = CSV, $2 = OUT/canary, $3 = BEST_CONFIGS (hparam dir)
echo ""
echo "── 1. Canary re-run (ga + adaptiforget + ft, both datasets) ──"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$(dirname "$ROOT")/bench/metadata"

CANB=$(sbatch --parsable \
    "$ROOT/scripts/slurm_canary.sh" \
    "$BENCH/dataset.csv" \
    "$ROOT/results/balanced/canary" \
    "$ROOT/results/balanced/hparam")
echo "  CANB (balanced): $CANB"

CANI=$(sbatch --parsable \
    "$ROOT/scripts/slurm_canary.sh" \
    "$BENCH/dataset_imbalanced.csv" \
    "$ROOT/results/imbalanced/canary" \
    "$ROOT/results/imbalanced/hparam")
echo "  CANI (imbalanced): $CANI"

# ── 2. Reports (both, after their canary lane) — picks up verify_ft.json ─
echo ""
echo "── 2. Reports (both, after canary) ──"
REPB=$(sbatch --parsable \
    --dependency=afterok:$CANB \
    "$ROOT/scripts/slurm_report.sh" "$ROOT/results/balanced")
echo "  REPB: $REPB"

REPI=$(sbatch --parsable \
    --dependency=afterok:$CANI \
    "$ROOT/scripts/slurm_report.sh" "$ROOT/results/imbalanced")
echo "  REPI: $REPI"

# ── 3. PACKAGE (all — balanced + imbalanced + logs) ────────────────────
echo ""
echo "── 3. PACKAGE (all) ──"
STAMP="$(date +%Y%m%d_%H%M)"
PKG=$(sbatch --parsable \
    --dependency=afterok:$REPB:$REPI \
    "$ROOT/scripts/slurm_package.sh" all "results_fplus_${STAMP}")
echo "  PKG: $PKG   (→ results_fplus_${STAMP}.zip at repo root)"

echo ""
echo "── MONITOR ──"
echo "  squeue -u $USER"
echo ""
echo "  After: verify FT canary gaps ≈ 0 (positive-erasure control):"
echo "    grep -h '\"gap\"' results/balanced/canary/verify_ft.json"
echo "  Expect: 4 identities with gap ≈ 0.000 (like adaptiforget),"
echo "          NOT ≫ 0 (which would mean FT leaves the pattern)."
