#!/bin/bash
# ============================================================================
# FRESH CANONICAL RUN — 17 Aug 2026 (supersedes salvage_20260817.sh)
# ============================================================================
# Run from: /mnt/scratch/kxvs0578/core/code   (repo root, AFTER git pull)
#
# Why fresh: the 15-17 Aug run is NOT canonical —
#   1. hparam ran with identity_classes un-injected (SRL relabeled into a
#      600-class space vs the 750-class head) → SRL's tuned config tainted
#      → single_shot_best / iterative / Protocol C all inherited it.
#   2. feasibility gates errored on srl/budget_scaled (CUDA assert, fixed
#      in 775dfd7).
#   3. 7137171 died to a home-dir I/O error → 4-seed retrain + cancelled
#      imbalanced downstream.
#   FIXED commits: 775dfd7 (identity_classes injection, all call sites),
#                   d30b707 (hparam logging), 6312077 (salvage, superseded).
#
# This runs the FULL pipeline from scratch (train included) for BOTH
# datasets, with the feasibility gate wired in (Stage 2c) and the report
# rendering the verdict matrix.  One clean provenance story.
# ============================================================================
set -euo pipefail

echo "═══ PRE-FLIGHT ═══"
echo "── 0a. Home dir health (the 7137171 killer) ──"
ls -la /users/kxvs0578 >/dev/null && echo "  home dir OK" || { echo "  home dir UNREACHABLE — stop, contact support"; exit 1; }
df -h /users/kxvs0578 | tail -1

echo "── 0b. Bench data = new release (750-id)? ──"
PYTHONPATH=src python - <<'EOF'
from dataset import infer_identity_classes
import sys
for ds, f in [("balanced", "dataset.csv"), ("imbalanced", "dataset_imbalanced.csv")]:
    n = infer_identity_classes(f"../bench/metadata/{f}")
    print(f"  {ds}: {n} identity classes", "OK" if n == 750 else "!! EXPECTED 750")
    if n != 750: sys.exit(1)
EOF

echo "── 0c. Archive previous results (if not already) ──"
for ds in balanced imbalanced; do
    if [ -d "results/$ds" ]; then
        echo "  results/$ds exists — move to results/$ds.prev_$(date +%Y%m%d) first, or delete"
        exit 1
    fi
done
echo "  results/ clean"

echo ""
echo "═══ SUBMIT: BALANCED PIPELINE ═══"
echo "  (train → single_shot ∥ 9× hparam ∥ feasibility → single_shot_best →"
echo "   iterative uniform ∥ poisson → stability → ablation ∥ canary → report)"
DATASET=balanced bash scripts/hpc_full_pipeline.sh

echo ""
echo "═══ SUBMIT: IMBALANCED PIPELINE ═══"
echo "  (train → single_shot ∥ 9× hparam ∥ feasibility → single_shot_best →"
echo "   equity ∥ per-bin oracles ∥ Protocol C → budget sweep → report)"
DATASET=imbalanced bash scripts/hpc_full_pipeline.sh

echo ""
echo "═══ MONITOR ═══"
echo "  watch:      squeue -u \$USER"
echo "  on failure: sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed"
echo ""
echo "═══ POST-RUN VERIFICATION (before trusting the numbers) ═══"
echo "  1. HP logs now show per-trial lines (d30b707):"
echo "     grep -c 'UF=' logs/unlearn_hp_*.out            # > 0 per job"
echo "  2. Feasibility verdicts all 9 methods, no ERROR:"
echo "     grep -E '→' logs/unlearn_feas_*.out            # srl/budget_scaled not ERROR"
echo "  3. single_shot_best retrain = 5-seed μ±σ (both datasets):"
echo "     grep 'Retrain Oracle' logs/unlearn_single_*.out | grep '±'"
echo "  4. Reports exist for both + feasibility matrix rendered:"
echo "     ls results/balanced/report/report.md results/imbalanced/report/report.md"
echo "     grep 'Feasibility Gate' results/*/report/report.md"
echo "  5. Stability plots = 16 per lane (balanced):"
echo "     ls results/balanced/iterative/plots/*.png | wc -l   # 16"
echo "     ls results/balanced/iterative_poisson/plots/*.png | wc -l  # 16"
