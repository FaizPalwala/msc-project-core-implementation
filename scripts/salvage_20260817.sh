#!/bin/bash
# ============================================================================
# SALVAGE SEQUENCE — Aire run of 15-17 Aug 2026 (jobs 7137139–7137178)
# ============================================================================
# Run from: /mnt/scratch/kxvs0578/core/code   (repo root, AFTER git pull)
#
# What failed and why:
#   1. 7137171 (imbalanced single_shot_best) — HOME-DIR I/O ERROR mid-run:
#      OSError [Errno 5] '/users/kxvs0578' (torch.hub cache makedirs).
#      Retrain oracle aggregated over 4 seeds silently (bootstrap bias).
#   2. Feasibility gates (7137150/7137170) — srl/budget_scaled CUDA
#      device-side assert (identity_classes not injected → relabel 0..599
#      vs 12-class head).  FIXED in commit 775dfd7.
#   3. 7137173-7137176 + 7137178 (imbalanced plots/oracles/select/sweep/
#      report) — never ran: afterok:7137171, cancelled when it failed.
#
# Balanced chain 7137139-7137158: ALL COMPLETED — do not touch.
# ============================================================================
set -euo pipefail

CSV=../bench/metadata/dataset_imbalanced.csv
MODEL=results/imbalanced/checkpoints/original_model_best.pt

echo "── 0. CHECK Aire home dir health (this killed 7137171) ──"
ls -la /users/kxvs0578 >/dev/null && echo "  home dir OK" || { echo "  home dir UNREACHABLE — stop, contact support"; exit 1; }
df -h /users/kxvs0578 | tail -1

echo ""
echo "── 1. RE-RUN imbalanced single_shot_best (clean 5-seed retrain) ──"
# ~1.5-2h GPU.  Same as pipeline Stage 2b.  Overwrites the 4-seed
# aggregated JSON with a correct 5-seed one.
SBEST=$(sbatch --parsable \
    scripts/slurm_single_shot.sh "$CSV" "$MODEL" \
    results/imbalanced/single_shot_best results/imbalanced/hparam)
echo "  single_shot_best job: $SBEST"

echo ""
echo "── 2. RE-RUN BOTH FEASIBILITY GATES (fixed verdicts for srl/budget_scaled) ──"
# ~4h each, parallel, no deps.  Needs the 775dfd7 fix (git pull first!).
FEAS_B=$(sbatch --parsable scripts/slurm_feasibility.sh balanced)
FEAS_I=$(sbatch --parsable scripts/slurm_feasibility.sh imbalanced)
echo "  feas balanced: $FEAS_B | feas imbalanced: $FEAS_I"

echo ""
echo "── 3. RE-RUN IMBALANCED DOWNSTREAM CHAIN (after SBEST) ──"
# Equity plots — after single_shot_best
IMBPLOT=$(sbatch --parsable \
    --dependency=afterok:$SBEST \
    scripts/slurm_imbalanced_plots.sh \
    results/imbalanced/single_shot_best/single_shot_aggregated.json \
    "$CSV" results/imbalanced/equity)
echo "  equity plots: $IMBPLOT"

# Protocol B — per-bin oracles (after single_shot_best)
ORACLE=$(sbatch --parsable \
    --dependency=afterok:$SBEST \
    scripts/slurm_per_bin_oracle.sh "$CSV" "$MODEL" results/imbalanced/oracles)
echo "  per-bin oracles: $ORACLE"

# Protocol C selection (after single_shot_best)
SELC=$(sbatch --parsable \
    --dependency=afterok:$SBEST \
    scripts/slurm_select_protocol_c.sh \
    results/imbalanced/single_shot_best/single_shot_aggregated.json \
    results/imbalanced)
echo "  protocol C select: $SELC"

# Budget sweep (after oracles + selection)
SWEEP=$(sbatch --parsable \
    --dependency=afterok:$ORACLE:$SELC \
    scripts/slurm_budget_sweep.sh "$CSV" "$MODEL" \
    results/imbalanced/protocol_c_methods.json \
    results/imbalanced/oracles results/imbalanced)
echo "  budget sweep: $SWEEP"

echo ""
echo "── 4. RE-RUN IMBALANCED REPORT (after all lanes + feas) ──"
REPORT=$(sbatch --parsable \
    --dependency=afterok:$SBEST:$IMBPLOT:$ORACLE:$SELC:$SWEEP:$FEAS_I \
    scripts/slurm_report.sh results/imbalanced)
echo "  imbalanced report: $REPORT"

echo ""
echo "── 5. REGENERATE BALANCED REPORT (picks up fixed feasibility verdicts) ──"
BREPORT=$(sbatch --parsable \
    --dependency=afterok:$FEAS_B \
    scripts/slurm_report.sh results/balanced)
echo "  balanced report: $BREPORT"

echo ""
echo "── MONITOR ──"
echo "  watch:  squeue -u \$USER"
echo "  after completion, verify:"
echo "    grep -E 'srl|budget_scaled' logs/unlearn_feas_*.out        # → GO/TUNE, not ERROR"
echo "    grep 'Retrain Oracle' logs/unlearn_single_*7171*.out       # → 5-seed row"
echo "    ls results/imbalanced/report/report.md                     # → exists"
