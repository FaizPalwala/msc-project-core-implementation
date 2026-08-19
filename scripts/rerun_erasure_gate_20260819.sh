#!/bin/bash
# ============================================================================
# PARTIAL RE-RUN — erasure-gate correction (after commit f8ed8c4)
# ============================================================================
# Why: the final run's hparam exported suppression configs as 'best'
# (AdaptiForget forget 0.82 with MIA 0.026 — confidence collapse, not
# erasure).  The erasure gate now rejects forget_acc>0.15 / probe>0.30.
#
# Re-run scope (verified against final-run trial JSONLs):
#   RE-TUNE (5 methods, both datasets): adaptiforget budget_scaled ga msg ng_plus
#     - ng_plus has 0/108 passing trials → gate falls back to YAML default
#       (it genuinely cannot erase at 750-id; honest outcome)
#   KEEP configs (4): ct ft srl msg_kd  (their best configs already pass)
#   RE-USE as-is: train checkpoints, default single_shot, feasibility gates
#     (they run on YAML defaults — unaffected by the gate)
#
# Run from: /mnt/scratch/kxvs0578/core/code   (AFTER git pull)
# ============================================================================
set -euo pipefail

CSVB=../bench/metadata/dataset.csv
CSVI=../bench/metadata/dataset_imbalanced.csv
MODELB=results/balanced/checkpoints/original_model_best.pt
MODELI=results/imbalanced/checkpoints/original_model_best.pt
METHODS="adaptiforget budget_scaled ga msg ng_plus"

echo "── 0. Home-dir health (killed 7137171 last time) ──"
ls -la /users/kxvs0578 >/dev/null && echo "  home OK" || { echo "  home UNREACHABLE — stop"; exit 1; }

echo ""
echo "── 1. RE-TUNE hparam ×5 (balanced + imbalanced, all parallel) ──"
HP_B=""; HP_I=""
for M in $METHODS; do
    J=$(sbatch --parsable scripts/slurm_hparam.sh "$CSVB" "$MODELB" results/balanced/hparam "$M" grid)
    HP_B="$HP_B:$J"
    J=$(sbatch --parsable scripts/slurm_hparam.sh "$CSVI" "$MODELI" results/imbalanced/hparam "$M" grid)
    HP_I="$HP_I:$J"
done
HP_B="${HP_B#:}"; HP_I="${HP_I#:}"
echo "  balanced HP deps: $HP_B"
echo "  imbalanced HP deps: $HP_I"

echo ""
echo "── 2. single_shot_best (both, after their HP) ──"
SSB=$(sbatch --parsable --dependency=afterok:$HP_B \
    scripts/slurm_single_shot.sh "$CSVB" "$MODELB" \
    results/balanced/single_shot_best results/balanced/hparam)
SSI=$(sbatch --parsable --dependency=afterok:$HP_I \
    scripts/slurm_single_shot.sh "$CSVI" "$MODELI" \
    results/imbalanced/single_shot_best results/imbalanced/hparam)
echo "  SSB: $SSB | SSI: $SSI"

echo ""
echo "── 3. Iterative (balanced, both lanes) + stability ──"
ITU=$(sbatch --parsable --dependency=afterok:$HP_B \
    scripts/slurm_iterative.sh "$CSVB" "$MODELB" \
    results/balanced/iterative uniform results/balanced/hparam)
ITP=$(sbatch --parsable --dependency=afterok:$HP_B \
    scripts/slurm_iterative.sh "$CSVB" "$MODELB" \
    results/balanced/iterative_poisson poisson results/balanced/hparam)
STU=$(sbatch --parsable --dependency=afterok:$ITU:$SSB \
    scripts/slurm_stability.sh \
    results/balanced/iterative/iterative_combined_aggregated.csv \
    results/balanced/iterative/plots \
    results/balanced/single_shot_best/single_shot_per_identity.csv \
    results/balanced/single_shot_best/single_shot_demographic.csv)
STP=$(sbatch --parsable --dependency=afterok:$ITP:$SSB \
    scripts/slurm_stability.sh \
    results/balanced/iterative_poisson/iterative_combined_aggregated.csv \
    results/balanced/iterative_poisson/plots \
    results/balanced/single_shot_best/single_shot_per_identity.csv \
    results/balanced/single_shot_best/single_shot_demographic.csv)
echo "  ITU: $ITU | ITP: $ITP | STU: $STU | STP: $STP"

echo ""
echo "── 4. Canary + ablation (both, after HP — tuned unlearning/base) ──"
CANB=$(sbatch --parsable --dependency=afterok:$HP_B \
    scripts/slurm_canary.sh "$CSVB" results/balanced/canary results/balanced/hparam)
CANI=$(sbatch --parsable --dependency=afterok:$HP_I \
    scripts/slurm_canary.sh "$CSVI" results/imbalanced/canary results/imbalanced/hparam)
ABLB=$(sbatch --parsable --dependency=afterok:$HP_B \
    scripts/slurm_ablation.sh "$CSVB" results/balanced/ablation results/balanced/hparam)
ABLI=$(sbatch --parsable --dependency=afterok:$HP_I \
    scripts/slurm_ablation.sh "$CSVI" results/imbalanced/ablation results/imbalanced/hparam)
echo "  CANB: $CANB | CANI: $CANI | ABLB: $ABLB | ABLI: $ABLI"

echo ""
echo "── 5. Imbalanced chain (equity → oracles ∥ select → sweep) ──"
IMB=$(sbatch --parsable --dependency=afterok:$SSI \
    scripts/slurm_imbalanced_plots.sh \
    results/imbalanced/single_shot_best/single_shot_aggregated.json \
    "$CSVI" results/imbalanced/equity)
ORB=$(sbatch --parsable --dependency=afterok:$SSI \
    scripts/slurm_per_bin_oracle.sh "$CSVI" "$MODELI" results/imbalanced/oracles)
SEL=$(sbatch --parsable --dependency=afterok:$SSI \
    scripts/slurm_select_protocol_c.sh \
    results/imbalanced/single_shot_best/single_shot_aggregated.json \
    results/imbalanced)
SWP=$(sbatch --parsable --dependency=afterok:$ORB:$SEL \
    scripts/slurm_budget_sweep.sh "$CSVI" "$MODELI" \
    results/imbalanced/protocol_c_methods.json \
    results/imbalanced/oracles results/imbalanced)
echo "  IMB: $IMB | ORB: $ORB | SEL: $SEL | SWP: $SWP"

echo ""
echo "── 6. Reports (both, after ALL their lanes) ──"
REPB=$(sbatch --parsable \
    --dependency=afterok:$SSB:$ITU:$ITP:$STU:$STP:$CANB:$ABLB \
    scripts/slurm_report.sh results/balanced)
REPI=$(sbatch --parsable \
    --dependency=afterok:$SSI:$IMB:$ORB:$SEL:$SWP:$CANI:$ABLI \
    scripts/slurm_report.sh results/imbalanced)
echo "  REPB: $REPB | REPI: $REPI"

echo ""
echo "── 7. PACKAGE (after both reports) ──"
PKG=$(sbatch --parsable --dependency=afterok:$REPB:$REPI \
    scripts/slurm_package.sh balanced)
echo "  PKG: $PKG   (pkg TimeLimit 1:30 — user-confirmed, do not change)"
echo ""
echo "── MONITOR ──"
echo "  squeue -u \$USER"
echo "  After: check hparam logs for '[GATE]' lines and 'NO config passed' fallbacks:"
echo "    grep -E 'GATE|NO config passed' logs/unlearn_hp_*.out"
echo "  Verify AdaptiForget tuned now erases:"
echo "    grep 'AdaptiForget' results/balanced/single_shot_best/single_shot_aggregated.csv"
