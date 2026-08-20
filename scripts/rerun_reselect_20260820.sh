#!/bin/bash
# ============================================================================
# RECOVERY RE-RUN — offline best-config re-selection + downstream (after f2)
# ============================================================================
# Why: f2's hparam grids COMPLETED but the gate's (now-retired) probe arm
# rejected 100% of trials → all 5 re-tuned methods lost their best configs
# and ran at YAML defaults downstream (msg retain 0.0101 collapse in f2).
#
# Fix (committed): probe arm removed (probe-identity saturates at ~1.0 even
# for the retrain oracle at full scale — measures backbone separability,
# not erasure).  Trials themselves are complete and valid, so:
#
#   STEP 1 (CPU, ~seconds): offline re-select best configs from the f2
#     trial JSONLs — gate = forget_acc <= 0.15 (pre-registered), rank by
#     v2 erasure-conditioned UF, require beating the YAML default's v2 UF.
#     → src/reselect_best_configs.py results/{balanced,imbalanced}/hparam
#   STEP 2+: re-run ONLY the stages that consume tuned configs
#     (single_shot_best + iterative + canary + ablation + imbalanced chain
#     + reports + package).  NO hparam re-run, NO train, NO feasibility,
#     NO default single_shot — all re-used from f2.
#
# Corrected selections (verified against f2 trial data):
#   adaptiforget  tuned forget 0.0267 (UF 0.857 > default 0.687)
#   msg           tuned forget 0.0126 (UF 0.864 > default -0.027)
#   budget_scaled tuned forget 0.0422 (UF 0.180 > default 0.067) [retain 0.33]
#   ga            DEFAULT KEPT (tuned -0.0014 < default 0.4725 — cannot be
#                 tuned at full scale; feasibility TUNE(retain-collapse) ✓)
#   ng_plus       DEFAULT (0/108 trials erase at 750-id)
#   ct/ft/srl/msg_kd untouched (gate-valid from the final run)
#
# Run from: /mnt/scratch/kxvs0578/core/code   (AFTER git pull)
# ============================================================================
set -euo pipefail

CSVB=../bench/metadata/dataset.csv
CSVI=../bench/metadata/dataset_imbalanced.csv
MODELB=results/balanced/checkpoints/original_model_best.pt
MODELI=results/imbalanced/checkpoints/original_model_best.pt

echo "── 0. Home-dir health (killed 7137171 before) ──"
ls -la /users/kxvs0578 >/dev/null && echo "  home OK" || { echo "  home UNREACHABLE — stop"; exit 1; }

echo ""
echo "── 1. OFFLINE re-selection (CPU, seconds) — corrected gate + v2 UF ──"
python3 src/reselect_best_configs.py results/balanced/hparam
python3 src/reselect_best_configs.py results/imbalanced/hparam
echo "  → adaptiforget/msg/budget_scaled best configs restored; ga/ng_plus → default"

echo ""
echo "── 1b. MULTI-SCALE FEASIBILITY GATE (12id + 100id, both datasets) ──"
# Canonical gate (2026-08): same budget sweep at 12-id and 100-id, with the
# 750-id column coming from the hparam grids.  100-id = 90 retain + 10 forget
# (~13% of full, keeps the 9:1 ratio) — large enough that head-width
# mechanisms bite (ng_plus scale-cliff, FT reverse-artefact).  Runs in
# parallel with single_shot_best; feeds the report's trajectory table.
FEAS_B12=$(sbatch --parsable scripts/slurm_feasibility.sh balanced 12id)
FEAS_B100=$(sbatch --parsable scripts/slurm_feasibility.sh balanced 100id)
FEAS_I12=$(sbatch --parsable scripts/slurm_feasibility.sh imbalanced 12id)
FEAS_I100=$(sbatch --parsable scripts/slurm_feasibility.sh imbalanced 100id)
echo "  FEAS_B12: $FEAS_B12 | FEAS_B100: $FEAS_B100 | FEAS_I12: $FEAS_I12 | FEAS_I100: $FEAS_I100"

echo ""
echo "── 2. single_shot_best (both datasets, parallel) ──"
SSB=$(sbatch --parsable \
    scripts/slurm_single_shot.sh "$CSVB" "$MODELB" \
    results/balanced/single_shot_best results/balanced/hparam)
SSI=$(sbatch --parsable \
    scripts/slurm_single_shot.sh "$CSVI" "$MODELI" \
    results/imbalanced/single_shot_best results/imbalanced/hparam)
echo "  SSB: $SSB | SSI: $SSI"

echo ""
echo "── 3. Iterative (balanced, both lanes) + stability ──"
ITU=$(sbatch --parsable --dependency=afterok:$SSB \
    scripts/slurm_iterative.sh "$CSVB" "$MODELB" \
    results/balanced/iterative uniform results/balanced/hparam)
ITP=$(sbatch --parsable --dependency=afterok:$SSB \
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
echo "── 4. Canary + ablation (both, tuned unlearning/base) ──"
CANB=$(sbatch --parsable --dependency=afterok:$SSB \
    scripts/slurm_canary.sh "$CSVB" results/balanced/canary results/balanced/hparam)
CANI=$(sbatch --parsable --dependency=afterok:$SSI \
    scripts/slurm_canary.sh "$CSVI" results/imbalanced/canary results/imbalanced/hparam)
ABLB=$(sbatch --parsable --dependency=afterok:$SSB \
    scripts/slurm_ablation.sh "$CSVB" results/balanced/ablation results/balanced/hparam)
ABLI=$(sbatch --parsable --dependency=afterok:$SSI \
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
echo "── 6. Reports (both, after ALL their lanes + feasibility) ──"
REPB=$(sbatch --parsable \
    --dependency=afterok:$SSB:$ITU:$ITP:$STU:$STP:$CANB:$ABLB:$FEAS_B12:$FEAS_B100 \
    scripts/slurm_report.sh results/balanced)
REPI=$(sbatch --parsable \
    --dependency=afterok:$SSI:$IMB:$ORB:$SEL:$SWP:$CANI:$ABLI:$FEAS_I12:$FEAS_I100 \
    scripts/slurm_report.sh results/imbalanced)
echo "  REPB: $REPB | REPI: $REPI"

echo ""
echo "── 7. PACKAGE (all — balanced + imbalanced + logs) ──"
PKG=$(sbatch --parsable --dependency=afterok:$REPB:$REPI \
    scripts/slurm_package.sh all)
echo "  PKG: $PKG   (→ results_all_<date>_<time>.zip at repo root)"

echo ""
echo "── MONITOR ──"
echo "  squeue -u \$USER"
echo "  After: verify adaptiforget tuned now erases:"
echo "    grep 'adaptiforget' results/balanced/single_shot_best/single_shot_aggregated.csv"
echo "  Expect: adaptiforget forget ≈ 0.0267 (NOT 0.0124 default, NOT 0.82 suppressor)"
