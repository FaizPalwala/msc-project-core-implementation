#!/bin/bash
# ==========================================
# hpc_full_pipeline.sh — Full unlearning pipeline on Aire HPC
# ==========================================
# ONE submission runs the whole suite at the designed parallelism:
#
#   train ──→ single_shot ∥ hparam×8 ──→ single_shot_best ──┐
#                     │                                       ├──→ report
#                     ├─→ iterative(uniform) → stability     ┘
#                     ├─→ iterative(poisson) → stability        (balanced lanes)
#                     └─→ canary (independent)
#
#   imbalanced replaces the schedule lanes with:
#       single_shot_best → per-bin oracles (B) ∥ Protocol C select
#                        → budget sweep (C) → report
#
# The two schedule lanes (uniform + poisson) run IN PARALLEL after the
# shared upstream stages; they write separate dirs (iterative/ vs
# iterative_poisson/) and never clobber each other.  HP search is one
# job per method, all parallel, each writing its own {method}_best_config.json.
#
# Usage:
#   bash scripts/hpc_full_pipeline.sh                    # balanced, both schedules
#   DATASET=imbalanced bash scripts/hpc_full_pipeline.sh
#   SCHEDULES=uniform bash scripts/hpc_full_pipeline.sh  # single-schedule override
#   HP_METHODS="ng_plus ft" bash scripts/hpc_full_pipeline.sh
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
# Schedule lanes to run (balanced only).  Default: BOTH in parallel.
# Override with SCHEDULES="uniform" or SCHEDULES="poisson".
SCHEDULES="${SCHEDULES:-uniform poisson}"
echo "[$(date)] Pipeline — dataset=$DATASET ($CSV)"
echo "  Output → $OUT"
echo "  Schedule lanes (balanced): $SCHEDULES"

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

# ── HP search: ONE job per method, ALL in parallel ──────────────────────
# Each job writes its own {method}_best_config.json (no shared-file races).
# HP_METHODS overrides the default set (methods with grids defined in
# hparam_search.GRIDS; no_unlearning/retrain have no tunable params).
HP_METHODS="${HP_METHODS:-ng_plus msg ct msg_kd adaptiforget ga srl ft budget_scaled}"
HP_DEPS=""
echo "[$(date)] Submitting HP search — one job per method, in parallel…"
for M in $HP_METHODS; do
    HPJ=$(sbatch --parsable \
        --dependency=afterok:$TRAIN_JOB \
        "$PROJECT_DIR/scripts/slurm_hparam.sh" \
        "$CSV" "$MODEL" "$OUT/hparam" "$M" "grid")
    echo "  HP($M) job: $HPJ"
    HP_DEPS="$HP_DEPS:$HPJ"
done
HP_DEPS="${HP_DEPS#:}"

# ── Stage 2c: C2/C3 feasibility gate (multi-scale pre-flight, parallel) ─
# Trains its own small subsampled model (12-id by default; 100-id via the
# second arg — see slurm_feasibility.sh) and sweeps each method's budget
# at 1x/3x/10x → verdicts (GO/TUNE/BROKEN).  No dependency on the train
# job (it subsamples from the dataset CSV directly), so it runs alongside
# single-shot + HP.  Results → results/feasibility_<dataset>_<scale>/.
echo "[$(date)] Submitting feasibility gate…"
FEAS_JOB=$(sbatch --parsable \
    "$PROJECT_DIR/scripts/slurm_feasibility.sh" "$DATASET" "12id")
echo "  Feasibility gate job: $FEAS_JOB"

# ── Stage 2b: Single-shot with BEST configs (after all HP jobs) ─────────
# The user's compare-and-contrast: default-config single-shot (above) vs
# tuned single-shot.  Both feed the report; iterative uses the tuned set.
BEST_DIR="$OUT/hparam"
echo "[$(date)] Submitting single-shot-best (dependency: all HP jobs)…"
SINGLE_BEST_JOB=$(sbatch --parsable \
    --dependency=afterok:$HP_DEPS \
    "$PROJECT_DIR/scripts/slurm_single_shot.sh" \
    "$CSV" "$MODEL" "$OUT/single_shot_best" "$BEST_DIR")
echo "  Single-shot-best job: $SINGLE_BEST_JOB"

# ── Stage 2a: AdaptiForget ablation — uses the HP-tuned best config so it
#    waits on all HP jobs (the "Full" variant is the method as deployed).
#    Must come after BEST_DIR/HP_DEPS are defined above.
echo "[$(date)] Submitting ablation study (dependency: $HP_DEPS)…"
ABLATION_JOB=$(sbatch --parsable \
    --dependency=afterok:$HP_DEPS \
    "$PROJECT_DIR/scripts/slurm_ablation.sh" \
    "$CSV" "$OUT/ablation" "$BEST_DIR")
echo "  Ablation job: $ABLATION_JOB"

# ── Imbalanced equity plots (only when DATASET=imbalanced) ─────────────
# Uses the TUNED single-shot results (best configs) — the equity story is
# the headline, not the default-config baseline.  Depends on the best
# single-shot (which itself waits on all HP jobs), so the plots reflect
# tuned behaviour.  (Must come AFTER SINGLE_BEST_JOB is defined above.)
if [ "$DATASET" = "imbalanced" ]; then
    echo "[$(date)] Submitting imbalanced plots (dependency: $SINGLE_BEST_JOB)…"
    IMBPLOT_JOB=$(sbatch --parsable \
        --dependency=afterok:$SINGLE_BEST_JOB \
        "$PROJECT_DIR/scripts/slurm_imbalanced_plots.sh" \
        "$OUT/single_shot_best/single_shot_aggregated.json" \
        "$CSV" "$OUT/imbalanced")
    echo "  Imbalanced plots job: $IMBPLOT_JOB"
fi

# ── Stage 3: Iterative + stability — one PARALLEL LANE per schedule ─────
# Balanced only — imbalanced has no forget_step schedule to iterate over
# (its experimental axis is the popularity gradient, not time).
# Uniform and Poisson are independent experiments: each lane depends on
# the shared upstream (single-shot + all HP jobs, so best configs are
# guaranteed by afterok) and writes its own dir.  The report waits on
# ALL lanes.
STAB_DEPS=""
if [ "$DATASET" = "balanced" ]; then
    for SCHEDULE in $SCHEDULES; do
        ITER_SUBDIR="iterative"
        [ "$SCHEDULE" = "poisson" ] && ITER_SUBDIR="iterative_poisson"
        echo "[$(date)] Submitting iterative($SCHEDULE; dependency: $SINGLE_JOB:$HP_DEPS)…"
        ITER_JOB=$(sbatch --parsable \
            --dependency=afterok:$SINGLE_JOB:$HP_DEPS \
            "$PROJECT_DIR/scripts/slurm_iterative.sh" \
            "$CSV" "$MODEL" "$OUT/$ITER_SUBDIR" "$SCHEDULE" "$BEST_DIR")
        echo "  Iterative($SCHEDULE) job: $ITER_JOB"

        echo "[$(date)] Submitting stability($SCHEDULE; dependency: $ITER_JOB:$SINGLE_BEST_JOB)…"
        STAB_JOB=$(sbatch --parsable \
            --dependency=afterok:$ITER_JOB:$SINGLE_BEST_JOB \
            "$PROJECT_DIR/scripts/slurm_stability.sh" \
            "$OUT/$ITER_SUBDIR/iterative_combined_aggregated.csv" \
            "$OUT/$ITER_SUBDIR/plots" \
            "$OUT/single_shot_best/single_shot_per_identity.csv" \
            "$OUT/single_shot_best/single_shot_demographic.csv")
        echo "  Stability($SCHEDULE) job: $STAB_JOB"
        STAB_DEPS="$STAB_DEPS:$STAB_JOB"
    done
    STAB_DEPS="${STAB_DEPS#:}"
else
    echo "[$(date)] Skipping iterative + stability (imbalanced has no schedule axis)"

    # ── Imbalanced stress-test chain (Protocols B + C) ─────────────────
    # B: per-bin retrain oracles — one oracle PER bin (the oracle changes
    #    per bin: bin-B's oracle still trains on the other bins' forgets).
    # C: dynamic selection — 1 best method per category (baseline/SOTA/
    #    novel) by UF score from single_shot_best; the 3 picks feed the
    #    budget sweep.
    echo "[$(date)] Submitting per-bin oracles (Protocol B; dependency: $SINGLE_BEST_JOB)…"
    ORACLE_JOB=$(sbatch --parsable \
        --dependency=afterok:$SINGLE_BEST_JOB \
        "$PROJECT_DIR/scripts/slurm_per_bin_oracle.sh" \
        "$CSV" "$MODEL" "$OUT/oracles")
    echo "  Per-bin oracle job: $ORACLE_JOB"

    echo "[$(date)] Submitting Protocol C selection (dependency: $SINGLE_BEST_JOB)…"
    SELC_JOB=$(sbatch --parsable \
        --dependency=afterok:$SINGLE_BEST_JOB \
        "$PROJECT_DIR/scripts/slurm_select_protocol_c.sh" \
        "$OUT/single_shot_best/single_shot_aggregated.json" \
        "$OUT")
    echo "  Protocol C selection job: $SELC_JOB"

    # C (sweep): budget sweep on the dynamically-picked methods, using the
    # per-bin oracles as the distance reference.  Needs BOTH B and C-selection.
    echo "[$(date)] Submitting budget sweep (dependency: $ORACLE_JOB:$SELC_JOB)…"
    SWEEP_JOB=$(sbatch --parsable \
        --dependency=afterok:$ORACLE_JOB:$SELC_JOB \
        "$PROJECT_DIR/scripts/slurm_budget_sweep.sh" \
        "$CSV" "$MODEL" "$OUT/protocol_c_methods.json" \
        "$OUT/oracles" "$OUT" "$OUT/hparam")
    echo "  Budget sweep job: $SWEEP_JOB"
fi

# ── Stage 5: Canary experiment (independent train — but unlearning uses
#    the HP-tuned best configs, so it waits on all HP jobs) ──────────────
echo "[$(date)] Submitting canary experiment (dependency: $HP_DEPS)…"
CANARY_JOB=$(sbatch --parsable \
    --dependency=afterok:$HP_DEPS \
    "$PROJECT_DIR/scripts/slurm_canary.sh" \
    "$CSV" "$OUT/canary" "$BEST_DIR")
echo "  Canary job: $CANARY_JOB"

# ── Stage 6: Report (after ALL lanes + single_shot + single_shot_best +
#    canary + ablation; SINGLE_JOB included so the default-config evidence
#    table is never dropped — on imbalanced no stage else depends on it)
REPORT_DEPS="$CANARY_JOB:$SINGLE_BEST_JOB:$SINGLE_JOB"
[ -n "${ABLATION_JOB:-}" ] && REPORT_DEPS="$ABLATION_JOB:$REPORT_DEPS"
[ -n "${IMBPLOT_JOB:-}" ] && REPORT_DEPS="$IMBPLOT_JOB:$REPORT_DEPS"
[ -n "${FEAS_JOB:-}" ] && REPORT_DEPS="$FEAS_JOB:$REPORT_DEPS"
[ -n "$STAB_DEPS" ] && REPORT_DEPS="$STAB_DEPS:$REPORT_DEPS"
[ -n "${ORACLE_JOB:-}" ] && REPORT_DEPS="$ORACLE_JOB:$REPORT_DEPS"
[ -n "${SELC_JOB:-}" ] && REPORT_DEPS="$SELC_JOB:$REPORT_DEPS"
[ -n "${SWEEP_JOB:-}" ] && REPORT_DEPS="$SWEEP_JOB:$REPORT_DEPS"
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
echo "  Job IDs: train=$TRAIN_JOB single=$SINGLE_JOB single_best=$SINGLE_BEST_JOB hp=[$HP_METHODS] stab=[$STAB_DEPS] oracle=${ORACLE_JOB:-—} selc=${SELC_JOB:-—} sweep=${SWEEP_JOB:-—} canary=$CANARY_JOB report=$REPORT_JOB"
