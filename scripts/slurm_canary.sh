#!/bin/bash
# ==========================================
# slurm_canary.sh — Tier-4 canary ground-truth experiment
# ==========================================
#SBATCH --job-name=unlearn_canary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail   # any python failure → job FAILED (never phantom COMPLETED)
# Usage: sbatch scripts/slurm_canary.sh <csv_path> <out_dir> [identity_ids...]
#
# One-off experiment: insert pixel canaries into 4 identities' images,
# train on the canary-tagged dataset, unlearn those identities, then
# verify the canary is gone from the unlearned model's features.
#
# Default identities: first 4 forget identities, derived from the CSV at
# submit time (forget_step=0 on balanced; first 4 forget ids on imbalanced).
# The canary dataset is written to <out_dir>/dataset_canary.csv; the
# canary model + unlearned models go under <out_dir>/canary/.

# ── Repo root ────────────────────────────────────────────────────────────
# Under sbatch, $0 points at the spool copy (/var/spool/slurmd/...), so
# resolve the repo from SLURM_SUBMIT_DIR — the dir sbatch was invoked
# from. Fall back to $0 for interactive runs. Climb one level if the
# script was submitted from scripts/ itself. Guarded by the configs/
# marker (present only in the repo root).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi
[ -d "$PROJECT_DIR/configs" ] || PROJECT_DIR="$(cd "$PROJECT_DIR/.." 2>/dev/null && pwd)"
[ -d "$PROJECT_DIR/configs" ] || { echo "ERROR: repo root not found (no configs/ in $PROJECT_DIR)" >&2; exit 1; }
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"   # canonicalize (no "..")

# ── Environment ──────────────────────────────────────────────────────────
module purge
module load miniforge
module load cuda/12.6.2
conda activate core

DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
source "$PROJECT_DIR/scripts/slurm_dataset_helper.sh"  # sets DATASET, CSV_DEFAULT, OUT_BASE
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
CSV="${1:-$CSV_DEFAULT}"
OUT="${2:-$OUT_BASE/canary}"
shift 2
IDENTITIES=("$@")
if [ ${#IDENTITIES[@]} -eq 0 ]; then
    # Default: first 4 forget identities (forget_step=0 on balanced,
    # first 4 forget identities on imbalanced — no schedule column).
    # Derived from the CSV, never hardcoded (identity ids change with
    # the dataset; the old '0 1 2 3' default were retain ids at 750-id).
    IDENTITIES=($(python -c "
import pandas as pd, sys
df = pd.read_csv('$CSV')
fg = df[df['split'] == 'forget']
if 'forget_step' in fg.columns:
    s0 = fg[fg['forget_step'] == fg['forget_step'].min()]['identity_id'].unique()
else:
    s0 = fg['identity_id'].unique()
print(' '.join(map(str, sorted(s0)[:4])))
"))
fi
echo "[$(date)] Canary identities: ${IDENTITIES[*]}"

CANARY_CSV="$OUT/dataset_canary.csv"
CKPT_DIR="$OUT/checkpoints"
UNLEARN_DIR="$OUT/unlearned"

mkdir -p "$OUT" "$CKPT_DIR" "$UNLEARN_DIR"

cd "$PROJECT_DIR"
echo "[$(date)] Canary experiment on $CSV → $OUT"

# ── Stage 1: Insert canaries (tags has_canary; pixel insertion is
#    documented as a placeholder — see canary.py CLI) ────────────────────
echo "[$(date)] Inserting canaries…"
python src/canary.py insert \
    --csv "$CSV" \
    --identities "${IDENTITIES[@]}" \
    --out "$CANARY_CSV" 2>&1

# ── Stage 2: Train on canary-tagged dataset ─────────────────────────────
echo "[$(date)] Training on canary dataset…"
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1
python src/train.py \
    --csv "$CANARY_CSV" \
    --save_dir "$CKPT_DIR" \
    --epochs 30 --lr 1e-3 --batch_size 64 \
    --age_weight 0.5 --seed 42 2>&1

CANARY_MODEL="$CKPT_DIR/original_model_best.pt"

# ── Stage 3: Unlearn the canary identities (single-shot, GA + AdaptiForget) ─
echo "[$(date)] Unlearning canary identities…"
python src/single_shot.py \
    --csv "$CANARY_CSV" \
    --model "$CANARY_MODEL" \
    --out "$UNLEARN_DIR" \
    --methods ga adaptiforget \
    --skip_retrain 2>&1

# ── Stage 4: Verify canary removal on the unlearned models ──────────────
# single_shot (multi-seed) writes seed_<seed>/<method>_unlearned.pt —
# use the default seed_42 lane for verification.
echo "[$(date)] Verifying canary unlearning…"
for method in ga adaptiforget; do
    MODEL="$UNLEARN_DIR/seed_42/${method}_unlearned.pt"
    if [ -f "$MODEL" ]; then
        echo "  [verify] $method"
        # Pure JSON to the file (stdout); logs stay on the job stream.
        # The old `2>&1 | tee` merged logger lines into the JSON and
        # crashed report.py's json.load (JSONDecodeError, both report
        # jobs 7102724/7102745).  Filter timestamped log lines BEFORE
        # tee so the file receives only the JSON payload.
        python src/canary.py verify \
            --csv "$CANARY_CSV" \
            --src_csv "$CSV" \
            --model "$MODEL" \
            --identities "${IDENTITIES[@]}" \
            2>&1 | grep -vE "^[0-9]{4}-[0-9]{2}-[0-9]{2} " \
            | tee "$OUT/verify_${method}.json"
    else
        echo "  [WARN] $MODEL not found — skipping verify for $method"
    fi
done

echo "[$(date)] Canary experiment complete."
echo "  Canary dataset: $CANARY_CSV"
echo "  Results: $OUT/verify_*.json"
