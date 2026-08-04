#!/bin/bash
# ==========================================
# slurm_canary.sh — Tier-4 canary ground-truth experiment
# ==========================================
#SBATCH --job-name=unlearn_canary
# Usage: sbatch scripts/slurm_canary.sh <csv_path> <out_dir> [identity_ids...]
#
# One-off experiment: insert pixel canaries into 4 identities' images,
# train on the canary-tagged dataset, unlearn those identities, then
# verify the canary is gone from the unlearned model's features.
#
# Default identities: first 4 forget-step-0 identities (id 0-3 of step 0).
# The canary dataset is written to <out_dir>/dataset_canary.csv; the
# canary model + unlearned models go under <out_dir>/canary/.

module purge
module load miniforge
module load cuda/12.6.2
conda activate data_gen

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(dirname "$PROJECT_DIR")/bench"       # bench is a sister of the repo
CSV="${1:-$DATA_DIR/metadata/dataset.csv}"
OUT="${2:-$PROJECT_DIR/results/canary}"
shift 2
IDENTITIES=("$@")
if [ ${#IDENTITIES[@]} -eq 0 ]; then
    IDENTITIES=(0 1 2 3)   # first 4 identities in forget_step_0
fi

CANARY_CSV="$OUT/dataset_canary.csv"
CKPT_DIR="$OUT/checkpoints"
UNLEARN_DIR="$OUT/unlearned"

mkdir -p "$OUT" "$CKPT_DIR" "$UNLEARN_DIR"

cd "$PROJECT_DIR"
echo "[$(date)] Canary experiment on $CSV → $OUT"
echo "[$(date)] Canary identities: ${IDENTITIES[*]}"

# ── Stage 1: Insert canaries (tags has_canary; pixel insertion is
#    documented as a placeholder — see canary.py CLI) ────────────────────
echo "[$(date)] Inserting canaries…"
python src/canary.py insert \
    --csv "$CSV" \
    --identities "${IDENTITIES[@]}" \
    --out "$CANARY_CSV"

# ── Stage 2: Train on canary-tagged dataset ─────────────────────────────
echo "[$(date)] Training on canary dataset…"
bash "$PROJECT_DIR/scripts/gpu_preflight.sh" || exit 1
python src/train.py \
    --csv "$CANARY_CSV" \
    --save_dir "$CKPT_DIR" \
    --epochs 30 --lr 1e-3 --batch_size 64 \
    --age_weight 0.5 --seed 42

CANARY_MODEL="$CKPT_DIR/original_model_best.pt"

# ── Stage 3: Unlearn the canary identities (single-shot, GA + AdaptiForget) ─
echo "[$(date)] Unlearning canary identities…"
python src/single_shot.py \
    --csv "$CANARY_CSV" \
    --model "$CANARY_MODEL" \
    --out "$UNLEARN_DIR" \
    --methods ga adaptiformet \
    --skip_retrain

# ── Stage 4: Verify canary removal on the unlearned models ──────────────
echo "[$(date)] Verifying canary unlearning…"
for method in ga adaptiformet; do
    MODEL="$UNLEARN_DIR/${method}_unlearned.pt"
    if [ -f "$MODEL" ]; then
        echo "  [verify] $method"
        python src/canary.py verify \
            --csv "$CANARY_CSV" \
            --model "$MODEL" \
            --identities "${IDENTITIES[@]}" \
            | tee "$OUT/verify_${method}.json"
    else
        echo "  [WARN] $MODEL not found — skipping verify for $method"
    fi
done

echo "[$(date)] Canary experiment complete."
echo "  Canary dataset: $CANARY_CSV"
echo "  Results: $OUT/verify_*.json"
