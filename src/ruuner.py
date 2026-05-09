"""
runner.py — Master runner for Phase 2 baselines and MIA.
            to extended with more methods in the future.

Orchestrates:
  Step 1: Train original model M on (retain + forget) splits
  Step 2: Run single-shot experiments on all baselines
  Step 3: Run MIA on the original model (sanity check)

Usage:
    # Full run (GPU recommended)
    python run_phase2.py --csv ../data/dataset/sfhq_dataset.csv \
                         --out ../results \
                         --epochs 30 --forget_step 0

    # Quick smoke-test (CPU, small dataset)
    python run_phase2.py --csv ../data/dataset/sfhq_dataset.csv \
                         --out ../results --epochs 5 \
                         --forget_step 0 --ga_steps 50 \
                         --srl_epochs 2 --ft_epochs 2 \
                         --skip_retrain_oracle
"""

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Master Runner")
    parser.add_argument("--csv",          type=str,   required=True)
    parser.add_argument("--out",          type=str,   default="../results")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--forget_step",  type=int,   default=0)
    parser.add_argument("--methods",      type=str,   nargs="+",
                        default=["no_unlearning", "retrain", "ga", "srl", "ft"])
    parser.add_argument("--ga_steps",     type=int,   default=300)
    parser.add_argument("--ga_lr",        type=float, default=1e-4)
    parser.add_argument("--srl_epochs",   type=int,   default=5)
    parser.add_argument("--srl_lr",       type=float, default=1e-4)
    parser.add_argument("--ft_epochs",    type=int,   default=5)
    parser.add_argument("--ft_lr",        type=float, default=1e-4)
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--skip_train",   action="store_true",
                        help="Skip Step 1 if checkpoint already exists")
    parser.add_argument("--skip_retrain_oracle", action="store_true",
                        help="Skip retrain oracle (saves time for smoke-test)")
    parser.add_argument("--model_path",   type=str,   default=None,
                        help="Path to existing model checkpoint (skips training)")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        if args.device == "auto" else torch.device(args.device)
    )
    print(f"\n{'='*60}")
    print("  PHASE 2: BASELINE UNLEARNING IMPLEMENTATION")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  CSV        : {args.csv}")
    print(f"  Output dir : {args.out}")
    print(f"  Forget step: {args.forget_step}")
    print(f"  Methods    : {args.methods}")
    print(f"{'='*60}\n")

    out_path = Path(args.out)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_path / "single_shot"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Train original model ──────────────────────────────────────────
    model_path = args.model_path
    if model_path is None:
        model_path = str(ckpt_dir / "original_model_best.pt")

    if args.skip_train and Path(model_path).exists():
        print(f"[SKIP] Training — loading existing model: {model_path}")
    else:
        print("\n[STEP 1] Training original model M on retain + forget data...")
        t0 = time.time()
        from train import train
        _, history = train(
            csv_path=args.csv,
            save_dir=str(ckpt_dir),
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            run_name="original_model",
        )
        print(f"[OK] Training complete in {(time.time()-t0)/60:.1f} min")

    # ── Step 2: Single-shot unlearning experiments ────────────────────────────
    print("\n[STEP 2] Running single-shot unlearning experiments...")
    methods = args.methods
    if args.skip_retrain_oracle and "retrain" in methods:
        methods = [m for m in methods if m != "retrain"]
        print("[INFO] Skipping retrain oracle (--skip_retrain_oracle set)")

    from single_shot_experiment import run_single_shot
    results = run_single_shot(
        csv_path=args.csv,
        model_path=model_path,
        forget_step=args.forget_step,
        methods=methods,
        out_dir=str(results_dir),
        device_str=args.device,
        seed=args.seed,
        ga_steps=args.ga_steps,
        ga_lr=args.ga_lr,
        srl_epochs=args.srl_epochs,
        srl_lr=args.srl_lr,
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        retrain_epochs=args.epochs,
    )

    # ── Step 3: MIA on original model (pre-unlearning baseline) ──────────────
    print("\n[STEP 3] Running MIA on original model (pre-unlearning baseline)...")
    from model import load_model
    from mia import run_mia_full
    original_model = load_model(model_path, device=str(device))
    mia_baseline = run_mia_full(
        original_model, args.csv, device,
        forget_step=args.forget_step,
        score_type="confidence",
        verbose=True,
    )

    # Save MIA baseline
    mia_path = out_path / "mia_baseline.json"
    with open(mia_path, "w") as f:
        json.dump({"original_model": mia_baseline}, f, indent=2)
    print(f"[OK] MIA baseline saved → {mia_path}")

    print(f"\n{'='*60}")
    print("  PHASE 2 COMPLETE")
    print(f"  Results: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
