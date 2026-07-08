"""
single_shot_experiment.py — Single-shot unlearning validation experiment.

Runs all baseline methods on a single forget set (one identity),
evaluates utility and forgetting quality, and saves results.

Usage:
    python single_shot_experiment.py \
        --csv   ../data/dataset/sfhq_dataset.csv \
        --model ../checkpoints/original_model_best.pt \
        --forget_step 0 \
        --out   ../results/single_shot \
        --methods no_unlearning retrain ga srl ft
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VirtualIdentityDataset, get_val_transform
from device_utils import resolve_device
from model import load_model
from baselines import BASELINE_REGISTRY
from evaluate import evaluate_full
from mia import run_mia_full


# ──────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_single_shot(
    csv_path: str,
    model_path: str,
    forget_step: int = 0,
    methods: list = None,
    out_dir: str = "../results/single_shot",
    device_str: str = "auto",
    seed: int = 42,
    # Baseline hyperparameters (defaults are conservative)
    ga_steps: int = 300,
    ga_lr: float = 1e-4,
    srl_epochs: int = 5,
    srl_lr: float = 1e-4,
    ft_epochs: int = 5,
    ft_lr: float = 1e-4,
    retrain_epochs: int = 30,
):
    torch.manual_seed(seed)
    device = resolve_device(device_str)
    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Forget step: {forget_step}")
    print(f"[INFO] Methods: {methods}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load original model
    original_model = load_model(model_path, device=str(device))

    if methods is None:
        methods = list(BASELINE_REGISTRY.keys())

    all_results = {}

    for method_name in methods:
        if method_name not in BASELINE_REGISTRY:
            print(f"[WARN] Unknown method '{method_name}', skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"  Method: {method_name.upper()}")
        print(f"{'='*55}")

        method_fn = BASELINE_REGISTRY[method_name]
        t0 = time.time()

        # ── Run unlearning ──
        result = method_fn(
            model=original_model,
            csv_path=csv_path,
            device=device,
            forget_step=forget_step,
            # GA args
            ga_steps=ga_steps,
            ga_lr=ga_lr,
            # SRL args
            srl_epochs=srl_epochs,
            srl_lr=srl_lr,
            # FT args
            ft_epochs=ft_epochs,
            ft_lr=ft_lr,
            # Retrain args
            epochs=retrain_epochs,
            seed=seed,
        )

        unlearned_model = result["model"]
        method_metrics  = result["metrics"]

        # ── Evaluate utility + forgetting ──
        print("\n[Evaluation]")
        eval_results = evaluate_full(
            unlearned_model, csv_path, device,
            forget_step=forget_step, verbose=True
        )

        # ── MIA (both confidence and loss-based) ──
        print("\n[MIA — Confidence]")
        mia_conf = run_mia_full(
            unlearned_model, csv_path, device,
            forget_step=forget_step, score_type="confidence", verbose=True
        )
        print("\n[MIA — Loss]")
        mia_loss = run_mia_full(
            unlearned_model, csv_path, device,
            forget_step=forget_step, score_type="loss", verbose=True
        )

        total_time = time.time() - t0
        all_results[method_name] = {
            "method": method_name,
            "method_metrics": method_metrics,
            "evaluation": eval_results,
            "mia_confidence": mia_conf,
            "mia_loss": mia_loss,
            "total_time_s": round(total_time, 2),
        }

        # Save unlearned model checkpoint
        ckpt_path = out_path / f"{method_name}_step{forget_step}.pt"
        torch.save(unlearned_model.state_dict(), ckpt_path)
        print(f"\n[OK] Saved unlearned model → {ckpt_path}")

    # ── Save all results ──
    results_path = out_path / f"single_shot_step{forget_step}_results.json"
    with open(results_path, "w") as f:
        # Remove non-serialisable items (logits, long lists)
        clean = {}
        for k, v in all_results.items():
            c = dict(v)
            if "evaluation" in c:
                for split, sm in c["evaluation"].items():
                    sm.pop("predictions", None)
                    sm.pop("labels", None)
                    sm.pop("confidences", None)
                    sm.pop("logits", None)
            clean[k] = c
        json.dump(clean, f, indent=2)

    print(f"\n[RESULTS SAVED] → {results_path}")

    # ── Print summary table ──
    print_summary_table(all_results, forget_step)
    return all_results


def print_summary_table(results: dict, forget_step: int):
    """Print a formatted comparison table of all methods."""
    header = (
        f"\n{'Method':<22} {'RetainAcc':>10} {'ForgetAcc':>10} "
        f"{'TestAcc':>9} {'MIA_AUC':>9} {'Time(s)':>9}"
    )
    print("\n" + "=" * 75)
    print(f"  SINGLE-SHOT UNLEARNING SUMMARY  (forget_step={forget_step})")
    print("=" * 75)
    print(header)
    print("-" * 75)

    for method, data in results.items():
        ev   = data.get("evaluation", {})
        mia  = data.get("mia_confidence", {})
        t    = data.get("total_time_s", 0)
        r_acc = ev.get("retain",  {}).get("accuracy", float("nan"))
        f_acc = ev.get(f"forget_step_{forget_step}", {}) or ev.get("forget", {})
        f_acc = f_acc.get("accuracy", float("nan")) if isinstance(f_acc, dict) else float("nan")
        te_acc = ev.get("test",   {}).get("accuracy", float("nan"))
        mia_auc = mia.get("forget_test_auc", float("nan"))

        print(
            f"{method:<22} {r_acc:>10.4f} {f_acc:>10.4f} "
            f"{te_acc:>9.4f} {mia_auc:>9.4f} {t:>9.1f}"
        )

    print("=" * 75)
    print("  MIA AUC → 0.5 = perfectly forgotten | 1.0 = fully memorised")
    print("  ForgetAcc → should match retrain oracle after successful unlearning")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-shot unlearning experiment on SFHQ"
    )
    parser.add_argument("--csv",          type=str, required=True)
    parser.add_argument("--model",        type=str, required=True,
                        help="Path to original model checkpoint (.pt)")
    parser.add_argument("--forget_step",  type=int, default=0)
    parser.add_argument("--out",          type=str, default="../results/single_shot")
    parser.add_argument("--methods",      type=str, nargs="+",
                        default=["no_unlearning", "retrain", "ga", "srl", "ft"])
    parser.add_argument("--device",       type=str, default="auto")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--ga_steps",     type=int, default=300)
    parser.add_argument("--ga_lr",        type=float, default=1e-4)
    parser.add_argument("--srl_epochs",   type=int, default=5)
    parser.add_argument("--srl_lr",       type=float, default=1e-4)
    parser.add_argument("--ft_epochs",    type=int, default=5)
    parser.add_argument("--ft_lr",        type=float, default=1e-4)
    parser.add_argument("--retrain_epochs", type=int, default=30)
    args = parser.parse_args()

    run_single_shot(
        csv_path=args.csv,
        model_path=args.model,
        forget_step=args.forget_step,
        methods=args.methods,
        out_dir=args.out,
        device_str=args.device,
        seed=args.seed,
        ga_steps=args.ga_steps,
        ga_lr=args.ga_lr,
        srl_epochs=args.srl_epochs,
        srl_lr=args.srl_lr,
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        retrain_epochs=args.retrain_epochs,
    )
