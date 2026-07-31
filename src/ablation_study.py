"""
ablation_study.py — AdaptiForget Component Ablation (dual-head).

Runs controlled single-shot experiments where each component of
AdaptiForget is disabled in turn.  Evaluates on the full forget set.

Ablations:
  Full AdaptiForget   — all 4 components active
  w/o adaptive λ      — constant KL weight (0.5)
  w/o mask refresh    — never recompute saliency mask
  w/o early stopping  — run all max_steps
  w/o KL distillation — pure MSG (no KL anchor)
  w/o masking         — all params updated (reduces to NG+)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from config_loader import get_method_config
from device_utils import resolve_device
from evaluate import evaluate_full
from mia import run_mia_full, run_mia_per_identity
from model import load_model
from novel_variant import adaptiformet


ABLATIONS = {
    "Full AdaptiForget":  {},
    "w/o adaptive λ":    {"kl_weight_init": 0.5, "kl_weight_max": 0.5},
    "w/o mask refresh":  {"mask_refresh_every": 999999},
    "w/o early stop":    {"early_stop_adv": -1.0},
    "w/o KL distil":     {"kl_weight_max": 0.0, "kl_weight_init": 0.0},
    "w/o masking (NG+)": {"topk_fraction": 1.0},
}


def run_ablation(
    csv_path: str,
    model_path: str,
    out_dir: str = "results/ablation",
    device_str: str = "auto",
    seed: int = 42,
    scale: float = 1.0,
) -> dict[str, dict]:
    """Run AdaptiForget component ablation."""
    device = resolve_device(device_str)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    original = load_model(model_path, device=str(device))

    # Load base config from methods/adaptiformet.yaml
    base_cfg = get_method_config("adaptiformet", scale=scale)

    results: dict[str, dict] = {}

    print(f"\n  Running {len(ABLATIONS)} ablation variants…")
    for name, overrides in ABLATIONS.items():
        cfg = {**base_cfg, **overrides}
        print(f"\n  ── {name}")
        res = adaptiformet(original, csv_path, device, seed=seed, **cfg)
        m = res["model"]

        ev = evaluate_full(m, csv_path, device, verbose=False)
        per_id = run_mia_per_identity(m, csv_path, device, head="identity")

        results[name] = {
            "retain_id_acc": ev.get("retain", {}).get("identity", {}).get("accuracy", 0),
            "retain_age_acc": ev.get("retain", {}).get("age", {}).get("accuracy", 0),
            "test_id_acc": ev.get("test", {}).get("identity", {}).get("accuracy", 0),
            "forget_id_acc": ev.get("forget", {}).get("identity", {}).get("accuracy", 0),
            "mia_mean_auc": per_id.get("mean_auc", 0.5),
            "mia_max_auc": per_id.get("max_auc", 0.5),
            "forget_advantage": abs(per_id.get("mean_auc", 0.5) - 0.5),
            "fraction_leaked": per_id.get("fraction_leaked", 0),
            "steps_used": res["metrics"].get("steps_used", 0),
            "time_s": res["metrics"].get("unlearning_time_s", 0),
            "early_stopped": res["metrics"].get("early_stopped", False),
        }
        print(
            f"    retain={results[name]['retain_id_acc']:.4f} | "
            f"mia={results[name]['mia_mean_auc']:.4f} | "
            f"adv={results[name]['forget_advantage']:.4f} | "
            f"steps={results[name]['steps_used']}"
        )

    # Save
    rows = [{"variant": k, **v} for k, v in results.items()]
    csv_out = out_path / "ablation_results.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_path / "ablation_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  ✓ Ablation results → {csv_out}")

    # Print table
    print(f"\n{'='*80}")
    print("  ADAPTIFORMET ABLATION STUDY")
    print(f"{'='*80}")
    print(f"{'Variant':<26} {'IdAcc':>7} {'MIA-AUC':>9} {'F-Adv':>7} "
          f"{'Leaked':>7} {'Steps':>6}")
    print("─" * 65)
    for name, r in results.items():
        print(
            f"{name:<26} {r['retain_id_acc']:>7.4f} "
            f"{r['mia_mean_auc']:>9.4f} {r['forget_advantage']:>7.4f} "
            f"{r['fraction_leaked']:>7.4f} {r['steps_used']:>6}"
        )
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str, required=True)
    parser.add_argument("--model",  type=str, required=True)
    parser.add_argument("--out",    type=str, default="results/ablation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--scale",  type=float, default=1.0)
    args = parser.parse_args()

    run_ablation(
        csv_path=args.csv,
        model_path=args.model,
        out_dir=args.out,
        device_str=args.device,
        seed=args.seed,
        scale=args.scale,
    )
