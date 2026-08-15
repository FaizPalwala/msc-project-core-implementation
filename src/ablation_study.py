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
from novel_variant import adaptiforget
from interfaces import validate_unlearning_result, prepare_method_call



import logging

logger = logging.getLogger(__name__)
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
    best_configs_path: str | None = None,
) -> dict[str, dict]:
    """Run AdaptiForget component ablation.

    best_configs_path: optional HP-tuned best-configs dir (hparam/) — the
    "Full AdaptiForget" variant then uses the TUNED config (the method as
    deployed), so the component contributions are measured from the
    deployed baseline, not the default one.
    """
    device = resolve_device(device_str)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    original = load_model(model_path, device=str(device))

    # Load base config from methods/adaptiforget.yaml, overlaid with the
    # HP-tuned best config when supplied (--best_configs hparam dir) so the
    # ablation studies the method AS DEPLOYED, not the default baseline.
    base_cfg = get_method_config("adaptiforget", scale=scale,
                                 best_configs_path=best_configs_path)

    results: dict[str, dict] = {}

    logger.info(f"\n  Running {len(ABLATIONS)} ablation variants…")
    for name, overrides in ABLATIONS.items():
        cfg = prepare_method_call({**base_cfg, **overrides})  # pins subset="train"
        logger.info(f"\n  ── {name}")
        res = adaptiforget(original, csv_path, device, seed=seed, **cfg)
        validate_unlearning_result(res, "adaptiforget")
        m = res["model"]

        ev = evaluate_full(m, csv_path, device, verbose=False, subset="holdout")
        per_id = run_mia_per_identity(m, csv_path, device, head="identity",
                                      subset="holdout")

        results[name] = {
            "retain_id_acc": ev.get("retain", {}).get("identity", {}).get("accuracy", 0),
            "retain_age_acc": ev.get("retain", {}).get("age", {}).get("accuracy", 0),
            "forget_id_acc": ev.get("forget", {}).get("identity", {}).get("accuracy", 0),
            "mia_mean_auc": per_id.get("mean_auc", 0.5),
            "mia_max_auc": per_id.get("max_auc", 0.5),
            # Signed forget advantage (C1 semantics): below-0.5 MIA is the
            # erasure signal.  max(0, 1-2*MIA) ∈ [0,1]; 1 = fully erased,
            # 0 = no erasure/leak.  abs(MIA-0.5) would reward the no-signal
            # 0.5 baseline and conflate leak (0.9) with erasure (0.1).
            "forget_advantage": max(0.0, 1.0 - 2.0 * per_id.get("mean_auc", 0.5)),
            "fraction_leaked": per_id.get("fraction_leaked", 0),
            "steps_used": res["metrics"].get("steps_used", 0),
            "time_s": res["metrics"].get("unlearning_time_s", 0),
            "early_stopped": res["metrics"].get("early_stopped", False),
        }
        logger.info(
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
    logger.info(f"\n  ✓ Ablation results → {csv_out}")

    # Print table
    logger.info(f"\n{'='*80}")
    logger.info("  ADAPTIFORGET ABLATION STUDY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Variant':<26} {'Retain':>7} {'Forget':>7} {'MIA-AUC':>9} {'F-Adv':>7} "
          f"{'Leaked':>7} {'Steps':>6}")
    logger.info("─" * 72)
    for name, r in results.items():
        logger.info(
            f"{name:<26} {r['retain_id_acc']:>7.4f} "
            f"{r['forget_id_acc']:>7.4f} "
            f"{r['mia_mean_auc']:>9.4f} {r['forget_advantage']:>7.4f} "
            f"{r['fraction_leaked']:>7.4f} {r['steps_used']:>6}"
        )
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str, required=True)
    parser.add_argument("--model",  type=str, required=True)
    parser.add_argument("--out",    type=str, default="results/ablation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--scale",  type=float, default=1.0)
    parser.add_argument("--best_configs", type=str, default=None,
                        help="HP-tuned best-configs dir (hparam/) — the Full "
                             "variant then uses the TUNED AdaptiForget config "
                             "(method as deployed), not the YAML default.")
    args = parser.parse_args()

    run_ablation(
        csv_path=args.csv,
        model_path=args.model,
        out_dir=args.out,
        device_str=args.device,
        seed=args.seed,
        scale=args.scale,
        best_configs_path=args.best_configs,
    )
