"""
single_shot.py — Single-shot unlearning evaluation (combined forget set).

Runs ALL methods on the full forget split (all 60 identities at once),
evaluates with the five-tier framework, and saves per-method results.

Usage:
    python single_shot.py --csv data/dataset/dataset.csv \\
                          --model results/checkpoints/original_model_best.pt \\
                          --out results/single_shot --methods ga ng_plus adaptiformet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from config_loader import load_method_configs
from device_utils import resolve_device
from evaluate import evaluate_full, evaluate_per_identity, evaluate_per_demographic
from mia import (
    run_mia_full, run_mia_per_identity,
    run_max_confidence_attack, run_mia_per_demographic,
    print_per_identity_summary,
)
from probes import probe_all, measure_forgetting, print_probe_summary
from model import load_model
from baselines import BASELINE_REGISTRY
from sota_methods import SOTA_REGISTRY
from novel_variant import NOVEL_REGISTRY

METHOD_REGISTRY = {**BASELINE_REGISTRY, **SOTA_REGISTRY, **NOVEL_REGISTRY}

METHOD_DISPLAY = {
    "no_unlearning": "No-Unlearning",
    "retrain":       "Retrain Oracle*",
    "ga":            "GA",
    "srl":           "SRL",
    "ft":            "FT",
    "ng_plus":       "NG+",
    "msg":           "MSG",
    "ct":            "CT",
    "msg_kd":        "MSG-KD †",
    "adaptiformet":  "AdaptiForget ‡",
}


def run_single_shot(
    csv_path: str,
    model_path: str,
    methods: list[str] | None = None,
    out_dir: str = "results/single_shot",
    device_str: str = "auto",
    seed: int = 42,
    scale: float = 1.0,
    skip_retrain: bool = False,
    mia_score_types: tuple[str, ...] = ("confidence", "loss"),
) -> dict[str, Any]:
    """Run single-shot evaluation on the combined forget set.

    Args:
        csv_path:    Path to dataset CSV.
        model_path:  Path to original model checkpoint.
        methods:     Method names to evaluate (default: all registered).
        out_dir:     Output directory for results and checkpoints.
        device_str:  Device string for PyTorch.
        seed:        Random seed.
        scale:       Smoke-test scaling factor (1.0 = full).
        skip_retrain: Skip retrain oracle (expensive).
        mia_score_types: MIA score types to compute.

    Returns:
        Dict keyed by method name → per-method results.
    """
    torch.manual_seed(seed)
    device = resolve_device(device_str)

    method_configs = load_method_configs(scale=scale)

    if methods is None:
        methods = sorted(m for m in METHOD_REGISTRY if m in method_configs)
    if skip_retrain and "retrain" in methods:
        methods = [m for m in methods if m != "retrain"]
        print("[INFO] Skipping retrain oracle")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    original_model = load_model(model_path, device=str(device))
    all_results: dict[str, Any] = {}

    print(f"\n{'='*70}")
    print(f"  SINGLE-SHOT UNLEARNING EVALUATION")
    print(f"  Device: {device} | Seed: {seed} | Scale: {scale}")
    print(f"  Methods: {methods}")
    print(f"{'='*70}")

    for method_name in methods:
        if method_name not in METHOD_REGISTRY:
            print(f"[WARN] Unknown method '{method_name}', skipping")
            continue

        display = METHOD_DISPLAY.get(method_name, method_name)
        print(f"\n{'─'*60}")
        print(f"  {display}")
        print(f"{'─'*60}")

        cfg = method_configs.get(method_name, {})
        t0 = time.time()

        try:
            result = METHOD_REGISTRY[method_name](
                model=original_model,
                csv_path=csv_path,
                device=device,
                seed=seed,
                **cfg,
            )
            unlearned_model = result["model"]
            method_metrics  = result["metrics"]

            # ── Tier 0: Standard evaluation ───────────────────────────────
            eval_res = evaluate_full(unlearned_model, csv_path, device, verbose=True)

            # ── Tier 1: Per-identity evaluation ───────────────────────────
            per_id_eval = evaluate_per_identity(unlearned_model, csv_path, device)
            per_id_mia = run_mia_per_identity(
                unlearned_model, csv_path, device, head="identity",
            )
            print_per_identity_summary(per_id_mia)

            # ── Max-confidence attack ─────────────────────────────────────
            max_conf = run_max_confidence_attack(
                unlearned_model, csv_path, device, head="identity",
            )
            print(f"  Max-confidence AUC: {max_conf['max_confidence_auc']:.4f}")

            # ── Demographic MIA ───────────────────────────────────────────
            demog_mia = run_mia_per_demographic(
                unlearned_model, csv_path, device, head="identity",
            )

            # ── Tier 2: Representation probing ────────────────────────────
            probes_forget = probe_all(
                unlearned_model, csv_path, device, split="forget",
                include_identity=(method_name != "no_unlearning"),
            )
            print_probe_summary(probes_forget)
            forgetting_metrics = measure_forgetting(
                unlearned_model, original_model, csv_path, device,
            )

            total_time = time.time() - t0

            # ── Save checkpoint ───────────────────────────────────────────
            ckpt = out_path / f"{method_name}_unlearned.pt"
            torch.save(unlearned_model.state_dict(), ckpt)

            all_results[method_name] = {
                "method": display,
                "config": cfg,
                "method_metrics": method_metrics,
                "evaluation": _clean_eval(eval_res),
                "per_identity_mia": {str(k): v for k, v in per_id_mia.items()
                                     if not isinstance(v, dict)},
                "max_confidence_attack": max_conf,
                "demographic_mia": demog_mia,
                "probes": probes_forget,
                "forgetting": forgetting_metrics,
                "total_time_s": round(total_time, 2),
            }
            print(f"  ✓ Done in {total_time:.1f}s")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[method_name] = {"method": display, "error": str(e)}

    # ── Save all results ──────────────────────────────────────────────────
    results_path = out_path / "single_shot_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Full results → {results_path}")

    # ── Print comprehensive table ─────────────────────────────────────────
    _print_table(all_results)
    _save_csv(all_results, out_path)

    return all_results


def _clean_eval(eval_res: dict) -> dict:
    """Strip raw arrays from evaluation results for JSON serialisation."""
    clean = {}
    for split, m in eval_res.items():
        clean[split] = {
            "n_samples": m.get("n_samples", 0),
            "identity": m.get("identity", {}),
            "age": m.get("age", {}),
        }
    return clean


def _print_table(results: dict[str, Any]) -> None:
    """Print formatted comparison table."""
    header = (
        f"\n{'Method':<16} {'IdAcc-Ret':>10} {'IdAcc-Test':>10} "
        f"{'MIA-F:Id':>9} {'F-Adv':>7} {'MaxConfAUC':>11} "
        f"{'Probe-Id':>9} {'Time(s)':>8}"
    )
    print("\n" + "=" * 90)
    print("  SINGLE-SHOT UNLEARNING SUMMARY")
    print("=" * 90)
    print(header)
    print("─" * 90)

    for method, data in results.items():
        if "error" in data:
            print(f"{METHOD_DISPLAY.get(method, method):<16}  ERROR: {data['error']}")
            continue
        ev  = data.get("evaluation", {})
        per_id = data.get("per_identity_mia", {})
        max_c = data.get("max_confidence_attack", {})
        probes = data.get("probes", {})
        t = data.get("total_time_s", 0)

        r_id_acc = ev.get("retain", {}).get("identity", {}).get("accuracy", float("nan"))
        t_id_acc = ev.get("test", {}).get("identity", {}).get("accuracy", float("nan"))
        mia_auc = per_id.get("mean_auc", float("nan"))
        f_adv = per_id.get("mean_forget_advantage",
                           abs(float(mia_auc) - 0.5) if isinstance(mia_auc, float) else float("nan"))
        max_auc = max_c.get("max_confidence_auc", float("nan"))
        probe_id = probes.get("identity", {}).get("accuracy", float("nan"))

        print(
            f"{METHOD_DISPLAY.get(method, method):<16} "
            f"{r_id_acc:>10.4f} {t_id_acc:>10.4f} "
            f"{mia_auc:>9.4f} {f_adv:>7.4f} {max_auc:>11.4f} "
            f"{probe_id:>9.4f} {t:>8.1f}"
        )

    print("=" * 90)
    print("  MIA-F:Id = per-identity mean MIA AUC (identity head)")
    print("  MaxConfAUC = worst-case single-image attack")
    print("  Probe-Id = identity probe accuracy (target: near 0%)")
    print("=" * 90)


def _save_csv(results: dict, out_path: Path) -> None:
    """Save flat CSV of key metrics."""
    import csv

    rows = []
    for method, data in results.items():
        if "error" in data:
            continue
        ev  = data.get("evaluation", {})
        per_id = data.get("per_identity_mia", {})
        max_c = data.get("max_confidence_attack", {})
        probes = data.get("probes", {})

        rows.append({
            "method": METHOD_DISPLAY.get(method, method),
            "retain_id_acc": ev.get("retain", {}).get("identity", {}).get("accuracy", ""),
            "retain_age_acc": ev.get("retain", {}).get("age", {}).get("accuracy", ""),
            "test_id_acc": ev.get("test", {}).get("identity", {}).get("accuracy", ""),
            "forget_id_acc": ev.get("forget", {}).get("identity", {}).get("accuracy", ""),
            "mia_mean_auc": per_id.get("mean_auc", ""),
            "mia_std_auc": per_id.get("std_auc", ""),
            "mia_max_auc": per_id.get("max_auc", ""),
            "mia_fraction_leaked": per_id.get("fraction_leaked", ""),
            "max_conf_auc": max_c.get("max_confidence_auc", ""),
            "probe_identity_acc": probes.get("identity", {}).get("accuracy", ""),
            "probe_age_acc": probes.get("age", {}).get("accuracy", ""),
            "time_s": data.get("total_time_s", ""),
        })

    csv_path = out_path / "single_shot_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] CSV → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          type=str, required=True)
    parser.add_argument("--model",        type=str, required=True)
    parser.add_argument("--out",          type=str, default="results/single_shot")
    parser.add_argument("--methods",      type=str, nargs="*", default=None)
    parser.add_argument("--device",       type=str, default="auto")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--scale",        type=float, default=1.0)
    parser.add_argument("--skip_retrain", action="store_true")
    args = parser.parse_args()

    run_single_shot(
        csv_path=args.csv,
        model_path=args.model,
        methods=args.methods or None,
        out_dir=args.out,
        device_str=args.device,
        seed=args.seed,
        scale=args.scale,
        skip_retrain=args.skip_retrain,
    )
