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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
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


# ── Multi-seed wrapper ────────────────────────────────────────────────────────

_AGGREGATABLE_KEYS: set[str] = {
    "mia_mean_auc", "mia_max_auc", "mia_std_auc", "mia_fraction_leaked",
    "max_conf_auc", "max_confidence_auc", "forget_advantage", "fraction_leaked",
    "probe_identity_acc", "probe_age_acc", "probe_gender_acc",
    "retain_id_acc", "retain_age_acc", "test_id_acc", "forget_id_acc",
    "total_time_s",
}


def run_single_shot_multi_seed(
    csv_path: str,
    model_path: str,
    methods: list[str] | None = None,
    out_dir: str = "results/single_shot",
    device_str: str = "auto",
    base_seed: int = 42,
    n_seeds: int = 5,
    scale: float = 1.0,
    skip_retrain: bool = False,
) -> dict[str, Any]:
    """Run single-shot evaluation with multiple seeds, reporting μ ± σ.

    Each seed writes its own results.json + summary.csv into a per-seed
    subdirectory.  The aggregated results (μ ± σ across seeds per metric
    per method) are saved at the top level.
    """
    import csv

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_seed_results: list[dict[str, Any]] = []

    for si in range(n_seeds):
        seed = base_seed + si
        seed_dir = out_path / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"  SEED {si+1}/{n_seeds}  (seed={seed})")
        print(f"{'#'*70}")

        result = run_single_shot(
            csv_path=csv_path,
            model_path=model_path,
            methods=methods,
            out_dir=str(seed_dir),
            device_str=device_str,
            seed=seed,
            scale=scale,
            skip_retrain=skip_retrain,
        )
        all_seed_results.append(result)

    # ── Aggregate across seeds ────────────────────────────────────────────
    aggregated: dict[str, dict[str, Any]] = {}
    method_names = sorted(all_seed_results[0].keys())

    # Collect per-method metric values across seeds
    collector: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )

    for seed_result in all_seed_results:
        for method, data in seed_result.items():
            if "error" in data:
                continue
            flat = _flatten_metrics(data)
            for key, value in flat.items():
                if key in _AGGREGATABLE_KEYS and isinstance(value, (int, float)):
                    collector[method][key].append(float(value))

    for method in method_names:
        if method not in collector:
            aggregated[method] = {
                "method": METHOD_DISPLAY.get(method, method), "error": "no valid seeds",
            }
            continue
        agg: dict[str, Any] = {"method": METHOD_DISPLAY.get(method, method)}
        for key, values in collector[method].items():
            arr = np.array(values)
            agg[key] = round(float(arr.mean()), 4)
            agg[f"{key}_std"] = round(float(arr.std()), 4)
        aggregated[method] = agg

    # ── Save aggregated ───────────────────────────────────────────────────
    agg_path = out_path / "single_shot_aggregated.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n[OK] Aggregated results (μ ± σ over {n_seeds} seeds) → {agg_path}")

    # ── Print aggregated table ────────────────────────────────────────────
    _print_aggregated_table(aggregated, n_seeds)

    # ── Save aggregated CSV ───────────────────────────────────────────────
    rows = []
    fieldnames = set()
    for method, agg in aggregated.items():
        row = {"method": agg.get("method", method)}
        for k, v in agg.items():
            if k == "method":
                continue
            row[k] = v
            fieldnames.add(k)
        rows.append(row)
    fieldnames = ["method"] + sorted(f for f in fieldnames if not f.endswith("_std")) + \
                 sorted(f for f in fieldnames if f.endswith("_std"))

    agg_csv = out_path / "single_shot_aggregated.csv"
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # fill missing
        for row in rows:
            for fn in fieldnames:
                row.setdefault(fn, "")
        writer.writerows(rows)
    print(f"[OK] Aggregated CSV → {agg_csv}")

    return {"per_seed": all_seed_results, "aggregated": aggregated}


def _flatten_metrics(data: dict) -> dict[str, Any]:
    """Flatten nested evaluation dict into flat key-value pairs for aggregation."""
    flat: dict[str, Any] = {}
    ev = data.get("evaluation", {})
    per_id = data.get("per_identity_mia", {})
    max_c = data.get("max_confidence_attack", {})
    probes = data.get("probes", {})

    flat["retain_id_acc"] = ev.get("retain", {}).get("identity", {}).get("accuracy")
    flat["retain_age_acc"] = ev.get("retain", {}).get("age", {}).get("accuracy")
    flat["test_id_acc"] = ev.get("test", {}).get("identity", {}).get("accuracy")
    flat["forget_id_acc"] = ev.get("forget", {}).get("identity", {}).get("accuracy")
    flat["mia_mean_auc"] = per_id.get("mean_auc")
    flat["mia_std_auc"] = per_id.get("std_auc")
    flat["mia_max_auc"] = per_id.get("max_auc")
    flat["mia_fraction_leaked"] = per_id.get("fraction_leaked")
    flat["max_conf_auc"] = max_c.get("max_confidence_auc")
    flat["probe_identity_acc"] = probes.get("identity", {}).get("accuracy")
    flat["probe_age_acc"] = probes.get("age", {}).get("accuracy")
    flat["probe_gender_acc"] = probes.get("gender", {}).get("accuracy") if probes.get("gender") else None
    flat["total_time_s"] = data.get("total_time_s")
    return {k: v for k, v in flat.items() if v is not None}


def _print_aggregated_table(aggregated: dict, n_seeds: int) -> None:
    """Print μ ± σ table across seeds."""
    header = (
        f"\n{'Method':<16} {'IdAcc-R':>14} {'MIA-AUC(μ±σ)':>18} "
        f"{'MaxAUC':>9} {'Probe-Id':>9} {'Time':>8}"
    )
    print(f"\n{'='*85}")
    print(f"  AGGREGATED RESULTS (μ ± σ over {n_seeds} seeds)")
    print(f"{'='*85}")
    print(header)
    print("─" * 85)

    for method, agg in aggregated.items():
        if "error" in agg:
            print(f"{agg['method']:<16}  ERROR: {agg['error']}")
            continue
        r_acc = agg.get("retain_id_acc", float("nan"))
        r_std = agg.get("retain_id_acc_std", 0)
        mia = agg.get("mia_mean_auc", float("nan"))
        mia_s = agg.get("mia_mean_auc_std", 0)
        max_a = agg.get("max_conf_auc", float("nan"))
        probe = agg.get("probe_identity_acc", float("nan"))
        t = agg.get("total_time_s", 0)

        print(
            f"{agg['method']:<16} {r_acc:>8.4f}±{r_std:.4f} "
            f"{mia:>8.4f}±{mia_s:.4f} {max_a:>9.4f} "
            f"{probe:>9.4f} {t:>8.1f}"
        )

    print("=" * 85)


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
    parser.add_argument("--n_seeds",      type=int, default=5)
    parser.add_argument("--scale",        type=float, default=1.0)
    parser.add_argument("--skip_retrain", action="store_true")
    args = parser.parse_args()

    if args.n_seeds > 1:
        run_single_shot_multi_seed(
            csv_path=args.csv,
            model_path=args.model,
            methods=args.methods or None,
            out_dir=args.out,
            device_str=args.device,
            base_seed=args.seed,
            n_seeds=args.n_seeds,
            scale=args.scale,
            skip_retrain=args.skip_retrain,
        )
    else:
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
