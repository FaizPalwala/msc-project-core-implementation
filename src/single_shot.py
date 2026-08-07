"""
single_shot.py — Single-shot unlearning evaluation (combined forget set).

Runs ALL methods on the full forget split (all forget identities at once),
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
from interfaces import validate_unlearning_result, prepare_method_call

import logging

logger = logging.getLogger(__name__)
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
        logger.info("Skipping retrain oracle")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Infer identity classes from the CSV so the pipeline tracks the
    # dataset (600 vs 750 identities) instead of hardcoding.
    from dataset import infer_identity_classes
    n_id_classes = infer_identity_classes(csv_path)
    logger.info(f"  Identity classes (inferred from CSV): {n_id_classes}")

    original_model = load_model(model_path, device=str(device))
    all_results: dict[str, Any] = {}

    logger.info(f"\n{'='*70}")
    logger.info(f"  SINGLE-SHOT UNLEARNING EVALUATION")
    logger.info(f"  Device: {device} | Seed: {seed} | Scale: {scale}")
    logger.info(f"  Methods: {methods}")
    logger.info(f"{'='*70}")

    for method_name in methods:
        if method_name not in METHOD_REGISTRY:
            logger.warning(f"[WARN] Unknown method '{method_name}', skipping")
            continue

        display = METHOD_DISPLAY.get(method_name, method_name)
        logger.info(f"\n{'─'*60}")
        logger.info(f"  {display}")
        logger.info(f"{'─'*60}")

        cfg = method_configs.get(method_name, {})
        cfg = prepare_method_call(cfg, identity_classes=n_id_classes)
        t0 = time.time()

        try:
            result = METHOD_REGISTRY[method_name](
                model=original_model,
                csv_path=csv_path,
                device=device,
                seed=seed,
                **cfg,
            )
            validate_unlearning_result(result, method_name)
            unlearned_model = result["model"]
            method_metrics  = result["metrics"]

            # ── Tier 0: Standard evaluation (holdout images = generalisation) ─
            eval_res = evaluate_full(unlearned_model, csv_path, device,
                                     subset="holdout", verbose=True)

            # ── Forget-train evaluation (same images used for unlearning)
            #     The gap between train and holdout forget accuracy detects
            #     overfitting-to-forgetting (see §4.3 of METRICS_GUIDE.md).
            eval_train = evaluate_full(unlearned_model, csv_path, device,
                                       subset="train", verbose=False)

            # ── Tier 1: Per-identity evaluation ───────────────────────────
            per_id_eval = evaluate_per_identity(unlearned_model, csv_path, device,
                                                subset="holdout")
            per_id_mia = run_mia_per_identity(
                unlearned_model, csv_path, device, head="identity",
                subset="holdout",
            )
            print_per_identity_summary(per_id_mia)

            # ── Max-confidence attack ─────────────────────────────────────
            max_conf = run_max_confidence_attack(
                unlearned_model, csv_path, device, head="identity",
                subset="holdout",
            )
            logger.info(f"  Max-confidence AUC: {max_conf['max_confidence_auc']:.4f}")

            # ── Demographic MIA ───────────────────────────────────────────
            demog_mia = run_mia_per_demographic(
                unlearned_model, csv_path, device, head="identity",
                subset="holdout",
            )

            # ── Demographic evaluation (accuracy by age/popularity bin) ───
            demog_eval = evaluate_per_demographic(
                unlearned_model, csv_path, device,
                subset="holdout",
            )

            # ── Tier 2: Representation probing ────────────────────────────
            probes_forget = probe_all(
                unlearned_model, csv_path, device, split="forget",
                subset="holdout",
            )
            print_probe_summary(probes_forget)
            forgetting_metrics = measure_forgetting(
                unlearned_model, original_model, csv_path, device,
                subset="holdout",
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
                "evaluation_train": _clean_eval(eval_train),  # forget-train gap
                # Scalar aggregates only for per_identity_mia (drop nested
                # per-identity dicts); the per-identity AUC map is stored
                # separately for the CSV export / plot 10.
                "per_identity_mia": {str(k): v for k, v in per_id_mia.items()
                                     if not isinstance(v, dict)},
                "per_identity_auc": per_id_mia.get("per_identity_auc", {}),
                "max_confidence_attack": max_conf,
                "demographic_mia": demog_mia,
                "demographic_eval": demog_eval,
                "probes": probes_forget,
                "forgetting": forgetting_metrics,
                "total_time_s": round(total_time, 2),
            }
            logger.info(f"  ✓ Done in {total_time:.1f}s")

        except Exception as e:
            logger.info(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[method_name] = {"method": display, "error": str(e)}

    # ── Save all results ──────────────────────────────────────────────────
    results_path = out_path / "single_shot_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n[OK] Full results → {results_path}")

    # ── Print comprehensive table ─────────────────────────────────────────
    _print_table(all_results)
    _save_csv(all_results, out_path)

    return all_results


# ── Multi-seed wrapper ────────────────────────────────────────────────────────

_AGGREGATABLE_KEYS: set[str] = {
    "mia_mean_auc", "mia_max_auc", "mia_std_auc", "mia_fraction_leaked",
    "max_conf_auc", "max_confidence_auc", "forget_advantage", "fraction_leaked",
    "probe_identity_acc", "probe_age_acc", "probe_gender_acc",
    "retain_id_acc", "retain_age_acc", "forget_id_acc",
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

        logger.info(f"\n{'#'*70}")
        logger.info(f"  SEED {si+1}/{n_seeds}  (seed={seed})")
        logger.info(f"{'#'*70}")

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

    # ── Bootstrap oracle comparison ──────────────────────────────────────
    p_values = _bootstrap_oracle_comparison(aggregated, n_seeds)
    _add_p_values(aggregated, p_values)
    if p_values:
        logger.info(f"\n[OK] Bootstrap p-values computed for {len(p_values)} methods vs retrain oracle")

    # ── Save aggregated ───────────────────────────────────────────────────
    agg_path = out_path / "single_shot_aggregated.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"\n[OK] Aggregated results (μ ± σ over {n_seeds} seeds) → {agg_path}")

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
    logger.info(f"[OK] Aggregated CSV → {agg_csv}")

    return {"per_seed": all_seed_results, "aggregated": aggregated}


def _flatten_metrics(data: dict) -> dict[str, Any]:
    """Flatten nested evaluation dict into flat key-value pairs for aggregation."""
    flat: dict[str, Any] = {}
    ev = data.get("evaluation", {})
    ev_train = data.get("evaluation_train", {})
    per_id = data.get("per_identity_mia", {})
    max_c = data.get("max_confidence_attack", {})
    probes = data.get("probes", {})

    flat["retain_id_acc"] = ev.get("retain", {}).get("identity", {}).get("accuracy")
    flat["retain_age_acc"] = ev.get("retain", {}).get("age", {}).get("accuracy")
    flat["forget_id_acc"] = ev.get("forget", {}).get("identity", {}).get("accuracy")
    flat["forget_train_id_acc"] = ev_train.get("forget", {}).get("identity", {}).get("accuracy")
    flat["mia_mean_auc"] = per_id.get("mean_auc")
    flat["mia_std_auc"] = per_id.get("std_auc")
    flat["mia_max_auc"] = per_id.get("max_auc")
    flat["mia_fraction_leaked"] = per_id.get("fraction_leaked")
    flat["max_conf_auc"] = max_c.get("max_confidence_auc")
    flat["probe_identity_acc"] = probes.get("identity", {}).get("accuracy")
    flat["probe_age_acc"] = probes.get("age", {}).get("accuracy")
    flat["probe_gender_acc"] = probes.get("gender", {}).get("accuracy") if probes.get("gender") else None
    flat["total_time_s"] = data.get("total_time_s")

    # ── Per-demographic-bin metrics (populated only for imbalanced dataset) ──
    demog_mia = data.get("demographic_mia", {})
    demog_eval = data.get("demographic_eval", {})
    for bin_name in ("high", "medium", "low"):
        # MIA AUC by popularity bin
        key_mia = f"popularity_{bin_name}"
        if key_mia in demog_mia and isinstance(demog_mia[key_mia], dict):
            flat[f"mia_auc_{bin_name}"] = demog_mia[key_mia].get("auc")
        # Forget identity accuracy by popularity bin (from demographic_eval)
        if key_mia in demog_eval and isinstance(demog_eval[key_mia], dict):
            flat[f"forget_id_acc_{bin_name}"] = (
                demog_eval[key_mia].get("identity", {}).get("accuracy")
            )
    return {k: v for k, v in flat.items() if v is not None}


def _print_aggregated_table(aggregated: dict, n_seeds: int) -> None:
    """Print μ ± σ table across seeds."""
    header = (
        f"\n{'Method':<16} {'IdAcc-R':>14} {'MIA-AUC(μ±σ)':>18} "
        f"{'MaxAUC':>9} {'Probe-Id':>9} {'FgTr-Gap':>9} {'Time':>8}"
    )
    logger.info(f"\n{'='*95}")
    logger.info(f"  AGGREGATED RESULTS (μ ± σ over {n_seeds} seeds)")
    logger.info(f"{'='*95}")
    logger.info(header)
    logger.info("─" * 95)

    for method, agg in aggregated.items():
        if "error" in agg:
            logger.info(f"{agg['method']:<16}  ERROR: {agg['error']}")
            continue
        r_acc = agg.get("retain_id_acc", float("nan"))
        r_std = agg.get("retain_id_acc_std", 0)
        mia = agg.get("mia_mean_auc", float("nan"))
        mia_s = agg.get("mia_mean_auc_std", 0)
        max_a = agg.get("max_conf_auc", float("nan"))
        probe = agg.get("probe_identity_acc", float("nan"))
        fg_hold = agg.get("forget_id_acc", float("nan"))
        fg_train = agg.get("forget_train_id_acc", float("nan"))
        gap = (fg_train - fg_hold) if not (isinstance(fg_train, float) and np.isnan(fg_train)) else float("nan")
        t = agg.get("total_time_s", 0)

        logger.info(
            f"{agg['method']:<16} {r_acc:>8.4f}±{r_std:.4f} "
            f"{mia:>8.4f}±{mia_s:.4f} {max_a:>9.4f} "
            f"{probe:>9.4f} {gap:>9.4f} {t:>8.1f}"
        )

    logger.info("=" * 95)

    # ── Imbalanced subset: per-popularity-bin breakdown ──────────────────
    has_imb = any(
        agg.get("forget_id_acc_high") is not None for agg in aggregated.values()
        if isinstance(agg, dict)
    )
    if has_imb:
        logger.info(f"\n{'='*80}")
        logger.info(f"  IMBALANCED: PER-POPULARITY-BIN BREAKDOWN")
        logger.info(f"  (high=~70 imgs, medium=~40 imgs, low=~20 imgs per identity)")
        logger.info(f"{'='*80}")
        logger.info(f"{'Method':<16} {'FgAcc-H':>8} {'FgAcc-M':>8} {'FgAcc-L':>8} "
              f"{'MIA-H':>8} {'MIA-M':>8} {'MIA-L':>8}")
        logger.info("-" * 62)
        for method, agg in aggregated.items():
            if "error" in agg:
                continue
            f_h = agg.get("forget_id_acc_high", float("nan"))
            f_m = agg.get("forget_id_acc_medium", float("nan"))
            f_l = agg.get("forget_id_acc_low", float("nan"))
            m_h = agg.get("mia_auc_high", float("nan"))
            m_m = agg.get("mia_auc_medium", float("nan"))
            m_l = agg.get("mia_auc_low", float("nan"))
            if all(np.isnan(v) for v in [f_h, f_m, f_l, m_h, m_m, m_l]):
                continue
            logger.info(
                f"{agg['method']:<16} {f_h:>8.4f} {f_m:>8.4f} {f_l:>8.4f} "
                f"{m_h:>8.4f} {m_m:>8.4f} {m_l:>8.4f}"
            )
        logger.info("=" * 80)


def _bootstrap_oracle_comparison(
    aggregated: dict,
    n_seeds: int,
    n_bootstrap: int = 10_000,
) -> dict[str, dict[str, float]]:
    """Bootstrap comparison of each method vs retrain oracle.

    Returns dict[method] → {metric: p_value}.
    p < 0.05 = significantly different from oracle on that metric.
    For MIA AUC / forget advantage / probe acc: one-tailed (method worse if higher).
    For retain/test acc: one-tailed (method worse if lower).
    """
    if "retrain" not in aggregated or "error" in aggregated.get("retrain", {}):
        return {}

    oracle = aggregated["retrain"]
    higher_is_worse = {
        "mia_mean_auc", "mia_max_auc", "forget_advantage",
        "mia_fraction_leaked", "max_conf_auc",
        "probe_identity_acc", "probe_age_acc", "probe_gender_acc",
    }
    lower_is_worse = {"retain_id_acc", "retain_age_acc", "forget_id_acc"}

    rng = np.random.RandomState(42)
    p_values: dict[str, dict[str, float]] = {}

    for method, agg in aggregated.items():
        if method == "retrain" or "error" in agg:
            continue
        p_vals: dict[str, float] = {}
        for metric in _AGGREGATABLE_KEYS:
            std_key = f"{metric}_std"
            if metric not in agg or metric not in oracle:
                continue
            mv, ov = agg[metric], oracle[metric]
            ms = agg.get(std_key, max(abs(mv) * 0.01, 1e-6))
            os_std = oracle.get(std_key, max(abs(ov) * 0.01, 1e-6))

            o_samples = rng.normal(ov, max(os_std, 1e-6), n_bootstrap)
            m_samples = rng.normal(mv, max(ms, 1e-6), n_bootstrap)

            if metric in higher_is_worse:
                p = float(np.mean(m_samples >= o_samples))
            elif metric in lower_is_worse:
                p = float(np.mean(m_samples <= o_samples))
            else:
                p = float(np.mean(np.abs(m_samples - o_samples)
                                  >= np.abs(mv - ov)))
            p_vals[metric] = round(p, 4)
        p_values[method] = p_vals

    return p_values


def _add_p_values(aggregated: dict, p_values: dict) -> None:
    for method, agg in aggregated.items():
        if method in p_values:
            agg["p_vs_oracle"] = p_values[method]


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
        f"\n{'Method':<16} {'IdAcc-Ret':>10} {'FgTr-Gap':>10} "
        f"{'MIA-F:Id':>9} {'F-Adv':>7} {'MaxConfAUC':>11} "
        f"{'Probe-Id':>9} {'Time(s)':>8}"
    )
    logger.info("\n" + "=" * 90)
    logger.info("  SINGLE-SHOT UNLEARNING SUMMARY")
    logger.info("=" * 90)
    logger.info(header)
    logger.info("─" * 90)

    for method, data in results.items():
        if "error" in data:
            logger.info(f"{METHOD_DISPLAY.get(method, method):<16}  ERROR: {data['error']}")
            continue
        ev  = data.get("evaluation", {})
        ev_train = data.get("evaluation_train", {})
        per_id = data.get("per_identity_mia", {})
        max_c = data.get("max_confidence_attack", {})
        probes = data.get("probes", {})
        t = data.get("total_time_s", 0)

        r_id_acc = ev.get("retain", {}).get("identity", {}).get("accuracy", float("nan"))
        fg_hold = ev.get("forget", {}).get("identity", {}).get("accuracy", float("nan"))
        fg_train = ev_train.get("forget", {}).get("identity", {}).get("accuracy", float("nan"))
        fg_gap = (fg_train - fg_hold) if isinstance(fg_train, float) and not np.isnan(fg_train) else float("nan")
        mia_auc = per_id.get("mean_auc", float("nan"))
        f_adv = per_id.get("mean_forget_advantage",
                           abs(float(mia_auc) - 0.5) if isinstance(mia_auc, float) else float("nan"))
        max_auc = max_c.get("max_confidence_auc", float("nan"))
        probe_id = probes.get("identity", {}).get("accuracy", float("nan"))

        logger.info(
            f"{METHOD_DISPLAY.get(method, method):<16} "
            f"{r_id_acc:>10.4f} {fg_gap:>10.4f} "
            f"{mia_auc:>9.4f} {f_adv:>7.4f} {max_auc:>11.4f} "
            f"{probe_id:>9.4f} {t:>8.1f}"
        )

    logger.info("=" * 90)
    logger.info("  MIA-F:Id = per-identity mean MIA AUC (identity head)")
    logger.info("  MaxConfAUC = worst-case single-image attack")
    logger.info("  Probe-Id = identity probe accuracy (target: near 0%)")
    logger.info("  FgTr-Gap = forget-train acc − forget-holdout acc (≤0.10 = genuine forgetting)")
    logger.info("=" * 90)


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
    logger.info(f"[OK] CSV → {csv_path}")

    # Per-identity MIA AUC export (feeds stability plot 10_identity_signatures).
    # Columns: method, identity_id, mia_auc
    per_id_rows = []
    for method, data in results.items():
        if "error" in data:
            continue
        auc_map = data.get("per_identity_auc", {})
        for cid, auc in auc_map.items():
            per_id_rows.append({
                "method": METHOD_DISPLAY.get(method, method),
                "identity_id": cid,
                "mia_auc": auc,
            })
    if per_id_rows:
        per_id_path = out_path / "single_shot_per_identity.csv"
        with open(per_id_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "identity_id", "mia_auc"])
            writer.writeheader()
            writer.writerows(per_id_rows)
        logger.info(f"[OK] Per-identity CSV → {per_id_path}")

    # Demographic MIA export (feeds stability plot 11_demographic_heatmap).
    # Columns: method, demographic_group, mia_auc
    demog_rows = []
    for method, data in results.items():
        if "error" in data:
            continue
        demog = data.get("demographic_mia", {})
        for group_key, group_res in demog.items():
            if isinstance(group_res, dict) and "auc" in group_res:
                demog_rows.append({
                    "method": METHOD_DISPLAY.get(method, method),
                    "demographic_group": group_key,
                    "mia_auc": group_res["auc"],
                })
    if demog_rows:
        demog_path = out_path / "single_shot_demographic.csv"
        with open(demog_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "demographic_group", "mia_auc"])
            writer.writeheader()
            writer.writerows(demog_rows)
        logger.info(f"[OK] Demographic CSV → {demog_path}")


def main() -> None:
    """CLI entry point for single-shot evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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


if __name__ == "__main__":
    main()
