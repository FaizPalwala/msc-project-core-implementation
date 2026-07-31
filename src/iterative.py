"""
iterative.py — Iterative Unlearning Protocol (dual-head, 15-step × 4-id).

Executes all methods over 15 sequential forget steps (4 identities each),
tracks per-step metrics for stability analysis, and monitors temporal
re-emergence of previously-forgotten identities.

Protocol
────────
  For each forget_step t in {0, …, 14}:
    1. Start from the model state after step t-1 (cumulative) or
       from the original model (fresh).
    2. Run the unlearning method on the 4 identities assigned to step t.
    3. Evaluate both heads, run per-step MIA, record drift from original.
    4. At re-emergence checkpoints, re-test identities from earlier steps.

Usage:
    python iterative.py --csv data/dataset/dataset.csv \\
                        --model results/checkpoints/original_model_best.pt \\
                        --out results/iterative --methods ga ng_plus adaptiformet
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from config_loader import load_method_configs
from device_utils import resolve_device
from evaluate import evaluate_full
from mia import (
    run_mia_full, run_mia_per_identity,
    print_per_identity_summary,
)
from model import load_model, copy_model
from baselines import BASELINE_REGISTRY
from sota_methods import SOTA_REGISTRY
from novel_variant import NOVEL_REGISTRY
from interfaces import validate_unlearning_result

import logging

logger = logging.getLogger(__name__)
METHOD_REGISTRY = {**BASELINE_REGISTRY, **SOTA_REGISTRY, **NOVEL_REGISTRY}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _weight_l2(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    """L2 distance of floating-point parameters between two models."""
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    dist = 0.0
    for k in sd_a:
        if sd_a[k].dtype.is_floating_point:
            dist += (sd_a[k].float() - sd_b[k].float()).pow(2).sum().item()
    return float(dist**0.5)


def _step_eval(
    model: torch.nn.Module,
    original_model: torch.nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: int,
) -> dict[str, Any]:
    """Evaluate model after unlearning one forget step."""
    eval_res = evaluate_full(model, csv_path, device, verbose=False)
    per_id_mia = run_mia_per_identity(
        model, csv_path, device, head="identity",
        score_type="confidence",
    )

    r_acc = eval_res.get("retain", {}).get("identity", {}).get("accuracy", float("nan"))
    te_acc = eval_res.get("test", {}).get("identity", {}).get("accuracy", float("nan"))
    f_acc = eval_res.get("forget", {}).get("identity", {}).get("accuracy", float("nan"))
    r_age = eval_res.get("retain", {}).get("age", {}).get("accuracy", float("nan"))
    drift = _weight_l2(model, original_model)

    return {
        "retain_acc": round(r_acc, 4),
        "test_acc": round(te_acc, 4),
        "forget_acc": round(f_acc, 4),
        "retain_age_acc": round(r_age, 4),
        "mia_mean_auc": per_id_mia.get("mean_auc", 0.5),
        "mia_max_auc": per_id_mia.get("max_auc", 0.5),
        "forget_advantage": round(
            abs(per_id_mia.get("mean_auc", 0.5) - 0.5), 4,
        ),
        "model_drift": round(drift, 4),
        "fraction_leaked": per_id_mia.get("fraction_leaked", 0.0),
    }


# ── Single-method iterative run ───────────────────────────────────────────────


def run_iterative(
    method_name: str,
    method_cfg: dict[str, Any],
    csv_path: str,
    model_path: str,
    n_steps: int = 15,
    mode: str = "cumulative",
    out_dir: str = "results/iterative",
    device_str: str = "auto",
    seed: int = 42,
    checkpoint_every: int = 5,
    re_emergence_checks: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run one unlearning method over sequential forget steps.

    Returns list of per-step result dicts.
    """
    if re_emergence_checks is None:
        re_emergence_checks = [5, 10, 15]

    device = resolve_device(device_str)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{method_name}'. Available: {sorted(METHOD_REGISTRY)}"
        )

    original_model = load_model(model_path, device=str(device))
    current_model = copy_model(original_model, device)

    logger.info(f"\n{'─'*65}")
    logger.info(f"  Iterative: {method_name.upper()} | {n_steps} steps | {mode}")
    logger.info(f"{'─'*65}")

    # Baseline (step 0)
    logger.info(f"\n  [Step 0 / baseline] Evaluating original model…")
    baseline = _step_eval(original_model, original_model, csv_path, device, forget_step=0)
    baseline.update({
        "step": 0, "method": method_name, "mode": mode,
        "step_time_s": 0.0, "cumulative_time_s": 0.0,
        "is_baseline": True,
    })
    logger.info(f"    retain={baseline['retain_acc']:.4f} | "
          f"mia_mean={baseline['mia_mean_auc']:.4f} | "
          f"drift={baseline['model_drift']:.2f}")

    all_records: list[dict[str, Any]] = [baseline]
    cumulative_time = 0.0

    # Store models at each step for re-emergence checks
    step_models: dict[int, torch.nn.Module] = {}
    # Map forget_step → set of identity cluster IDs
    step_identities: dict[int, set[int]] = {}

    for step in range(n_steps):
        logger.info(f"\n  [Step {step+1}/{n_steps}] {method_name} | forget_step={step}")
        t0 = time.time()

        start_model = original_model if mode == "fresh" else current_model

        try:
            result = METHOD_REGISTRY[method_name](
                model=start_model,
                csv_path=csv_path,
                device=device,
                forget_step=step,
                seed=seed + step,
                **method_cfg,
            )
            validate_unlearning_result(result, method_name)
            current_model = result["model"]
            step_time = time.time() - t0
            cumulative_time += step_time

            metrics = _step_eval(
                current_model, original_model, csv_path, device,
                forget_step=step,
            )
            record = {
                "step": step + 1,
                "forget_step_idx": step,
                "method": method_name,
                "mode": mode,
                "step_time_s": round(step_time, 2),
                "cumulative_time_s": round(cumulative_time, 2),
                "is_baseline": False,
                **metrics,
                "method_metrics": {
                    k: v for k, v in result.get("metrics", {}).items()
                    if k != "history"
                },
            }
            logger.info(
                f"    retain={metrics['retain_acc']:.4f} | "
                f"mia={metrics['mia_mean_auc']:.4f} | "
                f"adv={metrics['forget_advantage']:.4f} | "
                f"drift={metrics['model_drift']:.2f} | "
                f"t={step_time:.1f}s"
            )

            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                ckpt = out_path / f"{method_name}_step{step+1}.pt"
                torch.save(current_model.state_dict(), ckpt)

            step_models[step] = copy_model(current_model, device)

        except Exception as e:
            logger.warning(f"  [WARN] Step {step+1} failed: {e}")
            import traceback
            traceback.print_exc()
            record = {
                "step": step + 1, "method": method_name, "mode": mode,
                "error": str(e), "step_time_s": time.time() - t0,
            }

        all_records.append(record)
        _append_jsonl(out_path / f"{method_name}_log.jsonl", record)

    # ── Re-emergence checks ────────────────────────────────────────────
    for check_step in re_emergence_checks:
        if check_step > n_steps or check_step not in step_models:
            continue
        model_at_check = step_models[check_step]
        logger.info(f"\n  [Re-emergence @ step {check_step}]")

        for earlier_step in range(check_step):
            if earlier_step not in step_models:
                continue
            per_id = run_mia_per_identity(
                model_at_check, csv_path, device,
            )
            re_record = {
                "type": "re_emergence",
                "check_step": check_step,
                "forgotten_step": earlier_step,
                "method": method_name,
                "mia_mean_auc": per_id.get("mean_auc", 0.5),
                "forget_advantage": round(
                    abs(per_id.get("mean_auc", 0.5) - 0.5), 4,
                ),
            }
            _append_jsonl(out_path / f"{method_name}_re_emergence.jsonl", re_record)
            all_records.append(re_record)

    # ── CSV ───────────────────────────────────────────────────────────
    _write_csv(all_records, out_path / f"{method_name}_summary.csv")
    logger.info(f"\n  ✓ {method_name} done. {len(all_records)} records")
    return all_records


# ── Multi-method runner ───────────────────────────────────────────────────────


def run_all_iterative(
    csv_path: str,
    model_path: str,
    methods: list[str] | None = None,
    n_steps: int = 15,
    mode: str = "cumulative",
    out_dir: str = "results/iterative",
    device_str: str = "auto",
    seed: int = 42,
    scale: float = 1.0,
    checkpoint_every: int = 5,
    re_emergence_checks: list[int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run iterative unlearning for all methods."""
    method_configs = load_method_configs(scale=scale)

    if methods is None:
        methods = sorted(m for m in METHOD_REGISTRY if m in method_configs)
    if "retrain" in methods:
        methods = [m for m in methods if m != "retrain"]
        logger.info("[INFO] Retrain oracle skipped in iterative mode (too expensive)")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict[str, Any]]] = {}

    for method in methods:
        cfg = method_configs.get(method, {})
        records = run_iterative(
            method_name=method,
            method_cfg=cfg,
            csv_path=csv_path,
            model_path=model_path,
            n_steps=n_steps,
            mode=mode,
            out_dir=out_dir,
            device_str=device_str,
            seed=seed,
            checkpoint_every=checkpoint_every,
            re_emergence_checks=re_emergence_checks,
        )
        all_results[method] = records

    # ── Combined CSV ─────────────────────────────────────────────────────
    combined = []
    for method, records in all_results.items():
        combined.extend(records)
    _write_csv(combined, out_path / "iterative_combined.csv")
    logger.info(f"\n[OK] Combined CSV → {out_path}/iterative_combined.csv")

    _print_summary(all_results)
    return all_results


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


_FLAT_COLS = [
    "step", "method", "mode", "retain_acc", "test_acc", "forget_acc",
    "retain_age_acc", "mia_mean_auc", "mia_max_auc", "forget_advantage",
    "model_drift", "step_time_s", "cumulative_time_s",
    "is_baseline", "error", "type", "check_step", "forgotten_step",
    "fraction_leaked",
]


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for r in records:
        row = {}
        for c in _FLAT_COLS:
            v = r.get(c, "")
            if isinstance(v, float):
                row[c] = round(v, 4)
            else:
                row[c] = v
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[c for c in _FLAT_COLS
                                                if any(c in r for r in rows)])
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(all_results: dict) -> None:
    logger.info(f"\n{'='*80}")
    logger.info("  ITERATIVE UNLEARNING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Method':<16} {'MIA@5':>8} {'MIA@10':>8} {'MIA@15':>8} "
          f"{'ΔRetain':>9} {'Drift@End':>10} {'Time(min)':>10}")
    logger.info("─" * 72)

    for method, records in all_results.items():
        data = [r for r in records
                if not r.get("is_baseline") and "error" not in r
                and r.get("type") != "re_emergence"]
        if not data:
            continue
        steps = [r["step"] for r in data]
        mias  = [r.get("mia_mean_auc", 0.5) for r in data]
        rets  = [r.get("retain_acc", 0.0) for r in data]
        drift = [r.get("model_drift", 0.0) for r in data]

        def _at(n, vals):
            idx = next((i for i, s in enumerate(steps) if s >= n), -1)
            return vals[idx] if idx >= 0 else float("nan")

        d_ret = rets[-1] - rets[0] if len(rets) >= 2 else 0.0
        ttl = records[-1].get("cumulative_time_s", 0) / 60

        logger.info(
            f"{method:<16} {_at(5, mias):>8.4f} {_at(10, mias):>8.4f} "
            f"{_at(15, mias):>8.4f} {d_ret:>+9.4f} "
            f"{_at(len(drift), drift):>10.2f} {ttl:>10.1f}"
        )
    logger.info("=" * 80)


def main() -> None:
    """
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
    CLI entry point for iterative unlearning protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",               type=str, required=True)
    parser.add_argument("--model",             type=str, required=True)
    parser.add_argument("--out",               type=str, default="results/iterative")
    parser.add_argument("--methods",           type=str, nargs="*", default=None)
    parser.add_argument("--n_steps",           type=int, default=15)
    parser.add_argument("--mode",              type=str, default="cumulative",
                        choices=["cumulative", "fresh"])
    parser.add_argument("--device",            type=str, default="auto")
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--n_seeds",           type=int, default=5)
    parser.add_argument("--scale",             type=float, default=1.0)
    parser.add_argument("--checkpoint_every",  type=int, default=5)
    parser.add_argument("--re_emergence",      type=int, nargs="*",
                        default=[5, 10, 15])
    args = parser.parse_args()

    if args.n_seeds > 1:
        for si in range(args.n_seeds):
            seed = args.seed + si
            seed_dir = f"{args.out}/seed_{seed}"
            logger.info(f"\n{'#'*70}\n  SEED {si+1}/{args.n_seeds} (seed={seed})\n{'#'*70}")
            run_all_iterative(
                csv_path=args.csv,
                model_path=args.model,
                methods=args.methods or None,
                n_steps=args.n_steps,
                mode=args.mode,
                out_dir=seed_dir,
                device_str=args.device,
                seed=seed,
                scale=args.scale,
                checkpoint_every=args.checkpoint_every,
                re_emergence_checks=args.re_emergence or None,
            )
    else:
        run_all_iterative(
            csv_path=args.csv,
            model_path=args.model,
            methods=args.methods or None,
            n_steps=args.n_steps,
            mode=args.mode,
            out_dir=args.out,
            device_str=args.device,
            seed=args.seed,
            scale=args.scale,
            checkpoint_every=args.checkpoint_every,
            re_emergence_checks=args.re_emergence or None,
        )


if __name__ == "__main__":
    main()
