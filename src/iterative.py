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
from torch.utils.data import DataLoader

from config_loader import load_method_configs
from dataset import VirtualIdentityDataset, get_val_transform
from device_utils import resolve_device
from evaluate import evaluate_full, evaluate_model
from mia import (
    run_mia_full, run_mia_per_identity,
    print_per_identity_summary,
)
from model import load_model, copy_model
from baselines import BASELINE_REGISTRY
from sota_methods import SOTA_REGISTRY
from novel_variant import NOVEL_REGISTRY
from interfaces import validate_unlearning_result, prepare_method_call

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
    subset: str = "all",
) -> dict[str, Any]:
    """Evaluate model after unlearning one forget step.

    Records cumulative metrics (over all forget identities seen so far)
    plus step-local metrics (this step's identities only):
      - step_forget_acc: forget identity accuracy on THIS step's holdout
        images (subset='holdout').
      - step_forget_train_acc: forget identity accuracy on THIS step's
        train images — the overfitting-to-forgetting detector.  If the
        method memorised the unlearning images, train acc drops to ~0
        while holdout acc stays high.
    """
    eval_res = evaluate_full(model, csv_path, device, verbose=False, subset=subset)
    per_id_mia = run_mia_per_identity(
        model, csv_path, device, head="identity",
        score_type="confidence",
        subset=subset,
    )

    r_acc = eval_res.get("retain", {}).get("identity", {}).get("accuracy", float("nan"))
    f_acc = eval_res.get("forget", {}).get("identity", {}).get("accuracy", float("nan"))
    r_age = eval_res.get("retain", {}).get("age", {}).get("accuracy", float("nan"))
    drift = _weight_l2(model, original_model)

    # Step-local forget accuracy (this step's identities only).
    step_ds_hold = VirtualIdentityDataset(
        csv_path, split=f"forget_step_{forget_step}",
        transform=get_val_transform(), subset="holdout",
    )
    step_ds_train = VirtualIdentityDataset(
        csv_path, split=f"forget_step_{forget_step}",
        transform=get_val_transform(), subset="train",
    )
    step_forget_acc = float("nan")
    step_forget_train_acc = float("nan")
    if len(step_ds_hold) > 0:
        loader = DataLoader(step_ds_hold, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True)
        step_forget_acc = evaluate_model(model, loader, device)["identity"]["accuracy"]
    if len(step_ds_train) > 0:
        loader = DataLoader(step_ds_train, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True)
        step_forget_train_acc = evaluate_model(model, loader, device)["identity"]["accuracy"]

    return {
        "retain_acc": round(r_acc, 4),
        "forget_acc": round(f_acc, 4),
        "step_forget_acc": round(float(step_forget_acc), 4),
        "step_forget_train_acc": round(float(step_forget_train_acc), 4),
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
    subset: str = "all",
) -> list[dict[str, Any]]:
    """Run one unlearning method over sequential forget steps.

    Returns list of per-step result dicts.
    """
    if re_emergence_checks is None:
        re_emergence_checks = [5, 10, 15]

    # Infer the forget schedule (n_steps, identities per step) from the CSV
    # so the protocol tracks the dataset (e.g. 15×4 today, 15×5 at 75 forget
    # identities) instead of hardcoding.
    from dataset import infer_forget_schedule
    csv_steps, csv_per_step = infer_forget_schedule(csv_path)
    if n_steps == 15 and csv_steps != 15:
        # 15 is the historical default; prefer the CSV's actual step count
        # unless the caller explicitly requested a different n_steps.
        n_steps = csv_steps
        logger.info(f"  [infer] forget schedule from CSV: {n_steps} steps x {csv_per_step} ids")

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
    baseline = _step_eval(original_model, original_model, csv_path, device, forget_step=0, subset=subset)
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
                forget_step=step, subset=subset,
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
                subset=subset,
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
    subset: str = "all",
) -> dict[str, list[dict[str, Any]]]:
    """Run iterative unlearning for all methods."""
    method_configs = load_method_configs(scale=scale)

    if methods is None:
        methods = sorted(m for m in METHOD_REGISTRY if m in method_configs)
    if "retrain" in methods:
        methods = [m for m in methods if m != "retrain"]
        logger.info("Retrain oracle skipped in iterative mode (too expensive)")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict[str, Any]]] = {}

    # Infer identity classes from CSV once — fresh-head methods
    # (retrain oracle, SRL) need it to track the dataset size.
    from dataset import infer_identity_classes
    n_id_classes = infer_identity_classes(csv_path)
    logger.info(f"  Identity classes (inferred from CSV): {n_id_classes}")

    for method in methods:
        cfg = prepare_method_call(method_configs.get(method, {}),
                                  identity_classes=n_id_classes)  # pins subset + classes
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
            subset=subset,
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
    "step", "method", "mode", "retain_acc", "forget_acc",
    "step_forget_acc", "step_forget_train_acc",
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


def aggregate_iterative_seeds(out_dir: str) -> dict[str, Any]:
    """Aggregate multi-seed iterative results into μ ± σ.

    Reads every `seed_*/iterative_combined.csv` under out_dir, groups by
    (method, step, mode), and computes mean ± std for each numeric metric.
    Writes `iterative_combined_aggregated.csv` at the top level — this is
    the file the stability plots should consume for error bands.

    Returns dict keyed by (method, step) → aggregated row.
    """
    import pandas as pd

    out_path = Path(out_dir)
    seed_dirs = sorted(out_path.glob("seed_*/iterative_combined.csv"))
    if not seed_dirs:
        logger.warning(f"  [WARN] No seed_*/iterative_combined.csv found in {out_path}")
        return {}

    logger.info(f"  [Aggregate] Found {len(seed_dirs)} seed runs, aggregating…")
    frames = []
    for path in seed_dirs:
        df = pd.read_csv(path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Identify numeric metric columns (exclude identity/step/error columns)
    exclude = {"step", "method", "mode", "is_baseline", "error",
               "type", "check_step", "forgotten_step"}
    metric_cols = [c for c in combined.columns
                   if c not in exclude and pd.api.types.is_numeric_dtype(combined[c])]

    # Group by (method, step, mode) — ignore re-emergence rows in aggregation
    group_cols = ["method", "step", "mode"]
    grouped = combined[
        combined.get("type", "") != "re_emergence"
    ].groupby(group_cols, dropna=False)

    agg_rows = []
    for (method, step, mode), group in grouped:
        row = {"method": method, "step": step, "mode": mode,
               "n_seeds": len(group)}
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{col}_mean"] = round(float(values.mean()), 4)
            row[f"{col}_std"] = round(float(values.std()), 4)
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    out_csv = out_path / "iterative_combined_aggregated.csv"
    agg_df.to_csv(out_csv, index=False)
    logger.info(f"  [OK] Aggregated μ ± σ → {out_csv} "
                f"({len(agg_df)} rows across {len(seed_dirs)} seeds)")

    # Quick summary table
    logger.info(f"\n{'='*72}")
    logger.info("  ITERATIVE MULTI-SEED SUMMARY (μ ± σ)")
    logger.info(f"{'='*72}")
    for method in sorted(agg_df["method"].unique()):
        sub = agg_df[agg_df["method"] == method]
        if "mia_mean_auc_mean" not in sub.columns or "retain_acc_mean" not in sub.columns:
            continue
        final = sub[sub["step"] == sub["step"].max()]
        if final.empty:
            continue
        mia = final["mia_mean_auc_mean"].iloc[0]
        mia_s = final["mia_mean_auc_std"].iloc[0]
        ret = final["retain_acc_mean"].iloc[0]
        ret_s = final["retain_acc_std"].iloc[0]
        logger.info(
            f"  {method:<16} MIA={mia:.4f}±{mia_s:.4f}  "
            f"Retain={ret:.4f}±{ret_s:.4f}  (step {sub['step'].max()})"
        )
    logger.info("=" * 72)

    return {"aggregated_csv": str(out_csv), "rows": agg_rows}


def main() -> None:
    """CLI entry point for iterative unlearning protocol."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    parser.add_argument("--subset",            type=str, default="all",
                        choices=["all", "train", "holdout"],
                        help="Per-image subset filter (default: all)")
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
                subset=args.subset,
            )
        # After all seeds complete, aggregate into μ ± σ
        aggregate_iterative_seeds(args.out)
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
            subset=args.subset,
        )


if __name__ == "__main__":
    main()
