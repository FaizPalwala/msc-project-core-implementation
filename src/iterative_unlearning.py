"""
iterative_unlearning.py  —  Full Iterative Unlearning Protocol

Executes all methods over a configurable sequence of forget steps
(default: 15-20 sequential identity deletions) and records per-step
metrics for stability analysis.

Protocol
────────────────────────────────────────────────────────────────────────
  For each forget_step t in {0, 1, …, T-1}:
    1. Start from the model state AFTER step t-1 (cumulative unlearning).
       The original model is used only at step 0.
    2. Run the unlearning method on the identity assigned to forget_step t.
    3. Evaluate:
         - retain_acc     : accuracy on all retained identities so far
         - test_acc       : accuracy on the held-out test split
         - forget_acc     : accuracy on the identities forgotten so far
         - mia_auc        : MIA AUC on the latest forget step
         - forget_adv     : |MIA_AUC - 0.5|
         - model_drift    : L2 distance of weights from original model
         - step_time_s    : wall-clock time for this step
    4. Save the model checkpoint and append metrics to a JSONL log.

Two unlearning modes:
  "cumulative"  (default)  model accumulates all deletions step-by-step
  "fresh"                  each step starts from the original model
                           (useful as an ablation / oracle)

Output
──────────  results/iterative/
              {method}_iterative_log.jsonl    one JSON line per step
              {method}_iterative_summary.csv  tidy CSV for plotting
              {method}_step{t}.pt             model checkpoint
              iterative_combined.csv          all methods, all steps
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from evaluate import evaluate_full
from mia import run_mia_full
from model import load_model, copy_model


# ──────────────────────────────────────────────────────────────────────────────
# Weight-distance helper (tracks model drift from original)
# ──────────────────────────────────────────────────────────────────────────────

def _weight_l2_distance(model_a: torch.nn.Module,
                        model_b: torch.nn.Module) -> float:
    dist = 0.0
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    for k in sd_a:
        if sd_a[k].dtype.is_floating_point:
            dist += (sd_a[k].float() - sd_b[k].float()).pow(2).sum().item()
    return float(dist ** 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Per-step evaluation (combines utility + MIA)
# ──────────────────────────────────────────────────────────────────────────────

def _step_eval(model, original_model, csv_path, device,
               forget_step: int, verbose: bool = False) -> dict:
    eval_res = evaluate_full(model, csv_path, device,
                             forget_step=forget_step, verbose=verbose)
    mia_res  = run_mia_full(model, csv_path, device,
                            forget_step=forget_step,
                            score_type="confidence", verbose=False)

    r_acc  = eval_res.get("retain",   {}).get("accuracy", float("nan"))
    te_acc = eval_res.get("test",     {}).get("accuracy", float("nan"))
    f_acc  = (eval_res.get(f"forget_step_{forget_step}", {})
              or eval_res.get("forget", {})).get("accuracy", float("nan"))
    mia    = mia_res.get("forget_test_auc", 0.5)
    fadv   = mia_res.get("forget_advantage", 0.0)
    drift  = _weight_l2_distance(model, original_model)

    return {
        "retain_acc":       round(r_acc,  4),
        "test_acc":         round(te_acc, 4),
        "forget_acc":       round(f_acc,  4),
        "mia_auc":          round(mia,    4),
        "forget_advantage": round(fadv,   4),
        "model_drift":      round(drift,  4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build method registry (merge all phases)
# ──────────────────────────────────────────────────────────────────────────────

def _get_registry():
    reg = {}
    try:
        from baselines import BASELINE_REGISTRY
        reg.update(BASELINE_REGISTRY)
    except ImportError:
        pass
    try:
        from methods_sota import SOTA_REGISTRY
        reg.update(SOTA_REGISTRY)
    except ImportError:
        pass
    try:
        from novel_variant import NOVEL_REGISTRY
        reg.update(NOVEL_REGISTRY)
    except ImportError:
        pass
    return reg


# ──────────────────────────────────────────────────────────────────────────────
# Single-method iterative run
# ──────────────────────────────────────────────────────────────────────────────

def run_iterative(
    method_name: str,
    method_cfg: dict,
    csv_path: str,
    model_path: str,
    n_steps: int = 20,
    mode: str = "cumulative",          # "cumulative" | "fresh"
    out_dir: str = "../results/iterative",
    device_str: str = "auto",
    seed: int = 42,
    save_checkpoints: bool = True,
    checkpoint_every: int = 5,
    verbose_eval: bool = False,
) -> List[dict]:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_str == "auto" else torch.device(device_str))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / f"{method_name}_iterative_log.jsonl"
    csv_path_out = out_path / f"{method_name}_iterative_summary.csv"

    registry = _get_registry()
    if method_name not in registry:
        raise ValueError(f"Unknown method '{method_name}'. Available: {list(registry)}")

    original_model = load_model(model_path, device=str(device))
    current_model  = copy_model(original_model, device)

    print(f"\n{'─'*65}")
    print(f"  Iterative Unlearning: {method_name.upper()} | {n_steps} steps | {mode}")
    print(f"{'─'*65}")

    # Baseline: evaluate original model before any unlearning
    print(f"\n  [Step 0 / baseline] Evaluating original model…")
    baseline = _step_eval(original_model, original_model, csv_path,
                          device, forget_step=0, verbose=False)
    baseline.update({"step": 0, "method": method_name, "mode": mode,
                     "step_time_s": 0.0, "cumulative_time_s": 0.0,
                     "is_baseline": True})
    print(f"    retain={baseline['retain_acc']:.4f} | "
          f"mia={baseline['mia_auc']:.4f} | "
          f"drift={baseline['model_drift']:.2f}")

    all_records = [baseline]
    cumulative_time = 0.0

    for step in range(n_steps):
        print(f"\n  [Step {step+1}/{n_steps}] {method_name} | forget_step={step}")
        t0 = time.time()

        # Fresh mode: always start from original
        start_model = (original_model if mode == "fresh"
                       else current_model)

        try:
            result = registry[method_name](
                model=start_model,
                csv_path=csv_path,
                device=device,
                forget_step=step,
                seed=seed + step,     # different seed per step
                **method_cfg,
            )
            current_model = result["model"]
            step_time     = time.time() - t0
            cumulative_time += step_time

            metrics = _step_eval(current_model, original_model, csv_path,
                                 device, forget_step=step,
                                 verbose=verbose_eval)
            record = {
                "step": step + 1,
                "forget_step_idx": step,
                "method": method_name,
                "mode": mode,
                "step_time_s": round(step_time, 2),
                "cumulative_time_s": round(cumulative_time, 2),
                "is_baseline": False,
                **metrics,
                "method_metrics": {k: v for k, v in
                                   result.get("metrics", {}).items()
                                   if k != "history"},   # omit verbose history
            }

            print(f"    retain={metrics['retain_acc']:.4f} | "
                  f"mia={metrics['mia_auc']:.4f} | "
                  f"adv={metrics['forget_advantage']:.4f} | "
                  f"drift={metrics['model_drift']:.2f} | "
                  f"t={step_time:.1f}s")

            # Save checkpoint
            if save_checkpoints and (step + 1) % checkpoint_every == 0:
                ckpt = out_path / f"{method_name}_step{step+1}.pt"
                torch.save(current_model.state_dict(), ckpt)

        except Exception as e:
            print(f"  [WARN] Step {step+1} failed: {e}")
            record = {
                "step": step + 1, "method": method_name, "mode": mode,
                "error": str(e), "step_time_s": time.time() - t0,
            }

        all_records.append(record)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    _write_csv(all_records, csv_path_out)
    print(f"\n  ✓ {method_name} done. Log → {log_path}")
    return all_records


def _write_csv(records: List[dict], path: Path):
    import csv
    flat_cols = ["step", "method", "mode", "retain_acc", "test_acc",
                 "forget_acc", "mia_auc", "forget_advantage",
                 "model_drift", "step_time_s", "cumulative_time_s",
                 "is_baseline", "error"]
    rows = []
    for r in records:
        row = {c: r.get(c, "") for c in flat_cols}
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_cols)
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-method runner
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_ITER_CONFIGS = {
    "no_unlearning":  {},
    "ga":             {"ga_steps": 300, "ga_lr": 1e-4},
    "srl":            {"srl_epochs": 5,  "srl_lr": 1e-4},
    "ft":             {"ft_epochs": 5,   "ft_lr": 1e-4},
    "ng_plus":        {"ng_steps": 400,  "ng_lr_ascent": 5e-5,
                       "ng_lr_retain": 1e-4, "kl_weight": 0.5},
    "msg":            {"msg_steps": 300, "msg_lr": 1e-4,
                       "topk_fraction": 0.2},
    "msg_kd":         {"msg_steps": 300, "msg_lr": 1e-4,
                       "topk_fraction": 0.2, "kl_weight": 0.5},
    "ct":             {"ct_steps": 300,  "ct_lr": 1e-4,
                       "saliency_threshold_pct": 75.0, "dampen_factor": 0.1},
    "adaptiformet":   {"max_steps": 600, "topk_fraction": 0.2,
                       "lr_ascent": 5e-5, "lr_retain": 1e-4,
                       "kl_weight_init": 0.1, "kl_weight_max": 0.8},
}


def run_all_iterative(
    csv_path: str,
    model_path: str,
    methods: Optional[List[str]] = None,
    n_steps: int = 20,
    mode: str = "cumulative",
    out_dir: str = "../results/iterative",
    device_str: str = "auto",
    seed: int = 42,
    method_overrides: Optional[Dict[str, dict]] = None,
    checkpoint_every: int = 5,
) -> dict:
    if methods is None:
        methods = list(DEFAULT_ITER_CONFIGS.keys())

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for method in methods:
        cfg = dict(DEFAULT_ITER_CONFIGS.get(method, {}))
        if method_overrides and method in method_overrides:
            cfg.update(method_overrides[method])

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
        )
        all_results[method] = records

    # ── Combined CSV (all methods, all steps) ─────────────────────────────────
    combined_rows = []
    for method, records in all_results.items():
        combined_rows.extend(records)
    _write_csv(combined_rows, out_path / "iterative_combined.csv")
    print(f"\n[OK] Combined CSV → {out_path}/iterative_combined.csv")

    # ── Print stability summary table ─────────────────────────────────────────
    _print_stability_summary(all_results)
    return all_results


def _print_stability_summary(all_results: dict):
    print(f"\n{'='*80}")
    print("  ITERATIVE UNLEARNING STABILITY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<16} {'Step-5 MIA':>10} {'Step-10 MIA':>12} "
          f"{'Step-20 MIA':>12} {'Δ Retain':>10} {'Total Time':>12}")
    print("─" * 74)

    for method, records in all_results.items():
        data = [r for r in records if not r.get("is_baseline") and "error" not in r]
        if not data:
            continue
        steps = [r["step"] for r in data]
        mias  = [r["mia_auc"] for r in data]
        rets  = [r["retain_acc"] for r in data]

        def _at(n):
            idx = next((i for i, s in enumerate(steps) if s >= n), -1)
            return f"{mias[idx]:.4f}" if idx >= 0 else "—"

        d_ret = rets[-1] - rets[0] if len(rets) >= 2 else 0.0
        ttl   = records[-1].get("cumulative_time_s", 0)
        print(f"{method:<16} {_at(5):>10} {_at(10):>12} {_at(20):>12} "
              f"{d_ret:>+10.4f} {ttl:>12.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, required=True)
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--out",        type=str, default="../results/iterative")
    parser.add_argument("--methods",    type=str, nargs="+", default=None)
    parser.add_argument("--n_steps",    type=int, default=20)
    parser.add_argument("--mode",       type=str, default="cumulative",
                        choices=["cumulative", "fresh"])
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_all_iterative(
        csv_path=args.csv,
        model_path=args.model,
        methods=args.methods,
        n_steps=args.n_steps,
        mode=args.mode,
        out_dir=args.out,
        device_str=args.device,
        seed=args.seed,
    )
