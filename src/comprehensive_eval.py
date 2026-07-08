"""
comprehensive_eval.py  —  Comprehensive Single-Shot Evaluation

Runs ALL methods (baselines +  SOTA) on a single forget
identity and produces:
  1. Full metric table (retain acc, test acc, forget acc, MIA AUC, time)
  2. Per-method JSON results
  3. Pareto analysis: utility vs. forgetting quality scatter
  4. Best-config results (using HP search results if available)

Usage:
    python comprehensive_eval.py \
        --csv   ../../data/dataset/sfhq_dataset.csv \
        --model ../../results/checkpoints/original_model_best.pt \
        --out   ../../results/comprehensive \
        --forget_step 0 \
        --hparam_dir  ../../results/hparam   # optional: load best configs
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from dataset import SFHQDataset, get_val_transform
from device_utils import resolve_device
from evaluate import evaluate_full
from mia import run_mia_full
from model import load_model, copy_model
from novel_variant import NOVEL_REGISTRY


# ──────────────────────────────────────────────────────────────────────────────
# Best configs from HP search (overrides defaults when search results exist)
# ──────────────────────────────────────────────────────────────────────────────

def load_best_config(hparam_dir: Optional[str], method: str) -> dict:
    """Load best config from HP search CSV if it exists; else return {}."""
    if hparam_dir is None:
        return {}
    import csv, os
    for search_type in ["grid", "random"]:
        csv_path = Path(hparam_dir) / f"{method}_{search_type}_summary.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                best = max(rows, key=lambda r: float(r.get("uf_score", 0)))
                raw_cfg = best.get("config", "{}")
                try:
                    cfg = json.loads(raw_cfg)
                    print(f"  [HP] Loaded best {method} config from {csv_path.name}")
                    return cfg
                except Exception:
                    pass
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Method configurations (defaults — overridden by HP search results)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIGS = {
    # Baselines
    "no_unlearning":  {},
    "retrain":        {"epochs": 30, "lr": 1e-3},
    "ga":             {"ga_steps": 300, "ga_lr": 1e-4},
    "srl":            {"srl_epochs": 5, "srl_lr": 1e-4},
    "ft":             {"ft_epochs": 5,  "ft_lr": 1e-4},
    # SOTA
    "ng_plus":        {"ng_steps": 400, "ng_lr_ascent": 5e-5,
                       "ng_lr_retain": 1e-4, "kl_weight": 0.5},
    "msg":            {"msg_steps": 300, "msg_lr": 1e-4, "topk_fraction": 0.2},
    "ct":             {"ct_steps": 300, "ct_lr": 1e-4,
                       "saliency_threshold_pct": 75.0, "dampen_factor": 0.1},
    # Novel variant 
    "msg_kd":         {"msg_steps": 300, "msg_lr": 1e-4,
                       "topk_fraction": 0.2, "kl_weight": 0.5},
    "adaptiformet":   {"max_steps": 600, "lr_ascent": 5e-5,
                       "lr_retain": 1e-4, "topk_fraction": 0.2,
                       "kl_weight_init": 0.1, "kl_weight_max": 0.8,
                       "mask_refresh_every": 100},  
}

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

PHASE_LABEL = {
    "no_unlearning": "P2",
    "retrain":       "P2",
    "ga":            "P2",
    "srl":           "P2",
    "ft":            "P2",
    "ng_plus":       "P3",
    "msg":           "P3",
    "ct":            "P3",
    "msg_kd":        "P4",
    "adaptiformet":  "P4",
}


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_comprehensive_eval(
    csv_path: str,
    model_path: str,
    forget_step: int = 0,
    methods: Optional[List[str]] = None,
    out_dir: str = "../results/comprehensive",
    device_str: str = "auto",
    seed: int = 42,
    skip_retrain: bool = False,
    hparam_dir: Optional[str] = None,
    mia_score_types: List[str] = ("confidence", "loss"),
) -> dict:
    torch.manual_seed(seed)
    device = resolve_device(device_str)

    print(f"\n{'='*65}")
    print("  PHASE 3 — COMPREHENSIVE SINGLE-SHOT EVALUATION")
    print(f"{'='*65}")
    print(f"  Device      : {device}")
    print(f"  Forget step : {forget_step}")
    print(f"  HP dir      : {hparam_dir or 'not set (defaults)'}")
    print(f"{'='*65}\n")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = list(DEFAULT_CONFIGS.keys())
    if skip_retrain and "retrain" in methods:
        methods = [m for m in methods if m != "retrain"]
        print("[INFO] Skipping retrain oracle")

    # Import all method functions
    from sota_methods import SOTA_REGISTRY
    from novel_variant import NOVEL_REGISTRY
    from baselines import BASELINE_REGISTRY
    METHOD_REGISTRY = {**BASELINE_REGISTRY, **SOTA_REGISTRY, **NOVEL_REGISTRY}

    original_model = load_model(model_path, device=str(device))
    all_results: Dict[str, dict] = {}

    for method_name in methods:
        if method_name not in METHOD_REGISTRY:
            print(f"[WARN] Unknown method '{method_name}', skipping")
            continue

        display = METHOD_DISPLAY.get(method_name, method_name)
        phase   = PHASE_LABEL.get(method_name, "?")
        print(f"\n{'─'*55}")
        print(f"  [{phase}] {display}")
        print(f"{'─'*55}")

        # Merge default config with HP-search best config
        cfg = {**DEFAULT_CONFIGS.get(method_name, {})}
        hp_cfg = load_best_config(hparam_dir, method_name)
        cfg.update(hp_cfg)

        t0 = time.time()
        try:
            result = METHOD_REGISTRY[method_name](
                model=original_model,
                csv_path=csv_path,
                device=device,
                forget_step=forget_step,
                seed=seed,
                **cfg,
            )
            unlearned_model = result["model"]
            method_metrics  = result["metrics"]

            # Evaluate utility
            eval_res = evaluate_full(unlearned_model, csv_path, device,
                                     forget_step=forget_step, verbose=True)

            # MIA — both attack types
            mia_results = {}
            for score_type in mia_score_types:
                mia_results[score_type] = run_mia_full(
                    unlearned_model, csv_path, device,
                    forget_step=forget_step,
                    score_type=score_type, verbose=(score_type == "confidence")
                )

            total_time = time.time() - t0

            # Save model checkpoint
            ckpt = out_path / f"{method_name}_step{forget_step}.pt"
            torch.save(unlearned_model.state_dict(), ckpt)

            all_results[method_name] = {
                "method": display,
                "phase": phase,
                "config": cfg,
                "method_metrics": method_metrics,
                "evaluation": {
                    k: {kk: vv for kk, vv in v.items()
                        if kk not in ("predictions", "labels",
                                      "confidences", "logits")}
                    for k, v in eval_res.items()
                },
                "mia": mia_results,
                "total_time_s": round(total_time, 2),
            }
            print(f"  ✓ Done in {total_time:.1f}s")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            all_results[method_name] = {"method": display, "error": str(e)}

    # ── Save all results ──────────────────────────────────────────────────────
    results_path = out_path / f"comprehensive_step{forget_step}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Full results saved → {results_path}")

    # ── Print comprehensive table ─────────────────────────────────────────────
    print_comprehensive_table(all_results, forget_step)

    # ── Save CSV for plotting ─────────────────────────────────────────────────
    _save_csv(all_results, forget_step, out_path)

    return all_results


def _save_csv(results: dict, forget_step: int, out_path: Path):
    """Save flat CSV of key metrics for all methods."""
    import csv
    rows = []
    for method_name, data in results.items():
        if "error" in data:
            continue
        ev  = data.get("evaluation", {})
        mia = data.get("mia", {}).get("confidence", {})
        rows.append({
            "method":            data.get("method", method_name),
            "phase":             data.get("phase", "?"),
            "retain_acc":        ev.get("retain", {}).get("accuracy", ""),
            "test_acc":          ev.get("test",   {}).get("accuracy", ""),
            "forget_acc":        (ev.get(f"forget_step_{forget_step}", {})
                                  or ev.get("forget", {})).get("accuracy", ""),
            "mia_retain_auc":    mia.get("retain_test_auc", ""),
            "mia_forget_auc":    mia.get("forget_test_auc", ""),
            "forget_advantage":  mia.get("forget_advantage", ""),
            "time_s":            data.get("total_time_s", ""),
        })
    csv_path = out_path / f"comprehensive_step{forget_step}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] CSV saved → {csv_path}")


def print_comprehensive_table(results: dict, forget_step: int):
    """Print a formatted results table."""
    header = (
        f"\n{'Ph':>3} {'Method':<16} {'RetainAcc':>10} {'TestAcc':>9} "
        f"{'ForgetAcc':>10} {'MIA_AUC':>9} {'F_Adv':>7} {'Time(s)':>8}"
    )
    div = "─" * 80
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE UNLEARNING EVALUATION (single-shot)")
    print("=" * 80)
    print(header)
    print(div)

    for method_name, data in results.items():
        if "error" in data:
            print(f"{'?':>3} {data['method']:<16}  ERROR: {data['error']}")
            continue
        ev  = data.get("evaluation", {})
        mia = data.get("mia", {}).get("confidence", {})
        t   = data.get("total_time_s", 0)
        ph  = data.get("phase", "?")
        display = data.get("method", method_name)

        r_acc  = ev.get("retain", {}).get("accuracy", float("nan"))
        te_acc = ev.get("test",   {}).get("accuracy", float("nan"))
        f_acc  = (ev.get(f"forget_step_{forget_step}", {})
                  or ev.get("forget", {})).get("accuracy", float("nan"))
        mia_auc = mia.get("forget_test_auc",   float("nan"))
        f_adv   = mia.get("forget_advantage",  float("nan"))

        print(
            f"{ph:>3} {display:<16} {r_acc:>10.4f} {te_acc:>9.4f} "
            f"{f_acc:>10.4f} {mia_auc:>9.4f} {f_adv:>7.4f} {t:>8.1f}"
        )

    print("=" * 80)
    print("  * Retrain Oracle = gold standard target for all methods")
    print("  † MSG-KD = novel Phase 4 variant (MSG + KL distillation)")
    print("  MIA AUC: 0.50 = perfect forgetting | 1.00 = no forgetting")
    print("  F_Adv (Forget Advantage) = |MIA_AUC - 0.50|; target: 0.00")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          type=str, required=True)
    parser.add_argument("--model",        type=str, required=True)
    parser.add_argument("--out",          type=str, default="../results/comprehensive")
    parser.add_argument("--forget_step",  type=int, default=0)
    parser.add_argument("--methods",      type=str, nargs="+", default=None)
    parser.add_argument("--hparam_dir",   type=str, default=None)
    parser.add_argument("--device",       type=str, default="auto")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--skip_retrain", action="store_true")
    args = parser.parse_args()

    run_comprehensive_eval(
        csv_path=args.csv,
        model_path=args.model,
        forget_step=args.forget_step,
        methods=args.methods,
        out_dir=args.out,
        device_str=args.device,
        seed=args.seed,
        skip_retrain=args.skip_retrain,
        hparam_dir=args.hparam_dir,
    )
