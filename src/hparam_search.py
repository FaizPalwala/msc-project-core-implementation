"""
hparam_search.py  —  Hyperparameter Sensitivity Analysis

Implements two search strategies:
  GridSearch   — exhaustive Cartesian product over discrete param grids
  RandomSearch — uniform/log-uniform sampling over continuous param ranges

After each trial the model is evaluated on retain, test, and forget splits
and a scalar utility-forgetting score (UF-score) is computed:

    UF = w_r * retain_acc  +  w_f * (1 - forget_advantage)  -  w_p * time

    where forget_advantage = |MIA_AUC - 0.5|
    (0 = perfectly forgotten, 0.5 = no forgetting at all)

The search saves every trial result to a JSONL file so that runs can be
resumed, and outputs a summary CSV for easy analysis.

Usage
-----
  python hparam_search.py \
    --csv    ../../data/dataset/sfhq_dataset.csv \
    --model  ../../results/checkpoints/original_model_best.pt \
    --method ng_plus \
    --search grid \
    --out    ../../results/hparam \
    --forget_step 0
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from dataset import SFHQDataset, get_val_transform
from evaluate import evaluate_full
from mia import run_mia_full
from model import load_model


# ──────────────────────────────────────────────────────────────────────────────
# Default search grids / ranges for each method
# ──────────────────────────────────────────────────────────────────────────────

GRIDS = {
    # SOTA methods
    "ng_plus": {
        "ng_steps":               [200, 400, 600],
        "ng_lr_ascent":           [1e-5, 5e-5, 1e-4],
        "ng_lr_retain":           [5e-5, 1e-4, 5e-4],
        "retain_steps_per_ascent":[1, 2],
        "kl_weight":              [0.0, 0.5],
    },
    "msg": {
        "msg_steps":              [200, 400],
        "msg_lr":                 [5e-5, 1e-4, 5e-4],
        "topk_fraction":          [0.1, 0.2, 0.3],
        "kl_weight":              [0.0, 0.5],
        "retain_reg_every":       [1, 2],
    },
    "ct": {
        "ct_steps":               [200, 400],
        "ct_lr":                  [5e-5, 1e-4],
        "saliency_threshold_pct": [50.0, 75.0, 90.0],
        "dampen_factor":          [0.05, 0.1, 0.2],
        "kl_weight":              [0.0, 0.3],
    },
    # Novel variants of SOTA methods
        "msg_kd": {
        "msg_steps":              [200, 400],
        "msg_lr":                 [5e-5, 1e-4],
        "topk_fraction":          [0.1, 0.2, 0.3],
        "kl_weight":              [0.3, 0.5, 0.8],
    },
    # Baselines — included so we can compare sensitivity
    "ga": {
        "ga_steps": [100, 200, 400],
        "ga_lr":    [5e-5, 1e-4, 5e-4],
    },
    "srl": {
        "srl_epochs": [3, 5, 10],
        "srl_lr":     [5e-5, 1e-4, 5e-4],
    },
    "ft": {
        "ft_epochs": [3, 5, 10],
        "ft_lr":     [5e-5, 1e-4, 5e-4],
    },
}

RANDOM_RANGES = {
    "ng_plus": {
        "ng_steps":               ("int",   100, 800),
        "ng_lr_ascent":           ("log",   1e-6, 1e-3),
        "ng_lr_retain":           ("log",   1e-6, 1e-2),
        "retain_steps_per_ascent":("int",   1, 4),
        "kl_weight":              ("float", 0.0, 1.0),
    },
    "msg": {
        "msg_steps":              ("int",   100, 600),
        "msg_lr":                 ("log",   1e-5, 1e-3),
        "topk_fraction":          ("float", 0.05, 0.5),
        "kl_weight":              ("float", 0.0, 1.0),
    },
    "ct": {
        "ct_steps":               ("int",   100, 600),
        "ct_lr":                  ("log",   1e-5, 1e-3),
        "saliency_threshold_pct": ("float", 25.0, 95.0),
        "dampen_factor":          ("log",   0.01, 0.5),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# UF-Score (Utility–Forgetting composite)
# ──────────────────────────────────────────────────────────────────────────────

def uf_score(
    retain_acc: float,
    forget_advantage: float,   # |MIA_AUC - 0.5|, lower is better
    time_s: float,
    w_retain: float = 0.5,
    w_forget: float = 0.4,
    w_time: float = 0.1,
    time_budget_s: float = 300.0,   # normalise time against budget
) -> float:
    """
    Composite Utility-Forgetting score ∈ [0, 1].
      retain_acc        : higher → better utility
      forget_advantage  : lower → better forgetting
      time_s            : lower → faster

    UF = w_r * retain_acc
       + w_f * (1 - forget_advantage / 0.5)   # normalised: 0 adv → 1.0 score
       - w_t * min(time_s / time_budget_s, 1)
    """
    forget_score = max(0.0, 1.0 - forget_advantage / 0.5)
    time_score   = min(1.0, time_s / time_budget_s)
    return (w_retain * retain_acc
            + w_forget * forget_score
            - w_time   * time_score)


# ──────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sample_random_config(ranges: dict, seed: int = None) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    cfg = {}
    for k, spec in ranges.items():
        kind = spec[0]
        if kind == "int":
            cfg[k] = random.randint(spec[1], spec[2])
        elif kind == "float":
            cfg[k] = random.uniform(spec[1], spec[2])
        elif kind == "log":
            cfg[k] = float(np.exp(np.random.uniform(
                np.log(spec[1]), np.log(spec[2]))))
    return cfg


def _grid_configs(grid: dict) -> List[dict]:
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ──────────────────────────────────────────────────────────────────────────────
# Single trial runner
# ──────────────────────────────────────────────────────────────────────────────

def run_trial(
    method_name: str,
    cfg: dict,
    original_model,
    csv_path: str,
    device: torch.device,
    forget_step: int,
    trial_idx: int,
) -> dict:
    """Run one hyperparameter trial and return evaluation results."""
    from sota_methods import SOTA_REGISTRY
    try:
        from baselines import BASELINE_REGISTRY
        METHOD_REGISTRY = {**BASELINE_REGISTRY, **SOTA_REGISTRY}
    except ImportError:
        METHOD_REGISTRY = SOTA_REGISTRY

    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}")

    print(f"\n  Trial {trial_idx:>3} | {method_name} | {cfg}")
    t0 = time.time()

    result = METHOD_REGISTRY[method_name](
        model=original_model,
        csv_path=csv_path,
        device=device,
        forget_step=forget_step,
        **cfg,
    )
    unlearned = result["model"]
    elapsed   = time.time() - t0

    # Evaluate
    eval_res = evaluate_full(unlearned, csv_path, device,
                             forget_step=forget_step, verbose=False)
    mia_res  = run_mia_full(unlearned, csv_path, device,
                            forget_step=forget_step,
                            score_type="confidence", verbose=False)

    retain_acc = eval_res.get("retain", {}).get("accuracy", 0.0)
    f_adv      = mia_res.get("forget_advantage", 0.5)
    score      = uf_score(retain_acc, f_adv, elapsed)

    return {
        "trial": trial_idx,
        "method": method_name,
        "config": cfg,
        "retain_acc":        round(retain_acc, 4),
        "test_acc":          round(eval_res.get("test",   {}).get("accuracy", 0.0), 4),
        "forget_acc":        round((eval_res.get(f"forget_step_{forget_step}", {})
                                    or eval_res.get("forget", {}))
                                    .get("accuracy", 0.0), 4),
        "mia_forget_auc":    mia_res.get("forget_test_auc", 0.5),
        "forget_advantage":  round(f_adv, 4),
        "unlearning_time_s": round(elapsed, 2),
        "uf_score":          round(score, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main search loop
# ──────────────────────────────────────────────────────────────────────────────

def run_search(
    method_name: str,
    csv_path: str,
    model_path: str,
    forget_step: int = 0,
    search_type: str = "grid",     # "grid" or "random"
    n_random_trials: int = 30,
    out_dir: str = "../results/hparam",
    device_str: str = "auto",
    seed: int = 42,
) -> List[dict]:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_str == "auto" else torch.device(device_str))
    print(f"\n[HPSearch] Method={method_name} | Type={search_type} | Device={device}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_path / f"{method_name}_{search_type}_trials.jsonl"
    out_csv   = out_path / f"{method_name}_{search_type}_summary.csv"

    original_model = load_model(model_path, device=str(device))

    # Build trial configs
    if search_type == "grid":
        grid = GRIDS.get(method_name)
        if grid is None:
            raise ValueError(f"No grid defined for '{method_name}'")
        configs = _grid_configs(grid)
    else:
        ranges = RANDOM_RANGES.get(method_name) or GRIDS.get(method_name)
        if ranges is None:
            raise ValueError(f"No ranges defined for '{method_name}'")
        # If ranges is a grid dict (discrete), sample from it
        if all(isinstance(v, list) for v in ranges.values()):
            configs = [_sample_random_config(
                {k: ("choice", v) for k, v in ranges.items()}, seed=seed+i)
                for i in range(n_random_trials)]
            # Convert choice dicts
            configs = []
            for i in range(n_random_trials):
                random.seed(seed + i)
                configs.append({k: random.choice(v) for k, v in ranges.items()})
        else:
            configs = [_sample_random_config(ranges, seed=seed+i)
                       for i in range(n_random_trials)]

    print(f"[HPSearch] {len(configs)} trials to run")

    all_results = []
    for i, cfg in enumerate(configs):
        try:
            trial = run_trial(method_name, cfg, original_model,
                              csv_path, device, forget_step, trial_idx=i+1)
            all_results.append(trial)
            # Write to JSONL incrementally
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(trial) + "\n")
            print(f"    UF={trial['uf_score']:.4f} | retain={trial['retain_acc']:.4f} "
                  f"| MIA_AUC={trial['mia_forget_auc']:.4f} "
                  f"| t={trial['unlearning_time_s']:.1f}s")
        except Exception as e:
            print(f"  [WARN] Trial {i+1} failed: {e}")
            continue

    # Save summary CSV
    if all_results:
        import csv
        keys = list(all_results[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in all_results:
                flat = {k: (json.dumps(v) if isinstance(v, dict) else v)
                        for k, v in row.items()}
                writer.writerow(flat)

        # Print top-5
        sorted_results = sorted(all_results, key=lambda x: x["uf_score"], reverse=True)
        print(f"\n[HPSearch] Top-5 configs by UF score:")
        print(f"{'Rank':>5} {'UF':>7} {'RetainAcc':>10} {'MIA_AUC':>9} {'Time':>8}")
        print("-" * 45)
        for rank, r in enumerate(sorted_results[:5], 1):
            print(f"{rank:>5} {r['uf_score']:>7.4f} {r['retain_acc']:>10.4f} "
                  f"{r['mia_forget_auc']:>9.4f} {r['unlearning_time_s']:>8.1f}s")
        print(f"\n  Best config: {sorted_results[0]['config']}")
        print(f"  Results saved → {out_csv}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, required=True)
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--method",     type=str, default="ng_plus")
    parser.add_argument("--search",     type=str, default="grid",
                        choices=["grid", "random"])
    parser.add_argument("--n_random",   type=int, default=30)
    parser.add_argument("--out",        type=str, default="../results/hparam")
    parser.add_argument("--forget_step",type=int, default=0)
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_search(
        method_name=args.method,
        csv_path=args.csv,
        model_path=args.model,
        forget_step=args.forget_step,
        search_type=args.search,
        n_random_trials=args.n_random,
        out_dir=args.out,
        device_str=args.device,
        seed=args.seed,
    )
