"""
hparam_search.py  —  Hyperparameter Sensitivity Analysis

Implements two search strategies:
  GridSearch   — exhaustive Cartesian product over discrete param grids
  RandomSearch — uniform/log-uniform sampling over continuous param ranges

After each trial the model is evaluated on retain and forget splits
and a scalar utility-forgetting score (UF-score) is computed:

    UF = w_r * retain_acc  +  w_f * max(0, 1 − 2·MIA_AUC)  −  w_p * time

    where MIA_AUC is the signed per-identity mean AUC:
    (0.0 = identity erased — retrain oracle sits here,
     0.5 = no signal — the no-unlearning control sits here,
     >0.5 = leak — forget still distinguishable)

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

from dataset import VirtualIdentityDataset, get_val_transform
from device_utils import resolve_device
from evaluate import evaluate_full
from mia import run_mia_per_identity
from model import load_model
from interfaces import validate_unlearning_result, prepare_method_call


import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Erasure gate (option-1 guard, 2026-08)
# ──────────────────────────────────────────────────────────────────────────────
# The final run exposed a false optimum: a config with tiny lr_ascent
# collapsed confidence on forget images → MIA AUC ~0.026 (UF's forget term
# read it as "erased") while forget acc stayed 0.82 — output suppression,
# not erasure.  Any config that does not actually erase is rejected
# regardless of UF score:
#   forget_id_acc > ERASURE_FORGET_ACC_MAX  → rejected
# Threshold aligns with the pre-registered target in the Evaluation
# Protocol (forget acc ≤ 0.15).
# NOTE (2026-08-20, f2): ERASURE_PROBE_ACC_MAX is RETIRED from the gate —
# probe-identity-on-forget saturates at ~1.0 for every method INCLUDING
# the retrain oracle at full scale (it measures backbone feature
# separability, which survives head-level unlearning).  A probe arm
# rejected 100% of trials and reverted everything to defaults.  Probe
# remains a diagnostic in the trial dict; forget acc is the gate signal.
ERASURE_FORGET_ACC_MAX = 0.15
ERASURE_PROBE_ACC_MAX = 0.30  # diagnostic reference only — NOT used by the gate


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
    "adaptiforget": {
        "max_steps":              [400, 600],
        "lr_ascent":              [1e-5, 5e-5],
        "lr_retain":              [5e-5, 1e-4],
        "topk_fraction":          [0.1, 0.2, 0.3],
        "kl_weight_init":         [0.0, 0.1],
        "kl_weight_max":          [0.5, 0.8],
        "mask_refresh_every":     [50, 100],
    },
    "budget_scaled": {
        "base_steps":             [150, 300, 600],
        "budget_exponent":        [0.25, 0.5, 0.75],
        "budget_ref_images":      [150, 300],
        "ga_lr":                  [5e-5, 1e-4],
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
    },
    "msg_kd": {
        "msg_steps":              ("int",   100, 600),
        "msg_lr":                 ("log",   1e-5, 1e-3),
        "topk_fraction":          ("float", 0.05, 0.5),
        "kl_weight":              ("float", 0.0, 1.0),
    },
    "adaptiforget": {
        "max_steps":              ("int",   200, 800),
        "lr_ascent":              ("log",   1e-6, 1e-4),
        "lr_retain":              ("log",   1e-6, 1e-3),
        "topk_fraction":          ("float", 0.05, 0.5),
        "kl_weight_init":         ("float", 0.0, 0.3),
        "kl_weight_max":          ("float", 0.3, 1.0),
        "mask_refresh_every":     ("int",   25, 150),
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
    mia_auc: float,        # SIGNED MIA AUC — 0.0 = erased, 0.5 = no signal, 1.0 = leak
    time_s: float,
    forget_id_acc: float | None = None,   # v2: erasure-conditioned term
    w_retain: float = 0.5,
    w_forget: float = 0.4,
    w_time: float = 0.1,
    time_budget_s: float = 300.0,   # normalise time against budget
) -> float:
    """
    Composite Utility-Forgetting score ∈ [0, 1].
      retain_acc  : higher → better utility
      mia_auc     : LOWER → better forgetting.  Empirically in this
                    framework the retrain oracle (identity erased) sits at
                    AUC ≈ 0.00 and the no-unlearning control (nothing
                    happened) at AUC ≈ 0.50, so the forget term rewards
                    AUC BELOW 0.5 and zeroes at/above 0.5 (leak).
      time_s      : lower → faster
      forget_id_acc (v2): the ERASURE-CONDITIONED bracket.  The final
                    run exposed a false optimum: a config with tiny
                    lr_ascent collapsed confidence on forget images →
                    MIA AUC ~0.026 while forget acc stayed 0.82 and
                    probe-identity 1.0 — output suppression, not
                    erasure.  The MIA term alone cannot tell
                    "confidence collapse" from "identity erased", so
                    when forget_id_acc is provided the forget term earns
                    credit ONLY if the config actually erases
                    (forget_id_acc ≤ ERASURE_FORGET_ACC_MAX, matching
                    the pre-registered target).  Suppressors get a
                    forget term of exactly 0 and can never out-rank a
                    genuine eraser.

    UF = w_r * retain_acc
       + w_f * max(0, 1 − 2·mia_auc) * [forget_id_acc ≤ 0.15]   # v2
       − w_t * min(time_s / time_budget_s, 1)
    """
    forget_score = max(0.0, 1.0 - 2.0 * mia_auc)
    if forget_id_acc is not None and forget_id_acc > ERASURE_FORGET_ACC_MAX:
        forget_score = 0.0
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
    trial_idx: int,
) -> dict:
    """Run one hyperparameter trial and return evaluation results."""
    from sota_methods import SOTA_REGISTRY
    from novel_variant import NOVEL_REGISTRY
    from dataset import infer_identity_classes
    try:
        from baselines import BASELINE_REGISTRY
        METHOD_REGISTRY = {**BASELINE_REGISTRY, **SOTA_REGISTRY, **NOVEL_REGISTRY}
    except ImportError:
        METHOD_REGISTRY = {**SOTA_REGISTRY, **NOVEL_REGISTRY}

    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}")

    logger.info(f"\n  Trial {trial_idx:>3} | {method_name} | {cfg}")
    t0 = time.time()

    n_id_classes = infer_identity_classes(csv_path)
    result = METHOD_REGISTRY[method_name](
        model=original_model,
        csv_path=csv_path,
        device=device,
        # pins subset="train" (never see holdout) + the CSV-inferred class
        # count — SRL/budget_scaled default to the FULL 750-class head and
        # would relabel into the wrong space otherwise (same bug family as
        # the feasibility CUDA-assert; silent at full scale).
        **prepare_method_call(cfg, identity_classes=n_id_classes),
    )
    validate_unlearning_result(result, method_name)
    unlearned = result["model"]
    elapsed = time.time() - t0

    # HP evaluation is a utility probe — use holdout images (generalisation)
    eval_res = evaluate_full(unlearned, csv_path, device, verbose=False, subset="holdout")
    per_id = run_mia_per_identity(unlearned, csv_path, device, head="identity",
                                  subset="holdout")

    retain_acc = eval_res.get("retain", {}).get("identity", {}).get("accuracy", 0.0)
    forget_id_acc = eval_res.get("forget", {}).get("identity", {}).get("accuracy", 0.0)
    mia_auc = per_id.get("mean_auc", 0.5)
    f_adv = abs(mia_auc - 0.5)   # reported for diagnostics; NOT used in UF
    # v2: erasure-conditioned forget term — the MIA term earns credit only
    # when the config actually erases (forget acc ≤ 0.15).  A suppressor
    # (confidence collapse, forget acc ~0.82) gets forget credit 0.
    score = uf_score(retain_acc, mia_auc, elapsed, forget_id_acc=forget_id_acc)

    # Identity probe on the FORGET split — the erasure-gate signal.  The
    # final run's false optimum suppressed confidence (MIA AUC ~0.026)
    # while probe-identity stayed 1.0: features still fully encode the
    # identity.  Probe accuracy > 0.30 ⇒ features intact ⇒ not erased.
    probe_forget = None
    try:
        from probes import probe_identity
        _pf = probe_identity(unlearned, csv_path, device, split="forget")
        probe_forget = (_pf or {}).get("accuracy")
    except Exception:
        probe_forget = None

    return {
        "trial": trial_idx,
        "method": method_name,
        "config": cfg,
        "retain_id_acc":      round(retain_acc, 4),
        "retain_age_acc":     round(eval_res.get("retain", {}).get("age", {}).get("accuracy", 0.0), 4),
        "forget_id_acc":      round(eval_res.get("forget", {}).get("identity", {}).get("accuracy", 0.0), 4),
        "mia_mean_auc":       per_id.get("mean_auc", 0.5),
        "mia_max_auc":        per_id.get("max_auc", 0.5),
        "forget_advantage":   round(f_adv, 4),
        "fraction_leaked":    per_id.get("fraction_leaked", 0.0),
        "unlearning_time_s":  round(elapsed, 2),
        "uf_score":           round(score, 4),
        "probe_identity_forget_acc": (round(float(probe_forget), 4)
                                      if probe_forget is not None else None),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main search loop
# ──────────────────────────────────────────────────────────────────────────────

def run_search(
    method_name: str,
    csv_path: str,
    model_path: str,
    search_type: str = "grid",
    n_random_trials: int = 30,
    out_dir: str = "../results/hparam",
    device_str: str = "auto",
    seed: int = 42,
) -> list[dict]:
    device = resolve_device(device_str)
    logger.info(f"\n[HPSearch] Method={method_name} | Type={search_type} | Device={device}")

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

    logger.info(f"[HPSearch] {len(configs)} trials to run")

    all_results = []
    n_gated = 0
    for i, cfg in enumerate(configs):
        try:
            trial = run_trial(method_name, cfg, original_model,
                              csv_path, device, trial_idx=i+1)
            # ── Erasure gate (option-1 guard, commit 2026-08) ──────────────
            # The final run exposed a false optimum: a config with tiny
            # lr_ascent collapsed confidence on forget images → MIA AUC
            # dropped to ~0.026 (UF's forget term read it as "erased")
            # while forget acc stayed 0.82 — output suppression, not
            # erasure.  Reject any config that does not actually erase:
            # forget_id_acc > 0.15 (ERASURE_FORGET_ACC_MAX, the
            # pre-registered target).
            # NOTE (2026-08-20, f2): the probe arm was REMOVED.  At full
            # scale probe_identity_forget_acc is saturated at ~1.0 for
            # EVERYTHING — the retrain oracle (forget 0.0) also reads
            # probe 1.0, because the probe measures backbone feature
            # separability, which survives head-level unlearning.  A
            # probe threshold of 0.30 rejected 100% of trials (even
            # genuine erasers) and silently reverted every method to
            # YAML defaults.  Probe stays in the trial dict as a
            # DIAGNOSTIC only; forget acc is the gate signal.
            # Gated trials are still written to the JSONL (audit trail)
            # but excluded from ranking/best-config export; if NO trial
            # passes, no best config is exported and the downstream
            # stages fall back to the YAML default (which, for
            # AdaptiForget, erases correctly: forget acc 0.0124).
            probe_f = trial.get("probe_identity_forget_acc")
            if trial.get("forget_id_acc", 1.0) > ERASURE_FORGET_ACC_MAX:
                n_gated += 1
                trial["erasure_gate"] = "REJECTED"
                with open(out_jsonl, "a") as f:
                    f.write(json.dumps(trial) + "\n")
                logger.warning(
                    f"  [GATE] Trial {i+1} REJECTED: forget_acc={trial.get('forget_id_acc'):.4f} "
                    f"(max {ERASURE_FORGET_ACC_MAX}) — suppression, not erasure")
                continue
            trial["erasure_gate"] = "PASS"
            all_results.append(trial)
            # Write to JSONL incrementally
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(trial) + "\n")
            logger.info(f"    UF={trial['uf_score']:.4f} | retain={trial['retain_id_acc']:.4f} "
                  f"| MIA_AUC={trial['mia_mean_auc']:.4f} "
                  f"| t={trial['unlearning_time_s']:.1f}s")
        except Exception as e:
            logger.warning(f"  [WARN] Trial {i+1} failed: {e}")
            continue
    if n_gated:
        logger.warning(f"[HPSearch] {n_gated}/{len(configs)} trials REJECTED by the erasure "
                       f"gate (forget_acc>{ERASURE_FORGET_ACC_MAX})")

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
        logger.info(f"\n[HPSearch] Top-5 configs by UF score:")
        logger.info(f"{'Rank':>5} {'UF':>7} {'RetIdAcc':>10} {'MIA-AUC':>9} {'Time':>8}")
        logger.info("-" * 45)
        for rank, r in enumerate(sorted_results[:5], 1):
            logger.info(f"{rank:>5} {r['uf_score']:>7.4f} {r['retain_id_acc']:>10.4f} "
                  f"{r['mia_mean_auc']:>9.4f} {r['unlearning_time_s']:>8.1f}s")
        logger.info(f"\n  Best config: {sorted_results[0]['config']}")
        logger.info(f"  Results saved → {out_csv}")

        # ── Export best config for downstream stages ─────────────────────
        # Per-method file {method}_best_config.json so hparam jobs for
        # different methods can run IN PARALLEL without racing on a shared
        # file.  config_loader.load_method_configs(best_configs_path=<dir>)
        # globs *_best_config.json and merges each method's tuned values
        # over the YAML defaults.
        best_json = out_path / f"{method_name}_best_config.json"
        with open(best_json, "w") as fh:
            json.dump(sorted_results[0]["config"], fh, indent=2)
        logger.info(f"  Best config exported → {best_json}")

    else:
        # ── No trial passed the erasure gate: fall back to YAML default ─
        # Every config either failed or was gated as suppression.  Delete
        # any STALE best-config from a previous run so config_loader's
        # glob cannot resurrect a rejected config downstream — the method
        # then runs at its YAML default (documented, auditable fallback).
        stale = out_path / f"{method_name}_best_config.json"
        if stale.exists():
            stale.unlink()
            logger.warning(f"[HPSearch] {method_name}: NO config passed the erasure gate — "
                           f"removed stale {stale.name}; downstream runs at YAML default")
        else:
            logger.warning(f"[HPSearch] {method_name}: NO config passed the erasure gate — "
                           f"no best config written; downstream runs at YAML default")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, required=True)
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--method",     type=str, default="ng_plus")
    parser.add_argument("--search",     type=str, default="grid",
                        choices=["grid", "random"])
    parser.add_argument("--n_random",   type=int, default=30)
    parser.add_argument("--out",        type=str, default="../results/hparam")
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_search(
        method_name=args.method,
        csv_path=args.csv,
        model_path=args.model,
        search_type=args.search,
        n_random_trials=args.n_random,
        out_dir=args.out,
        device_str=args.device,
        seed=args.seed,
    )
