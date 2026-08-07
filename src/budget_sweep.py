"""
budget_sweep.py — Protocol C: budget sweep (the cleanest causal difficulty test).

Dynamic by design: reads protocol_c_methods.json (the 1-best-per-category
selection from single_shot_best) and sweeps each picked method's unlearning
budget over a grid, measuring per-bin behavioral distance to the Protocol B
per-bin oracles.  The budget at which each bin reaches its oracle IS the
bin's difficulty: high-bin needing more budget = over-learned = harder to
forget (H1, FaLW-grounded).

Budget semantics (imbalanced has no schedule axis — "budget" = method
internal unlearning iterations):
    ga→ga_steps  srl→srl_epochs  ft→ft_epochs  ng_plus→ng_steps
    msg→msg_steps  ct→ct_steps  msg_kd→msg_steps  adaptiformet→max_steps

Distance-to-oracle (behavioral, per bin):
    |forget_holdout_acc(method@budget, bin) − forget_holdout_acc(oracle_bin, bin)|
  where oracle_bin is the Protocol B retrain that excluded ONLY bin B's
  forget identities (so oracle_bin's bin-B forget acc ≈ 0 — never trained).

Usage:
    python src/budget_sweep.py \\
        --csv ../bench/metadata/dataset_imbalanced.csv \\
        --model results/imbalanced/checkpoints/original_model_best.pt \\
        --methods results/imbalanced/protocol_c_methods.json \\
        --oracles results/imbalanced/oracles \\
        --best_configs results/imbalanced/hparam \\
        --out results/imbalanced

Outputs budget_sweep.json:
    {method: {budget: {bin: {"forget_acc": .., "dist_to_oracle": ..}}}}
    + budget_to_oracle: {method: {bin: budget}} (first budget with dist ≤ tol)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from baselines import BASELINE_REGISTRY
from config_loader import load_method_configs
from dataset import infer_identity_classes
from device_utils import resolve_device
from evaluate import evaluate_per_demographic
from interfaces import prepare_method_call, validate_unlearning_result
from model import load_model
from novel_variant import NOVEL_REGISTRY
from sota_methods import SOTA_REGISTRY

METHOD_REGISTRY = {**BASELINE_REGISTRY, **SOTA_REGISTRY, **NOVEL_REGISTRY}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

BINS = ["high", "medium", "low"]

# method → unlearning-budget config key
BUDGET_KEY = {
    "ga": "ga_steps", "srl": "srl_epochs", "ft": "ft_epochs",
    "ng_plus": "ng_steps", "msg": "msg_steps", "ct": "ct_steps",
    "msg_kd": "msg_steps", "adaptiformet": "max_steps",
}

# Budget grid as multiples of the tuned best-config value.
# 0.5×, 1×, 2×, 4× of the best budget → difficulty curve around the
# tuned operating point (and one step below for the easy-bin case).
BUDGET_MULTIPLES = [0.5, 1.0, 2.0, 4.0]


def load_oracle_reference(oracles_dir: str, csv_path: str,
                          device: torch.device) -> dict[str, float]:
    """Per-bin forget-holdout acc of each Protocol B oracle.

    oracle_{bin}.pt was retrained excluding ONLY bin's forget identities,
    so its forget acc on that bin ≈ 0 (never trained).  This is the
    reference each unlearned model is measured against per bin.
    """
    ref: dict[str, float] = {}
    for b in BINS:
        ckpt = Path(oracles_dir) / f"oracle_{b}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Protocol B oracle missing: {ckpt} — run per_bin_oracle.py first"
            )
        m = load_model(str(ckpt), device=str(device))
        demog = evaluate_per_demographic(m, csv_path, device, subset="holdout")
        key = f"popularity_{b}"
        acc = None
        if key in demog and isinstance(demog[key], dict):
            acc = demog[key].get("identity", {}).get("accuracy")
        if acc is None:
            logger.warning(f"  [warn] no per-bin forget acc for {b} from oracle — using 0.0")
            acc = 0.0
        ref[b] = float(acc)
        logger.info(f"  oracle[{b}] forget-holdout acc = {acc:.4f}")
    return ref


def run_budget_sweep(csv_path: str, model_path: str, methods_json: str,
                     oracles_dir: str, best_configs_dir: str, out_dir: str,
                     device_str: str = "auto", seed: int = 42,
                     tol: float = 0.05, max_budget: int | None = None) -> dict:
    device = resolve_device(device_str)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(methods_json) as fh:
        picked = json.load(fh)["categories"]
    methods = list(picked.values())
    logger.info(f"\n[BudgetSweep] Methods (Protocol C picks): {methods}")

    n_id = infer_identity_classes(csv_path)
    original = load_model(model_path, device=str(device))
    oracle_ref = load_oracle_reference(oracles_dir, csv_path, device)

    # Tuned configs: best-config files override YAML defaults per method.
    configs = load_method_configs(scale=1.0, best_configs_path=best_configs_dir)

    sweep: dict[str, dict] = {}
    budget_to_oracle: dict[str, dict[str, int | None]] = {}

    for method in methods:
        bkey = BUDGET_KEY.get(method)
        if bkey is None:
            logger.warning(f"  [skip] no budget key for {method} — cannot sweep")
            continue
        base_cfg = dict(configs.get(method, {}))
        base_budget = int(base_cfg.get(bkey, 0))
        if base_budget <= 0:
            logger.warning(f"  [skip] {method} has no {bkey} in config")
            continue

        # Budget grid around the tuned value
        budgets = sorted({max(1, int(base_budget * m)) for m in BUDGET_MULTIPLES})
        if max_budget:
            budgets = [b for b in budgets if b <= max_budget]
        logger.info(f"\n  {method}: sweeping {bkey} ∈ {budgets} "
                    f"(tuned base {base_budget})")

        sweep[method] = {}
        for budget in budgets:
            cfg = dict(base_cfg)
            cfg[bkey] = budget
            cfg = prepare_method_call(cfg, identity_classes=n_id,
                                      schedule="uniform")
            t0 = time.time()
            try:
                res = METHOD_REGISTRY[method](
                    model=original, csv_path=csv_path, device=device,
                    seed=seed, **cfg,
                )
                validate_unlearning_result(res, method)
                unlearned = res["model"]
            except Exception as e:
                logger.warning(f"  [warn] {method}@{bkey}={budget} failed: {e}")
                continue

            demog = evaluate_per_demographic(unlearned, csv_path, device,
                                             subset="holdout")
            sweep[method][str(budget)] = {}
            for b in BINS:
                key = f"popularity_{b}"
                acc = 0.0
                if key in demog and isinstance(demog[key], dict):
                    acc = float(demog[key].get("identity", {}).get("accuracy") or 0.0)
                dist = abs(acc - oracle_ref[b])
                sweep[method][str(budget)][b] = {
                    "forget_acc": round(acc, 4),
                    "dist_to_oracle": round(dist, 4),
                }
            logger.info(f"    {bkey}={budget:<4d} t={time.time()-t0:.0f}s "
                        + "  ".join(
                            f"{b}:d={sweep[method][str(budget)][b]['dist_to_oracle']:.3f}"
                            for b in BINS))

        # Budget at which each bin reaches its oracle (dist ≤ tol)
        budget_to_oracle[method] = {}
        for b in BINS:
            reached = [int(k) for k, v in sweep[method].items()
                       if v[b]["dist_to_oracle"] <= tol]
            budget_to_oracle[method][b] = min(reached) if reached else None

    result = {
        "methods": methods,
        "budget_key": BUDGET_KEY,
        "tol": tol,
        "sweep": sweep,
        "budget_to_oracle": budget_to_oracle,
        "oracle_reference": oracle_ref,
    }
    out_file = out_path / "budget_sweep.json"
    with open(out_file, "w") as fh:
        json.dump(result, fh, indent=2)

    logger.info(f"\n[BudgetSweep] budget-to-oracle (dist ≤ {tol}):")
    logger.info(f"{'method':<14} " + "  ".join(f"{b:>8}" for b in BINS))
    for method, d in budget_to_oracle.items():
        logger.info(f"{method:<14} " + "  ".join(
            f"{str(d[b]):>8}" for b in BINS))
    logger.info(f"\n  Results → {out_file}")
    logger.info("  Interpretation: higher budget-to-oracle = harder to forget "
                "(H1: high-bin should need more)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--methods", type=str, required=True,
                        help="protocol_c_methods.json from select_protocol_c_methods")
    parser.add_argument("--oracles", type=str, required=True,
                        help="dir of Protocol B oracle_{bin}.pt")
    parser.add_argument("--best_configs", type=str, default=None,
                        help="hparam dir of *_best_config.json (tuned budgets)")
    parser.add_argument("--out", type=str, default="results/imbalanced")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=0.05,
                        help="distance-to-oracle tolerance (budget reached)")
    parser.add_argument("--max_budget", type=int, default=None,
                        help="cap the largest swept budget")
    args = parser.parse_args()

    run_budget_sweep(args.csv, args.model, args.methods, args.oracles,
                     args.best_configs, args.out,
                     device_str=args.device, seed=args.seed,
                     tol=args.tol, max_budget=args.max_budget)


if __name__ == "__main__":
    main()
