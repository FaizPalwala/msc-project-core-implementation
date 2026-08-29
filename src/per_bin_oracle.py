"""
per_bin_oracle.py — Protocol B: per-bin retrain oracles (imbalanced artifact).

The global retrain oracle (single_shot's `retrain` method) retrains on
retain-only — it answers "what would the model look like if NO forget
identities were ever trained".  Protocol B asks a *per-bin* counterfactual:
for bin B, what would the model look like if ONLY bin-B's forget identities
had never been trained?  That oracle still trains on the other bins' forget
identities (they were not requested for deletion in that counterfactual).

This is the honest fix for the plan's open question 1: the oracle DOES
change per bin, so we compute one oracle PER BIN (3 retrains), not one
global oracle reused for all bins.

Usage:
    python src/per_bin_oracle.py \\
        --csv ../bench/metadata/dataset_imbalanced.csv \\
        --model results/imbalanced/checkpoints/original_model_best.pt \\
        --out results/imbalanced/oracles

Outputs (per bin B):
    oracle_{B}.pt              — retrain excluding bin-B forget identities
    oracle_{B}_eval.json       — retain/forget eval of the oracle itself
    per_bin_distance.json      — behavioral distance from each single-shot
                                 unlearned model to its bin oracle
                                 (|forget_acc(unlearned, bin) −
                                  forget_acc(oracle_bin, bin)|)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from baselines import retrain_oracle
from dataset import infer_identity_classes
from device_utils import resolve_device
from evaluate import evaluate_full
from model import copy_model, load_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

BINS = ["high", "medium", "low"]


def bin_forget_identities(csv_path: str) -> dict[str, list[int]]:
    """Map bin name → sorted forget identity_ids from the imbalanced CSV."""
    df = pd.read_csv(csv_path)
    fg = df[df["split"] == "forget"]
    out: dict[str, list[int]] = {}
    for b in BINS:
        ids = sorted(fg[fg["popularity_bin"] == b]["identity_id"].unique().tolist())
        out[b] = ids
        logger.info(f"  bin {b}: {len(ids)} forget identities")
    return out


def run_per_bin_oracle(csv_path: str, model_path: str, out_dir: str,
                       device_str: str = "auto", seed: int = 42,
                       epochs: int = 30) -> dict:
    device = resolve_device(device_str)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_id = infer_identity_classes(csv_path)
    logger.info(f"  Identity classes: {n_id} | Device: {device}")

    bin_ids = bin_forget_identities(csv_path)

    oracles: dict[str, str] = {}
    for b in BINS:
        t0 = time.time()
        logger.info(f"\n[Oracle {b}] retraining WITHOUT {b}-bin forget identities…")
        res = retrain_oracle(
            model=None,  # unused by retrain path
            csv_path=csv_path,
            device=device,
            epochs=epochs,
            seed=seed,
            identity_classes=n_id,
            exclude_identity_ids=bin_ids[b],
            pretrained=True,
        )
        m = res["model"]
        ckpt = out_path / f"oracle_{b}.pt"
        # Save in the canonical dual-head checkpoint format (not a bare
        # state_dict): load_model() expects architecture/identity_classes/
        # age_classes/model_state_dict — a bare state_dict is misdetected
        # as legacy single-head and crashes with KeyError downstream
        # (budget_sweep's load_oracle_reference hit exactly that).
        torch.save({
            "architecture": "dual_head",
            "identity_classes": n_id,
            "age_classes": getattr(m, "age_classes", 4),
            "model_state_dict": m.state_dict(),
        }, ckpt)
        oracles[b] = str(ckpt)

        ev = evaluate_full(m, csv_path, device, verbose=False, subset="holdout")
        with open(out_path / f"oracle_{b}_eval.json", "w") as fh:
            json.dump(ev, fh, indent=2, default=str)
        logger.info(f"  [{b}] oracle saved → {ckpt} "
                    f"({time.time()-t0:.0f}s) retain_id_acc="
                    f"{ev.get('retain',{}).get('identity',{}).get('accuracy'):.4f}")

    # ── Per-bin distance: each single-shot unlearned model vs its bin oracle ──
    # Behavioral distance per bin:
    #   |forget_holdout_acc(unlearned, bin) − forget_holdout_acc(oracle_bin, bin)|
    # where oracle_bin's bin-B forget acc ≈ 0 (never trained on them).
    from evaluate import evaluate_per_demographic

    single_shot_dir = out_path.parent / "single_shot_best"
    distance: dict[str, dict] = {}
    if single_shot_dir.exists():
        agg_path = single_shot_dir / "single_shot_aggregated.json"
        if agg_path.exists():
            with open(agg_path) as fh:
                agg = json.load(fh)
            # oracle per-bin forget-holdout acc (demographic eval, no leak)
            oracle_ref: dict[str, float] = {}
            for b in BINS:
                oracle_m = load_model(oracles[b], device=str(device))
                demog = evaluate_per_demographic(oracle_m, csv_path, device,
                                                 subset="holdout")
                key = f"popularity_{b}"
                acc = None
                if key in demog and isinstance(demog[key], dict):
                    acc = demog[key].get("identity", {}).get("accuracy")
                oracle_ref[b] = float(acc) if acc is not None else 0.0
                logger.info(f"  oracle[{b}] per-bin forget-holdout acc = {oracle_ref[b]:.4f}")
            for b in BINS:
                bin_rows: dict[str, dict] = {}
                for method, data in agg.items():
                    if not isinstance(data, dict):
                        continue
                    # per-bin forget acc of the unlearned model (flat key);
                    # fall back to overall forget_id_acc only if per-bin
                    # aggregation was unavailable (balanced datasets).
                    fg_acc = data.get(f"forget_id_acc_{b}")
                    if fg_acc is None:
                        fg_acc = data.get("forget_id_acc")
                    if fg_acc is None:
                        continue
                    bin_rows[method] = {
                        "forget_acc_unlearned": fg_acc,
                        "forget_acc_oracle": oracle_ref[b],
                        "behavioral_distance": abs(fg_acc - oracle_ref[b]),
                    }
                distance[b] = bin_rows
            with open(out_path / "per_bin_distance.json", "w") as fh:
                json.dump(distance, fh, indent=2, default=str)
            logger.info(f"\n  Per-bin behavioral distance → {out_path/'per_bin_distance.json'}")
        else:
            logger.info("  [skip] single_shot_aggregated.json not found — "
                        "oracles only (no distance table)")
    else:
        logger.info("  [skip] single_shot_best/ not found — oracles only")

    return {"oracles": oracles, "bins": bin_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/imbalanced/oracles")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Retrain epochs per oracle (3 oracles total)")
    args = parser.parse_args()

    run_per_bin_oracle(args.csv, args.model, args.out,
                       device_str=args.device, seed=args.seed,
                       epochs=args.epochs)


if __name__ == "__main__":
    main()
