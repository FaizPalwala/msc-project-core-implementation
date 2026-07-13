"""
ablation_study.py  —  Phase 4B: AdaptiForget Component Ablation

Runs controlled single-shot unlearning experiments where each component
of AdaptiForget is disabled in turn:

  Full model:          all 4 components active
  w/o adaptive λ:      kl_weight_init == kl_weight_max (constant λ)
  w/o mask refresh:    mask_refresh_every = 999999 (never refresh)
  w/o early stopping:  early_stop_adv = -1 (never stop early)
  w/o KL distillation: kl_weight_max = 0 (pure MSG)
  w/o masking:         topk_fraction = 1.0 (all params updated = NG+)

Outputs a CSV and printed table showing contribution of each component.
"""

import json
from pathlib import Path
from typing import Optional
import torch
from device_utils import resolve_device
from evaluate import evaluate_full
from mia import run_mia_full
from model import load_model
from novel_variant import adaptiformet


ABLATIONS = {
    "Full AdaptiForget": {},
    "w/o adaptive λ":   {"kl_weight_init": 0.5, "kl_weight_max": 0.5},
    "w/o mask refresh": {"mask_refresh_every": 999999},
    "w/o early stop":   {"early_stop_adv": -1.0},
    "w/o KL distil":    {"kl_weight_max": 0.0, "kl_weight_init": 0.0},
    "w/o masking (NG+)":{"topk_fraction": 1.0},
}

BASE_CFG = {
    "max_steps": 300,
    "topk_fraction": 0.2,
    "lr_ascent": 5e-5,
    "lr_retain": 1e-4,
    "kl_weight_init": 0.1,
    "kl_weight_max": 0.8,
    "mask_refresh_every": 100,
    "early_stop_adv": 0.05,
}


def run_ablation(csv_path, model_path, out_dir, device_str="auto",
                 seed=42, smoke_test=False):
    device = resolve_device(device_str)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    original = load_model(model_path, device=str(device))
    results  = {}

    cfg_overrides = {"max_steps": 50} if smoke_test else {}

    print(f"\n  Running {len(ABLATIONS)} ablation variants…")
    for name, overrides in ABLATIONS.items():
        cfg = {**BASE_CFG, **overrides, **cfg_overrides}
        print(f"\n  ── {name}")
        res = adaptiformet(original, csv_path, device,
                           forget_step=0, seed=seed, **cfg)
        m   = res["model"]
        ev  = evaluate_full(m, csv_path, device, forget_step=0, verbose=False)
        mia = run_mia_full(m, csv_path, device, forget_step=0,
                           score_type="confidence", verbose=False)
        results[name] = {
            "retain_acc":       ev.get("retain", {}).get("accuracy", 0),
            "forget_acc":       (ev.get("forget_step_0", {})
                                 or ev.get("forget", {})).get("accuracy", 0),
            "mia_auc":          mia.get("forget_test_auc", 0.5),
            "forget_advantage": mia.get("forget_advantage", 0),
            "steps_used":       res["metrics"].get("steps_used", 0),
            "time_s":           res["metrics"].get("unlearning_time_s", 0),
        }
        print(f"    retain={results[name]['retain_acc']:.4f} | "
              f"mia={results[name]['mia_auc']:.4f} | "
              f"adv={results[name]['forget_advantage']:.4f}")

    # Save
    import csv
    rows = [{"variant": k, **v} for k, v in results.items()]
    csv_out = out_path / "ablation_results.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_path / "ablation_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  ✓ Ablation results → {csv_out}")

    # Print table
    print(f"\n{'='*75}")
    print("  ADAPTIFORMET ABLATION STUDY")
    print(f"{'='*75}")
    print(f"{'Variant':<26} {'RetainAcc':>10} {'MIA_AUC':>9} {'F_Adv':>8} {'Steps':>7}")
    print("─" * 62)
    for name, r in results.items():
        print(f"{name:<26} {r['retain_acc']:>10.4f} {r['mia_auc']:>9.4f} "
              f"{r['forget_advantage']:>8.4f} {r['steps_used']:>7}")
    print("=" * 75)
