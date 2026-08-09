"""
feasibility_study.py — CHEAP method-feasibility gate (C2/C3 triage).

Runs each unlearning method on a SMALL subsample of the REAL benchmark
data at increasing budget multipliers (1x / 3x / 10x of the default
step budget) and reports whether the method shows the expected
DIRECTION of forgetting (forget_id_acc → 0, MIA AUC ↓ below 0.5) while
retaining utility (retain_id_acc stays high).

Why: the 750-id preliminary run showed GA/NG+/AdaptiForget with
forget_id_acc ≈ 0.999 (they did nothing) and MSG/MSG-KD with retain
collapse (0.009).  Before paying for a full-scale re-run, this gate
answers: is it a CONFIG problem (method responds to more budget →
re-run with tuned configs) or a STRUCTURAL problem (method shows no
response even at 10x → needs a redesign or removal)?

Scale: ~8 retain + 4 forget identities x 16 images (real crops,
subsampled), 3 training epochs, CPU-capable (~10-20 min).

Usage:
    python src/feasibility_study.py \
        --src_csv metadata/dataset.csv \
        --out results/feasibility \
        [--methods ga ng_plus adaptiformet msg msg_kd ct ft srl] \
        [--multipliers 1 3 10] \
        [--epochs 3] [--n_retain 8] [--n_forget 4] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── Method budget knobs (primary step parameter per method) ──────────────
# multiplier scales the default value from configs/methods/<m>.yaml
STEP_SCALE = 1.0   # set from --step_scale; shrinks base budget on CPU
BUDGET_KEYS: dict[str, str | None] = {
    "ga": "ga_steps",
    "ng_plus": "ng_steps",
    "msg": "msg_steps",
    "msg_kd": "msg_steps",
    "adaptiformet": "max_steps",
    "ct": "ct_steps",
    "ft": "ft_epochs",
    "srl": "srl_epochs",      # falls back to 3 if key absent
    "no_unlearning": None,
    "retrain": None,
}

# Methods the feasibility gate is designed to triage (all registries).
DEFAULT_METHODS = ["ga", "ng_plus", "adaptiformet", "msg", "msg_kd",
                   "ct", "ft", "srl"]


def subsample_csv(src_csv: str, out_csv: str, n_retain: int, n_forget: int,
                  imgs_per_id: int = 16, seed: int = 42) -> str:
    """Subsample real identities (images_per_identity-contract aware) and
    rewrite image_path to ABSOLUTE paths so the CSV is self-contained
    anywhere (dataset.py resolves relative paths against csv_dir)."""
    df = pd.read_csv(src_csv)
    rng = np.random.RandomState(seed)

    retain_ids = sorted(df[df["split"] == "retain"]["identity_id"].unique())
    forget_ids = sorted(df[df["split"] == "forget"]["identity_id"].unique())
    assert len(retain_ids) >= n_retain, f"need {n_retain} retain ids, have {len(retain_ids)}"
    assert len(forget_ids) >= n_forget, f"need {n_forget} forget ids, have {len(forget_ids)}"

    pick_r = rng.choice(retain_ids, n_retain, replace=False)
    pick_f = rng.choice(forget_ids, n_forget, replace=False)
    keep_ids = set(pick_r) | set(pick_f)

    sub = df[df["identity_id"].isin(keep_ids)].copy()
    # Deterministic per-id image cap: first imgs_per_id rows per identity
    sub = (sub.sort_values(["identity_id", "image_path"])
              .groupby("identity_id", group_keys=False)
              .head(imgs_per_id).reset_index(drop=True))

    # REMAP identity_id → contiguous 0..N-1: the model's identity head has
    # N classes and the dataset uses raw identity_id as the class label —
    # real ids (0..749) would be out of bounds at subsample scale.
    id_map = {old: new for new, old in enumerate(sorted(sub["identity_id"].unique()))}
    sub["identity_id"] = sub["identity_id"].map(id_map)

    # Absolute paths: resolve against the source CSV's data root (parent.parent)
    data_root = Path(src_csv).resolve().parent.parent
    sub["image_path"] = sub["image_path"].apply(
        lambda p: str((data_root / p).resolve()) if not Path(p).is_absolute() else p
    )
    sub.to_csv(out_csv, index=False)
    return out_csv


def run_method(method: str, model, csv_path: str, device,
               cfg: dict, multiplier: float) -> dict:
    """Run one method at a given budget multiplier via the real registry."""
    from config_loader import load_method_configs
    from baselines import BASELINE_REGISTRY
    from sota_methods import SOTA_REGISTRY
    from novel_variant import NOVEL_REGISTRY
    from interfaces import prepare_method_call, validate_unlearning_result

    registry = None
    for r in (BASELINE_REGISTRY, SOTA_REGISTRY, NOVEL_REGISTRY):
        if method in r:
            registry = r
            break
    if registry is None:
        return {"error": f"unknown method {method}"}

    merged = dict(cfg)
    key = BUDGET_KEYS.get(method)
    if key and key in merged:
        # step_scale shrinks the BASE budget (CPU feasibility runs);
        # multiplier then scales from that base.  Direction of response
        # is what matters, not absolute step counts.
        base = max(1, int(round(merged[key] * STEP_SCALE)))
        merged[key] = max(1, int(round(base * multiplier)))

    t0 = time.time()
    try:
        result = registry[method](
            model=model, csv_path=csv_path, device=device,
            **prepare_method_call(merged),
        )
        validate_unlearning_result(result, method)
        unlearned = result["model"]
        elapsed = time.time() - t0
    except Exception as e:  # feasibility gate: never let one method kill the run
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-500:]}

    # ── Evaluate (holdout = generalisation; forget acc → 0 is the signal) ──
    from evaluate import evaluate_full
    from mia import run_mia_per_identity

    eval_res = evaluate_full(unlearned, csv_path, device, verbose=False, subset="holdout")
    per_id = run_mia_per_identity(unlearned, csv_path, device,
                                  head="identity", subset="holdout")
    return {
        "method": method,
        "multiplier": multiplier,
        "budget": {key: merged.get(key)} if key else {},
        "retain_id_acc": eval_res.get("retain", {}).get("identity", {}).get("accuracy"),
        "forget_id_acc": eval_res.get("forget", {}).get("identity", {}).get("accuracy"),
        "mia_mean_auc": per_id.get("mean_auc"),
        "time_s": round(elapsed, 1),
    }


def verdict(rows: list[dict], method: str) -> str:
    """GO / TUNE / BROKEN from the multiplier response."""
    rs = [r for r in rows if r["method"] == method and "error" not in r]
    if not rs:
        return "ERROR"
    base = rs[0]
    best_forget = min(r.get("forget_id_acc") or 1.0 for r in rs)
    max_retain = max(r.get("retain_id_acc") or 0.0 for r in rs)
    mia_at_best = [r for r in rs if (r.get("forget_id_acc") or 1.0) == best_forget][0].get("mia_mean_auc")
    responded = best_forget < (base.get("forget_id_acc") or 1.0) - 0.05
    forgot = best_forget < 0.2
    retain_ok = max_retain >= 0.6
    if forgot and retain_ok:
        return "GO"
    if responded and retain_ok:
        return "TUNE"
    if not responded:
        return "BROKEN"
    return "TUNE(retain-collapse)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Cheap method-feasibility gate (C2/C3)")
    ap.add_argument("--src_csv", required=True, help="full dataset.csv (balanced or imbalanced)")
    ap.add_argument("--out", default="results/feasibility")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--multipliers", nargs="*", type=float, default=[1.0, 3.0, 10.0])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n_retain", type=int, default=8)
    ap.add_argument("--n_forget", type=int, default=4)
    ap.add_argument("--imgs_per_id", type=int, default=16)
    ap.add_argument("--step_scale", type=float, default=1.0,
                    help="shrink the BASE budget (e.g. 0.1 for CPU checks); "
                         "multipliers scale from that base")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    global STEP_SCALE
    STEP_SCALE = args.step_scale

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Subsampled real dataset ────────────────────────────────────────
    sub_csv = out / "feasibility_dataset.csv"
    subsample_csv(args.src_csv, str(sub_csv), args.n_retain, args.n_forget,
                  imgs_per_id=args.imgs_per_id)
    print(f"[feasibility] Subsample → {sub_csv}")
    df = pd.read_csv(sub_csv)
    print(f"  {len(df)} rows | ids={df['identity_id'].nunique()} "
          f"(retain {df[df['split']=='retain']['identity_id'].nunique()}, "
          f"forget {df[df['split']=='forget']['identity_id'].nunique()})")

    # ── 2. Train a small model ────────────────────────────────────────────
    ckpts = out / "checkpoints"
    from train import train
    print(f"[feasibility] Training {args.epochs} epochs ({args.device})…")
    train(csv_path=str(sub_csv), save_dir=str(ckpts), epochs=args.epochs,
          batch_size=16, device_str=args.device, age_weight=0.5, seed=42,
          pretrained=False)
    model_path = ckpts / "original_model_best.pt"

    from model import load_model
    device = args.device
    model = load_model(str(model_path), device=device)

    # ── 3. Method × multiplier grid ───────────────────────────────────────
    from config_loader import load_method_configs
    cfgs = load_method_configs(scale=1.0)   # YAML defaults

    all_rows: list[dict] = []
    for method in args.methods:
        cfg = dict(cfgs.get(method, {}))
        for mult in args.multipliers:
            r = run_method(method, model, str(sub_csv), device, cfg, mult)
            r["config_source"] = "yaml-default"
            all_rows.append(r)
            if "error" in r:
                print(f"  [{method} x{mult}] ERROR: {r['error'][:80]}")
            else:
                print(f"  [{method} x{mult:<4}] retain={r['retain_id_acc']:.3f} "
                      f"forget={r['forget_id_acc']:.3f} "
                      f"mia={r['mia_mean_auc']:.3f} ({r['time_s']}s)")

    # ── 4. Verdicts ───────────────────────────────────────────────────────
    print("\n=== FEASIBILITY VERDICTS ===")
    results: dict = {"subsample": str(sub_csv), "rows": all_rows, "verdicts": {}}
    for method in args.methods:
        v = verdict(all_rows, method)
        results["verdicts"][method] = v
        print(f"  {method:<14} → {v}")

    with open(out / "feasibility_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n[OK] → {out}/feasibility_results.json")
    print("Verdict guide: GO = forgets+retains at some budget (full run OK); "
          "TUNE = responds to budget, needs config work; "
          "BROKEN = no response even at 10x (remove/redesign); "
          "TUNE(retain-collapse) = forgets but kills retain.")


if __name__ == "__main__":
    main()
