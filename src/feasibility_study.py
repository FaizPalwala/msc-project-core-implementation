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
        [--methods ga ng_plus adaptiforget msg msg_kd ct ft srl budget_scaled] \
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
import torch


# ── Method budget knobs (primary step parameter per method) ──────────────
# multiplier scales the default value from configs/methods/<m>.yaml
STEP_SCALE = 1.0   # set from --step_scale; shrinks base budget on CPU
BUDGET_KEYS: dict[str, str | None] = {
    "ga": "ga_steps",
    "ng_plus": "ng_steps",
    "msg": "msg_steps",
    "msg_kd": "msg_steps",
    "adaptiforget": "max_steps",
    "budget_scaled": "base_steps",  # scaled base; final budget = f(forget-set size)
    "ct": "ct_steps",
    "ft": "ft_epochs",
    "srl": "srl_epochs",      # falls back to 3 if key absent
    "no_unlearning": None,
    "retrain": None,
}

# Methods the feasibility gate is designed to triage (all registries).
DEFAULT_METHODS = ["ga", "ng_plus", "adaptiforget", "msg", "msg_kd",
                   "ct", "ft", "srl", "budget_scaled"]


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
    """Run one method at a given budget multiplier via the real registry.

    identity_classes is inferred from the (subsampled) CSV and injected via
    prepare_method_call — the subsample remaps identity_id to 0..N-1, so
    methods that default to the FULL 750-class head (e.g. SRL's
    _RelabelledDataset(num_classes=600)) would relabel out of range →
    CUDA device-side assert → poisons the context for every later method.
    Mirrors single_shot.py which passes the CSV-inferred class count.
    """
    from config_loader import load_method_configs
    from baselines import BASELINE_REGISTRY
    from sota_methods import SOTA_REGISTRY
    from novel_variant import NOVEL_REGISTRY
    from interfaces import prepare_method_call, validate_unlearning_result
    from dataset import infer_identity_classes

    registry = None
    for r in (BASELINE_REGISTRY, SOTA_REGISTRY, NOVEL_REGISTRY):
        if method in r:
            registry = r
            break
    if registry is None:
        # Always carry method/multiplier so verdict()'s filter
        # `r["method"] == method` cannot KeyError on error rows
        # (the 20 Aug run: "unknown method 12id" crashed the verdict
        #  loop because the error dict lacked these keys).
        return {"method": method, "multiplier": multiplier,
                "error": f"unknown method {method}"}

    n_id_classes = infer_identity_classes(csv_path)

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
            model=model, csv_path=csv_path, device=torch.device(device),
            **prepare_method_call(merged, identity_classes=n_id_classes),
        )
        validate_unlearning_result(result, method)
        unlearned = result["model"]
        elapsed = time.time() - t0
    except Exception as e:  # feasibility gate: never let one method kill the run
        import traceback
        return {"method": method, "multiplier": multiplier,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-500:]}

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
    """GO / TUNE / BROKEN from the multiplier response.

    forget_id_acc == 0.0 is the PERFECT forget signal — must not be
    conflated with missing (None).  The old code did `x or 1.0`, which
    mapped 0.0 -> 1.0 (falsy-zero bug): every method that forgot got
    'BROKEN'.  Use explicit None checks.

    v2 (2026-08-20): retain is judged AT THE BUDGET WHERE FORGETTING
    HAPPENS, not the max across budgets.  The old `max_retain` masked
    retain damage: budget_scaled (12-id: 1× → 0.300/1.000, 3× →
    0.000/0.190) scored retain_ok via the healthy 1× row and got GO,
    despite destroying retain exactly where it erases.  Now: find the
    best-forgetting row, judge retain THERE.
    """
    def _acc(r: dict, key: str, default: float) -> float:
        v = r.get(key)
        return float(v) if v is not None else default

    rs = [r for r in rows if r.get("method") == method and "error" not in r]
    if not rs:
        return "ERROR"
    base = rs[0]
    # The row where forgetting is strongest = the budget we'd actually use
    best = min(rs, key=lambda r: _acc(r, "forget_id_acc", 1.0))
    base_forget = _acc(base, "forget_id_acc", 1.0)
    best_forget = _acc(best, "forget_id_acc", 1.0)
    retain_at_erase = _acc(best, "retain_id_acc", 0.0)
    responded = best_forget < base_forget - 0.05
    forgot = best_forget < 0.2
    retain_ok = retain_at_erase >= 0.6
    if forgot and retain_ok:
        return "GO"
    if forgot and not retain_ok:
        return "TUNE(retain-collapse)"
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
    ap.add_argument("--scale", default="12id",
                    help="scale tag written into results (e.g. 12id, 75id).  "
                         "Multi-scale feasibility (2026-08): the SAME study at "
                         "12-id AND 75-id (≈10% of full) exposes head-width "
                         "mechanisms the 12-id gate cannot see — ng_plus "
                         "(GO@12, never forgets@750), FT (BROKEN@12, "
                         "perfect@750).  The 750-id column of the trajectory "
                         "table comes from the hparam grids.")
    ap.add_argument("--step_scale", type=float, default=1.0,
                    help="shrink the BASE budget (e.g. 0.1 for CPU checks); "
                         "multipliers scale from that base")
    ap.add_argument("--no_pretrain", action="store_true",
                    help="random-init backbone (matches real pipeline: "
                         "pretrained=True is the default)")
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
          pretrained=not args.no_pretrain)
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
                # Metrics can be None when a method produces a degenerate
                # model at extreme step_scale (eval returns nothing) — never
                # let the progress line crash the whole gate.
                def _f(v):
                    return "n/a" if v is None else f"{v:.3f}"
                print(f"  [{method} x{mult:<4}] retain={_f(r.get('retain_id_acc'))} "
                      f"forget={_f(r.get('forget_id_acc'))} "
                      f"mia={_f(r.get('mia_mean_auc'))} ({r.get('time_s')}s)")

    # ── 4. Verdicts ───────────────────────────────────────────────────────
    print("\n=== FEASIBILITY VERDICTS ===")
    results: dict = {"subsample": str(sub_csv), "scale": args.scale,
                     "rows": all_rows, "verdicts": {}}
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
