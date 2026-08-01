"""
smoke_test.py — End-to-end pipeline smoke test (CPU, ~2 min).

Generates a minimal synthetic dataset, trains for 2 epochs, runs one
unlearning method, and asserts basic output integrity.  Designed to catch
import errors, tensor shape mismatches, and config loading failures
without requiring GPU or real data.

Usage:
    python tests/smoke_test.py          # standalone
    smoke-test                          # via pyproject.toml console_script
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _make_synthetic_dataset(tmpdir: str) -> str:
    """Create a minimal dataset.csv + fake images for smoke testing.

    Returns path to dataset.csv.
    """
    root = Path(tmpdir)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rng = np.random.RandomState(42)

    for cid in range(4):  # 4 identities
        split = "retain" if cid < 2 else "forget" if cid == 2 else "test"
        fs = cid - 2 if split == "forget" else -1
        for img_idx in range(3):  # 3 images per identity
            fname = f"identity_{cid:03d}_img_{img_idx:02d}.jpg"
            fpath = img_dir / fname
            # 224×224 random RGB image
            arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(fpath)
            rows.append({
                "image_path": f"images/{fname}",
                "identity_id": cid,
                "age_group": cid % 4,
                "age": 20 + cid * 10,
                "gender": cid % 2,
                "split": split,
                "forget_step": fs,
                "forget_variant": 0 if fs >= 0 else -1,
                "arcface_similarity": round(0.5 + img_idx * 0.1, 4),
                "laplacian_variance": round(50.0 + img_idx * 20.0, 2),
                "detection_confidence": round(0.90 + img_idx * 0.02, 4),
            })

    csv_path = root / "dataset.csv"
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(csv_path)


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="msc_smoke_")
    results = Path(tmp) / "results"
    ckpts = results / "checkpoints"
    ckpts.mkdir(parents=True, exist_ok=True)

    csv_path = _make_synthetic_dataset(tmp)
    print(f"[smoke] Dataset: {csv_path}")

    # ── Train ────────────────────────────────────────────────────────────
    print("[smoke] Training (2 epochs, CPU)…")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from train import train

    train(
        csv_path=csv_path,
        save_dir=str(ckpts),
        epochs=2,
        batch_size=4,
        device_str="cpu",
        age_weight=0.5,
        seed=42,
        pretrained=False,  # CPU smoke: skip ImageNet download
    )
    model_path = str(ckpts / "original_model_best.pt")
    assert os.path.exists(model_path), f"Model not saved: {model_path}"
    print(f"[smoke] Model: {model_path}")

    # ── Single-shot ──────────────────────────────────────────────────────
    print("[smoke] Single-shot (GA only, scale=0.05)…")
    from single_shot import run_single_shot

    result = run_single_shot(
        csv_path=csv_path,
        model_path=model_path,
        methods=["ga"],
        out_dir=str(results / "single_shot"),
        device_str="cpu",
        seed=42,
        scale=0.05,  # heavily scaled down
        skip_retrain=True,
    )

    # ── Assertions ───────────────────────────────────────────────────────
    ga = result.get("ga", {})
    assert "error" not in ga, f"GA failed: {ga.get('error')}"

    mia_auc = ga.get("per_identity_mia", {}).get("mean_auc")
    assert isinstance(mia_auc, (int, float)), f"mia_auc not numeric: {mia_auc}"
    assert 0.3 <= mia_auc <= 1.0, f"mia_auc out of range: {mia_auc}"

    retain_acc = (
        ga.get("evaluation", {}).get("retain", {}).get("identity", {}).get("accuracy", 0)
    )
    assert isinstance(retain_acc, (int, float)), f"retain_acc not numeric: {retain_acc}"
    assert retain_acc >= 0.20, f"retain_acc below chance: {retain_acc}"

    # Verify output files exist
    out_dir = results / "single_shot"
    assert (out_dir / "single_shot_results.json").exists(), "results.json missing"
    assert (out_dir / "single_shot_summary.csv").exists(), "summary.csv missing"

    print(f"[smoke] ✓ All assertions passed")
    print(f"[smoke]   MIA AUC: {mia_auc:.4f}  |  Retain acc: {retain_acc:.4f}")

    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[smoke] Cleaned up {tmp}")


if __name__ == "__main__":
    main()
