"""
canary.py — Pixel-level canary insertion for ground-truth deletion verification.

Inserts subtle, identity-specific pixel patterns into selected forget
identities' images before training.  After unlearning, verifies that the
model no longer encodes the canary pattern.

References:
  Thudi et al. (2022), Unrolling SGD — canary insertion for unlearning proof
  Jagielski et al. (2023), Measuring Forgetting of Memorized Training Examples

Protocol
────────
  1. Select N forget identities (default: first 4 of forget_step 0).
  2. Insert a unique 8×8 pixel canary in the top-left corner of each image
     belonging to those identities.  Pattern: identity-specific metameric
     colour grid, visually imperceptible (mean pixel shift < 2 LSB).
  3. Train original model M with canaries.
  4. Run unlearning method on the canaried identities.
  5. Verify: after unlearning, a simple classifier trained on canary-vs-clean
     images cannot distinguish them (accuracy ≈ 50%).
  6. Gradient-based canary extraction: compute input gradient at canary
     location.  After unlearning, gradient magnitude should match noise.
  7. Reference comparison: verify unlearned model produces features identical
     to a shadow model that never saw the canary.

Usage:
    # Phase 1: insert canaries (pixel insertion — writes canary copies of
    # the selected identities' images alongside the CSV, absolutizes all
    # image_paths so the canary CSV is self-contained)
    python canary.py insert --csv data/dataset/dataset.csv \\
        --identities 300 301 302 303 --out data/dataset/dataset_canary.csv

    # Phase 2: train normally on dataset_canary.csv (see train.py)

    # Phase 3: verify after unlearning
    python canary.py verify --csv data/dataset/dataset_canary.csv \\
        --model results/single_shot/adaptiforget_unlearned.pt \\
        --identities 300 301 302 303
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from model import load_model
from device_utils import resolve_device

logger = logging.getLogger(__name__)

# ── Canary pattern generator ──────────────────────────────────────────────────


def _make_canary(identity_id: int, size: int = 8) -> np.ndarray:
    """Generate a deterministic 8×8×3 pixel canary unique to this identity.

    Uses identity_id as seed so the pattern is reproducible.
    The pattern uses low-magnitude colour offsets (Δ ≤ 4 per channel at 8-bit)
    to remain visually imperceptible.
    """
    rng = np.random.RandomState(identity_id)
    # Small offsets: ±2 in [0,255] range
    canary = (rng.randint(0, 5, size=(size, size, 3)) - 2).astype(np.int16)
    return canary


def insert_canary(
    img: np.ndarray,
    identity_id: int,
    size: int = 8,
    position: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Add identity-specific canary to an image array in-place.

    Args:
        img:         H×W×3 numpy array (uint8, 0–255).
        identity_id: Which identity's pattern to use.
        size:        Canary size (square).
        position:    (y, x) top-left corner.

    Returns:
        Modified image (copy of input).
    """
    img = img.copy().astype(np.int16)
    canary = _make_canary(identity_id, size)
    y, x = position
    h, w = img.shape[:2]
    end_y, end_x = min(y + size, h), min(x + size, w)
    cy, cx = end_y - y, end_x - x
    img[y:end_y, x:end_x] += canary[:cy, :cx]
    return np.clip(img, 0, 255).astype(np.uint8)


# ── Verification ──────────────────────────────────────────────────────────────


def _clean_path_for(img_path: Path, canary_marker: str = "canary_images") -> Path | None:
    """Recover the ORIGINAL (canary-free) image path from a canary copy.

    insert_canary() writes canary-pattern copies under
    <out_dir>/canary_images/<rel_path> and points the canary CSV at them;
    non-canary rows keep their original absolute paths.  A canary identity's
    clean counterpart is the original path with the canary_images segment
    removed.  Returns None if the path isn't a canary copy (no marker).
    """
    parts = img_path.parts
    if canary_marker not in parts:
        return None
    idx = parts.index(canary_marker)
    # Drop everything up to and including canary_images/ → original relative
    # path from the data root; keep the original image filename.
    rel = Path(*parts[idx + 1:])
    return rel


def verify_canary_unlearning(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    identity_ids: list[int],
    src_csv: str | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Verify that canaries have been unlearned (Thudi et al. Tier-4).

    For each canary identity, collect 512-d feature vectors of:
      - CANARY-tagged images   (rows whose image_path points into
                                canary_images/ — the pixel-pattern copies)
      - CLEAN originals        (same images WITHOUT the canary pattern,
                                recovered from the original data root)

    Then fit a logistic-regression detector on the two groups.  If the
    unlearned model no longer encodes the canary pattern, the two feature
    distributions overlap → detector accuracy ≈ chance (0.5).  If the
    pattern persists, the detector separates them → accuracy → 1.0.

    Metrics per identity:
      canary_detection_acc   : detector accuracy (0.5 = erased, 1.0 = persists)
      feature_similarity     : mean cosine similarity between canary and
                               clean features of the SAME image (1.0 = the
                               pattern has no feature-level effect)
      mean_feature_norm      : canary-group feature norm (context)

    src_csv: the ORIGINAL (pre-canary) dataset CSV — its location defines
    the data root (bench/) where clean images live.  Falls back to the
    canary CSV's parent.parent when not given (works when the canary CSV
    is written beside the source metadata, as in smoke tests).
    """
    from dataset import VirtualIdentityDataset, get_val_transform

    # Clean images live under the ORIGINAL data root (bench/).  Derive it
    # from the CSV's own NON-canary rows: those keep absolute paths into
    # <data_root>/images/... (insert_canary only rewrites canary identities'
    # paths; retain/other rows are untouched).  Robust to wherever the CSV
    # is mounted — do NOT derive from csv_path.parent.parent, which is only
    # correct when the CSV sits exactly two levels below the data root
    # (e.g. bench/metadata/dataset.csv → bench/) and breaks for temp copies.
    df0 = VirtualIdentityDataset(csv_path, split="retain+forget", transform=get_val_transform(),
                                 subset="all").df
    non_can = df0[~df0["image_path"].astype(str).str.contains("canary_images", na=False)]
    data_root: Path | None = None
    if len(non_can):
        sample = str(non_can["image_path"].iloc[0])
        p = Path(sample)
        if "/images/" in sample:
            data_root = Path(sample.split("/images/")[0])
        elif p.is_absolute():
            data_root = p.parents[0]
    if data_root is None:  # last resort: CSV two levels above images/
        src_p = Path(src_csv).resolve() if src_csv else Path(csv_path).resolve()
        data_root = src_p.parent.parent

    results: dict[str, Any] = {}

    for cid in identity_ids:
        # All images of the canary identity (train + holdout — the pattern
        # was inserted into every image before training).
        ds = VirtualIdentityDataset(
            csv_path, split="forget",
            transform=get_val_transform(),
            subset="all",
        )
        mask = ds.df["identity_id"] == cid
        if not mask.any():
            results[str(cid)] = {"error": f"identity_id {cid} not in forget split"}
            continue
        sub = ds.df[mask].reset_index(drop=True)
        # Split rows into canary-tagged (pointing into canary_images/) and
        # the rest (original clean paths).
        can_rows = sub[sub["image_path"].str.contains("canary_images", na=False)]
        clean_rows = sub[~sub["image_path"].str.contains("canary_images", na=False)]
        if len(can_rows) == 0:
            results[str(cid)] = {"error": f"no canary-tagged rows for id {cid}"}
            continue

        model.eval()
        feats = {"canary": [], "clean": []}

        def _feats(df_rows) -> np.ndarray:
            out = []
            d = VirtualIdentityDataset(csv_path, split="forget",
                                       transform=get_val_transform(),
                                       subset="all")
            d.df = df_rows.reset_index(drop=True)
            loader = torch.utils.data.DataLoader(
                d, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )
            with torch.no_grad():
                for imgs, _, _ in loader:
                    imgs = imgs.to(device)
                    out.append(model.feature_vector(imgs).cpu().numpy())
            return np.concatenate(out, axis=0) if out else np.zeros((0, 512))

        feats["canary"] = _feats(can_rows)

        # Clean counterparts: for canary rows, recover the ORIGINAL path
        # (strip the canary_images segment); for any clean rows, use as-is.
        clean_paths: list[str] = []
        for _, row in can_rows.iterrows():
            p = Path(row["image_path"])
            rel = _clean_path_for(p)
            if rel is not None:
                clean_paths.append(str((data_root / rel).resolve()))
            else:
                clean_paths.append(str(p))
        clean_rows2 = sub.copy()
        clean_rows2["image_path"] = clean_paths
        feats["clean"] = _feats(clean_rows2)

        n_c = len(feats["canary"])
        n_cl = len(feats["clean"])
        if n_c < 4 or n_cl < 4:
            results[str(cid)] = {"error": f"too few images: canary={n_c} clean={n_cl}"}
            continue

        # 1) Canary-detection classifier (Thudi et al.): chance ⇒ erased.
        X = np.concatenate([feats["canary"], feats["clean"]], axis=0)
        y = np.concatenate([np.ones(n_c), np.zeros(n_cl)])
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        try:
            clf = LogisticRegression(max_iter=500, C=1.0)
            accs = cross_val_score(clf, X, y, cv=min(4, n_c, n_cl), scoring="accuracy")
            det_acc = float(np.mean(accs))
        except Exception:
            det_acc = float("nan")

        # 2) Same-image cosine similarity (canary vs clean originals).
        n_pair = min(n_c, n_cl)
        a = feats["canary"][:n_pair]
        b = feats["clean"][:n_pair]
        sims = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
        )

        # 3) Output-space diagnostic: does the model STILL recognise the
        # canary identity on canary-tagged vs clean images?  After true
        # unlearning BOTH should drop to chance; before unlearning the
        # canary-tagged images (which the model was trained on) typically
        # show HIGHER identity confidence than clean holdout.  This is
        # the Thudi-style membership signal in output space.
        def _id_conf(df_rows) -> tuple[float, float]:
            d = VirtualIdentityDataset(csv_path, split="forget",
                                       transform=get_val_transform(),
                                       subset="all")
            d.df = df_rows.reset_index(drop=True)
            loader = torch.utils.data.DataLoader(
                d, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )
            correct = total = 0
            confs: list[float] = []
            with torch.no_grad():
                for imgs, id_lbls, _ in loader:
                    imgs = imgs.to(device)
                    id_lbls = id_lbls.to(device)
                    id_logits, _ = model(imgs)
                    probs = torch.softmax(id_logits, dim=1)
                    correct += (id_logits.argmax(1) == id_lbls).sum().item()
                    total += len(id_lbls)
                    confs.extend(probs[torch.arange(len(id_lbls)), id_lbls].cpu().tolist())
            return (correct / max(total, 1), float(np.mean(confs)) if confs else 0.0)

        can_id_acc, can_conf = _id_conf(can_rows)
        cln_id_acc, cln_conf = _id_conf(clean_rows2)

        results[str(cid)] = {
            "n_images": n_c,
            "mean_feature_norm": float(np.linalg.norm(feats["canary"], axis=1).mean()),
            "canary_detection_acc": round(float(det_acc), 4),
            "feature_similarity_to_clean": round(float(np.mean(sims)), 4),
            "id_acc_canary": round(float(can_id_acc), 4),
            "id_acc_clean": round(float(cln_id_acc), 4),
            "id_conf_canary": round(float(can_conf), 4),
            "id_conf_clean": round(float(cln_conf), 4),
        }

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    # Insert
    ins = sub.add_parser("insert")
    ins.add_argument("--csv",        type=str, required=True)
    ins.add_argument("--identities", type=int, nargs="+", required=True)
    ins.add_argument("--out",        type=str, required=True)

    # Verify
    ver = sub.add_parser("verify")
    ver.add_argument("--csv",        type=str, required=True)
    ver.add_argument("--model",      type=str, required=True)
    ver.add_argument("--identities", type=int, nargs="+", required=True)
    ver.add_argument("--src_csv",    type=str, default=None,
                     help="ORIGINAL (pre-canary) dataset CSV — defines the data "
                          "root where clean images live (bench/).  Required for "
                          "a meaningful canary-vs-clean comparison.")
    ver.add_argument("--device",     type=str, default="auto")

    args = parser.parse_args()

    if args.command == "insert":
        import pandas as pd

        df = pd.read_csv(args.csv)
        # Resolve the data root: bench/ is the parent of the CSV's dir
        # (bench/metadata/dataset.csv → bench/).  Relative image_paths in
        # the source CSV are resolved against it.
        data_root = Path(args.csv).resolve().parent.parent
        canary_ids = set(args.identities)

        # Tag canaried identities
        df["has_canary"] = df["identity_id"].apply(
            lambda c: int(c in canary_ids),
        )

        # Rewrite image_path to ABSOLUTE paths (Bug fix): the canary CSV is
        # written under results/<dataset>/canary/, so dataset.py's relative
        # resolution (csv_dir → data_dir) would look in results/<dataset>/
        # where images don't exist.  Absolutizing against the source data
        # root makes the canary CSV self-contained wherever it lands.
        # NOTE: --out may be RELATIVE on HPC (slurm passes
        # 'results/<dataset>/canary'), so resolve() against CWD first —
        # otherwise the written paths stay relative and dataset.py
        # re-resolves them, doubling the prefix.
        out_path = Path(args.out).resolve()
        out_dir = out_path.parent
        canary_img_dir = out_dir / "canary_images"
        new_paths: list[str] = []
        inserted = 0
        for _, row in df.iterrows():
            img_rel = Path(row["image_path"])
            if img_rel.is_absolute():
                new_paths.append(str(img_rel))
                continue
            src_img = (data_root / img_rel).resolve()
            if not src_img.exists():
                raise FileNotFoundError(
                    f"Source image missing: {src_img} (from {row['image_path']})"
                )
            if row["identity_id"] in canary_ids:
                # Real pixel insertion: read the image, apply the
                # identity-specific canary, save a copy alongside the
                # canary CSV, and point the CSV at the copy.  Keep the
                # full relative path (identity_XXX/crop_YYY.jpg) so
                # same-named crops from different identities never collide.
                dst = canary_img_dir / img_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                img = np.array(Image.open(src_img).convert("RGB"))
                img = insert_canary(img, int(row["identity_id"]))
                Image.fromarray(img).save(dst, quality=95)
                new_paths.append(str(dst))
                inserted += 1
            else:
                new_paths.append(str(src_img))
        df["image_path"] = new_paths
        df.to_csv(args.out, index=False)
        logger.info(f"[OK] Canary-tagged CSV → {args.out}")
        logger.info(f"     Identities: {args.identities}")
        logger.info(f"     Pixel canaries inserted into {inserted} images "
                    f"({len(df)} rows total, {len(df) - inserted} untouched)")

    elif args.command == "verify":
        device = resolve_device(args.device)
        model = load_model(args.model, device=str(device))
        results = verify_canary_unlearning(
            model, args.csv, device, args.identities,
            src_csv=args.src_csv,
        )
        # Data output → stdout so `| tee file.json` captures raw JSON;
        # human-readable confirmation → logger (stderr).
        print(json.dumps(results, indent=2))
        logger.info(f"[OK] Canary verification complete for {len(results)} identities")
