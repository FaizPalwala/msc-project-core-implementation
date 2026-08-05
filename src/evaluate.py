"""
evaluate.py — Dual-head evaluation utilities.

evaluate_model()         → per-split metrics, both heads
evaluate_full()          → all splits + per-identity breakdown
evaluate_per_identity()  → one dict per forget identity
evaluate_per_demographic() → breakdown by age group / popularity bin
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import VirtualIdentityDataset, get_val_transform



import logging

logger = logging.getLogger(__name__)
# ── Per-sample model evaluation (both heads) ──────────────────────────────────


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate both heads on a dual-label DataLoader.

    Returns dict with per-head accuracy, loss, per-class metrics,
    and raw logits/confidences/labels for downstream MIA.
    """
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()

    # Accumulators
    all_id_preds: list[int]   = []
    all_id_labels: list[int]  = []
    all_id_confs: list[float] = []
    all_id_logits: list[list[float]] = []

    all_age_preds: list[int]   = []
    all_age_labels: list[int]  = []
    all_age_confs: list[float] = []
    all_age_logits: list[list[float]] = []

    total_id_loss = 0.0
    total_age_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, id_labels, age_labels in loader:
            imgs = imgs.to(device)
            id_labels = id_labels.to(device)
            age_labels = age_labels.to(device)

            id_logits, age_logits = model(imgs)

            loss_id = criterion(id_logits, id_labels)
            loss_age = criterion(age_logits, age_labels)
            total_id_loss += loss_id.item() * imgs.size(0)
            total_age_loss += loss_age.item() * imgs.size(0)
            total_samples += imgs.size(0)

            # Identity head
            id_probs = torch.softmax(id_logits, dim=1)
            id_conf, id_pred = id_probs.max(dim=1)
            all_id_preds.extend(id_pred.cpu().tolist())
            all_id_labels.extend(id_labels.cpu().tolist())
            all_id_confs.extend(id_conf.cpu().tolist())
            all_id_logits.extend(id_logits.cpu().tolist())

            # Age head
            age_probs = torch.softmax(age_logits, dim=1)
            age_conf, age_pred = age_probs.max(dim=1)
            all_age_preds.extend(age_pred.cpu().tolist())
            all_age_labels.extend(age_labels.cpu().tolist())
            all_age_confs.extend(age_conf.cpu().tolist())
            all_age_logits.extend(age_logits.cpu().tolist())

    def _head_summary(preds, labels, confs, loss_total) -> dict:
        preds_arr  = np.array(preds)
        labels_arr = np.array(labels)
        confs_arr  = np.array(confs)

        acc = float((preds_arr == labels_arr).mean())
        avg_loss = loss_total / max(total_samples, 1)
        avg_conf = float(confs_arr.mean())

        # Per-class accuracy
        num_classes = int(max(labels_arr.max(), preds_arr.max()) + 1)
        per_class = {}
        for c in range(num_classes):
            mask = labels_arr == c
            if mask.sum() > 0:
                per_class[c] = float((preds_arr[mask] == c).mean())

        return {
            "accuracy": acc,
            "loss": avg_loss,
            "avg_confidence": avg_conf,
            "per_class_accuracy": per_class,
        }

    id_summary = _head_summary(all_id_preds, all_id_labels, all_id_confs, total_id_loss)
    age_summary = _head_summary(all_age_preds, all_age_labels, all_age_confs, total_age_loss)

    return {
        "n_samples": total_samples,
        "identity": id_summary,
        "age": age_summary,
        # Raw arrays for MIA / probing
        "id_predictions": all_id_preds,
        "id_labels": all_id_labels,
        "id_confidences": all_id_confs,
        "id_logits": all_id_logits,
        "age_predictions": all_age_preds,
        "age_labels": all_age_labels,
        "age_confidences": all_age_confs,
        "age_logits": all_age_logits,
    }


# ── Full-split evaluation ─────────────────────────────────────────────────────


def evaluate_full(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
    verbose: bool = True,
    subset: str = "all",   # "all" | "train" | "holdout"
) -> dict[str, Any]:
    """Evaluate on retain, test, and forget splits (both heads)."""
    results: dict[str, Any] = {}

    for split in ["retain", "test", "forget"]:
        ds = VirtualIdentityDataset(csv_path, split=split, transform=get_val_transform(),
                                    subset=subset)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        results[split] = evaluate_model(model, loader, device)

    if verbose:
        logger.info(f"\n{'Split':<12} {'IdAcc':>9}  {'IdLoss':>9}  "
              f"{'AgeAcc':>9}  {'AgeLoss':>9}  {'N':>7}")
        logger.info("-" * 58)
        for split, m in results.items():
            logger.info(
                f"{split:<12} {m['identity']['accuracy']:>9.4f}  "
                f"{m['identity']['loss']:>9.4f}  "
                f"{m['age']['accuracy']:>9.4f}  "
                f"{m['age']['loss']:>9.4f}  "
                f"{m['n_samples']:>7}"
            )

    return results


# ── Per-identity breakdown ────────────────────────────────────────────────────


def evaluate_per_identity(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
    subset: str = "all",
) -> dict[int, dict[str, Any]]:
    """Evaluate each forget identity (forget_variant) separately.

    Returns dict keyed by identity_id → per-head eval summary.
    Uses the entire forget split, iterating over unique identity IDs.
    """
    ds = VirtualIdentityDataset(csv_path, split="forget", transform=get_val_transform(),
                                subset=subset)
    id_to_indices: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        cid = int(ds.df.iloc[i]["identity_id"])
        id_to_indices[cid].append(i)

    results: dict[int, dict[str, Any]] = {}
    for cid, indices in sorted(id_to_indices.items()):
        subset = torch.utils.data.Subset(ds, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
        results[cid] = evaluate_model(model, loader, device)

    return results


# ── Demographic-stratified evaluation ─────────────────────────────────────────


def evaluate_per_demographic(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, dict[str, Any]]:
    """Evaluate forget split stratified by age-group and popularity bin."""
    ds = VirtualIdentityDataset(csv_path, split="forget", transform=get_val_transform())
    results: dict[str, dict[str, Any]] = {}

    # By age group
    if "age_group" in ds.df.columns:
        for ag in sorted(ds.df["age_group"].unique()):
            indices = [i for i in range(len(ds))
                       if int(ds.df.iloc[i]["age_group"]) == ag]
            if not indices:
                continue
            subset = torch.utils.data.Subset(ds, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
            key = f"age_group_{int(ag)}"
            results[key] = evaluate_model(model, loader, device)

    # By popularity bin (imbalanced dataset only)
    if "popularity_bin" in ds.df.columns:
        for pb in sorted(ds.df["popularity_bin"].dropna().unique()):
            indices = [i for i in range(len(ds))
                       if ds.df.iloc[i].get("popularity_bin") == pb]
            if not indices:
                continue
            subset = torch.utils.data.Subset(ds, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
            key = f"popularity_{pb}"
            results[key] = evaluate_model(model, loader, device)

    return results
