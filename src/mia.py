"""
mia.py — Membership Inference Attack suite for identity-level unlearning.

Attacks
───────
  Confidence-threshold MIA   — standard attack (Yeom et al., 2018)
  Loss-threshold MIA         — negated-loss variant
  Per-identity MIA           — separate AUC per forget identity cluster
  Max-confidence attack      — worst-case: single highest-confidence image
  Demographic-stratified MIA — breakdown by age group / popularity bin

All attacks operate on either the identity head or the age head.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from dataset import VirtualIdentityDataset, get_val_transform



import logging

logger = logging.getLogger(__name__)
# ── Score extraction ──────────────────────────────────────────────────────────


def get_confidence_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head: str = "identity",
) -> np.ndarray:
    """Max-softmax confidence per sample for the specified head.

    Args:
        head: 'identity' or 'age' — which head to extract scores from.
    """
    model.eval()
    confs: list[float] = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = imgs.to(device)
            id_logits, age_logits = model(imgs)
            logits = id_logits if head == "identity" else age_logits
            probs = torch.softmax(logits, dim=1)
            conf, _ = probs.max(dim=1)
            confs.extend(conf.cpu().tolist())
    return np.array(confs)


def get_loss_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head: str = "identity",
) -> np.ndarray:
    """Per-sample cross-entropy loss (negated so higher = more member-like).

    Args:
        head: 'identity' or 'age' — which head's labels to use.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses: list[float] = []
    with torch.no_grad():
        for imgs, id_labels, age_labels in loader:
            imgs = imgs.to(device)
            labels = id_labels if head == "identity" else age_labels
            labels = labels.to(device)
            id_logits, age_logits = model(imgs)
            logits = id_logits if head == "identity" else age_logits
            loss = criterion(logits, labels)
            losses.extend((-loss).cpu().tolist())  # negate: higher = member
    return np.array(losses)


# ── AUC computation ───────────────────────────────────────────────────────────


def compute_mia_auc(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> float:
    """ROC-AUC for membership inference.

    0.50 = attack failed (perfect unlearning)
    1.00 = perfect attack (complete memorisation)
    """
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(nonmember_scores)),
    ])
    return float(roc_auc_score(labels, scores))


# ── Threshold-based MIA ───────────────────────────────────────────────────────


def run_mia_threshold(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    forget_loader: DataLoader,
    device: torch.device,
    score_type: str = "confidence",
    head: str = "identity",
) -> dict[str, Any]:
    """Standard threshold-based MIA on three loaders.

    Three-way evaluation:
      (A) Retain vs Test — baseline attack on normal training data
          (should show higher AUC — model distinguishes these)
      (B) Forget vs Test — KEY metric: should approach 0.50 after unlearning
    """
    score_fn = get_confidence_scores if score_type == "confidence" else get_loss_scores

    retain_scores = score_fn(model, member_loader, device, head=head)
    test_scores   = score_fn(model, nonmember_loader, device, head=head)
    forget_scores = score_fn(model, forget_loader, device, head=head)

    retain_test_auc = compute_mia_auc(retain_scores, test_scores)
    forget_test_auc = compute_mia_auc(forget_scores, test_scores)

    # Threshold accuracy at median
    all_scores = np.concatenate([forget_scores, test_scores])
    all_labels = np.concatenate([
        np.ones(len(forget_scores)), np.zeros(len(test_scores)),
    ])
    threshold = float(np.median(all_scores))
    preds = (all_scores >= threshold).astype(int)
    forget_test_acc = float(accuracy_score(all_labels, preds))

    return {
        "score_type": score_type,
        "head": head,
        "retain_test_auc": round(retain_test_auc, 4),
        "forget_test_auc": round(forget_test_auc, 4),
        "forget_test_acc": round(forget_test_acc, 4),
        "forget_advantage": round(abs(forget_test_auc - 0.5), 4),
        "n_retain": len(retain_scores),
        "n_test": len(test_scores),
        "n_forget": len(forget_scores),
    }


# ── Convenience: build loaders from CSV ───────────────────────────────────────


def _build_loaders(
    csv_path: str,
    batch_size: int = 128,
    forget_split: str = "forget",
    subset: str = "all",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build retain, test, forget DataLoaders.

    Non-member reference ("test_loader"): in the new dataset schema there is
    no identity-disjoint "test" split anymore (every identity is retain or
    forget, each with train/holdout image subsets).  The held-out retain
    images (subset='holdout') are the non-member reference — images the
    model never saw during training, from identities it was trained on.
    This is cleaner than the old unseen-identity split, which carried a
    systematic low-confidence confound.
    """
    kw = dict(batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    retain_loader = DataLoader(
        VirtualIdentityDataset(csv_path, "retain", transform=get_val_transform(),
                               subset=subset), **kw,
    )
    test_loader = DataLoader(
        VirtualIdentityDataset(csv_path, "retain", transform=get_val_transform(),
                               subset="holdout"), **kw,
    )
    forget_loader = DataLoader(
        VirtualIdentityDataset(csv_path, forget_split, transform=get_val_transform(),
                               subset=subset), **kw,
    )
    return retain_loader, test_loader, forget_loader


def run_mia_full(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
    score_type: str = "confidence",
    head: str = "identity",
    verbose: bool = True,
    subset: str = "all",
) -> dict[str, Any]:
    """Run MIA on full forget set (all 60 identities)."""
    retain_loader, test_loader, forget_loader = _build_loaders(
        csv_path, batch_size, subset=subset)

    result = run_mia_threshold(
        model, retain_loader, test_loader, forget_loader,
        device, score_type=score_type, head=head,
    )

    if verbose:
        _print_mia_result(result)

    return result


# ── Per-identity MIA ──────────────────────────────────────────────────────────


def run_mia_per_identity(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
    score_type: str = "confidence",
    head: str = "identity",
    subset: str = "all",
) -> dict[str, Any]:
    """Run MIA separately for each forget identity (identity_id).

    Returns dict with per-identity AUCs + aggregate statistics.
    """
    _, test_loader, _ = _build_loaders(csv_path, batch_size, subset=subset)

    # Pre-compute test scores once
    score_fn = get_confidence_scores if score_type == "confidence" else get_loss_scores
    test_scores = score_fn(model, test_loader, device, head=head)

    # Iterate over forget identities
    forget_ds = VirtualIdentityDataset(
        csv_path, split="forget", transform=get_val_transform(),
        subset=subset,
    )
    id_to_indices: dict[int, list[int]] = defaultdict(list)
    for i in range(len(forget_ds)):
        cid = int(forget_ds.df.iloc[i]["identity_id"])
        id_to_indices[cid].append(i)

    per_id_auc: dict[int, float] = {}
    per_id_max_conf: dict[int, float] = {}
    per_id_median_conf: dict[int, float] = {}

    for cid, indices in sorted(id_to_indices.items()):
        subset = torch.utils.data.Subset(forget_ds, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
        member_scores = score_fn(model, loader, device, head=head)
        auc = compute_mia_auc(member_scores, test_scores)
        per_id_auc[cid] = round(auc, 4)
        per_id_max_conf[cid] = round(float(member_scores.max()), 4)
        per_id_median_conf[cid] = round(float(np.median(member_scores)), 4)

    auc_values = list(per_id_auc.values())
    max_values = list(per_id_max_conf.values())
    median_values = list(per_id_median_conf.values())

    return {
        "head": head,
        "score_type": score_type,
        "per_identity_auc": per_id_auc,
        "per_identity_max_confidence": per_id_max_conf,
        "per_identity_median_confidence": per_id_median_conf,
        # Aggregates
        "mean_auc": round(float(np.mean(auc_values)), 4) if auc_values else 0.0,
        "std_auc": round(float(np.std(auc_values)), 4) if auc_values else 0.0,
        "max_auc": round(float(np.max(auc_values)), 4) if auc_values else 0.0,
        "min_auc": round(float(np.min(auc_values)), 4) if auc_values else 0.0,
        "max_image_confidence": round(float(np.max(max_values)), 4) if max_values else 0.0,
        "mean_max_confidence": round(float(np.mean(max_values)), 4) if max_values else 0.0,
        "fraction_leaked": round(
            float(np.mean([1.0 if a > 0.55 else 0.0 for a in auc_values])), 4,
        ) if auc_values else 0.0,
        "n_identities": len(per_id_auc),
        "n_test": len(test_scores),
    }


# ── Max-confidence attack ─────────────────────────────────────────────────────


def run_max_confidence_attack(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
    head: str = "identity",
    subset: str = "all",
) -> dict[str, Any]:
    """Single-image worst-case attack.

    For each forget identity, the attacker uses ONLY the image with the
    highest model confidence.  If even that single image can't be
    distinguished from test-set images, the identity is fully forgotten.
    """
    _, test_loader, _ = _build_loaders(csv_path, batch_size, subset=subset)

    # Pre-compute test confidence scores
    model.eval()
    test_confs: list[float] = []
    with torch.no_grad():
        for imgs, _, _ in test_loader:
            imgs = imgs.to(device)
            id_logits, age_logits = model(imgs)
            logits = id_logits if head == "identity" else age_logits
            probs = torch.softmax(logits, dim=1)
            conf, _ = probs.max(dim=1)
            test_confs.extend(conf.cpu().tolist())
    test_scores_arr = np.array(test_confs)

    # Per-identity max confidence
    forget_ds = VirtualIdentityDataset(
        csv_path, split="forget", transform=get_val_transform(),
        subset=subset,
    )
    id_to_indices: dict[int, list[int]] = defaultdict(list)
    for i in range(len(forget_ds)):
        cid = int(forget_ds.df.iloc[i]["identity_id"])
        id_to_indices[cid].append(i)

    max_per_id: list[float] = []
    for cid, indices in sorted(id_to_indices.items()):
        subset = torch.utils.data.Subset(forget_ds, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
        confs: list[float] = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs = imgs.to(device)
                id_logits, age_logits = model(imgs)
                logits = id_logits if head == "identity" else age_logits
                probs = torch.softmax(logits, dim=1)
                c, _ = probs.max(dim=1)
                confs.extend(c.cpu().tolist())
        max_per_id.append(float(np.max(confs)))

    max_per_id_arr = np.array(max_per_id)
    max_auc = compute_mia_auc(max_per_id_arr, test_scores_arr)

    return {
        "head": head,
        "max_confidence_auc": round(max_auc, 4),
        "max_confidence_advantage": round(abs(max_auc - 0.5), 4),
        "mean_max_confidence": round(float(max_per_id_arr.mean()), 4),
        "max_single_image_confidence": round(float(max_per_id_arr.max()), 4),
        "n_identities": len(max_per_id),
        "n_test": len(test_scores_arr),
    }


# ── Demographic-stratified MIA ────────────────────────────────────────────────


def run_mia_per_demographic(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
    score_type: str = "confidence",
    head: str = "identity",
    subset: str = "all",
) -> dict[str, Any]:
    """MIA broken down by age group and popularity bin.

    Returns nested dict: results[group_key] → standard MIA result dict.
    """
    _, test_loader, _ = _build_loaders(csv_path, batch_size, subset=subset)
    score_fn = get_confidence_scores if score_type == "confidence" else get_loss_scores
    test_scores = score_fn(model, test_loader, device, head=head)

    forget_ds = VirtualIdentityDataset(
        csv_path, split="forget", transform=get_val_transform(),
        subset=subset,
    )
    results: dict[str, Any] = {}

    # By age group
    if "age_group" in forget_ds.df.columns:
        for ag in sorted(forget_ds.df["age_group"].unique()):
            indices = [i for i in range(len(forget_ds))
                       if int(forget_ds.df.iloc[i]["age_group"]) == ag]
            if not indices:
                continue
            subset = torch.utils.data.Subset(forget_ds, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
            member_scores = score_fn(model, loader, device, head=head)
            auc = compute_mia_auc(member_scores, test_scores)
            key = f"age_group_{int(ag)}"
            results[key] = {
                "auc": round(auc, 4),
                "forget_advantage": round(abs(auc - 0.5), 4),
                "n_samples": len(indices),
            }

    # By popularity bin
    if "popularity_bin" in forget_ds.df.columns:
        for pb in sorted(forget_ds.df["popularity_bin"].dropna().unique()):
            indices = [i for i in range(len(forget_ds))
                       if forget_ds.df.iloc[i].get("popularity_bin") == pb]
            if not indices:
                continue
            subset = torch.utils.data.Subset(forget_ds, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
            member_scores = score_fn(model, loader, device, head=head)
            auc = compute_mia_auc(member_scores, test_scores)
            key = f"popularity_{str(pb)}"
            results[key] = {
                "auc": round(auc, 4),
                "forget_advantage": round(abs(auc - 0.5), 4),
                "n_samples": len(indices),
            }

    return results


# ── Display helpers ───────────────────────────────────────────────────────────


def _print_mia_result(result: dict[str, Any]) -> None:
    """Pretty-print a single MIA result."""
    logger.info(f"\n[MIA — {result.get('head', '?')} head / "
          f"{result.get('score_type', '?')}]")
    logger.info(f"  Retain vs Test  AUC : {result['retain_test_auc']:.4f}  "
          f"(higher = model distinguishes members)")
    logger.info(f"  Forget vs Test  AUC : {result['forget_test_auc']:.4f}  "
          f"(target: 0.5000 = perfectly forgotten)")
    logger.info(f"  Forget Advantage    : {result['forget_advantage']:.4f}  "
          f"(target: 0.0000)")
    logger.info(f"  Forget vs Test  Acc : {result['forget_test_acc']:.4f}  "
          f"(target: ~0.5000)")


def print_per_identity_summary(result: dict[str, Any]) -> None:
    """Print per-identity MIA summary."""
    logger.info(f"\n[Per-Identity MIA — {result.get('head', '?')} head / "
          f"{result.get('score_type', '?')}]")
    logger.info(f"  Identities: {result['n_identities']}")
    logger.info(f"  Mean AUC:   {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    logger.info(f"  Max  AUC:   {result['max_auc']:.4f}  (worst identity)")
    logger.info(f"  Min  AUC:   {result['min_auc']:.4f}  (best identity)")
    logger.info(f"  Max  image confidence: {result['max_image_confidence']:.4f}")
    logger.info(f"  Fraction leaked (AUC>0.55): {result['fraction_leaked']:.4f}")
