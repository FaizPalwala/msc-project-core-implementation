"""
probes.py — Representation-space linear probes for identity-level unlearning verification.

Tests whether the model's internal representations (512-d penultimate features)
still encode identity, age, or gender information after unlearning.

References:
  Golatkar et al. (2020) — representation-level vs output-level forgetting
  Choi & Na (2023)   — MUFAC identity-level verification

Probes
──────
  probe_identity()       — logistic regression on identity_id (600-class)
  probe_age()            — logistic regression on age_group (4-class)
  probe_gender()         — logistic regression on gender (2-class, if column present)
  probe_all()            — run all probes, return summary dict
  measure_forgetting()   — compare feature distributions before vs after unlearning
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import VirtualIdentityDataset, get_val_transform


import logging

logger = logging.getLogger(__name__)


# ── Feature extraction ────────────────────────────────────────────────────────


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract 512-d features, identity labels, and age labels from a loader.

    Returns (features, id_labels, age_labels) as numpy arrays.
    """
    model.eval()
    all_feats: list[np.ndarray] = []
    all_id_labels: list[int] = []
    all_age_labels: list[int] = []

    with torch.no_grad():
        for imgs, id_lbls, age_lbls in loader:
            imgs = imgs.to(device)
            feats = model.feature_vector(imgs)  # (B, 512)
            all_feats.append(feats.cpu().numpy())
            all_id_labels.extend(id_lbls.tolist())
            all_age_labels.extend(age_lbls.tolist())

    features = np.concatenate(all_feats, axis=0)
    return features, np.array(all_id_labels), np.array(all_age_labels)


def _get_meta_column(
    ds: VirtualIdentityDataset,
    column: str,
) -> np.ndarray | None:
    """Return a metadata column as numpy array if it exists, else None."""
    if column in ds.df.columns:
        return ds.df[column].values.astype(int)
    return None


# ── Individual probes ─────────────────────────────────────────────────────────


def probe_identity(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    split: str = "retain",
    batch_size: int = 128,
    C: float = 1.0,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Train a LogisticRegression probe to predict identity (identity_id).

    Returns accuracy and per-class metrics on the probe's predictions.
    """
    ds = VirtualIdentityDataset(csv_path, split=split, transform=get_val_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    features, id_labels, _ = extract_features(model, loader, device)

    if max_samples is not None and len(features) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(features), max_samples, replace=False)
        features = features[idx]
        id_labels = id_labels[idx]

    # 80/20 train/test split for probe evaluation
    n_train = int(0.8 * len(features))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(features))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_test  = scaler.transform(features[test_idx])
    y_train = id_labels[train_idx]
    y_test  = id_labels[test_idx]

    clf = LogisticRegression(
        C=C, max_iter=500, multi_class="multinomial",
        solver="lbfgs", random_state=42,
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    return {
        "probe": "identity",
        "split": split,
        "accuracy": round(float(acc), 4),
        "n_classes": int(np.max(id_labels) + 1),
        "n_train": n_train,
        "n_test": len(y_test),
        "chance_level": round(1.0 / (int(np.max(id_labels) + 1)), 4),
    }


def probe_age(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    split: str = "retain",
    batch_size: int = 128,
    C: float = 1.0,
) -> dict[str, Any]:
    """Train a LogisticRegression probe to predict age group (4-class)."""
    ds = VirtualIdentityDataset(csv_path, split=split, transform=get_val_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    features, _, age_labels = extract_features(model, loader, device)

    n_train = int(0.8 * len(features))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(features))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_test  = scaler.transform(features[test_idx])
    y_train = age_labels[train_idx]
    y_test  = age_labels[test_idx]

    clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    return {
        "probe": "age",
        "split": split,
        "accuracy": round(float(acc), 4),
        "n_classes": 4,
        "n_train": n_train,
        "n_test": len(y_test),
        "chance_level": 0.25,
    }


def probe_gender(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    split: str = "retain",
    batch_size: int = 128,
    C: float = 1.0,
) -> dict[str, Any] | None:
    """Train a LogisticRegression probe to predict gender.

    Returns None if the 'gender' column is not present in the CSV.
    """
    ds = VirtualIdentityDataset(csv_path, split=split, transform=get_val_transform())
    gender_arr = _get_meta_column(ds, "gender")
    if gender_arr is None:
        return None

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    features, _, _ = extract_features(model, loader, device)

    n_train = int(0.8 * len(features))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(features))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_test  = scaler.transform(features[test_idx])
    y_train = gender_arr[train_idx]
    y_test  = gender_arr[test_idx]

    clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    return {
        "probe": "gender",
        "split": split,
        "accuracy": round(float(acc), 4),
        "n_classes": 2,
        "n_train": n_train,
        "n_test": len(y_test),
        "chance_level": 0.50,
    }


# ── Combined probe runner ─────────────────────────────────────────────────────


def probe_all(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    split: str = "retain",
    batch_size: int = 128,
    include_identity: bool = True,
) -> dict[str, Any]:
    """Run all available probes and return a summary dict."""
    results: dict[str, Any] = {}

    if include_identity:
        results["identity"] = probe_identity(
            model, csv_path, device, split=split, batch_size=batch_size,
            max_samples=20_000,
        )

    results["age"] = probe_age(
        model, csv_path, device, split=split, batch_size=batch_size,
    )

    gender_result = probe_gender(
        model, csv_path, device, split=split, batch_size=batch_size,
    )
    if gender_result is not None:
        results["gender"] = gender_result

    return results


# ── Forgetting measurement ────────────────────────────────────────────────────


def measure_forgetting(
    model_unlearned: nn.Module,
    model_original: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Compare representations before and after unlearning.

    Computes:
      - feature_mse: mean squared error between original and unlearned features
        on the forget set (lower = more forgotten)
      - forget_vs_test_distance: cosine similarity between forget and test
        feature centroids (0 = indistinguishable)
      - retain_vs_test_distance: control — should remain stable
    """
    from sklearn.metrics.pairwise import cosine_similarity


    def _centroid(_model, _split):
        ds = VirtualIdentityDataset(csv_path, split=_split, transform=get_val_transform())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        feats, _, _ = extract_features(_model, loader, device)
        return feats.mean(axis=0, keepdims=True)

    forget_orig = _centroid(model_original, "forget")
    forget_unl  = _centroid(model_unlearned, "forget")
    retain_orig = _centroid(model_original, "retain")
    retain_unl  = _centroid(model_unlearned, "retain")
    test_orig   = _centroid(model_original, "test")

    # Feature drift on forget set
    feature_mse_forget = float(np.mean((forget_orig - forget_unl) ** 2))
    feature_mse_retain = float(np.mean((retain_orig - retain_unl) ** 2))

    # Centroid similarity (0 = indistinguishable, 1 = identical)
    forget_test_sim_orig = float(cosine_similarity(forget_orig, test_orig)[0, 0])
    forget_test_sim_unl  = float(cosine_similarity(forget_unl, test_orig)[0, 0])

    return {
        "feature_mse_forget": round(feature_mse_forget, 8),
        "feature_mse_retain": round(feature_mse_retain, 8),
        "forget_test_similarity_original":  round(forget_test_sim_orig, 4),
        "forget_test_similarity_unlearned":  round(forget_test_sim_unl, 4),
        "forgetting_delta": round(
            abs(forget_test_sim_orig - forget_test_sim_unl), 4,
        ),
    }


# ── Printed summary ───────────────────────────────────────────────────────────


def print_probe_summary(results: dict[str, Any]) -> None:
    """Pretty-print probe results."""
    logger.info(f"\n{'='*60}")
    logger.info("  REPRESENTATION PROBE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"{'Probe':<12} {'Accuracy':>10} {'Chance':>10} {'Δ':>10}")
    logger.info("-" * 44)
    for name, res in results.items():
        if res is None:
            continue
        acc = res["accuracy"]
        ch  = res["chance_level"]
        logger.info(f"{name:<12} {acc:>10.4f} {ch:>10.4f} {acc - ch:>+10.4f}")
    logger.info("=" * 60)
