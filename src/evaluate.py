"""
evaluate.py — Evaluation utilities.

evaluate_model()     → accuracy, loss, per-class metrics
evaluate_full()      → evaluates retain, forget, test splits and returns dict
"""

import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module = None,
) -> dict:
    """
    Evaluate a model on a DataLoader.
    Returns dict with: accuracy, loss (if criterion given),
    per_class_acc, confidences, labels, predictions.
    """
    model.eval()
    all_preds, all_labels, all_confs, all_logits = [], [], [], []
    total_loss = 0.0
    criterion = criterion or nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(conf.cpu().tolist())
            all_logits.extend(logits.cpu().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)

    accuracy = float((all_preds == all_labels).mean())
    avg_loss = total_loss / len(all_labels)
    avg_conf = float(all_confs.mean())

    # Per-class accuracy
    num_classes = max(all_labels.max(), all_preds.max()) + 1
    per_class = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[int(c)] = float((all_preds[mask] == all_labels[mask]).mean())

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "avg_confidence": avg_conf,
        "per_class_accuracy": per_class,
        "n_samples": len(all_labels),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "confidences": all_confs.tolist(),
        "logits": all_logits,
    }


def evaluate_full(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
    forget_step: int = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate model on retain, test, and forget splits.
    If forget_step is given, also evaluate that specific step's identity.

    Returns nested dict keyed by split name.
    """
    from dataset import SFHQDataset, get_val_transform

    criterion = nn.CrossEntropyLoss()
    results = {}

    splits_to_eval = ["retain", "test", "forget"]
    if forget_step is not None:
        splits_to_eval.append(f"forget_step_{forget_step}")

    for split in splits_to_eval:
        ds = SFHQDataset(csv_path, split=split, transform=get_val_transform())
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        metrics = evaluate_model(model, loader, device, criterion)
        results[split] = {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "avg_confidence": metrics["avg_confidence"],
            "per_class_accuracy": metrics["per_class_accuracy"],
            "n_samples": metrics["n_samples"],
        }

    if verbose:
        print(f"\n{'Split':<20} {'Accuracy':>10} {'Loss':>10} {'N':>8}")
        print("-" * 52)
        for split, m in results.items():
            print(f"{split:<20} {m['accuracy']:>10.4f} {m['loss']:>10.4f} "
                  f"{m['n_samples']:>8}")

    return results
