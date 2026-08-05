"""
train.py — Dual-Head ResNet-18 training pipeline.

Trains the original model M on the full training set (retain + forget
identities).  The model is shared across identity (600-class) and age
(4-class) heads, with a configurable loss weighting.

Usage:
    python train.py --csv ../data/dataset/dataset.csv \\
                    --save_dir ../checkpoints \\
                    --epochs 30 --lr 1e-3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import VirtualIdentityDataset, get_train_transform, get_val_transform
from device_utils import resolve_device, resolve_num_workers
from evaluate import evaluate_model
from model import build_dual_head_resnet18, save_model, NUM_IDENTITY_CLASSES, NUM_AGE_CLASSES


import logging

logger = logging.getLogger(__name__)
# ── Seed ──────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training helpers ──────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    age_weight: float = 0.5,
) -> tuple[float, float, float, float]:
    """Train one epoch on dual-label data.

    Returns (avg_loss, identity_acc, age_acc, combined_acc).
    """
    model.train()
    total_loss = 0.0
    id_correct = 0
    age_correct = 0
    total_samples = 0

    for imgs, id_labels, age_labels in loader:
        imgs = imgs.to(device)
        id_labels = id_labels.to(device)
        age_labels = age_labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                id_logits, age_logits = model(imgs)
                loss_id = criterion(id_logits, id_labels)
                loss_age = criterion(age_logits, age_labels)
                loss = loss_id + age_weight * loss_age
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            id_logits, age_logits = model(imgs)
            loss_id = criterion(id_logits, id_labels)
            loss_age = criterion(age_logits, age_labels)
            loss = loss_id + age_weight * loss_age
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        id_correct += (id_logits.argmax(dim=1) == id_labels).sum().item()
        age_correct += (age_logits.argmax(dim=1) == age_labels).sum().item()
        total_samples += imgs.size(0)

    return (
        total_loss / total_samples,
        id_correct / total_samples,
        age_correct / total_samples,
        (id_correct + age_correct) / (2 * total_samples),
    )


def _evaluate_both_heads(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    """Evaluate both heads on a dual-label DataLoader.

    Returns dict with identity_acc, age_acc, id_loss, age_loss.
    """
    model.eval()
    total_id_loss = 0.0
    total_age_loss = 0.0
    id_correct = 0
    age_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, id_labels, age_labels in loader:
            imgs = imgs.to(device)
            id_labels = id_labels.to(device)
            age_labels = age_labels.to(device)
            id_logits, age_logits = model(imgs)

            total_id_loss += criterion(id_logits, id_labels).item() * imgs.size(0)
            total_age_loss += criterion(age_logits, age_labels).item() * imgs.size(0)
            id_correct += (id_logits.argmax(dim=1) == id_labels).sum().item()
            age_correct += (age_logits.argmax(dim=1) == age_labels).sum().item()
            total_samples += imgs.size(0)

    return {
        "id_acc": id_correct / total_samples,
        "age_acc": age_correct / total_samples,
        "id_loss": total_id_loss / total_samples,
        "age_loss": total_age_loss / total_samples,
    }


# ── Main training loop ────────────────────────────────────────────────────────


def train(
    csv_path: str,
    save_dir: str,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    age_weight: float = 0.5,
    val_fraction: float = 0.1,
    seed: int = 42,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device_str: str = "auto",
    run_name: str = "original_model",
    identity_classes: int = NUM_IDENTITY_CLASSES,
    age_classes: int = NUM_AGE_CLASSES,
    subset: str = "all",
):
    """Train dual-head ResNet-18 on SFHQ-InstantID.

    Args:
        csv_path:         Path to dataset CSV.
        save_dir:         Directory for checkpoints and history.
        epochs:           Max training epochs.
        lr:               Adam learning rate.
        batch_size:       Mini-batch size.
        weight_decay:     Adam weight decay.
        age_weight:       Weight of age-head loss relative to identity loss
                          (λ in L = L_id + λ * L_age).
        val_fraction:     Fraction of training data held out for validation.
        seed:             Random seed for reproducibility.
        pretrained:       Use ImageNet-pretrained backbone.
        freeze_backbone:  Freeze convolutional layers (only train heads).
        device_str:       'auto', 'cuda', 'cpu', or 'mps'.
        run_name:         Prefix for saved checkpoint filenames.
        identity_classes: Number of identity classes (default 600).
        age_classes:      Number of age classes (default 4).
    """
    set_seed(seed)
    device = resolve_device(device_str)
    logger.info(f"[INFO] Device: {device} | Seed: {seed} | Epochs: {epochs}")

    # ── Datasets ──────────────────────────────────────────────────────────
    full_train = VirtualIdentityDataset(
        csv_path, split="retain+forget", transform=get_train_transform(),
        subset=subset,
    )
    test_ds = VirtualIdentityDataset(
        csv_path, split="test", transform=get_val_transform(),
        subset=subset,
    )

    n_val = max(1, int(len(full_train) * val_fraction))
    n_trn = len(full_train) - n_val
    trn_ds, val_ds = random_split(
        full_train, [n_trn, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                            num_workers=resolve_num_workers(), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=resolve_num_workers(), pin_memory=True)
    tst_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                            num_workers=resolve_num_workers(), pin_memory=True)

    logger.info(f"[INFO] Train: {len(trn_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_dual_head_resnet18(
        identity_classes=identity_classes,
        age_classes=age_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_val_id_acc = 0.0
    t0 = time.time()

    header = (
        f"{'Epoch':>6} {'Loss':>9} {'IdAcc':>8} {'AgeAcc':>8} "
        f"{'ValId':>8} {'ValAge':>8} {'TestId':>9} {'TestAge':>9} {'LR':>10}"
    )
    logger.info(f"\n{header}")
    logger.info("-" * 95)

    for epoch in range(1, epochs + 1):
        trn_loss, trn_id_acc, trn_age_acc, _ = train_one_epoch(
            model, trn_loader, criterion, optimizer, device, scaler, age_weight,
        )
        val_res = _evaluate_both_heads(model, val_loader, device, criterion)
        tst_res = _evaluate_both_heads(model, tst_loader, device, criterion)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": round(trn_loss, 4),
            "train_id_acc": round(trn_id_acc, 4),
            "train_age_acc": round(trn_age_acc, 4),
            "val_id_acc": round(val_res["id_acc"], 4),
            "val_age_acc": round(val_res["age_acc"], 4),
            "test_id_acc": round(tst_res["id_acc"], 4),
            "test_age_acc": round(tst_res["age_acc"], 4),
            "lr": round(lr_now, 6),
        }
        history.append(row)

        logger.info(
            f"{epoch:>6} {trn_loss:>9.4f} {trn_id_acc:>8.4f} {trn_age_acc:>8.4f} "
            f"{val_res['id_acc']:>8.4f} {val_res['age_acc']:>8.4f} "
            f"{tst_res['id_acc']:>9.4f} {tst_res['age_acc']:>9.4f} "
            f"{lr_now:>10.2e}"
        )

        # Save best (by validation identity accuracy)
        if val_res["id_acc"] > best_val_id_acc:
            best_val_id_acc = val_res["id_acc"]
            save_model(
                model, save_path / f"{run_name}_best.pt",
                metadata={
                    "epoch": epoch,
                    "val_id_acc": best_val_id_acc,
                    "val_age_acc": val_res["age_acc"],
                    "seed": seed,
                    "run_name": run_name,
                },
                optimizer=optimizer,
                epoch=epoch,
            )

    # Save final
    save_model(
        model, save_path / f"{run_name}_final.pt",
        metadata={
            "epochs": epochs,
            "final_test_id_acc": tst_res["id_acc"],
            "final_test_age_acc": tst_res["age_acc"],
            "seed": seed,
            "run_name": run_name,
        },
        epoch=epochs,
    )

    # Save history
    hist_path = save_path / f"{run_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump({
            "config": {
                "csv": csv_path, "epochs": epochs, "lr": lr,
                "batch_size": batch_size, "age_weight": age_weight,
                "seed": seed, "run_name": run_name,
            },
            "history": history,
        }, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\n[OK] Training complete in {elapsed / 60:.1f} min")
    logger.info(f"     Best val identity acc : {best_val_id_acc:.4f}")
    logger.info(f"     Final test identity acc: {tst_res['id_acc']:.4f}")
    logger.info(f"     Final test age acc     : {tst_res['age_acc']:.4f}")
    logger.info(f"     Saved to: {save_path}")
    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Train dual-head ResNet-18 on SFHQ-InstantID",
    )
    parser.add_argument("--csv",             type=str,   required=True)
    parser.add_argument("--save_dir",        type=str,   default="../checkpoints")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--age_weight",      type=float, default=0.5)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--run_name",        type=str,   default="original_model")
    parser.add_argument("--no_pretrain",     action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--device",          type=str,   default="auto")
    parser.add_argument("--subset",          type=str,   default="all",
                        choices=["all", "train", "holdout"],
                        help="Per-image subset filter (default: all, back-compat)")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        save_dir=args.save_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        age_weight=args.age_weight,
        seed=args.seed,
        pretrained=not args.no_pretrain,
        freeze_backbone=args.freeze_backbone,
        device_str=args.device,
        run_name=args.run_name,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
