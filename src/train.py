"""
train.py — ResNet-18 training pipeline.

Trains the original model M on the full training set (retain + forget).
This is the model that all unlearning methods start from.

Usage:
    python train.py --csv ../data/dataset/sfhq_dataset.csv \
                    --save_dir ../checkpoints \
                    --epochs 30 --lr 1e-3 --seed 42
"""

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
from model import build_resnet18, save_model
from evaluate import evaluate_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def train(
    csv_path: str,
    save_dir: str,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.1,
    seed: int = 42,
    pretrained: bool = True,
    device_str: str = "auto",
    run_name: str = "original_model",
):
    set_seed(seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device_str == "auto" else torch.device(device_str)
    print(f"[INFO] Device: {device} | Seed: {seed} | Epochs: {epochs}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    full_train = VirtualIdentityDataset(csv_path, split="retain+forget",
                                         transform=get_train_transform())
    test_ds    = VirtualIdentityDataset(csv_path, split="test",
                                         transform=get_val_transform())

    # Val split from training data
    n_val = max(1, int(len(full_train) * val_fraction))
    n_trn = len(full_train) - n_val
    trn_ds, val_ds = random_split(
        full_train, [n_trn, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    # Val subset gets val transform
    val_ds.dataset.transform = get_val_transform()

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)
    tst_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"[INFO] Train: {len(trn_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    model = build_resnet18(pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_acc = 0.0
    t0 = time.time()

    print(f"\n{'Epoch':>6} {'TrnLoss':>9} {'TrnAcc':>8} {'ValAcc':>8} {'TestAcc':>9} {'LR':>10}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        trn_loss, trn_acc = train_one_epoch(
            model, trn_loader, criterion, optimizer, device, scaler
        )
        val_metrics  = evaluate_model(model, val_loader,  device, criterion)
        test_metrics = evaluate_model(model, tst_loader,  device, criterion)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch, "trn_loss": round(trn_loss, 4),
            "trn_acc": round(trn_acc, 4),
            "val_acc":  round(val_metrics["accuracy"], 4),
            "test_acc": round(test_metrics["accuracy"], 4),
            "lr": round(lr_now, 6)
        }
        history.append(row)
        print(f"{epoch:>6} {trn_loss:>9.4f} {trn_acc:>8.4f} "
              f"{val_metrics['accuracy']:>8.4f} {test_metrics['accuracy']:>9.4f} "
              f"{lr_now:>10.2e}")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_model(model, save_path / f"{run_name}_best.pt",
                       metadata={"epoch": epoch, "val_acc": best_val_acc,
                                 "seed": seed, "run_name": run_name},
                       optimizer=optimizer, epoch=epoch)

    # Save final model
    save_model(model, save_path / f"{run_name}_final.pt",
               metadata={"epochs": epochs, "final_test_acc": test_metrics["accuracy"],
                         "seed": seed, "run_name": run_name},
               epoch=epochs)

    # Save training history
    hist_path = save_path / f"{run_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump({"config": {"csv": csv_path, "epochs": epochs, "lr": lr,
                               "seed": seed, "run_name": run_name},
                   "history": history}, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[OK] Training complete in {elapsed/60:.1f} min")
    print(f"     Best val acc : {best_val_acc:.4f}")
    print(f"     Final test acc: {test_metrics['accuracy']:.4f}")
    print(f"     Saved to: {save_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 on SFHQ")
    parser.add_argument("--csv",        type=str,   required=True)
    parser.add_argument("--save_dir",   type=str,   default="../checkpoints")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--run_name",   type=str,   default="original_model")
    parser.add_argument("--no_pretrain",action="store_true")
    parser.add_argument("--device",     type=str,   default="auto")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        save_dir=args.save_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        pretrained=not args.no_pretrain,
        device_str=args.device,
        run_name=args.run_name,
    )
