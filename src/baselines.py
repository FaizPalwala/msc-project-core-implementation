"""
baselines.py — Classical machine unlearning baseline methods.

Methods implemented:
  1. No-Unlearning Control  : returns original model unchanged
  2. Retrain Oracle (RT)    : retrains from scratch on retain-only data
  3. Gradient Ascent (GA)   : maximises loss on forget set
  4. Successive Random
     Relabelling (SRL)      : relabels forget samples randomly, then fine-tunes
  5. Fine-Tuning (FT)       : fine-tunes on retain set only (no forget data)

All methods share the same signature:
    result = method(model, csv_path, device, config) → dict
        result["model"]    : nn.Module (unlearned model)
        result["method"]   : str method name
        result["metrics"]  : dict of timing and step counts
"""

import copy
import time
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset

from dataset import VirtualIdentityDataset, get_train_transform, get_val_transform
from model import build_resnet18, copy_model


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build data loaders from csv + split name
# ──────────────────────────────────────────────────────────────────────────────

def _make_loader(
    csv_path, split, transform, batch_size, shuffle=False,
    forget_step=None, num_workers=2
):
    if forget_step is not None and split == "forget":
        split = f"forget_step_{forget_step}"
    ds = VirtualIdentityDataset(csv_path, split=split, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True), len(ds)


# ──────────────────────────────────────────────────────────────────────────────
# 1. No-Unlearning Control
# ──────────────────────────────────────────────────────────────────────────────

def no_unlearning(model: nn.Module, **kwargs) -> dict:
    """Returns the original model unchanged. Baseline upper-bound for utility."""
    return {
        "model": copy_model(model, next(model.parameters()).device),
        "method": "NoUnlearning",
        "metrics": {"unlearning_time_s": 0.0, "steps": 0},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Retrain Oracle (RT)
# ──────────────────────────────────────────────────────────────────────────────

def retrain_oracle(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    seed: int = 42,
    pretrained: bool = True,
    forget_step: int = None,   # if set, only removes that step's identity
    **kwargs,
) -> dict:
    """
    Retrain from scratch on retain set only (without forget data).
    This is the GOLD STANDARD that all unlearning methods are compared against.
    For a specific forget_step, removes only that identity from training.
    """
    torch.manual_seed(seed)
    t0 = time.time()

    # Build retain-only training set
    # If forget_step is given: retain + remaining forget steps (not this one)
    retain_ds = VirtualIdentityDataset(csv_path, split="retain", transform=get_train_transform())

    if forget_step is not None:
        # Include all forget steps EXCEPT the current one
        import pandas as pd
        df = pd.read_csv(csv_path)
        other_forget = df[
            (df["split"] == "forget") & (df["forget_step"] != forget_step)
        ]
        # Build a small extra dataset for already-forgotten identities
        from dataset import VirtualIdentityDataset
        # We wrap via indices on the full forget set
        full_forget_ds = VirtualIdentityDataset(csv_path, split="forget",
                                               transform=get_train_transform())
        retain_indices = [
            i for i, row in full_forget_ds.df.iterrows()
            if row["forget_step"] != forget_step
        ]
        if retain_indices:
            extra = Subset(full_forget_ds,
                           [full_forget_ds.df.index.get_loc(i)
                            for i in retain_indices
                            if i in full_forget_ds.df.index])
            retain_ds = ConcatDataset([retain_ds, extra])

    loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    new_model = build_resnet18(pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(new_model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        new_model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(new_model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    elapsed = time.time() - t0
    return {
        "model": new_model,
        "method": "RetrainOracle",
        "metrics": {"unlearning_time_s": round(elapsed, 2),
                    "epochs": epochs, "forget_step": forget_step},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Gradient Ascent (GA)
# ──────────────────────────────────────────────────────────────────────────────

def gradient_ascent(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    ga_steps: int = 300,
    ga_lr: float = 1e-4,
    batch_size: int = 32,
    forget_step: int = None,
    retain_reg: bool = True,       # mix in retain data to prevent catastrophic drift
    retain_reg_ratio: float = 0.5, # fraction of retain batches relative to forget
    **kwargs,
) -> dict:
    """
    Gradient Ascent (GA): maximise cross-entropy on forget set.
    Optionally interleaves retain batches to limit utility collapse.

    ga_steps  : total number of ascent gradient steps
    ga_lr     : learning rate for ascent (small: avoids divergence)
    retain_reg: if True, interleaves retain data to preserve utility
    """
    t0 = time.time()
    unlearn_model = copy_model(model, device)
    unlearn_model.train()

    split = f"forget_step_{forget_step}" if forget_step is not None else "forget"
    forget_loader, n_forget = _make_loader(csv_path, split,
                                           get_val_transform(), batch_size,
                                           shuffle=True)
    retain_loader = None
    if retain_reg:
        retain_loader, _ = _make_loader(csv_path, "retain",
                                        get_val_transform(), batch_size,
                                        shuffle=True)

    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=ga_lr,
                                momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    forget_iter  = iter(forget_loader)
    retain_iter  = iter(retain_loader) if retain_loader else None

    for step in range(ga_steps):
        # ── Ascent on forget ──
        try:
            imgs_f, labels_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            imgs_f, labels_f = next(forget_iter)

        imgs_f, labels_f = imgs_f.to(device), labels_f.to(device)
        optimizer.zero_grad()
        loss_forget = criterion(unlearn_model(imgs_f), labels_f)
        (-loss_forget).backward()   # ASCENT: negate loss
        optimizer.step()

        # ── Optional retain regularisation ──
        if retain_iter is not None and step % max(1, int(1/retain_reg_ratio)) == 0:
            try:
                imgs_r, labels_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                imgs_r, labels_r = next(retain_iter)

            imgs_r, labels_r = imgs_r.to(device), labels_r.to(device)
            optimizer.zero_grad()
            loss_retain = criterion(unlearn_model(imgs_r), labels_r)
            loss_retain.backward()
            optimizer.step()

    elapsed = time.time() - t0
    return {
        "model": unlearn_model,
        "method": "GradientAscent",
        "metrics": {
            "unlearning_time_s": round(elapsed, 2),
            "ga_steps": ga_steps, "ga_lr": ga_lr,
            "retain_reg": retain_reg, "forget_step": forget_step,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Successive Random Relabelling (SRL)
# ──────────────────────────────────────────────────────────────────────────────

class _RelabelledDataset(torch.utils.data.Dataset):
    """Wraps a Dataset and replaces labels with random wrong labels."""

    def __init__(self, base_ds, num_classes: int = 4):
        self.base = base_ds
        self.num_classes = num_classes
        # Pre-assign random wrong labels
        self.random_labels = [
            random.choice([c for c in range(num_classes) if c != base_ds[i][1]])
            for i in range(len(base_ds))
        ]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img, self.random_labels[idx]


def successive_random_relabelling(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    srl_epochs: int = 5,
    srl_lr: float = 1e-4,
    batch_size: int = 32,
    forget_step: int = None,
    num_classes: int = 4,
    **kwargs,
) -> dict:
    """
    Successive Random Relabelling (SRL):
    Replace forget-set labels with random wrong labels, then fine-tune.
    Forces the model to "unlearn" correct labels for forgotten identities.
    """
    t0 = time.time()
    unlearn_model = copy_model(model, device)

    split = f"forget_step_{forget_step}" if forget_step is not None else "forget"
    forget_ds = VirtualIdentityDataset(csv_path, split=split, transform=get_train_transform())
    relabelled_ds = _RelabelledDataset(forget_ds, num_classes=num_classes)
    loader = DataLoader(relabelled_ds, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=srl_lr)
    criterion = nn.CrossEntropyLoss()

    steps = 0
    for epoch in range(srl_epochs):
        unlearn_model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(unlearn_model(imgs), labels)
            loss.backward()
            optimizer.step()
            steps += 1

    elapsed = time.time() - t0
    return {
        "model": unlearn_model,
        "method": "RandomRelabelling",
        "metrics": {
            "unlearning_time_s": round(elapsed, 2),
            "srl_epochs": srl_epochs, "srl_lr": srl_lr,
            "total_steps": steps, "forget_step": forget_step,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Fine-Tuning on Retain Set (FT)
# ──────────────────────────────────────────────────────────────────────────────

def fine_tune_retain(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    ft_epochs: int = 5,
    ft_lr: float = 1e-4,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    **kwargs,
) -> dict:
    """
    Fine-Tuning (FT): fine-tune on retain set only.
    Relies on catastrophic forgetting of the forget set as a side effect.
    Relatively cheap but often under-forgets.
    """
    t0 = time.time()
    unlearn_model = copy_model(model, device)

    retain_ds = VirtualIdentityDataset(csv_path, split="retain", transform=get_train_transform())
    loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=ft_lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)
    criterion = nn.CrossEntropyLoss()

    steps = 0
    for epoch in range(ft_epochs):
        unlearn_model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(unlearn_model(imgs), labels)
            loss.backward()
            optimizer.step()
            steps += 1
        scheduler.step()

    elapsed = time.time() - t0
    return {
        "model": unlearn_model,
        "method": "FineTuneRetain",
        "metrics": {
            "unlearning_time_s": round(elapsed, 2),
            "ft_epochs": ft_epochs, "ft_lr": ft_lr,
            "total_steps": steps,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Registry — map method name → function
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "no_unlearning":  no_unlearning,
    "retrain":        retrain_oracle,
    "ga":             gradient_ascent,
    "srl":            successive_random_relabelling,
    "ft":             fine_tune_retain,
}
