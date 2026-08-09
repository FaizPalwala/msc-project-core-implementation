"""
baselines.py — Classical machine unlearning baseline methods (dual-head).

Methods:
  1. No-Unlearning Control  — returns original model unchanged
  2. Retrain Oracle (RT)    — retrains from scratch on retain-only data
  3. Gradient Ascent (GA)   — maximises combined loss on forget set
  4. Successive Random
     Relabelling (SRL)      — relabels forget samples randomly, then fine-tunes
  5. Fine-Tuning (FT)       — fine-tunes on retain set only

All methods share the same signature:
    result = method(model, csv_path, device, **config) → dict
        result["model"]    : nn.Module
        result["method"]   : str
        result["metrics"]  : dict
"""

from __future__ import annotations

import random
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset

from dataset import (
    VirtualIdentityDataset, forget_split_name,
    get_train_transform, get_val_transform,
)
from device_utils import resolve_num_workers
from model import build_dual_head_resnet18, copy_model


# ── Shared helper: AMP-aware backward + step ──────────────────────────────────

def _amp_backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> None:
    """Backward + optimizer step, with optional AMP GradScaler."""
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


# ── Shared helper: combined dual-head loss ────────────────────────────────────


def _combined_loss(
    id_logits: torch.Tensor,
    age_logits: torch.Tensor,
    id_labels: torch.Tensor,
    age_labels: torch.Tensor,
    criterion: nn.Module,
    age_weight: float = 0.5,
) -> torch.Tensor:
    """Compute L = L_id + λ · L_age."""
    return criterion(id_logits, id_labels) + age_weight * criterion(age_logits, age_labels)


# ── DataLoader helpers ────────────────────────────────────────────────────────


def _make_loader(
    csv_path: str,
    split: str,
    transform,
    batch_size: int,
    shuffle: bool = False,
    forget_step: int | None = None,
    num_workers: int | None = None,   # None → resolve_num_workers() (env/Slurm-aware)
    subset: str = "all",
    schedule: str = "uniform",
    order_seed: int | None = None,
) -> tuple[DataLoader, int]:
    """Build a dual-label DataLoader."""
    if forget_step is not None and split == "forget":
        split = forget_split_name(forget_step, schedule)
    ds = VirtualIdentityDataset(csv_path, split=split, transform=transform,
                                subset=subset, order_seed=order_seed)
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                   num_workers=resolve_num_workers(num_workers), pin_memory=True),
        len(ds),
    )


# ── 1. No-Unlearning Control ──────────────────────────────────────────────────

def no_unlearning(model: nn.Module, **kwargs) -> dict:
    """Returns the original model unchanged (zero-copy reference).

    Unlike all other methods, no_unlearning does not modify the model,
    so we skip the deep copy.  The caller must not mutate the returned model.
    """
    return {
        "model": model,  # zero-copy — caller must treat as read-only
        "method": "NoUnlearning",
        "metrics": {"unlearning_time_s": 0.0, "steps": 0},
    }


# ── 2. Retrain Oracle (RT) ───────────────────────────────────────────────────


def retrain_oracle(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    age_weight: float = 0.5,
    seed: int = 42,
    pretrained: bool = True,
    forget_step: int | None = None,
    identity_classes: int = 600,
    age_classes: int = 4,
    exclude_identity_ids: list[int] | None = None,
    **kwargs,
) -> dict:
    """Retrain from scratch on retain set (gold standard).

    Two exclusion modes (mutually exclusive):
      - forget_step=N: exclude schedule batch N from the forget set
        (the oracle for the global unlearning task).
      - exclude_identity_ids=[...]: exclude a specific identity set from
        the forget set — used for PER-BIN oracles (Protocol B): retrain
        excluding only one popularity bin's forget identities, so the
        oracle for bin B still trains on the other bins' forgets.
    """
    torch.manual_seed(seed)
    t0 = time.time()

    retain_ds = VirtualIdentityDataset(
        csv_path, split="retain", transform=get_train_transform(),
        subset=kwargs.get("subset", "all"),
        order_seed=kwargs.get("order_seed"),
    )

    if exclude_identity_ids is not None:
        full_forget_ds = VirtualIdentityDataset(
            csv_path, split="forget", transform=get_train_transform(),
            subset=kwargs.get("subset", "all"),
            order_seed=kwargs.get("order_seed"),
        )
        # Exclude ONLY the given identities — the oracle for bin B keeps
        # the other bins' forget identities in its training set (they were
        # not requested for deletion in that bin's counterfactual).
        keep_mask = ~full_forget_ds.df["identity_id"].isin(exclude_identity_ids)
        if keep_mask.any():
            extra = Subset(full_forget_ds,
                           keep_mask.index[keep_mask].tolist())
            retain_ds = ConcatDataset([retain_ds, extra])
    elif forget_step is not None:
        full_forget_ds = VirtualIdentityDataset(
            csv_path, split="forget", transform=get_train_transform(),
            subset=kwargs.get("subset", "all"),
            order_seed=kwargs.get("order_seed"),
        )
        # Exclude the current schedule batch from the oracle's retain set.
        # Column depends on the schedule: uniform → forget_step, poisson →
        # forget_step_poisson.
        schedule = kwargs.get("schedule", "uniform")
        fg_col = "forget_step_poisson" if schedule == "poisson" else "forget_step"
        retain_indices = [
            i for i in range(len(full_forget_ds))
            if int(full_forget_ds.df.iloc[i][fg_col]) != forget_step
        ]
        if retain_indices:
            extra = Subset(full_forget_ds, retain_indices)
            retain_ds = ConcatDataset([retain_ds, extra])

    loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True,
                        num_workers=resolve_num_workers(), pin_memory=True)

    new_model = build_dual_head_resnet18(
        identity_classes=identity_classes,
        age_classes=age_classes,
        pretrained=pretrained,
    ).to(device)

    optimizer = torch.optim.Adam(
        new_model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        new_model.train()
        for imgs, id_labels, age_labels in loader:
            imgs = imgs.to(device)
            id_labels = id_labels.to(device)
            age_labels = age_labels.to(device)

            optimizer.zero_grad()
            id_logits, age_logits = new_model(imgs)
            loss = _combined_loss(
                id_logits, age_logits, id_labels, age_labels,
                criterion, age_weight,
            )
            loss.backward()
            optimizer.step()
        scheduler.step()

    return {
        "model": new_model,
        "method": "RetrainOracle",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "epochs": epochs,
            "forget_step": forget_step,
        },
    }


# ── 3. Gradient Ascent (GA) ──────────────────────────────────────────────────


def gradient_ascent(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    ga_steps: int = 300,
    ga_lr: float = 1e-4,
    batch_size: int = 32,
    age_weight: float = 0.5,
    forget_step: int | None = None,
    retain_reg: bool = True,
    retain_reg_ratio: float = 0.5,
    **kwargs,
) -> dict:
    """Gradient Ascent: maximise combined loss on forget set."""
    t0 = time.time()
    unlearn_model = copy_model(model, device)
    unlearn_model.train()

    f_split = forget_split_name(forget_step, kwargs.get("schedule", "uniform")) if forget_step is not None else "forget"
    f_loader, _ = _make_loader(csv_path, f_split, get_val_transform(),
                               batch_size, shuffle=True,
                               subset=kwargs.get("subset", "all"),
                               order_seed=kwargs.get("order_seed"))
    r_loader = None
    if retain_reg:
        r_loader, _ = _make_loader(csv_path, "retain", get_val_transform(),
                                   batch_size, shuffle=True,
                                   subset=kwargs.get("subset", "all"),
                                   order_seed=kwargs.get("order_seed"))

    optimizer = torch.optim.SGD(
        unlearn_model.parameters(), lr=ga_lr, momentum=0.9,
    )
    criterion = nn.CrossEntropyLoss()

    f_iter = iter(f_loader)
    r_iter = iter(r_loader) if r_loader else None

    for step in range(ga_steps):
        try:
            imgs_f, id_f, age_f = next(f_iter)
        except StopIteration:
            f_iter = iter(f_loader)
            imgs_f, id_f, age_f = next(f_iter)

        imgs_f = imgs_f.to(device)
        id_f = id_f.to(device)
        age_f = age_f.to(device)

        optimizer.zero_grad()
        id_logits, age_logits = unlearn_model(imgs_f)
        loss_f = _combined_loss(id_logits, age_logits, id_f, age_f,
                                criterion, age_weight)
        (-loss_f).backward()  # ascent
        optimizer.step()

        if r_iter is not None and step % max(1, int(1 / retain_reg_ratio)) == 0:
            try:
                imgs_r, id_r, age_r = next(r_iter)
            except StopIteration:
                r_iter = iter(r_loader)
                imgs_r, id_r, age_r = next(r_iter)

            imgs_r = imgs_r.to(device)
            id_r = id_r.to(device)
            age_r = age_r.to(device)

            optimizer.zero_grad()
            id_logits_r, age_logits_r = unlearn_model(imgs_r)
            loss_r = _combined_loss(id_logits_r, age_logits_r, id_r, age_r,
                                    criterion, age_weight)
            loss_r.backward()
            optimizer.step()

    return {
        "model": unlearn_model,
        "method": "GradientAscent",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "ga_steps": ga_steps, "ga_lr": ga_lr,
            "retain_reg": retain_reg, "forget_step": forget_step,
        },
    }


# ── 4. Successive Random Relabelling (SRL) ────────────────────────────────────


class _RelabelledDataset(torch.utils.data.Dataset):
    """Wraps a dual-label Dataset and replaces identity labels with random wrong labels.

    Age labels are preserved (we want the model to retain age knowledge
    while forgetting identity)."""

    def __init__(self, base_ds, num_classes: int = 600):
        self.base = base_ds
        self.num_classes = num_classes
        # identity_label_at() avoids the full image decode+transform
        # that __getitem__ triggers — a ~200× speedup for init.
        self.fake_id_labels = [
            random.choice([c for c in range(num_classes) if c != base_ds.identity_label_at(i)])
            for i in range(len(base_ds))
        ]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _, age_label = self.base[idx]
        return img, self.fake_id_labels[idx], age_label


def successive_random_relabelling(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    srl_epochs: int = 5,
    srl_lr: float = 1e-4,
    batch_size: int = 32,
    age_weight: float = 0.5,
    forget_step: int | None = None,
    identity_classes: int = 600,
    **kwargs,
) -> dict:
    """Randomly relabel forget identities, then fine-tune."""
    t0 = time.time()
    unlearn_model = copy_model(model, device)

    f_split = forget_split_name(forget_step, kwargs.get("schedule", "uniform")) if forget_step is not None else "forget"
    forget_ds = VirtualIdentityDataset(
        csv_path, split=f_split, transform=get_train_transform(),
        subset=kwargs.get("subset", "all"),
        order_seed=kwargs.get("order_seed"),
    )
    relabelled = _RelabelledDataset(forget_ds, num_classes=identity_classes)
    loader = DataLoader(relabelled, batch_size=batch_size, shuffle=True,
                        num_workers=resolve_num_workers(), pin_memory=True)

    optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=srl_lr)
    criterion = nn.CrossEntropyLoss()
    steps = 0

    for _ in range(srl_epochs):
        unlearn_model.train()
        for imgs, fake_id, age_labels in loader:
            imgs = imgs.to(device)
            fake_id = fake_id.to(device)
            age_labels = age_labels.to(device)

            optimizer.zero_grad()
            id_logits, age_logits = unlearn_model(imgs)
            loss = _combined_loss(id_logits, age_logits, fake_id, age_labels,
                                  criterion, age_weight)
            loss.backward()
            optimizer.step()
            steps += 1

    return {
        "model": unlearn_model,
        "method": "RandomRelabelling",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "srl_epochs": srl_epochs, "srl_lr": srl_lr,
            "total_steps": steps, "forget_step": forget_step,
        },
    }


# ── 5. Fine-Tuning on Retain Set (FT) ─────────────────────────────────────────


def fine_tune_retain(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    ft_epochs: int = 5,
    ft_lr: float = 1e-4,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    age_weight: float = 0.5,
    **kwargs,
) -> dict:
    """Fine-tune on retain set only (identity + age heads jointly)."""
    t0 = time.time()
    unlearn_model = copy_model(model, device)

    retain_ds = VirtualIdentityDataset(
        csv_path, split="retain", transform=get_train_transform(),
        subset=kwargs.get("subset", "all"),
        order_seed=kwargs.get("order_seed"),
    )
    loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True,
                        num_workers=resolve_num_workers(), pin_memory=True)

    optimizer = torch.optim.Adam(
        unlearn_model.parameters(), lr=ft_lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)
    criterion = nn.CrossEntropyLoss()
    steps = 0

    for _ in range(ft_epochs):
        unlearn_model.train()
        for imgs, id_labels, age_labels in loader:
            imgs = imgs.to(device)
            id_labels = id_labels.to(device)
            age_labels = age_labels.to(device)

            optimizer.zero_grad()
            id_logits, age_logits = unlearn_model(imgs)
            loss = _combined_loss(id_logits, age_logits, id_labels, age_labels,
                                  criterion, age_weight)
            loss.backward()
            optimizer.step()
            steps += 1
        scheduler.step()

    return {
        "model": unlearn_model,
        "method": "FineTuneRetain",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "ft_epochs": ft_epochs, "ft_lr": ft_lr,
            "total_steps": steps,
        },
    }


# ── Registry ──────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "no_unlearning": no_unlearning,
    "retrain":       retrain_oracle,
    "ga":            gradient_ascent,
    "srl":           successive_random_relabelling,
    "ft":            fine_tune_retain,
}
