"""
sota_methods.py — State-of-the-Art Unlearning Methods (dual-head).

Implements three methods from the Deep Unlearn benchmark
(Cadet et al., 2024 — arXiv:2410.01276):

  1. NG+  — Negative Gradient Plus
  2. MSG  — Masked Small Gradients
  3. CT   — Convolution Transpose (Selective Weight Dampening)

All methods operate on the combined dual-head loss:
  L = CE(id_logits, id_labels) + age_weight · CE(age_logits, age_labels)
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines import _combined_loss as _closs
from dataset import VirtualIdentityDataset, forget_split_name, get_val_transform
from device_utils import resolve_num_workers
from model import copy_model



import logging

logger = logging.getLogger(__name__)
# ── Helpers ───────────────────────────────────────────────────────────────────


def _loader(csv, split, transform, batch_size, shuffle=False, num_workers=resolve_num_workers(), subset="all"):
    ds = VirtualIdentityDataset(csv, split=split, transform=transform, subset=subset)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True), len(ds)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _grad_vector(model, loader, criterion, device, n_batches=10) -> dict[str, torch.Tensor]:
    """Compute mean |gradient| per parameter over n_batches."""
    model.eval()
    accum = {n: torch.zeros_like(p) for n, p in model.named_parameters()
             if p.requires_grad}
    count = 0
    for imgs, id_lbls, age_lbls in loader:
        if count >= n_batches:
            break
        imgs = imgs.to(device)
        id_lbls = id_lbls.to(device)
        age_lbls = age_lbls.to(device)
        model.zero_grad()
        id_logits, age_logits = model(imgs)
        loss = criterion(id_logits, id_lbls)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                accum[n] += p.grad.abs().detach()
        count += 1
    if count > 0:
        for n in accum:
            accum[n] /= count
    model.zero_grad()
    return accum


# ── 1. NG+ ────────────────────────────────────────────────────────────────────


def neg_grad_plus(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: Optional[int] = None,
    ng_steps: int = 400,
    ng_lr_ascent: float = 5e-5,
    ng_lr_retain: float = 1e-4,
    retain_steps_per_ascent: int = 1,
    batch_size: int = 32,
    age_weight: float = 0.5,
    kl_weight: float = 0.5,
    **kwargs,
) -> dict:
    """NG+: ascent on forget + descent on retain + KL anchor."""
    t0 = time.time()
    unlearn_m = copy_model(model, device)

    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split = forget_split_name(forget_step, kwargs.get("schedule", "uniform")) if forget_step is not None else "forget"
    subset_ = kwargs.get("subset", "all")
    f_loader, _ = _loader(csv_path, f_split, get_val_transform(), batch_size, shuffle=True, subset=subset_)
    r_loader, _ = _loader(csv_path, "retain", get_val_transform(), batch_size, shuffle=True, subset=subset_)

    criterion = nn.CrossEntropyLoss()
    kl_crit   = nn.KLDivLoss(reduction="batchmean", log_target=True)
    opt_f = torch.optim.SGD(unlearn_m.parameters(), lr=ng_lr_ascent, momentum=0.9)
    opt_r = torch.optim.SGD(unlearn_m.parameters(), lr=ng_lr_retain, momentum=0.9)
    # AMP (mixed precision) — enabled on CUDA for the 400-step loops
    use_amp = device.type == "cuda"
    scaler_f = torch.amp.GradScaler(device.type) if use_amp else None
    scaler_r = torch.amp.GradScaler(device.type) if use_amp else None

    f_iter = _cycle(f_loader)
    r_iter = _cycle(r_loader)
    forget_losses, retain_losses = [], []

    for _ in range(ng_steps):
        unlearn_m.train()

        # Ascent on forget
        imgs_f, id_f, age_f = next(f_iter)
        imgs_f = imgs_f.to(device)
        id_f = id_f.to(device)
        age_f = age_f.to(device)
        opt_f.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda"):
                id_logits_f, age_logits_f = unlearn_m(imgs_f)
                loss_f = _closs(id_logits_f, age_logits_f, id_f, age_f, criterion, age_weight)
            scaler_f.scale(-loss_f).backward()   # ascent via negated scaled loss
            scaler_f.step(opt_f)
            scaler_f.update()
        else:
            id_logits_f, age_logits_f = unlearn_m(imgs_f)
            loss_f = _closs(id_logits_f, age_logits_f, id_f, age_f, criterion, age_weight)
            (-loss_f).backward()
            opt_f.step()
        forget_losses.append(loss_f.item())

        # Descent on retain
        for _ in range(retain_steps_per_ascent):
            imgs_r, id_r, age_r = next(r_iter)
            imgs_r = imgs_r.to(device)
            id_r = id_r.to(device)
            age_r = age_r.to(device)
            opt_r.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    id_logits_r, age_logits_r = unlearn_m(imgs_r)
                    loss_ce = _closs(id_logits_r, age_logits_r, id_r, age_r,
                                     criterion, age_weight)
                    if kl_weight > 0:
                        with torch.no_grad():
                            ref_id, ref_age = ref_model(imgs_r)
                        log_p = torch.log_softmax(id_logits_r, dim=1)
                        log_q = torch.log_softmax(ref_id, dim=1)
                        loss_r = loss_ce + kl_weight * kl_crit(log_p, log_q)
                    else:
                        loss_r = loss_ce
                scaler_r.scale(loss_r).backward()
                scaler_r.step(opt_r)
                scaler_r.update()
            else:
                id_logits_r, age_logits_r = unlearn_m(imgs_r)
                loss_ce = _closs(id_logits_r, age_logits_r, id_r, age_r,
                                 criterion, age_weight)
                if kl_weight > 0:
                    with torch.no_grad():
                        ref_id, ref_age = ref_model(imgs_r)
                    log_p = torch.log_softmax(id_logits_r, dim=1)
                    log_q = torch.log_softmax(ref_id, dim=1)
                    loss_r = loss_ce + kl_weight * kl_crit(log_p, log_q)
                else:
                    loss_r = loss_ce
                loss_r.backward()
                opt_r.step()
            retain_losses.append(loss_r.item())

    return {
        "model": unlearn_m,
        "method": "NG+",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "ng_steps": ng_steps, "ng_lr_ascent": ng_lr_ascent,
            "ng_lr_retain": ng_lr_retain, "kl_weight": kl_weight,
            "forget_step": forget_step,
            "final_forget_loss": round(float(np.mean(forget_losses[-20:])), 4),
            "final_retain_loss": round(float(np.mean(retain_losses[-20:])), 4),
        },
    }


# ── 2. MSG ────────────────────────────────────────────────────────────────────


def masked_small_gradients(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: Optional[int] = None,
    mask_grad_batches: int = 20,
    topk_fraction: float = 0.20,
    msg_steps: int = 300,
    msg_lr: float = 1e-4,
    batch_size: int = 32,
    age_weight: float = 0.5,
    retain_reg: bool = True,
    retain_reg_lr: float = 1e-4,
    retain_reg_every: int = 2,
    **kwargs,
) -> dict:
    """MSG: saliency masking + masked ascent."""
    t0 = time.time()
    unlearn_m = copy_model(model, device)

    f_split = forget_split_name(forget_step, kwargs.get("schedule", "uniform")) if forget_step is not None else "forget"
    subset_ = kwargs.get("subset", "all")
    f_loader, _ = _loader(csv_path, f_split, get_val_transform(), batch_size, shuffle=True, subset=subset_)
    r_loader, _ = _loader(csv_path, "retain", get_val_transform(), batch_size, shuffle=True, subset=subset_)
    criterion = nn.CrossEntropyLoss()

    # Build mask
    logger.info(f"  [MSG] Computing saliency mask (k={topk_fraction*100:.0f}%)…")
    unlearn_m.eval()
    G_f = _grad_vector(unlearn_m, f_loader, criterion, device, mask_grad_batches)
    G_r = _grad_vector(unlearn_m, r_loader, criterion, device, mask_grad_batches)

    scores = {n: (G_f[n] / (G_r[n] + 1e-8)).cpu() for n in G_f}
    all_scores = torch.cat([v.flatten() for v in scores.values()])
    n_topk = max(1, int(topk_fraction * all_scores.numel()))
    threshold = float(all_scores.topk(n_topk).values[-1])
    param_mask = {n: (scores[n] >= threshold) for n in scores}

    hooks = []
    for n, p in unlearn_m.named_parameters():
        if n in param_mask:
            mask_t = param_mask[n].to(device)
            def _hook(grad, _m=mask_t):
                return grad * _m.float()
            hooks.append(p.register_hook(_hook))

    n_masked = sum(m.sum().item() for m in param_mask.values())
    n_total = sum(p.numel() for p in unlearn_m.parameters())
    logger.info(f"  [MSG] Mask: {n_masked:,.0f}/{n_total:,.0f} params ({100*n_masked/n_total:.1f}%)")

    optimizer = torch.optim.Adam(unlearn_m.parameters(), lr=msg_lr)
    opt_r = torch.optim.Adam(unlearn_m.parameters(), lr=retain_reg_lr) if retain_reg else None
    f_iter = _cycle(f_loader)
    r_iter = _cycle(r_loader) if retain_reg else None
    forget_losses, retain_losses = [], []

    for step in range(msg_steps):
        unlearn_m.train()
        imgs_f, id_f, age_f = next(f_iter)
        imgs_f = imgs_f.to(device)
        id_f = id_f.to(device)
        age_f = age_f.to(device)
        optimizer.zero_grad()
        id_logits_f, age_logits_f = unlearn_m(imgs_f)
        loss_f = _closs(id_logits_f, age_logits_f, id_f, age_f, criterion, age_weight)
        (-loss_f).backward()
        optimizer.step()
        forget_losses.append(loss_f.item())

        if retain_reg and r_iter is not None and (step + 1) % retain_reg_every == 0:
            imgs_r, id_r, age_r = next(r_iter)
            imgs_r = imgs_r.to(device)
            id_r = id_r.to(device)
            age_r = age_r.to(device)
            opt_r.zero_grad()
            id_logits_r, age_logits_r = unlearn_m(imgs_r)
            loss_r = _closs(id_logits_r, age_logits_r, id_r, age_r, criterion, age_weight)
            loss_r.backward()
            opt_r.step()
            retain_losses.append(loss_r.item())

    for h in hooks:
        h.remove()

    return {
        "model": unlearn_m,
        "method": "MSG",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "msg_steps": msg_steps, "msg_lr": msg_lr,
            "topk_fraction": topk_fraction, "n_masked_params": int(n_masked),
            "forget_step": forget_step,
            "final_forget_loss": round(float(np.mean(forget_losses[-20:])), 4),
            "final_retain_loss": round(float(np.mean(retain_losses[-20:])
                                             if retain_losses else [0.0]), 4),
        },
    }


# ── 3. CT ─────────────────────────────────────────────────────────────────────


def convolution_transpose(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: Optional[int] = None,
    saliency_batches: int = 20,
    saliency_threshold_pct: float = 75.0,
    dampen_factor: float = 0.1,
    ct_steps: int = 300,
    ct_lr: float = 1e-4,
    batch_size: int = 32,
    age_weight: float = 0.5,
    retain_reg: bool = True,
    retain_lr: float = 1e-4,
    retain_every: int = 1,
    kl_weight: float = 0.0,
    **kwargs,
) -> dict:
    """CT: one-shot weight dampening + stabilisation fine-tune."""
    t0 = time.time()
    unlearn_m = copy_model(model, device)

    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split = forget_split_name(forget_step, kwargs.get("schedule", "uniform")) if forget_step is not None else "forget"
    subset_ = kwargs.get("subset", "all")
    f_loader, _ = _loader(csv_path, f_split, get_val_transform(), batch_size, shuffle=True, subset=subset_)
    r_loader, _ = _loader(csv_path, "retain", get_val_transform(), batch_size, shuffle=True, subset=subset_)
    criterion = nn.CrossEntropyLoss()
    kl_crit = nn.KLDivLoss(reduction="batchmean", log_target=True)

    logger.info(f"  [CT] Computing saliency (p{saliency_threshold_pct:.0f} threshold)…")
    unlearn_m.eval()
    G_f = _grad_vector(unlearn_m, f_loader, criterion, device, saliency_batches)
    G_r = _grad_vector(unlearn_m, r_loader, criterion, device, saliency_batches)

    scores = {n: (G_f[n] / (G_r[n] + 1e-8)).cpu() for n in G_f}
    all_scores = torch.cat([v.flatten() for v in scores.values()])
    threshold = float(torch.quantile(all_scores, saliency_threshold_pct / 100.0))

    n_dampened = 0
    with torch.no_grad():
        for n, p in unlearn_m.named_parameters():
            if n in scores:
                mask = (scores[n].to(device) >= threshold)
                p[mask] *= dampen_factor
                n_dampened += int(mask.sum().item())

    n_total = sum(p.numel() for p in unlearn_m.parameters())
    logger.info(f"  [CT] Dampened {n_dampened:,.0f}/{n_total:,.0f} params "
          f"({100*n_dampened/n_total:.1f}%) by {dampen_factor}")

    optimizer = torch.optim.Adam(unlearn_m.parameters(), lr=ct_lr)
    r_iter = _cycle(r_loader)
    retain_losses = []

    for step in range(ct_steps):
        unlearn_m.train()
        imgs_r, id_r, age_r = next(r_iter)
        imgs_r = imgs_r.to(device)
        id_r = id_r.to(device)
        age_r = age_r.to(device)
        optimizer.zero_grad()
        id_logits_r, age_logits_r = unlearn_m(imgs_r)
        loss_r = _closs(id_logits_r, age_logits_r, id_r, age_r, criterion, age_weight)

        if kl_weight > 0:
            with torch.no_grad():
                ref_id, ref_age = ref_model(imgs_r)
            log_p = torch.log_softmax(id_logits_r, dim=1)
            log_q = torch.log_softmax(ref_id, dim=1)
            loss_r = loss_r + kl_weight * kl_crit(log_p, log_q)

        loss_r.backward()
        optimizer.step()
        retain_losses.append(loss_r.item())

    return {
        "model": unlearn_m,
        "method": "CT",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "ct_steps": ct_steps, "ct_lr": ct_lr,
            "saliency_threshold_pct": saliency_threshold_pct,
            "dampen_factor": dampen_factor, "n_dampened_params": n_dampened,
            "kl_weight": kl_weight, "forget_step": forget_step,
            "final_retain_loss": round(float(np.mean(retain_losses[-20:])), 4),
        },
    }


# ── Registry ──────────────────────────────────────────────────────────────────

SOTA_REGISTRY = {
    "ng_plus": neg_grad_plus,
    "msg":     masked_small_gradients,
    "ct":      convolution_transpose,
}
