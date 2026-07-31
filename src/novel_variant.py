"""
novel_variant.py — Phase 4 Variants (dual-head).

Methods:
  MSG-KD       — MSG with KL distillation on retain outputs
  AdaptiForget — adaptive variant: scheduled KL, mask refresh, dual early stopping

Both operate on the combined dual-head loss.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines import _combined_loss as _closs
from dataset import VirtualIdentityDataset, get_val_transform
from model import copy_model



import logging

logger = logging.getLogger(__name__)
# ── Helpers ───────────────────────────────────────────────────────────────────


def _loader(csv, split, transform, batch_size, shuffle=False, workers=2):
    ds = VirtualIdentityDataset(csv, split=split, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True), len(ds)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _build_mask(model, f_loader, r_loader, criterion, device,
                n_batches: int, topk: float) -> dict[str, torch.Tensor]:
    """Build boolean saliency mask (True = forget-sensitive, keep for update)."""

    def _grad_mag(loader, n):
        accum = {nm: torch.zeros_like(p)
                 for nm, p in model.named_parameters() if p.requires_grad}
        count = 0
        model.eval()
        for imgs, id_lbls, age_lbls in loader:
            if count >= n:
                break
            imgs = imgs.to(device)
            id_lbls = id_lbls.to(device)
            age_lbls = age_lbls.to(device)
            model.zero_grad()
            id_logits, age_logits = model(imgs)
            loss = criterion(id_logits, id_lbls)
            loss.backward()
            for nm, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    accum[nm] += p.grad.abs().detach()
            count += 1
        model.zero_grad()
        if count:
            for nm in accum:
                accum[nm] /= count
        return accum

    G_f = _grad_mag(f_loader, n_batches)
    G_r = _grad_mag(r_loader, n_batches)
    scores = {nm: (G_f[nm] / (G_r[nm] + 1e-8)).cpu() for nm in G_f}
    all_s = torch.cat([v.flatten() for v in scores.values()])
    n_keep = max(1, int(topk * all_s.numel()))
    thresh = float(all_s.topk(n_keep).values[-1])
    return {nm: (scores[nm] >= thresh) for nm in scores}


def _quick_forget_advantage(model, f_loader, r_loader, device,
                            n_batches=8) -> float:
    """Lightweight proxy: fraction of forget samples with loss < median retain loss."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="none")
    f_losses, r_losses = [], []
    with torch.no_grad():
        for i, (imgs, id_lbls, age_lbls) in enumerate(f_loader):
            if i >= n_batches:
                break
            imgs = imgs.to(device)
            id_lbls = id_lbls.to(device)
            id_logits, _ = model(imgs)
            f_losses.extend(crit(id_logits, id_lbls).cpu().tolist())
        for i, (imgs, id_lbls, age_lbls) in enumerate(r_loader):
            if i >= n_batches:
                break
            imgs = imgs.to(device)
            id_lbls = id_lbls.to(device)
            id_logits, _ = model(imgs)
            r_losses.extend(crit(id_logits, id_lbls).cpu().tolist())
    if not f_losses or not r_losses:
        return 0.5
    med_r = float(np.median(r_losses))
    return float(np.mean([l < med_r for l in f_losses]))


# ── MSG-KD ────────────────────────────────────────────────────────────────────


def msg_kd(
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
    kl_weight: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> dict:
    """MSG with KL distillation on retain outputs."""
    torch.manual_seed(seed)
    t0 = time.time()

    unlearn_m = copy_model(model, device)
    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split = f"forget_step_{forget_step}" if forget_step is not None else "forget"
    f_loader, _ = _loader(csv_path, f_split, get_val_transform(), batch_size, shuffle=True)
    r_loader, _ = _loader(csv_path, "retain", get_val_transform(), batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    kl_crit = nn.KLDivLoss(reduction="batchmean", log_target=True)

    logger.info(f"  [MSG-KD] Building saliency mask (k={topk_fraction*100:.0f}%)…")
    unlearn_m.eval()
    mask = _build_mask(unlearn_m, f_loader, r_loader, criterion, device,
                       mask_grad_batches, topk_fraction)

    hooks = []
    for name, param in unlearn_m.named_parameters():
        if name in mask:
            mask_t = mask[name].to(device)
            def _hook(grad, _m=mask_t):
                return grad * _m.float()
            hooks.append(param.register_hook(_hook))

    n_masked = sum(m.sum().item() for m in mask.values())
    n_total = sum(p.numel() for p in unlearn_m.parameters())
    logger.info(f"  [MSG-KD] Mask: {n_masked:,.0f}/{n_total:,.0f} params ({100*n_masked/n_total:.1f}%)")

    optimizer = torch.optim.Adam(unlearn_m.parameters(), lr=msg_lr)
    opt_r = torch.optim.Adam(unlearn_m.parameters(), lr=retain_reg_lr)
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
            loss_ce = _closs(id_logits_r, age_logits_r, id_r, age_r, criterion, age_weight)

            if kl_weight > 0:
                with torch.no_grad():
                    ref_id, _ = ref_model(imgs_r)
                log_p = torch.log_softmax(id_logits_r, dim=1)
                log_q = torch.log_softmax(ref_id, dim=1)
                loss_r = loss_ce + kl_weight * kl_crit(log_p, log_q)
            else:
                loss_r = loss_ce

            loss_r.backward()
            opt_r.step()
            retain_losses.append(loss_r.item())

    for h in hooks:
        h.remove()

    return {
        "model": unlearn_m,
        "method": "MSG-KD",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "msg_steps": msg_steps, "msg_lr": msg_lr,
            "topk_fraction": topk_fraction, "kl_weight": kl_weight,
            "n_masked_params": int(n_masked), "forget_step": forget_step,
            "final_forget_loss": round(float(np.mean(forget_losses[-20:])), 4),
            "final_retain_loss": round(float(np.mean(
                retain_losses[-20:] if retain_losses else [0.0])), 4),
        },
    }


# ── AdaptiForget ──────────────────────────────────────────────────────────────


def adaptiformet(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: Optional[int] = None,
    topk_fraction: float = 0.20,
    mask_refresh_every: int = 100,
    mask_grad_batches: int = 20,
    max_steps: int = 600,
    lr_ascent: float = 5e-5,
    lr_retain: float = 1e-4,
    retain_steps_per: int = 2,
    batch_size: int = 32,
    age_weight: float = 0.5,
    kl_weight_init: float = 0.1,
    kl_weight_max: float = 0.8,
    kl_anneal_steps: int = 200,
    early_stop_adv: float = 0.05,
    early_stop_patience: int = 50,
    retain_drop_tol: float = 0.03,
    seed: int = 42,
    **kwargs,
) -> dict:
    """AdaptiForget: adaptive λ, periodic mask refresh, dual early stopping."""
    torch.manual_seed(seed)
    t0 = time.time()

    unlearn_m = copy_model(model, device)
    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split = f"forget_step_{forget_step}" if forget_step is not None else "forget"
    f_loader, n_f = _loader(csv_path, f_split, get_val_transform(), batch_size, shuffle=True)
    r_loader, n_r = _loader(csv_path, "retain", get_val_transform(), batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    kl_crit = nn.KLDivLoss(reduction="batchmean", log_target=True)
    opt_f = torch.optim.AdamW(unlearn_m.parameters(), lr=lr_ascent, weight_decay=0.0)
    opt_r = torch.optim.AdamW(unlearn_m.parameters(), lr=lr_retain, weight_decay=1e-4)
    f_iter = _cycle(f_loader)
    r_iter = _cycle(r_loader)

    logger.info(f"  [AdaptiForget] Building initial mask (k={topk_fraction*100:.0f}%)…")
    mask = _build_mask(unlearn_m, f_loader, r_loader, criterion, device,
                       mask_grad_batches, topk_fraction)

    def _attach_hooks(m, mask_dict):
        hs = []
        for nm, p in m.named_parameters():
            if nm in mask_dict:
                mt = mask_dict[nm].to(device)
                def _hook(grad, _m=mt):
                    return grad * _m.float()
                hs.append(p.register_hook(_hook))
        return hs

    hooks = _attach_hooks(unlearn_m, mask)
    n_masked = sum(v.sum().item() for v in mask.values())
    n_total = sum(p.numel() for p in unlearn_m.parameters())
    logger.info(f"  [AdaptiForget] Mask: {n_masked:,.0f}/{n_total:,.0f} params "
          f"({100*n_masked/n_total:.1f}%)")

    def _quick_retain_acc(m, n_batches=10):
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for i, (imgs, id_lbls, age_lbls) in enumerate(r_loader):
                if i >= n_batches:
                    break
                imgs = imgs.to(device)
                id_lbls = id_lbls.to(device)
                id_logits, _ = m(imgs)
                preds = id_logits.argmax(1)
                correct += (preds == id_lbls).sum().item()
                total += len(id_lbls)
        return correct / max(total, 1)

    retain_acc_ref = _quick_retain_acc(unlearn_m)
    retain_drop_threshold = retain_acc_ref - retain_drop_tol

    history: dict[str, list] = {
        "forget_loss": [], "retain_loss": [], "kl_weight": [],
        "proxy_adv": [], "step": [],
    }
    stopped_at = max_steps
    patience_count = 0

    for step in range(max_steps):
        kl_w = kl_weight_init + (kl_weight_max - kl_weight_init) * min(
            step / max(kl_anneal_steps, 1), 1.0)

        if step > 0 and step % mask_refresh_every == 0:
            for h in hooks:
                h.remove()
            mask = _build_mask(unlearn_m, f_loader, r_loader, criterion, device,
                               mask_grad_batches // 2, topk_fraction)
            hooks = _attach_hooks(unlearn_m, mask)

        unlearn_m.train()
        imgs_f, id_f, age_f = next(f_iter)
        imgs_f = imgs_f.to(device)
        id_f = id_f.to(device)
        age_f = age_f.to(device)
        opt_f.zero_grad()
        id_logits_f, age_logits_f = unlearn_m(imgs_f)
        loss_f = _closs(id_logits_f, age_logits_f, id_f, age_f, criterion, age_weight)
        (-loss_f).backward()
        opt_f.step()
        history["forget_loss"].append(loss_f.item())

        for _ in range(retain_steps_per):
            imgs_r, id_r, age_r = next(r_iter)
            imgs_r = imgs_r.to(device)
            id_r = id_r.to(device)
            age_r = age_r.to(device)
            opt_r.zero_grad()
            id_logits_r, age_logits_r = unlearn_m(imgs_r)
            loss_ce = _closs(id_logits_r, age_logits_r, id_r, age_r, criterion, age_weight)
            with torch.no_grad():
                ref_id, _ = ref_model(imgs_r)
            log_p = torch.log_softmax(id_logits_r, dim=1)
            log_q = torch.log_softmax(ref_id, dim=1)
            loss_kl = kl_crit(log_p, log_q)
            total_r = loss_ce + kl_w * loss_kl
            total_r.backward()
            opt_r.step()
            history["retain_loss"].append(total_r.item())

        history["kl_weight"].append(kl_w)

        if (step + 1) % 25 == 0:
            proxy = _quick_forget_advantage(unlearn_m, f_loader, r_loader, device, n_batches=5)
            history["proxy_adv"].append(proxy)
            history["step"].append(step + 1)
            r_acc = _quick_retain_acc(unlearn_m)

            if proxy < early_stop_adv:
                logger.info(f"  [AdaptiForget] Early stop at step {step+1}: proxy_adv={proxy:.3f}")
                stopped_at = step + 1
                break

            if r_acc < retain_drop_threshold:
                patience_count += 1
                if patience_count >= early_stop_patience // 25:
                    logger.info(f"  [AdaptiForget] Early stop at step {step+1}: "
                          f"retain_acc={r_acc:.3f} dropped {retain_drop_tol}")
                    stopped_at = step + 1
                    break
            else:
                patience_count = 0

    for h in hooks:
        h.remove()

    return {
        "model": unlearn_m,
        "method": "AdaptiForget",
        "metrics": {
            "unlearning_time_s": round(time.time() - t0, 2),
            "steps_used": stopped_at, "max_steps": max_steps,
            "final_forget_loss": round(float(np.mean(history["forget_loss"][-20:])), 4),
            "final_retain_loss": round(float(np.mean(history["retain_loss"][-20:])), 4),
            "final_kl_weight": round(float(history["kl_weight"][-1]), 4),
            "n_masked_params": int(n_masked), "retain_acc_ref": round(retain_acc_ref, 4),
            "forget_step": forget_step, "early_stopped": stopped_at < max_steps,
            "history": history,
        },
    }


# ── Registry ──────────────────────────────────────────────────────────────────

NOVEL_REGISTRY = {
    "msg_kd": msg_kd,
    "adaptiformet": adaptiformet,
}
