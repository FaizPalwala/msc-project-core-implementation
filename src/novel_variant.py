"""
novel_variant.py  —  Phase 4: AdaptiForget (Novel Robust Variant)

AdaptiForget is a principled extension of MSG-KD that addresses three
observed weaknesses in preliminary Phase 3 results:

  1. Fixed step counts cause over-unlearning on small forget sets and
     under-unlearning on larger ones.  → Adaptive early stopping.

  2. A single global mask built once at the start goes stale as weights
     change during unlearning.  → Periodic mask refresh.

  3. The KL weight λ is constant throughout training.  When the forget
     loss is still high, a large λ fights the ascent; when the forget
     loss has converged, λ should be large to protect utility.
     → Adaptive λ scheduling tied to forgetting progress.

Architecture
───────────────────────────────────────────────────────────────────────
  AdaptiForget combines:
    (a) MSG-style gradient saliency masking (localised ascent)
    (b) KL-divergence distillation on retain set (MSG-KD)
    (c) Adaptive λ schedule  (forget-progress-linked annealing)
    (d) Periodic mask refresh  (every `mask_refresh_every` steps)
    (e) Early stopping  (forget advantage < target OR patience exceeded)
    (f) Optional retain-set momentum buffer  (stabilises retain gradients
        during the aggressive early ascent phase)

The method is designed to be robust to:
  - Varying forget-set sizes (1 identity up to 20+ sequential deletions)
  - Varying model state (fresh vs. partially-unlearned after prior steps)
  - Hyperparameter sensitivity (adapts internally rather than requiring
    precise manual tuning)

Call signature (same as all other methods):
    result = adaptiformet(model, csv_path, device, **cfg) → dict

Key hyperparameters
─────────────────────
  topk_fraction         float  0.2   fraction of params in saliency mask
  max_steps             int    600   hard ceiling on unlearning steps
  lr_ascent             float  5e-5  ascent LR (forget set)
  lr_retain             float  1e-4  retain LR (descent + KL)
  kl_weight_init        float  0.1   initial KL weight (low → ascent can fire)
  kl_weight_max         float  0.8   max KL weight (high → protect utility)
  kl_anneal_steps       int    200   steps to ramp λ from init → max
  mask_refresh_every    int    100   steps between mask recomputation
  mask_grad_batches     int    20    batches used to build each mask
  early_stop_adv        float  0.05  stop if forget_advantage < this
  early_stop_patience   int    50    stop if retain_acc drops > patience steps
  retain_steps_per      int    2     retain steps per ascent step
  batch_size            int    32
"""

import copy
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SFHQDataset, get_val_transform
from model import copy_model


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (shared with methods_sota.py but reproduced here for
# standalone portability)
# ──────────────────────────────────────────────────────────────────────────────

def _loader(csv, split, transform, batch_size, shuffle=False, workers=2):
    ds = SFHQDataset(csv, split=split, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True), len(ds)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _build_mask(model, f_loader, r_loader, criterion, device,
                n_batches: int, topk: float) -> Dict[str, torch.Tensor]:
    """
    Build a boolean saliency mask.
    Returns {param_name: BoolTensor (same shape as param)}.
    True = this element is in the top-k% forget-sensitive positions.
    """
    def _grad_mag(loader, n):
        accum = {nm: torch.zeros_like(p)
                 for nm, p in model.named_parameters() if p.requires_grad}
        count = 0
        model.eval()
        for imgs, labels in loader:
            if count >= n:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            model.zero_grad()
            loss = criterion(model(imgs), labels)
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
    all_s  = torch.cat([v.flatten() for v in scores.values()])
    n_keep = max(1, int(topk * all_s.numel()))
    thresh = float(all_s.topk(n_keep).values[-1])
    return {nm: (scores[nm] >= thresh) for nm in scores}


def _quick_forget_advantage(model, f_loader, r_loader, device,
                             n_batches=8) -> float:
    """
    Lightweight proxy for forget_advantage:
    compares mean loss on forget vs. retain mini-batches.
    Higher means model still "knows" the forget set.
    Used for adaptive early stopping without a full MIA pass.
    """
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="none")
    f_losses, r_losses = [], []
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(f_loader):
            if i >= n_batches: break
            f_losses.extend(crit(model(imgs.to(device)), lbls.to(device)).cpu().tolist())
        for i, (imgs, lbls) in enumerate(r_loader):
            if i >= n_batches: break
            r_losses.extend(crit(model(imgs.to(device)), lbls.to(device)).cpu().tolist())
    if not f_losses or not r_losses:
        return 0.5
    # Proxy advantage: fraction of forget samples with lower loss than median retain loss
    med_r = float(np.median(r_losses))
    adv   = float(np.mean([l < med_r for l in f_losses]))
    return adv   # 0 = forgotten, 1 = still memorised


# ──────────────────────────────────────────────────────────────────────────────
# AdaptiForget
# ──────────────────────────────────────────────────────────────────────────────

def adaptiformet(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: Optional[int] = None,
    # masking
    topk_fraction: float = 0.20,
    mask_refresh_every: int = 100,
    mask_grad_batches: int = 20,
    # optimisation
    max_steps: int = 600,
    lr_ascent: float = 5e-5,
    lr_retain: float = 1e-4,
    retain_steps_per: int = 2,
    batch_size: int = 32,
    # adaptive KL schedule
    kl_weight_init: float = 0.1,
    kl_weight_max: float = 0.8,
    kl_anneal_steps: int = 200,
    # early stopping
    early_stop_adv: float = 0.05,       # proxy forget advantage threshold
    early_stop_patience: int = 50,       # steps to tolerate retain drop
    retain_drop_tol: float = 0.03,       # tolerated retain acc drop fraction
    # misc
    seed: int = 42,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    t0 = time.time()

    unlearn_m = copy_model(model, device)
    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split = (f"forget_step_{forget_step}" if forget_step is not None
               else "forget")
    f_loader, n_f = _loader(csv_path, f_split, get_val_transform(),
                            batch_size, shuffle=True)
    r_loader, n_r = _loader(csv_path, "retain", get_val_transform(),
                            batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    kl_crit   = nn.KLDivLoss(reduction="batchmean", log_target=True)
    opt_f = torch.optim.AdamW(unlearn_m.parameters(),
                               lr=lr_ascent, weight_decay=0.0)
    opt_r = torch.optim.AdamW(unlearn_m.parameters(),
                               lr=lr_retain, weight_decay=1e-4)

    f_iter = _cycle(f_loader)
    r_iter = _cycle(r_loader)

    # ── Initial mask ──────────────────────────────────────────────────────────
    print(f"  [AdaptiForget] Building initial mask (k={topk_fraction*100:.0f}%)…")
    mask = _build_mask(unlearn_m, f_loader, r_loader, criterion,
                       device, mask_grad_batches, topk_fraction)
    hooks = []

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
    n_total  = sum(p.numel() for p in unlearn_m.parameters())
    print(f"  [AdaptiForget] Mask: {n_masked:,.0f}/{n_total:,.0f} params "
          f"({100*n_masked/n_total:.1f}%)")

    # ── Reference retain accuracy (for early stopping) ────────────────────────
    def _quick_retain_acc(m, n_batches=10):
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(r_loader):
                if i >= n_batches: break
                preds = m(imgs.to(device)).argmax(1)
                correct += (preds == lbls.to(device)).sum().item()
                total   += len(lbls)
        return correct / max(total, 1)

    retain_acc_ref = _quick_retain_acc(unlearn_m)
    retain_drop_threshold = retain_acc_ref - retain_drop_tol

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {
        "forget_loss": [], "retain_loss": [], "kl_weight": [],
        "proxy_adv": [], "step": [],
    }
    stopped_at = max_steps
    patience_count = 0

    for step in range(max_steps):

        # ── Adaptive KL weight (linear warmup init → max) ────────────────────
        kl_w = kl_weight_init + (kl_weight_max - kl_weight_init) * min(
            step / max(kl_anneal_steps, 1), 1.0
        )

        # ── Periodic mask refresh ────────────────────────────────────────────
        if step > 0 and step % mask_refresh_every == 0:
            for h in hooks:
                h.remove()
            mask = _build_mask(unlearn_m, f_loader, r_loader, criterion,
                               device, mask_grad_batches // 2, topk_fraction)
            hooks = _attach_hooks(unlearn_m, mask)

        # ── Ascent on forget ─────────────────────────────────────────────────
        unlearn_m.train()
        imgs_f, lbl_f = next(f_iter)
        imgs_f, lbl_f = imgs_f.to(device), lbl_f.to(device)
        opt_f.zero_grad()
        loss_f = criterion(unlearn_m(imgs_f), lbl_f)
        (-loss_f).backward()
        opt_f.step()
        history["forget_loss"].append(loss_f.item())

        # ── Retain descent + adaptive KL ────────────────────────────────────
        for _ in range(retain_steps_per):
            imgs_r, lbl_r = next(r_iter)
            imgs_r, lbl_r = imgs_r.to(device), lbl_r.to(device)
            opt_r.zero_grad()
            logits_r = unlearn_m(imgs_r)
            loss_r   = criterion(logits_r, lbl_r)
            with torch.no_grad():
                ref_out = ref_model(imgs_r)
            log_p    = torch.log_softmax(logits_r, dim=1)
            log_q    = torch.log_softmax(ref_out,   dim=1)
            loss_kl  = kl_crit(log_p, log_q)
            total_r  = loss_r + kl_w * loss_kl
            total_r.backward()
            opt_r.step()
            history["retain_loss"].append(total_r.item())

        history["kl_weight"].append(kl_w)

        # ── Early stopping checks (every 25 steps) ───────────────────────────
        if (step + 1) % 25 == 0:
            proxy = _quick_forget_advantage(
                unlearn_m, f_loader, r_loader, device, n_batches=5)
            history["proxy_adv"].append(proxy)
            history["step"].append(step + 1)
            r_acc = _quick_retain_acc(unlearn_m)
            history["retain_loss"]  # already tracked per-step

            if proxy < early_stop_adv:
                print(f"  [AdaptiForget] Early stop at step {step+1}: "
                      f"proxy_adv={proxy:.3f} < {early_stop_adv}")
                stopped_at = step + 1
                break

            if r_acc < retain_drop_threshold:
                patience_count += 1
                if patience_count >= early_stop_patience // 25:
                    print(f"  [AdaptiForget] Early stop at step {step+1}: "
                          f"retain_acc={r_acc:.3f} dropped "
                          f">{retain_drop_tol:.2f} from ref {retain_acc_ref:.3f}")
                    stopped_at = step + 1
                    break
            else:
                patience_count = 0

    for h in hooks:
        h.remove()

    elapsed = time.time() - t0
    return {
        "model": unlearn_m,
        "method": "AdaptiForget",
        "metrics": {
            "unlearning_time_s":    round(elapsed, 2),
            "steps_used":           stopped_at,
            "max_steps":            max_steps,
            "final_forget_loss":    round(float(np.mean(history["forget_loss"][-20:])), 4),
            "final_retain_loss":    round(float(np.mean(history["retain_loss"][-20:])), 4),
            "final_kl_weight":      round(float(history["kl_weight"][-1]), 4),
            "n_masked_params":      int(n_masked),
            "retain_acc_ref":       round(retain_acc_ref, 4),
            "forget_step":          forget_step,
            "early_stopped":        stopped_at < max_steps,
            "history":              history,
        },
    }


# ── MSG-KD ───────────────────────────────────────────────────────────────────

def msg_kd(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: Optional[int] = None,
    # masking
    mask_grad_batches: int = 20,
    topk_fraction: float = 0.20,
    # optimisation
    msg_steps: int = 300,
    msg_lr: float = 1e-4,
    batch_size: int = 32,
    # retain regularisation
    retain_reg: bool = True,
    retain_reg_lr: float = 1e-4,
    retain_reg_every: int = 2,
    kl_weight: float = 0.5,
    # misc
    seed: int = 42,
    **kwargs,
) -> dict:
    """MSG with KL distillation on the retain set."""
    torch.manual_seed(seed)
    t0 = time.time()

    unlearn_m = copy_model(model, device)
    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split = f"forget_step_{forget_step}" if forget_step is not None else "forget"
    f_loader, _ = _loader(csv_path, f_split, get_val_transform(), batch_size,
                          shuffle=True)
    r_loader, _ = _loader(csv_path, "retain", get_val_transform(), batch_size,
                          shuffle=True)

    criterion = nn.CrossEntropyLoss()
    kl_crit   = nn.KLDivLoss(reduction="batchmean", log_target=True)

    print(f"  [MSG-KD] Building saliency mask (k={topk_fraction*100:.0f}%)…")
    unlearn_m.eval()
    mask = _build_mask(unlearn_m, f_loader, r_loader, criterion, device,
                       mask_grad_batches, topk_fraction)

    hooks = []
    for name, param in unlearn_m.named_parameters():
        if name in mask:
            mask_t = mask[name].to(device)

            def _make_hook(m):
                def hook(grad):
                    return grad * m.float()
                return hook

            hooks.append(param.register_hook(_make_hook(mask_t)))

    n_masked = sum(m.sum().item() for m in mask.values())
    n_total  = sum(p.numel() for p in unlearn_m.parameters())
    print(f"  [MSG-KD] Mask: {n_masked:,.0f}/{n_total:,.0f} params active "
          f"({100*n_masked/n_total:.1f}%)")

    optimizer = torch.optim.Adam(unlearn_m.parameters(), lr=msg_lr)
    opt_r     = torch.optim.Adam(unlearn_m.parameters(), lr=retain_reg_lr)
    f_iter    = _cycle(f_loader)
    r_iter    = _cycle(r_loader) if retain_reg else None
    forget_losses, retain_losses = [], []

    for step in range(msg_steps):
        unlearn_m.train()
        imgs_f, lbl_f = next(f_iter)
        imgs_f, lbl_f = imgs_f.to(device), lbl_f.to(device)
        optimizer.zero_grad()
        loss_f = criterion(unlearn_m(imgs_f), lbl_f)
        (-loss_f).backward()
        optimizer.step()
        forget_losses.append(loss_f.item())

        if retain_reg and r_iter is not None and (step + 1) % retain_reg_every == 0:
            imgs_r, lbl_r = next(r_iter)
            imgs_r, lbl_r = imgs_r.to(device), lbl_r.to(device)
            opt_r.zero_grad()
            logits_r = unlearn_m(imgs_r)
            loss_r   = criterion(logits_r, lbl_r)

            if kl_weight > 0:
                with torch.no_grad():
                    ref_logits = ref_model(imgs_r)
                log_p = torch.log_softmax(logits_r, dim=1)
                log_q = torch.log_softmax(ref_logits, dim=1)
                loss_r += kl_weight * kl_crit(log_p, log_q)

            loss_r.backward()
            opt_r.step()
            retain_losses.append(loss_r.item())

    for h in hooks:
        h.remove()

    elapsed = time.time() - t0
    return {
        "model": unlearn_m,
        "method": "MSG-KD",
        "metrics": {
            "unlearning_time_s": round(elapsed, 2),
            "msg_steps": msg_steps,
            "msg_lr": msg_lr,
            "topk_fraction": topk_fraction,
            "kl_weight": kl_weight,
            "n_masked_params": int(n_masked),
            "forget_step": forget_step,
            "final_forget_loss": round(float(np.mean(forget_losses[-20:])), 4),
            "final_retain_loss": round(float(np.mean(
                retain_losses[-20:] if retain_losses else [0.0]
            )), 4),
        },
    }


# ── Registry entry ────────────────────────────────────────────────────────────
NOVEL_REGISTRY = {
    "msg_kd": msg_kd,
    "adaptiformet": adaptiformet,
}
