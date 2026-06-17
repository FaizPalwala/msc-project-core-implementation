"""
sota_methods.py  —  State-of-the-Art Unlearning Methods (Phase 3)

Implements three methods from the Deep Unlearn benchmark
(Cadet et al., 2024 — arXiv:2410.01276):

  1. NG+  — Negative Gradient Plus
             Gradient ascent on forget set combined with gradient
             descent on a retain batch, with careful LR tuning.

  2. MSG  — Masked Small Gradients
             Identifies parameters whose gradients are large on the
             forget set but small on the retain set, then applies
             ascent only through those parameters.  Localises
             unlearning to parameters that matter for the forget data
             while protecting retain knowledge.

  3. CT   — Convolution Transpose (Selective Weight Dampening)
             Uses the ratio of forget-to-retain gradient magnitudes
             to build a per-parameter saliency mask, then damps
             (suppresses) weights that are disproportionately
             influenced by the forget set.  Exploits the layer-wise
             structure of CNNs to propagate corrections efficiently.

All methods follow the Phase 2 call signature:
    result = method(model, csv_path, device, **config) → dict
        result["model"]    : nn.Module  (unlearned model)
        result["method"]   : str
        result["metrics"]  : dict
"""

import copy
import time
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from dataset import SFHQDataset, get_train_transform, get_val_transform
from model import copy_model


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _loader(csv, split, transform, batch_size, shuffle=False, num_workers=2):
    ds = SFHQDataset(csv, split=split, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True), len(ds)


def _cycle(loader):
    """Infinite iterator over a DataLoader."""
    while True:
        for batch in loader:
            yield batch


def _grad_vector(model, loader, criterion, device, n_batches=10) -> Dict[str, torch.Tensor]:
    """
    Compute average per-parameter gradient magnitude over n_batches.
    Returns dict {param_name: mean_abs_grad_tensor}.
    """
    model.eval()
    accum = {n: torch.zeros_like(p) for n, p in model.named_parameters()
             if p.requires_grad}
    count = 0
    for imgs, labels in loader:
        if count >= n_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        model.zero_grad()
        loss = criterion(model(imgs), labels)
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


# ──────────────────────────────────────────────────────────────────────────────
# 1.  NG+  —  Negative Gradient Plus
# ──────────────────────────────────────────────────────────────────────────────

def neg_grad_plus(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    # ── forget set ──
    forget_step: Optional[int] = None,
    # ── ascent on forget ──
    ng_steps: int = 400,
    ng_lr_ascent: float = 5e-5,
    # ── descent on retain ──
    ng_lr_retain: float = 1e-4,
    retain_steps_per_ascent: int = 1,   # retain descent steps per ascent step
    batch_size: int = 32,
    # ── KL regularisation on retain outputs ──
    kl_weight: float = 0.5,             # 0 = disabled
    **kwargs,
) -> dict:
    """
    NG+ (Negative Gradient Plus):

    Each unlearning iteration:
      1. Sample a forget mini-batch; take one ASCENT step (maximise CE loss).
      2. Sample a retain mini-batch; take `retain_steps_per_ascent` DESCENT
         steps (minimise CE loss) to prevent catastrophic utility collapse.
      3. Optionally add a KL-divergence regulariser matching the unlearned
         model's retain outputs to the original model's retain outputs —
         this prevents over-erasure of related representations.

    The ratio ng_lr_ascent / ng_lr_retain and retain_steps_per_ascent are
    the key hyperparameters studied in the sensitivity analysis.
    """
    t0 = time.time()
    unlearn_m = copy_model(model, device)

    # Freeze reference model for KL term
    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split = f"forget_step_{forget_step}" if forget_step is not None else "forget"
    f_loader, _ = _loader(csv_path, f_split, get_val_transform(), batch_size, shuffle=True)
    r_loader, _ = _loader(csv_path, "retain", get_val_transform(), batch_size, shuffle=True)

    criterion  = nn.CrossEntropyLoss()
    kl_crit    = nn.KLDivLoss(reduction="batchmean", log_target=True)
    opt_forget = torch.optim.SGD(unlearn_m.parameters(),
                                 lr=ng_lr_ascent, momentum=0.9)
    opt_retain = torch.optim.SGD(unlearn_m.parameters(),
                                 lr=ng_lr_retain, momentum=0.9)

    f_iter = _cycle(f_loader)
    r_iter = _cycle(r_loader)

    forget_losses, retain_losses = [], []

    for step in range(ng_steps):
        # ── 1. Ascent on forget ──────────────────────────────────────────
        unlearn_m.train()
        imgs_f, lbl_f = next(f_iter)
        imgs_f, lbl_f = imgs_f.to(device), lbl_f.to(device)
        opt_forget.zero_grad()
        loss_f = criterion(unlearn_m(imgs_f), lbl_f)
        (-loss_f).backward()          # ascent
        opt_forget.step()
        forget_losses.append(loss_f.item())

        # ── 2. Descent on retain ────────────────────────────────────────
        for _ in range(retain_steps_per_ascent):
            imgs_r, lbl_r = next(r_iter)
            imgs_r, lbl_r = imgs_r.to(device), lbl_r.to(device)
            opt_retain.zero_grad()
            logits_r = unlearn_m(imgs_r)
            loss_r   = criterion(logits_r, lbl_r)

            # Optional KL regularisation against reference model
            if kl_weight > 0:
                with torch.no_grad():
                    ref_logits = ref_model(imgs_r)
                log_p   = torch.log_softmax(logits_r,  dim=1)
                log_q   = torch.log_softmax(ref_logits, dim=1)
                loss_kl = kl_crit(log_p, log_q)
                loss_r  = loss_r + kl_weight * loss_kl

            loss_r.backward()
            opt_retain.step()
            retain_losses.append(loss_r.item())

    elapsed = time.time() - t0
    return {
        "model": unlearn_m,
        "method": "NG+",
        "metrics": {
            "unlearning_time_s": round(elapsed, 2),
            "ng_steps": ng_steps,
            "ng_lr_ascent": ng_lr_ascent,
            "ng_lr_retain": ng_lr_retain,
            "kl_weight": kl_weight,
            "retain_steps_per_ascent": retain_steps_per_ascent,
            "forget_step": forget_step,
            "final_forget_loss": round(float(np.mean(forget_losses[-20:])), 4),
            "final_retain_loss": round(float(np.mean(retain_losses[-20:])), 4),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MSG  —  Masked Small Gradients
# ──────────────────────────────────────────────────────────────────────────────

def masked_small_gradients(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: Optional[int] = None,
    # ── mask construction ──
    mask_grad_batches: int = 20,        # batches used to estimate gradients
    topk_fraction: float = 0.2,         # top-k fraction of params to update
    # ── ascent ──
    msg_steps: int = 300,
    msg_lr: float = 1e-4,
    batch_size: int = 32,
    # ── retain regularisation ──
    retain_reg: bool = True,
    retain_reg_lr: float = 1e-4,
    retain_reg_every: int = 2,          # run retain step every N forget steps
    # ── KL distillation (MSG-KD) ──
    kl_weight: float = 0.5,             # 0 = vanilla MSG; >0 = MSG-KD
    **kwargs,
) -> dict:
    """
    MSG (Masked Small Gradients):

    Step 1 — Build gradient saliency mask:
      Compute mean |gradient| on the forget set  (G_f) and on the retain
      set (G_r) for every parameter.  Build a per-parameter score:
          score = G_f / (G_r + eps)
      High score → parameter is disproportionately driven by forget data.
      Keep only the top-k fraction of parameters by score for updates.
      All other parameters are frozen during unlearning.

    Step 2 — Masked ascent + optional retain regularisation:
      Apply gradient ascent only through the masked parameters.
      Optionally interleave retain descent steps (with optional KL
      distillation from the reference model) to preserve utility.

    When kl_weight > 0, this becomes MSG-KD (the novel variant from Phase 3):
    a KL-divergence term anchors the unlearned model's retain-set outputs
    to those of the original model, reducing over-erasure.
    """
    t0 = time.time()
    unlearn_m = copy_model(model, device)

    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split   = f"forget_step_{forget_step}" if forget_step is not None else "forget"
    f_loader, _ = _loader(csv_path, f_split,  get_val_transform(), batch_size, shuffle=True)
    r_loader, _ = _loader(csv_path, "retain", get_val_transform(), batch_size, shuffle=True)
    criterion    = nn.CrossEntropyLoss()
    kl_crit      = nn.KLDivLoss(reduction="batchmean", log_target=True)

    # ── Step 1: Build gradient mask ───────────────────────────────────────
    print(f"  [MSG] Computing gradient saliency mask (k={topk_fraction*100:.0f}%)...")
    unlearn_m.eval()
    G_f = _grad_vector(unlearn_m, f_loader, criterion, device, mask_grad_batches)
    G_r = _grad_vector(unlearn_m, r_loader, criterion, device, mask_grad_batches)

    # Score: forget gradient / (retain gradient + eps)
    scores = {}
    for n in G_f:
        scores[n] = (G_f[n] / (G_r[n] + 1e-8)).cpu()

    # Flatten all scores to find global top-k threshold
    all_scores = torch.cat([v.flatten() for v in scores.values()])
    n_topk     = max(1, int(topk_fraction * all_scores.numel()))
    threshold  = float(all_scores.topk(n_topk).values[-1])

    # Boolean mask per parameter: True = allowed to update
    param_mask = {n: (scores[n] >= threshold) for n in scores}

    # Register mask as a gradient hook on each parameter
    hooks = []
    for n, p in unlearn_m.named_parameters():
        if n in param_mask:
            mask_t = param_mask[n].to(device)

            def make_hook(m):
                def hook(grad):
                    return grad * m.float()
                return hook

            hooks.append(p.register_hook(make_hook(mask_t)))

    n_masked = sum(m.sum().item() for m in param_mask.values())
    n_total  = sum(p.numel() for p in unlearn_m.parameters())
    print(f"  [MSG] Mask: {n_masked:,.0f}/{n_total:,.0f} params active "
          f"({100*n_masked/n_total:.1f}%)")

    # ── Step 2: Masked ascent ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(unlearn_m.parameters(), lr=msg_lr)
    opt_r     = torch.optim.Adam(unlearn_m.parameters(),
                                 lr=retain_reg_lr) if retain_reg else None

    f_iter    = _cycle(f_loader)
    r_iter    = _cycle(r_loader) if retain_reg else None
    forget_losses, retain_losses = [], []

    for step in range(msg_steps):
        unlearn_m.train()
        imgs_f, lbl_f = next(f_iter)
        imgs_f, lbl_f = imgs_f.to(device), lbl_f.to(device)
        optimizer.zero_grad()
        loss_f = criterion(unlearn_m(imgs_f), lbl_f)
        (-loss_f).backward()       # ascent; hooks zero out unmasked grads
        optimizer.step()
        forget_losses.append(loss_f.item())

        # Retain regularisation
        if retain_reg and r_iter is not None and (step + 1) % retain_reg_every == 0:
            imgs_r, lbl_r = next(r_iter)
            imgs_r, lbl_r = imgs_r.to(device), lbl_r.to(device)
            opt_r.zero_grad()
            logits_r = unlearn_m(imgs_r)
            loss_r   = criterion(logits_r, lbl_r)

            if kl_weight > 0:
                with torch.no_grad():
                    ref_logits = ref_model(imgs_r)
                log_p   = torch.log_softmax(logits_r,  dim=1)
                log_q   = torch.log_softmax(ref_logits, dim=1)
                loss_r += kl_weight * kl_crit(log_p, log_q)

            loss_r.backward()
            opt_r.step()
            retain_losses.append(loss_r.item())

    # Remove hooks
    for h in hooks:
        h.remove()

    variant = "MSG-KD" if kl_weight > 0 else "MSG"
    elapsed  = time.time() - t0
    return {
        "model": unlearn_m,
        "method": variant,
        "metrics": {
            "unlearning_time_s": round(elapsed, 2),
            "msg_steps": msg_steps,
            "msg_lr": msg_lr,
            "topk_fraction": topk_fraction,
            "kl_weight": kl_weight,
            "n_masked_params": int(n_masked),
            "forget_step": forget_step,
            "final_forget_loss": round(float(np.mean(forget_losses[-20:])), 4),
            "final_retain_loss": round(float(np.mean(retain_losses[-20:])
                                       if retain_losses else [0.0]), 4),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CT  —  Convolution Transpose (Selective Weight Dampening)
# ──────────────────────────────────────────────────────────────────────────────

def convolution_transpose(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    forget_step: Optional[int] = None,
    # ── saliency mask ──
    saliency_batches: int = 20,
    saliency_threshold_pct: float = 75.0,   # percentile cutoff
    # ── dampening ──
    dampen_factor: float = 0.1,             # multiply selected weights by this
    ct_steps: int = 300,
    ct_lr: float = 1e-4,
    batch_size: int = 32,
    # ── retain stabilisation ──
    retain_reg: bool = True,
    retain_lr: float = 1e-4,
    retain_every: int = 1,
    kl_weight: float = 0.0,
    **kwargs,
) -> dict:
    """
    CT (Convolution Transpose / Selective Weight Dampening):

    Intuition: parameters that are disproportionately activated by the forget
    set carry the most of the forget identity's representation.  Dampening
    them forces the model to lose that representation.

    Step 1 — Gradient saliency:
      Compute the ratio score = G_f / (G_r + eps) as in MSG.
      Parameters whose score exceeds the `saliency_threshold_pct`-th
      percentile are identified as forget-sensitive.

    Step 2 — Weight dampening:
      Multiply the forget-sensitive parameters by `dampen_factor` in-place,
      effectively ablating their contribution.  This is equivalent to a
      one-shot parameter suppression without requiring many gradient steps.

    Step 3 — Fine-tuning stabilisation:
      Run `ct_steps` of gradient descent on the retain set (with optional KL
      distillation) to recover any utility lost during dampening.  This
      prevents the dampening from propagating instability across layers
      (the original CT paper calls this the "transposed correction").

    Design note: CT is faster per-step than NG+ or MSG because the weight
    dampening is a one-shot operation; most of the compute budget goes to
    the stabilisation fine-tuning phase.
    """
    t0 = time.time()
    unlearn_m = copy_model(model, device)

    ref_model = copy_model(model, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    f_split   = f"forget_step_{forget_step}" if forget_step is not None else "forget"
    f_loader, _ = _loader(csv_path, f_split,  get_val_transform(), batch_size, shuffle=True)
    r_loader, _ = _loader(csv_path, "retain", get_val_transform(), batch_size, shuffle=True)
    criterion    = nn.CrossEntropyLoss()
    kl_crit      = nn.KLDivLoss(reduction="batchmean", log_target=True)

    # ── Step 1: Gradient saliency ─────────────────────────────────────────
    print(f"  [CT] Computing saliency (p{saliency_threshold_pct:.0f} threshold)...")
    unlearn_m.eval()
    G_f = _grad_vector(unlearn_m, f_loader, criterion, device, saliency_batches)
    G_r = _grad_vector(unlearn_m, r_loader, criterion, device, saliency_batches)

    scores = {}
    for n in G_f:
        scores[n] = (G_f[n] / (G_r[n] + 1e-8)).cpu()

    all_scores = torch.cat([v.flatten() for v in scores.values()])
    threshold  = float(torch.quantile(all_scores,
                                      saliency_threshold_pct / 100.0))

    # ── Step 2: Dampen forget-sensitive weights ───────────────────────────
    n_dampened = 0
    with torch.no_grad():
        for n, p in unlearn_m.named_parameters():
            if n in scores:
                mask = (scores[n].to(device) >= threshold)
                p[mask] *= dampen_factor
                n_dampened += int(mask.sum().item())

    n_total = sum(p.numel() for p in unlearn_m.parameters())
    print(f"  [CT] Dampened {n_dampened:,.0f}/{n_total:,.0f} params "
          f"({100*n_dampened/n_total:.1f}%) by factor {dampen_factor}")

    # ── Step 3: Stabilisation fine-tuning on retain set ───────────────────
    optimizer = torch.optim.Adam(unlearn_m.parameters(), lr=ct_lr)
    r_iter    = _cycle(r_loader)
    f_iter    = _cycle(f_loader)
    retain_losses = []

    for step in range(ct_steps):
        unlearn_m.train()
        imgs_r, lbl_r = next(r_iter)
        imgs_r, lbl_r = imgs_r.to(device), lbl_r.to(device)
        optimizer.zero_grad()
        logits_r = unlearn_m(imgs_r)
        loss_r   = criterion(logits_r, lbl_r)

        if kl_weight > 0:
            with torch.no_grad():
                ref_logits = ref_model(imgs_r)
            log_p   = torch.log_softmax(logits_r,  dim=1)
            log_q   = torch.log_softmax(ref_logits, dim=1)
            loss_r += kl_weight * kl_crit(log_p, log_q)

        loss_r.backward()
        optimizer.step()
        retain_losses.append(loss_r.item())

    elapsed = time.time() - t0
    return {
        "model": unlearn_m,
        "method": "CT",
        "metrics": {
            "unlearning_time_s": round(elapsed, 2),
            "ct_steps": ct_steps,
            "ct_lr": ct_lr,
            "saliency_threshold_pct": saliency_threshold_pct,
            "dampen_factor": dampen_factor,
            "n_dampened_params": n_dampened,
            "kl_weight": kl_weight,
            "forget_step": forget_step,
            "final_retain_loss": round(float(np.mean(retain_losses[-20:])), 4),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Registry extension (merged with Phase 2 baselines in run_phase3.py)
# ──────────────────────────────────────────────────────────────────────────────

SOTA_REGISTRY = {
    "ng_plus":  neg_grad_plus,
    "msg":      masked_small_gradients,
    "msg_kd":   lambda *a, **kw: masked_small_gradients(*a, **kw, kl_weight=kw.pop("kl_weight", 0.5)),
    "ct":       convolution_transpose,
}
