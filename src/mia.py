"""
mia.py — Basic Membership Inference Attack (MIA)

Implements the standard confidence-threshold MIA:
  - For each sample, record the model's max softmax confidence
  - Members (training data) tend to have higher confidence than non-members
  - Threshold the confidence to classify "member" vs "non-member"

Also implements a simple shadow-model-free attack using the
"loss-based" threshold (Yeom et al., 2018) as a lightweight alternative.

Key functions:
  run_mia_threshold()   → threshold-based MIA using confidence scores
  mia_scores()          → raw confidence/loss scores per sample
  compute_mia_auc()     → ROC-AUC of attack (0.5 = random = perfectly unlearned)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score


# ──────────────────────────────────────────────────────────────────────────────
# Score extraction
# ──────────────────────────────────────────────────────────────────────────────

def get_confidence_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Returns max softmax confidence for each sample in loader.
    Higher confidence → more likely to be a training member.
    """
    model.eval()
    confs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)
            conf, _ = probs.max(dim=1)
            confs.extend(conf.cpu().tolist())
    return np.array(confs)


def get_loss_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Returns per-sample cross-entropy loss.
    Lower loss → more likely to be a training member (Yeom et al., 2018).
    We negate so that "higher score → more likely member" is consistent.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            losses.extend(loss.cpu().tolist())
    return -np.array(losses)   # negate: higher = more member-like


# ──────────────────────────────────────────────────────────────────────────────
# MIA evaluation
# ──────────────────────────────────────────────────────────────────────────────

def compute_mia_auc(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
) -> float:
    """
    Compute ROC-AUC of membership inference.
    AUC = 0.5 → attack fails (perfect unlearning signal)
    AUC = 1.0 → perfect attack (model completely memorises members)
    """
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(nonmember_scores))
    ])
    return float(roc_auc_score(labels, scores))


def run_mia_threshold(
    model: nn.Module,
    member_loader: DataLoader,     # retain set (members)
    nonmember_loader: DataLoader,  # test set (non-members)
    forget_loader: DataLoader,     # forget set (target of unlearning)
    device: torch.device,
    score_type: str = "confidence",  # "confidence" or "loss"
) -> dict:
    """
    Run threshold-based MIA to assess unlearning quality.

    Three-way evaluation:
      (A) Retain vs Test  → attack accuracy on normal training data
      (B) Forget vs Test  → attack accuracy on forgotten data (KEY metric)
          - Should approach 0.5 AUC after successful unlearning
          - Significantly above 0.5 means forget data is still "remembered"

    Returns dict with:
      retain_test_auc    : AUC of retain-vs-test attack (should stay ~high)
      forget_test_auc    : AUC of forget-vs-test attack (should approach 0.5)
      forget_test_acc    : threshold accuracy of forget-vs-test
      forget_advantage   : |forget_test_auc - 0.5|  (0 = perfect unlearning)
    """
    score_fn = get_confidence_scores if score_type == "confidence" else get_loss_scores

    retain_scores  = score_fn(model, member_loader,   device)
    test_scores    = score_fn(model, nonmember_loader, device)
    forget_scores  = score_fn(model, forget_loader,    device)

    # (A) Retain vs Test: how well the model distinguishes normal members
    retain_test_auc = compute_mia_auc(retain_scores, test_scores)

    # (B) Forget vs Test: key unlearning signal
    forget_test_auc = compute_mia_auc(forget_scores, test_scores)

    # Threshold accuracy at 0.5 for forget vs test
    all_scores = np.concatenate([forget_scores, test_scores])
    all_labels = np.concatenate([
        np.ones(len(forget_scores)), np.zeros(len(test_scores))
    ])
    threshold = np.median(all_scores)
    preds = (all_scores >= threshold).astype(int)
    forget_test_acc = float(accuracy_score(all_labels, preds))

    return {
        "score_type": score_type,
        "retain_test_auc":  round(retain_test_auc,  4),
        "forget_test_auc":  round(forget_test_auc,  4),
        "forget_test_acc":  round(forget_test_acc,  4),
        "forget_advantage": round(abs(forget_test_auc - 0.5), 4),
        "n_retain":  len(retain_scores),
        "n_test":    len(test_scores),
        "n_forget":  len(forget_scores),
    }


def run_mia_full(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
    batch_size: int = 128,
    forget_step: int = None,
    score_type: str = "confidence",
    verbose: bool = True,
) -> dict:
    """
    Convenience wrapper: builds loaders from CSV and runs full MIA.
    """
    from dataset import SFHQDataset, get_val_transform

    retain_ds  = SFHQDataset(csv_path, "retain",  transform=get_val_transform())
    test_ds    = SFHQDataset(csv_path, "test",    transform=get_val_transform())

    if forget_step is not None:
        forget_split = f"forget_step_{forget_step}"
    else:
        forget_split = "forget"
    forget_ds = SFHQDataset(csv_path, forget_split, transform=get_val_transform())

    kw = dict(batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    retain_loader = DataLoader(retain_ds, **kw)
    test_loader   = DataLoader(test_ds,   **kw)
    forget_loader = DataLoader(forget_ds, **kw)

    result = run_mia_threshold(
        model, retain_loader, test_loader, forget_loader,
        device, score_type=score_type
    )

    if verbose:
        print(f"\n[MIA Results — {score_type}]")
        print(f"  Retain vs Test  AUC : {result['retain_test_auc']:.4f}  "
              f"(higher = model still distinguishes members)")
        print(f"  Forget vs Test  AUC : {result['forget_test_auc']:.4f}  "
              f"(target: 0.5000 = perfectly forgotten)")
        print(f"  Forget Advantage    : {result['forget_advantage']:.4f}  "
              f"(target: 0.0000)")
        print(f"  Forget vs Test  Acc : {result['forget_test_acc']:.4f}  "
              f"(target: ~0.5000)")

    return result
