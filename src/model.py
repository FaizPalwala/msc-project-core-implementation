"""
model.py — Dual-Head ResNet-18 for Identity-Level Machine Unlearning.

Architecture
────────────
  ResNet-18 backbone (ImageNet pretrained, 224×224 input)
    → AdaptiveAvgPool2d → 512-d feature vector
    → fc_identity : 512 → 600   (primary task, MUFAC-aligned)
    → fc_age      : 512 →   4   (secondary task, Handover-aligned)

Forward returns:  (id_logits, age_logits)

API
───
  build_dual_head_resnet18()  → fresh model
  load_model()                → restore from checkpoint, auto-detects layout
  save_model()                → persist with metadata
  copy_model()                → deep copy for unlearning (avoids mutating original)
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


import logging

logger = logging.getLogger(__name__)
# ── Class counts ──────────────────────────────────────────────────────────────

NUM_IDENTITY_CLASSES = 600   # one per SFHQ-InstantID cluster
NUM_AGE_CLASSES      =   4   # Young / Adult / Middle-Aged / Senior

# ── Model factory ─────────────────────────────────────────────────────────────

class DualHeadResNet(nn.Module):
    """ResNet-18 with separate identity-classification and age-classification heads.

    The backbone is all layers up to (but excluding) the original FC.
    The penultimate pooled features (512-d) feed both heads independently.
    """

    def __init__(
        self,
        identity_classes: int = NUM_IDENTITY_CLASSES,
        age_classes: int = NUM_AGE_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        full = resnet18(weights=weights)

        # Split: everything before the original FC → backbone
        self.backbone = nn.Sequential(
            full.conv1,
            full.bn1,
            full.relu,
            full.maxpool,
            full.layer1,
            full.layer2,
            full.layer3,
            full.layer4,
            full.avgpool,
        )

        _in_features = full.fc.in_features   # 512

        self.fc_identity = nn.Linear(_in_features, identity_classes)
        self.fc_age       = nn.Linear(_in_features, age_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Store for introspection
        self.identity_classes = identity_classes
        self.age_classes      = age_classes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x).flatten(1)          # (B, 512)
        return self.fc_identity(f), self.fc_age(f)

    def forward_identity(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: identity logits only (e.g. for standalone probing)."""
        f = self.backbone(x).flatten(1)
        return self.fc_identity(f)

    def forward_age(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: age logits only."""
        f = self.backbone(x).flatten(1)
        return self.fc_age(f)

    def feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d penultimate features (for probing / t-SNE)."""
        return self.backbone(x).flatten(1)


def build_dual_head_resnet18(
    identity_classes: int = NUM_IDENTITY_CLASSES,
    age_classes: int = NUM_AGE_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> DualHeadResNet:
    """Create a fresh dual-head ResNet-18.

    Args:
        identity_classes: Output classes for identity head (default 600).
        age_classes:      Output classes for age head (default 4).
        pretrained:       Use ImageNet-1K weights (default True).
        freeze_backbone:  Freeze convolutional layers (default False).
    """
    return DualHeadResNet(
        identity_classes=identity_classes,
        age_classes=age_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_model(
    model: DualHeadResNet,
    path: str | Path,
    metadata: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
) -> None:
    """Save a dual-head model checkpoint.

    Includes identity/age class counts so ``load_model`` can reconstruct
    the exact architecture without requiring them as arguments.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
        "saved_at": datetime.now().isoformat(),
        "identity_classes": model.identity_classes,
        "age_classes": model.age_classes,
        "architecture": "dual_head",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch

    torch.save(payload, save_path)
    logger.info(f"[OK] Model saved → {save_path}")


def load_model(
    path: str | Path,
    device: str = "cpu",
    strict: bool = True,
) -> DualHeadResNet:
    """Load a dual-head ResNet-18 from a checkpoint.

    Auto-detects single-head (legacy) vs dual-head (current) checkpoints.
    Legacy checkpoints are loaded into the age head with identity head
    initialised randomly — suitable for evaluation, not for continued training.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    arch = checkpoint.get("architecture", "single_head")  # pre-dual-head default

    if arch == "dual_head":
        id_cls = checkpoint["identity_classes"]
        age_cls = checkpoint["age_classes"]
        model = build_dual_head_resnet18(
            identity_classes=id_cls,
            age_classes=age_cls,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        # Legacy single-head checkpoint → identity head random, age head loaded
        logger.info("Legacy single-head checkpoint detected — "
              "loading into age head; identity head will be randomly initialised.")
        age_cls = checkpoint.get("num_classes", NUM_AGE_CLASSES)
        model = build_dual_head_resnet18(
            identity_classes=NUM_IDENTITY_CLASSES,
            age_classes=age_cls,
            pretrained=False,
        )
        # Map old 'fc' keys to 'fc_age' in state dict
        old_sd = checkpoint["model_state_dict"]
        new_sd = {}
        for k, v in old_sd.items():
            if k == "fc.weight":
                new_sd["fc_age.weight"] = v
            elif k == "fc.bias":
                new_sd["fc_age.bias"] = v
            elif k.startswith("backbone."):
                new_sd[k] = v
            else:
                # Old backbone layers weren't named 'backbone.X' — they were
                # direct children. Map common ResNet-18 keys.
                new_sd[f"backbone.{k}"] = v
        model.load_state_dict(new_sd, strict=False)

    model.to(device)
    logger.info(f"[OK] Model loaded ← {path}")
    meta = checkpoint.get("metadata", {})
    if meta:
        logger.info(f"     Metadata: {meta}")
    return model


def copy_model(model: nn.Module, device: str = "cpu") -> nn.Module:
    """Deep-copy a model for unlearning — avoids mutating the original."""
    m = copy.deepcopy(model)
    m.to(device)
    return m
