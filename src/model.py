"""
model.py — ResNet-18 for Virtual Identity Classification.

build_resnet18()  → returns model ready for training
load_model()      → loads a saved checkpoint
save_model()      → saves checkpoint with metadata
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from datetime import datetime


NUM_CLASSES = 4   # age groups: Young / Adult / Middle-Aged / Senior


# ── Model factory ─────────────────────────────────────────────────────────────

def build_resnet18(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build ResNet-18 for age-group classification.

    Args:
        num_classes:      Number of output classes (default 4).
        pretrained:       Use ImageNet-pretrained weights (default True).
        freeze_backbone:  Freeze all layers except the final FC (default False).

    Returns:
        nn.Module ready for training/unlearning.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad_(False)

    return model


# ── Checkpoint utilities ───────────────────────────────────────────────────────

def save_model(
    model: nn.Module,
    path: str,
    metadata: dict = None,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = None,
):
    """Save model weights + optional metadata to a .pt checkpoint."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
        "saved_at": datetime.now().isoformat(),
        "num_classes": NUM_CLASSES,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch

    torch.save(payload, save_path)
    print(f"[OK] Model saved → {save_path}")


def load_model(
    path: str,
    num_classes: int = NUM_CLASSES,
    device: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load a ResNet-18 model from a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(device)
    print(f"[OK] Model loaded ← {path}")
    meta = checkpoint.get("metadata", {})
    if meta:
        print(f"     Metadata: {meta}")
    return model


def copy_model(model: nn.Module, device: str = "cpu") -> nn.Module:
    """Return a deep copy of a model (for unlearning — avoid mutating original)."""
    import copy
    m = copy.deepcopy(model)
    m.to(device)
    return m
