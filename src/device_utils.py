"""Shared device selection helpers for CLI entry points."""

from __future__ import annotations

import torch


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve a requested device string using CUDA > MPS > CPU for auto."""
    if device_str != "auto":
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")