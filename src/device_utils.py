"""Shared device selection and resource helpers for CLI entry points."""

from __future__ import annotations

import os

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


def resolve_num_workers(requested: int | None = None) -> int:
    """Return the number of DataLoader workers.

    Priority: explicit override > UNLEARN_NUM_WORKERS env > Slurm CPUs >
    os.cpu_count() > 4.  Capped at 8 to avoid I/O contention on shared HPC
    nodes.  Set UNLEARN_NUM_WORKERS=0 for CPU-only runs: per-step loader
    recreation with many workers deadlocks on macOS (main thread blocks in
    poll waiting on worker pipes).
    """
    if requested is not None and requested > 0:
        return min(requested, 16)

    env_workers = os.environ.get("UNLEARN_NUM_WORKERS")
    if env_workers is not None:
        try:
            return max(0, min(int(env_workers), 16))
        except ValueError:
            pass

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return min(int(slurm_cpus), 8)

    cpu_count = os.cpu_count() or 4
    return min(cpu_count, 8)