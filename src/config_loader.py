"""
config_loader.py — Single source of truth for method hyperparameters.

Loads per-method YAML files from configs/methods/ and returns them as
nested dicts.  Replaces the duplicated DEFAULT_CONFIGS / DEFAULT_ITER_CONFIGS
dicts previously scattered across comprehensive_eval.py and iterative_unlearning.py.

Also provides smoke-test scaling: when scale < 1.0, all step/epoch
counts are multiplied by the scale factor (floored to min 1), while learning
rates and fractions are left unchanged. Scale is an explicit CLI flag
(--scale) on each stage script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_METHODS_DIR = Path(__file__).resolve().parent.parent / "configs" / "methods"

# Cached after first load
_METHOD_CONFIGS: dict[str, dict[str, Any]] | None = None


def load_method_configs(scale: float = 1.0) -> dict[str, dict[str, Any]]:
    """Load all method configs from configs/methods/*.yaml.

    Args:
        scale: Smoke-test factor (0.0–1.0).  Multiplies all integer step/epoch
               counts.  Values < 1.0 are floored to min 1.  Learning rates,
               fractions, and boolean flags are unscaled.

    Returns:
        Dict keyed by method name (stem of YAML file), each value a dict of
        hyperparameter name → value.

    Raises:
        FileNotFoundError: if configs/methods/ is missing or empty.
    """
    global _METHOD_CONFIGS

    if _METHOD_CONFIGS is None:
        _METHOD_CONFIGS = {}
        yaml_files = sorted(_METHODS_DIR.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(
                f"No method configs found in {_METHODS_DIR}"
            )
        for path in yaml_files:
            with open(path) as fh:
                cfg = yaml.safe_load(fh) or {}
            name = path.stem
            _METHOD_CONFIGS[name] = cfg

    if scale >= 1.0:
        return dict(_METHOD_CONFIGS)  # shallow copy for safety

    # Apply smoke scaling
    scaled: dict[str, dict[str, Any]] = {}
    for name, cfg in _METHOD_CONFIGS.items():
        s: dict[str, Any] = {}
        for key, value in cfg.items():
            s[key] = _scale_value(key, value, scale)
        scaled[name] = s
    return scaled


def _scale_value(key: str, value: Any, scale: float) -> Any:
    """Scale integer step/epoch values by scale; leave everything else."""
    if not isinstance(value, int):
        return value
    # Keys that represent step/epoch counts (not fractions, not flags)
    if any(suffix in key for suffix in (
        "steps", "epochs", "max_steps", "patience",
        "retain_steps_per", "retain_reg_every", "retain_every",
        "anneal_steps", "refresh_every", "grad_batches",
        "saliency_batches",
    )):
        return max(1, int(value * scale))
    return value


def get_method_config(method: str, scale: float = 1.0) -> dict[str, Any]:
    """Convenience: load configs and return the one for `method`."""
    return load_method_configs(scale).get(method, {})


def available_methods() -> list[str]:
    """Return sorted list of method names with config files."""
    return sorted(load_method_configs().keys())
