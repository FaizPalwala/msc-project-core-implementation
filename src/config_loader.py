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

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_METHODS_DIR = Path(__file__).resolve().parent.parent / "configs" / "methods"

# Cached after first load
_METHOD_CONFIGS: dict[str, dict[str, Any]] | None = None


def load_method_configs(scale: float = 1.0,
                        best_configs_path: str | Path | None = None,
                        ) -> dict[str, dict[str, Any]]:
    """Load all method configs from configs/methods/*.yaml.

    Args:
        scale: Smoke-test factor (0.0–1.0).  Multiplies all integer step/epoch
               counts.  Values < 1.0 are floored to min 1.  Learning rates,
               fractions, and boolean flags are unscaled.
        best_configs_path: Optional JSON file of per-method best configs from
               hparam_search ({"method": {...}}).  When given, each method's
               keys are merged OVER the YAML defaults — the tuned values win,
               untuned keys fall back to the default.  This is how the
               "single-shot with best params after HP search" stage works.

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

    configs = dict(_METHOD_CONFIGS)  # shallow copy for safety

    # Merge HP-search best configs over the YAML defaults.
    if best_configs_path is not None:
        best_path = Path(best_configs_path)
        if best_path.is_dir():
            # Parallel-safe form: hparam writes one {method}_best_config.json
            # per method (each hparam job is per-method, all parallel).
            files = sorted(best_path.glob("*_best_config.json"))
            if not files:
                raise FileNotFoundError(
                    f"No *_best_config.json found in {best_path} — run "
                    f"hparam_search first (it writes per-method best configs)."
                )
            best: dict = {}
            import json
            for f in files:
                name = f.name.removesuffix("_best_config.json")
                with open(f) as fh:
                    best[name] = json.load(fh)
        else:
            if not best_path.exists():
                raise FileNotFoundError(
                    f"best_configs_path {best_path} does not exist — run "
                    f"hparam_search first (it writes per-method best configs)."
                )
            import json
            with open(best_path) as fh:
                best = json.load(fh)
        merged: list[str] = []
        for name, overrides in best.items():
            if name in configs and isinstance(overrides, dict):
                changed = [k for k in overrides if overrides[k] != configs[name].get(k)]
                configs[name] = {**configs[name], **overrides}
                if changed:
                    merged.append(f"{name}[{','.join(changed)}]")
            elif name not in configs:
                # Method searched but no YAML default (shouldn't happen) —
                # keep the searched config as-is.
                configs[name] = dict(overrides)
        if merged:
            logger.info(f"  [config] OPTIMIZED configs merged from {best_path}: {', '.join(merged)}")
        else:
            logger.info(f"  [config] best_configs_path given but no keys differed from YAML defaults: {best_path}")
    else:
        logger.info("  [config] DEFAULT configs (YAML configs/methods/*.yaml, no HP-search overrides)")

    if scale >= 1.0:
        return configs

    # Apply smoke scaling
    scaled: dict[str, dict[str, Any]] = {}
    for name, cfg in configs.items():
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


def get_method_config(method: str, scale: float = 1.0,
                      best_configs_path: str | Path | None = None) -> dict[str, Any]:
    """Convenience: load configs and return the one for `method`.

    best_configs_path: optional HP-search best-configs (dir of
    {method}_best_config.json files or a single merged JSON) — the tuned
    values override the YAML defaults, so callers like the ablation study
    evaluate the method AS DEPLOYED (tuned), not the default baseline.
    """
    return load_method_configs(scale, best_configs_path).get(method, {})


def available_methods() -> list[str]:
    """Return sorted list of method names with config files."""
    return sorted(load_method_configs().keys())
