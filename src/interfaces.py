"""
interfaces.py — TypedDict contracts for unlearning methods and metrics.

Formalises the dict-return convention shared by all 10 unlearning methods.
Every method returns an `UnlearningResult`; the runner validates it after
each call so contract violations surface immediately instead of failing
silently 12 hours into an iterative run.

Usage:
    from interfaces import UnlearningResult, validate_unlearning_result

    result: UnlearningResult = my_method(model, csv_path, device, **cfg)
    validate_unlearning_result(result, method_name)
"""

from __future__ import annotations

from typing import Any, TypedDict

import logging

logger = logging.getLogger(__name__)


class UnlearningResult(TypedDict, total=False):
    """Contract for all unlearning method return values.

    Required keys: model, method, metrics.
    Optional keys: extra (per-method diagnostics, e.g. AdaptiForget history).
    """
    model: Any            # nn.Module — unlearned model
    method: str           # display name, e.g. "AdaptiForget"
    metrics: dict[str, Any]  # timing, step counts, loss traces
    extra: dict[str, Any]    # method-specific diagnostics (optional)


REQUIRED_KEYS = ("model", "method", "metrics")


def prepare_method_call(cfg: dict) -> dict:
    """Pin ``subset='train'`` on an unlearning-method config.

    Unlearning methods must never see holdout images — the holdout is
    reserved for post-unlearning evaluation.  All runners (single_shot,
    iterative, hparam_search, ablation_study) must build method configs
    through this helper so the pin cannot be forgotten.

    ``subset`` is injected into the config dict, which methods receive via
    ``**cfg`` and forward to their dataset construction.
    """
    out = dict(cfg)
    out["subset"] = "train"
    return out


def validate_unlearning_result(result: dict, method_name: str) -> None:
    """Validate a method's return dict against the contract.

    Raises ValueError on missing keys or wrong types — surfaces contract
    violations at the call site rather than deep in the pipeline.
    """
    import torch.nn as nn

    for key in REQUIRED_KEYS:
        if key not in result:
            raise ValueError(
                f"[Contract] {method_name} returned dict missing required "
                f"key '{key}'. Expected keys: {REQUIRED_KEYS}"
            )
    if not isinstance(result["model"], nn.Module):
        raise ValueError(
            f"[Contract] {method_name} 'model' must be an nn.Module, "
            f"got {type(result['model']).__name__}"
        )
    if not isinstance(result["method"], str):
        raise ValueError(
            f"[Contract] {method_name} 'method' must be a str"
        )
    if not isinstance(result["metrics"], dict):
        raise ValueError(
            f"[Contract] {method_name} 'metrics' must be a dict"
        )

    # Warn on suspicious metric values (common contract violations)
    metrics = result["metrics"]
    if "unlearning_time_s" in metrics:
        t = metrics["unlearning_time_s"]
        if not isinstance(t, (int, float)):
            logger.warning(f"[Contract] {method_name} unlearning_time_s not numeric: {t}")
    if "steps" in metrics and metrics["steps"] == 0:
        logger.warning(f"[Contract] {method_name} reported 0 steps — verify loop ran")
