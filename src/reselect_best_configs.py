#!/usr/bin/env python3
"""reselect_best_configs.py — OFFLINE best-config re-selection (f2 recovery).

The 19-Aug partial re-run executed every hparam grid correctly, but the
gate's PROBE ARM (now retired) rejected 100% of trials — probe-identity
on the forget split saturates at ~1.0 even for the retrain oracle at
full scale (it measures backbone feature separability, not erasure).
All 5 re-tuned methods therefore lost their best configs and ran at
YAML defaults downstream.

The trials themselves are complete and valid.  This script re-selects
the best config per method from the EXISTING trial JSONLs using the
corrected gate (forget_id_acc <= 0.15, pre-registered target) ranked by
the v2 erasure-conditioned UF — no hparam re-run needed.

Usage:
  python3 src/reselect_best_configs.py results/balanced/hparam
  python3 src/reselect_best_configs.py results/imbalanced/hparam

For each method grid_trials.jsonl:
  - keeps only lines from the CURRENT run (probe_identity_forget_acc
    present — the old run's lines lack it)
  - gate: forget_id_acc <= ERASURE_FORGET_ACC_MAX
  - rank by stored v2 uf_score (Aire ran 20b7a7f)
  - writes {method}_best_config.json; if none pass, DELETES any stale
    file so the method runs at YAML default downstream
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reselect")

ERASURE_FORGET_ACC_MAX = 0.15


def _v2_uf(t: dict) -> float:
    """v2 erasure-conditioned UF, matching hparam_search.uf_score."""
    fg = t.get("forget_id_acc", 1.0)
    forget_score = max(0.0, 1.0 - 2.0 * t.get("mia_mean_auc", 0.5))
    if fg > ERASURE_FORGET_ACC_MAX:
        forget_score = 0.0
    return (0.5 * t.get("retain_id_acc", 0.0)
            + 0.4 * forget_score
            - 0.1 * min(t.get("total_time_s", t.get("unlearning_time_s", 300.0)) / 300.0, 1.0))


def _load_default_uf(hparam_dir: Path, method: str) -> float | None:
    """v2 UF of the YAML default config, from the defaults single_shot run.

    A tuned config must BEAT the default on the same objective — otherwise
    the method runs at its default (the honest outcome for methods whose
    grid cannot improve on defaults, e.g. GA: only erasing config destroys
    retain (UF -0.0014) vs default UF 0.4725).
    """
    agg = hparam_dir.parent / "single_shot" / "single_shot_aggregated.json"
    if not agg.exists():
        return None
    data = json.loads(agg.read_text())
    row = data.get(method)
    if not row:
        return None
    return _v2_uf(row)


def reselect(hparam_dir: Path) -> None:
    for jl_path in sorted(hparam_dir.glob("*_grid_trials.jsonl")):
        method = jl_path.name.replace("_grid_trials.jsonl", "")
        trials = []
        for line in jl_path.read_text().strip().splitlines():
            t = json.loads(line)
            if t.get("probe_identity_forget_acc") is not None:  # current run only
                trials.append(t)
        if not trials:
            # Method was NOT re-run in the partial run (ct/ft/srl/msg_kd):
            # its JSONL has no current-run lines.  Its best config from
            # the final run is already gate-valid — leave it untouched.
            logger.info(f"{method}: no current-run trials → keeping existing "
                        f"best config if present")
            continue
        passing = [t for t in trials if t.get("forget_id_acc", 1.0) <= ERASURE_FORGET_ACC_MAX]
        best_json = hparam_dir / f"{method}_best_config.json"
        if not passing:
            if best_json.exists():
                best_json.unlink()
                logger.warning(f"{method}: 0/{len(trials)} pass gate → stale best config "
                              f"removed; runs at YAML default")
            else:
                logger.warning(f"{method}: 0/{len(trials)} pass gate → YAML default")
            continue
        best = max(passing, key=lambda t: t.get("uf_score", 0.0))
        best_uf = _v2_uf(best)
        def_uf = _load_default_uf(hparam_dir, method)
        if def_uf is not None and best_uf < def_uf:
            if best_json.exists():
                best_json.unlink()
                logger.warning(
                    f"{method}: best passing v2UF {best_uf:.4f} < default {def_uf:.4f} "
                    f"→ keeping YAML default (tuned cannot beat default at full scale)")
            else:
                logger.warning(
                    f"{method}: best passing v2UF {best_uf:.4f} < default {def_uf:.4f} "
                    f"→ YAML default (tuned cannot beat default at full scale)")
            continue
        best_json.write_text(json.dumps(best["config"], indent=2) + "\n")
        logger.info(
            f"{method}: {len(passing)}/{len(trials)} pass gate → best forget="
            f"{best['forget_id_acc']:.4f} retain={best['retain_id_acc']:.4f} "
            f"v2UF={best_uf:.4f} (default {def_uf if def_uf is not None else 'n/a'}) "
            f"→ {best_json.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    reselect(Path(sys.argv[1]))
