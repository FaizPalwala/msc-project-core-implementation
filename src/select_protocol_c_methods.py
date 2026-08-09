"""
select_protocol_c_methods.py — Protocol C: dynamic method selection.

After single_shot_best completes, pick ONE best method per category
(baseline / SOTA / novel) by UF score, computed from the aggregated
results.  These 3 methods then feed the Protocol C budget sweep (the
cleanest causal difficulty test: sweep unlearning budget per bin until
each bin reaches its per-bin oracle).

Categories (fixed by method family, not by results):
    baseline : ga, srl, ft
    sota     : ng_plus, msg, ct
    novel    : msg_kd, adaptiforget, budget_scaled

Ranking metric: hparam_search.uf_score(retain_acc, forget_advantage,
time_s) — the same composite used by the HP search, so selection is
consistent with the tuning objective.

Usage:
    python src/select_protocol_c_methods.py \\
        --aggregated results/imbalanced/single_shot_best/single_shot_aggregated.json \\
        --out results/imbalanced

Writes protocol_c_methods.json:
    {"baseline": "ft", "sota": "ng_plus", "novel": "adaptiforget", "scores": {...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Fixed method-family categories.  no_unlearning/retrain excluded:
# control + oracle are not sweepable (no unlearning budget to vary).
CATEGORIES = {
    "baseline": ["ga", "srl", "ft"],
    "sota":     ["ng_plus", "msg", "ct"],
    "novel":    ["msg_kd", "adaptiforget", "budget_scaled"],
}


def _method_uf(data: dict) -> float:
    """UF score from aggregated (mean) results for one method."""
    from hparam_search import uf_score
    retain = data.get("retain_id_acc") or 0.0
    mia = data.get("mia_mean_auc")
    mia = float(mia) if mia is not None else 0.5
    t = data.get("total_time_s") or data.get("unlearning_time_s") or 0.0
    return uf_score(retain_acc=retain, mia_auc=mia, time_s=t)


def select(aggregated_path: str, out_dir: str) -> dict:
    agg_path = Path(aggregated_path)
    if not agg_path.exists():
        raise FileNotFoundError(f"Aggregated results not found: {agg_path}")
    with open(agg_path) as fh:
        agg = json.load(fh)

    # aggregated results are keyed by method under "aggregated"
    results = agg.get("aggregated", agg)
    logger.info(f"  Methods in aggregated results: {sorted(results)}")

    scores: dict[str, dict[str, float]] = {}
    for cat, methods in CATEGORIES.items():
        scores[cat] = {}
        for m in methods:
            if m in results and "error" not in results[m]:
                scores[cat][m] = _method_uf(results[m])
                logger.info(f"  [{cat:>8}] {m:<14} UF={scores[cat][m]:.4f}")

    picked: dict[str, str] = {}
    for cat, method_scores in scores.items():
        if not method_scores:
            logger.warning(f"  [warn] no scored methods in category '{cat}'")
            continue
        best = max(method_scores.items(), key=lambda kv: kv[1])[0]
        picked[cat] = best
        logger.info(f"  → best {cat}: {best} (UF={method_scores[best]:.4f})")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result = {"categories": picked, "scores": scores,
              "source": str(agg_path)}
    out_file = out_path / "protocol_c_methods.json"
    with open(out_file, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"\n  Protocol C methods → {out_file}")
    logger.info(f"  Budget sweep will run: {list(picked.values())}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregated", type=str, required=True,
                        help="single_shot_best/single_shot_aggregated.json")
    parser.add_argument("--out", type=str, default="results/imbalanced")
    args = parser.parse_args()
    select(args.aggregated, args.out)


if __name__ == "__main__":
    main()
