"""
report.py — State-of-the-art post-run reporting for the dissertation.

Reads the evaluation artifacts produced by the pipeline:
  - results/single_shot/single_shot_aggregated.json   (per-method μ ± σ, p-values)
  - results/iterative/iterative_combined_aggregated.csv (per-method×step μ ± σ)
  - results/iterative/plots/stability_summary.csv      (summary statistics)

and renders three artefacts in `results/report/`:
  - report.tex   — LaTeX tables (μ ± σ, significance stars, bold bests) for the paper
  - report.md    — markdown summary for quick reading / supervisor review
  - report.csv   — machine-readable full table (for further analysis)

Design rationale
----------------
The dissertation needs "state of the art" reporting so post-run inference
is easy.  LaTeX tables with correct ± formatting, significance stars from
the bootstrap oracle comparison, and best-value emphasis are directly
camera-ready.  The CSV keeps every number accessible for follow-up
analysis, and the markdown view serves as a human-readable digest.

Usage:
    python src/report.py [--results results] [--out results/report]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)

# ── Metric display metadata ──────────────────────────────────────────────────
# (key, human label, lower_is_better, format spec)
METRIC_SPECS: list[tuple[str, str, bool, str]] = [
    ("retain_id_acc",        "Retain Id Acc",            False, ".4f"),
    ("retain_age_acc",       "Retain Age Acc",           False, ".4f"),
    ("forget_id_acc",        "Forget Id Acc",            False, ".4f"),
    ("mia_mean_auc",         "MIA AUC (mean)",           True,  ".4f"),
    ("mia_max_auc",          "MIA AUC (max)",            True,  ".4f"),
    ("mia_fraction_leaked",  "MIA Frac Leaked",          True,  ".4f"),
    ("max_conf_auc",         "Max-Conf AUC",             True,  ".4f"),
    ("probe_identity_acc",   "Probe Id Acc",             True,  ".4f"),
    ("probe_age_acc",        "Probe Age Acc",            True,  ".4f"),
    ("probe_gender_acc",     "Probe Gender Acc",         True,  ".4f"),
    ("total_time_s",         "Total Time (s)",           True,  ".1f"),
]

UF_WEIGHTS = {"w_retain": 0.5, "w_forget": 0.4, "w_time": 0.1}
TIME_BUDGET_S = 300.0

# ── Helpers ──────────────────────────────────────────────────────────────────


def _latex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def _sig_stars(p: float) -> str:
    """Significance stars from p-value (one-tailed bootstrap)."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_musigma(mu: Any, sigma: Any, fmt: str = ".4f") -> str:
    """Format μ ± σ; if σ missing, format μ alone."""
    if mu is None or (isinstance(mu, float) and np.isnan(mu)):
        return "—"
    m = f"{float(mu):{fmt}}"
    if sigma is not None and not (isinstance(sigma, float) and np.isnan(sigma)):
        return rf"{m} $\pm$ {float(sigma):{fmt}}"
    return m


def _uf_score(retain_acc: float, mia_mean_auc: float, time_s: float) -> float:
    """Composite Utility-Forgetting score ∈ [0, 1] (see hparam_search.uf_score)."""
    forget_advantage = abs(mia_mean_auc - 0.5)
    forget_score = max(0.0, 1.0 - forget_advantage / 0.5)
    time_score = min(1.0, time_s / TIME_BUDGET_S)
    return (
        UF_WEIGHTS["w_retain"] * retain_acc
        + UF_WEIGHTS["w_forget"] * forget_score
        - UF_WEIGHTS["w_time"] * time_score
    )


# ── Loaders ──────────────────────────────────────────────────────────────────


def _load_single_shot(results_dir: Path) -> dict[str, Any] | None:
    path = results_dir / "single_shot" / "single_shot_aggregated.json"
    if not path.exists():
        logger.warning(f"[Report] Single-shot aggregated JSON not found: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    logger.info(f"[Report] Loaded single-shot aggregation from {path}")
    return data


def _load_iterative_aggregated(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "iterative" / "iterative_combined_aggregated.csv"
    if not path.exists():
        logger.warning(f"[Report] Iterative aggregated CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    logger.info(f"[Report] Loaded iterative aggregation: {path} ({len(df)} rows)")
    return df


def _load_stability_summary(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "iterative" / "plots" / "stability_summary.csv"
    if not path.exists():
        logger.warning(f"[Report] Stability summary not found: {path}")
        return None
    df = pd.read_csv(path)
    logger.info(f"[Report] Loaded stability summary: {path} ({len(df)} rows)")
    return df


def _load_canary(results_dir: Path) -> dict[str, Any] | None:
    """Load canary verification JSONs: canary/verify_<method>.json.

    Each file is a dict keyed by identity_id → {n_images, mean_feature_norm,
    feature_similarity_to_clean}.  Returns {method: {id: metrics}} or None.
    """
    canary_dir = results_dir / "canary"
    if not canary_dir.exists():
        logger.warning(f"[Report] Canary dir not found: {canary_dir}")
        return None
    files = sorted(canary_dir.glob("verify_*.json"))
    if not files:
        logger.warning(f"[Report] No verify_*.json in {canary_dir}")
        return None
    data: dict[str, Any] = {}
    for path in files:
        method = path.stem[len("verify_"):]
        with open(path) as f:
            data[method] = json.load(f)
    logger.info(f"[Report] Loaded canary verification: {[p.name for p in files]}")
    return data


# ── Renderers: LaTeX ─────────────────────────────────────────────────────────


def _render_single_shot_latex(data: dict) -> str:
    """LaTeX table: rows = methods, cols = core metrics, μ ± σ + stars."""
    methods = sorted(data.keys())
    if not methods:
        return ""

    # Which metrics are actually present?
    present = []
    for key, label, lower_better, fmt in METRIC_SPECS:
        if any(key in data[m] for m in methods):
            present.append((key, label, lower_better, fmt))

    # Best per metric (excluding the retrain oracle from the competition)
    best: dict[str, float] = {}
    for key, _label, lower_better, _fmt in present:
        vals = {
            m: float(data[m][key])
            for m in methods
            if key in data[m] and m != "retrain" and "error" not in data[m]
        }
        if not vals:
            continue
        best[key] = min(vals.values()) if lower_better else max(vals.values())

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Single-shot unlearning evaluation (mean $\\pm$ std across "
        f"{'seeds' if any(k.endswith('_std') for m in methods for k in data[m]) else 'seeds'}, "
        "bootstrap $p$ vs. retrain oracle: $* p<0.05$, $** p<0.01$, $*** p<0.001$)}",
        "\\label{tab:single_shot}",
        "\\small",
        "\\begin{tabular}{l" + "c" * (len(present) + 1) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{_latex_escape(label)}}}"
                                 for _k, label, _lb, _fmt in present) + " & UF \\\\",
        "\\midrule",
    ]

    for m in methods:
        row = data[m]
        if "error" in row:
            lines.append(f"{_latex_escape(row.get('method', m))} & ERROR & \\multicolumn{{{len(present)}}}{{c}}{{—}} \\\\")
            continue
        cells = []
        for key, _label, lower_better, fmt in present:
            mu = row.get(key)
            sigma = row.get(f"{key}_std")
            cell = _fmt_musigma(mu, sigma, fmt)
            # Bold best
            if key in best and mu is not None and not np.isnan(float(mu)):
                if (lower_better and float(mu) <= best[key] + 1e-9) or \
                   (not lower_better and float(mu) >= best[key] - 1e-9):
                    cell = f"\\textbf{{{cell}}}"
            # Append significance stars for MIA AUC
            p_key = f"p_vs_oracle_{key}"
            if key == "mia_mean_auc" and p_key in row:
                cell = f"{cell} {_sig_stars(float(row[p_key]))}"
            cells.append(cell)

        # UF score
        r_acc, mia, t = (row.get("retain_id_acc"), row.get("mia_mean_auc"),
                         row.get("total_time_s"))
        if all(v is not None for v in (r_acc, mia, t)):
            cells.append(f"{_uf_score(float(r_acc), float(mia), float(t)):.4f}")
        else:
            cells.append("—")

        lines.append(f"{_latex_escape(row.get('method', m))} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _render_iterative_latex(df: pd.DataFrame) -> str:
    """LaTeX table: final-step per method, μ ± σ from iterative aggregation."""
    if df is None or df.empty:
        return ""

    # Final step per method
    final_rows = df.loc[df.groupby("method")["step"].idxmax()].copy()
    final_rows = final_rows.sort_values("method")

    # Columns present with _mean/_std
    mean_cols = [c for c in final_rows.columns if c.endswith("_mean")]
    cols = []
    for key, label, lower_better, fmt in METRIC_SPECS:
        mc = f"{key}_mean"
        if mc in final_rows.columns:
            cols.append((key, label, lower_better, fmt))

    # Best per metric
    best = {}
    for key, _label, lower_better, _fmt in cols:
        vals = {r["method"]: float(r[f"{key}_mean"]) for _, r in final_rows.iterrows()
                if r["method"] != "retrain" and not np.isnan(r[f"{key}_mean"])}
        if vals:
            best[key] = min(vals.values()) if lower_better else max(vals.values())

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Iterative unlearning at final step (mean $\\pm$ std across seeds)}",
        "\\label{tab:iterative_final}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{_latex_escape(label)}}}"
                                 for _k, label, _lb, _fmt in cols) + " \\\\",
        "\\midrule",
    ]

    for _i, row in final_rows.iterrows():
        cells = []
        for key, _label, lower_better, fmt in cols:
            mu = row[f"{key}_mean"]
            sigma = row.get(f"{key}_std", np.nan)
            cell = _fmt_musigma(mu, sigma, fmt)
            if key in best and not np.isnan(float(mu)) and \
               ((lower_better and float(mu) <= best[key] + 1e-9) or
                (not lower_better and float(mu) >= best[key] - 1e-9)):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{_latex_escape(str(row['method']))} & " + " & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Renderers: markdown ──────────────────────────────────────────────────────


def _render_single_shot_md(data: dict) -> str:
    if not data:
        return "_No single-shot results available._\n"
    methods = sorted(data.keys())
    present = []
    for key, label, lower_better, fmt in METRIC_SPECS:
        if any(key in data[m] for m in methods):
            present.append((key, label, lower_better, fmt))

    best = {}
    for key, _l, lower_better, _f in present:
        vals = {m: float(data[m][key]) for m in methods
                if key in data[m] and m != "retrain" and "error" not in data[m]}
        if vals:
            best[key] = min(vals.values()) if lower_better else max(vals.values())

    header = "| Method | " + " | ".join(l for _k, l, _lb, _f in present) + " | UF |"
    sep = "|" + "---|" * (len(present) + 2)
    lines = [header, sep]
    for m in methods:
        row = data[m]
        if "error" in row:
            lines.append(f"| {row.get('method', m)} | ERROR |" + " — |" * (len(present) + 1))
            continue
        cells = []
        for key, _l, lower_better, fmt in present:
            mu, sigma = row.get(key), row.get(f"{key}_std")
            cell = _fmt_musigma(mu, sigma, fmt).replace("$\\pm$", "±")
            if key in best and mu is not None and not np.isnan(float(mu)) and \
               ((lower_better and float(mu) <= best[key] + 1e-9) or
                (not lower_better and float(mu) >= best[key] - 1e-9)):
                cell = f"**{cell}**"
            p_key = f"p_vs_oracle_{key}"
            if key == "mia_mean_auc" and p_key in row:
                cell = f"{cell} {_sig_stars(float(row[p_key]))}"
            cells.append(cell)
        r_acc, mia, t = (row.get("retain_id_acc"), row.get("mia_mean_auc"),
                         row.get("total_time_s"))
        uf = f"{_uf_score(float(r_acc), float(mia), float(t)):.4f}" if all(
            v is not None for v in (r_acc, mia, t)) else "—"
        lines.append(f"| {row.get('method', m)} | " + " | ".join(cells) + f" | {uf} |")
    lines.append("")
    lines.append("_Bold = best (excluding retrain oracle). Stars: * p<0.05, ** p<0.01, *** p<0.001 (bootstrap vs oracle). Retrain oracle itself is excluded from the best-value competition._")
    return "\n".join(lines)


def _render_iterative_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No iterative results available._\n"
    final_rows = df.loc[df.groupby("method")["step"].idxmax()].sort_values("method")
    cols = []
    for key, label, lower_better, fmt in METRIC_SPECS:
        if f"{key}_mean" in final_rows.columns:
            cols.append((key, label, lower_better, fmt))
    best = {}
    for key, _l, lower_better, _f in cols:
        vals = {r["method"]: float(r[f"{key}_mean"]) for _, r in final_rows.iterrows()
                if not np.isnan(r[f"{key}_mean"])}
        if vals:
            best[key] = min(vals.values()) if lower_better else max(vals.values())
    header = "| Method | " + " | ".join(l for _k, l, _lb, _f in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = ["", header, sep]
    for _i, row in final_rows.iterrows():
        cells = []
        for key, _l, lower_better, fmt in cols:
            mu, sigma = row[f"{key}_mean"], row.get(f"{key}_std", np.nan)
            cell = _fmt_musigma(mu, sigma, fmt).replace("$\\pm$", "±")
            if key in best and not np.isnan(float(mu)) and \
               ((lower_better and float(mu) <= best[key] + 1e-9) or
                (not lower_better and float(mu) >= best[key] - 1e-9)):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _render_canary_md(canary: dict[str, Any]) -> str:
    """Markdown table: canary feature-similarity per method.

    After unlearning, canary identities' features should no longer encode
    the canary pattern — feature_similarity_to_clean should be low
    (toward 0).  Higher = canary pattern persists (unlearning failed).
    """
    if not canary:
        return "_No canary verification results available._\n"

    lines = [
        "",
        "| Method | Identity | Images | Feat-norm | Sim-to-clean (↓ better) |",
        "|--------|----------|--------|-----------|--------------------------|",
    ]
    for method, per_id in sorted(canary.items()):
        if not isinstance(per_id, dict) or not per_id:
            lines.append(f"| {method} | — | — | — | — |")
            continue
        for cid, m in sorted(per_id.items(), key=lambda kv: int(kv[0])):
            sim = m.get("feature_similarity_to_clean", float("nan"))
            sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
            norm = m.get("mean_feature_norm", "—")
            norm_str = f"{norm:.2f}" if isinstance(norm, (int, float)) else str(norm)
            lines.append(
                f"| {method} | {cid} | {m.get('n_images', '—')} | "
                f"{norm_str} | {sim_str} |"
            )
    lines.append("")
    lines.append("_Lower feature-similarity-to-clean = canary pattern removed. "
                 "Interpretation requires comparison against the pre-unlearning "
                 "model (retain the `original_model_best.pt` from the canary run)._")
    return "\n".join(lines)


# ── CSV export ───────────────────────────────────────────────────────────────


def _export_single_shot_csv(data: dict, out_path: Path) -> None:
    if not data:
        return
    methods = sorted(data.keys())
    # Flatten: method, metric_μ, metric_σ, p_value
    rows = []
    for m in methods:
        row = data[m]
        flat = {"method": row.get("method", m)}
        for key, _l, _lb, _f in METRIC_SPECS:
            if key in row:
                flat[key] = row[key]
                if f"{key}_std" in row:
                    flat[f"{key}_std"] = row[f"{key}_std"]
            p_key = f"p_vs_oracle_{key}"
            if p_key in row:
                flat[p_key] = row[p_key]
        r_acc, mia, t = row.get("retain_id_acc"), row.get("mia_mean_auc"), row.get("total_time_s")
        if all(v is not None for v in (r_acc, mia, t)):
            flat["uf_score"] = round(_uf_score(float(r_acc), float(mia), float(t)), 4)
        rows.append(flat)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"[Report] CSV → {out_path}")


# ── Master ───────────────────────────────────────────────────────────────────


def generate_report(
    results_dir: str = "results",
    out_dir: str = "results/report",
) -> None:
    """Generate the full report: LaTeX tables + markdown + CSV."""
    results_path = Path(results_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    single_shot = _load_single_shot(results_path)
    iterative_df = _load_iterative_aggregated(results_path)
    _stability = _load_stability_summary(results_path)
    canary = _load_canary(results_path)

    single_shot_data: dict[str, Any] = single_shot or {}

    tex_parts = []
    md_parts = []

    # Single-shot
    tex_parts.append(_render_single_shot_latex(single_shot_data))
    md_parts.append("# Single-Shot Results")
    md_parts.append(_render_single_shot_md(single_shot_data))
    _export_single_shot_csv(single_shot_data, out_path / "single_shot_table.csv")

    # Iterative
    tex_parts.append(_render_iterative_latex(iterative_df))
    md_parts.append("\n# Iterative Results (final step)")
    md_parts.append(_render_iterative_md(iterative_df))

    # Canary (markdown only — verification detail, not a headline table)
    md_parts.append("\n# Canary Verification (ground-truth deletion proof)")
    md_parts.append(_render_canary_md(canary or {}))

    # Write LaTeX
    tex = "\n\n".join(part for part in tex_parts if part)
    if tex:
        tex_path = out_path / "report.tex"
        tex_path.write_text(tex)
        logger.info(f"[Report] LaTeX → {tex_path}")

    # Write markdown
    md = "\n".join(md_parts)
    md_path = out_path / "report.md"
    md_path.write_text(md)
    logger.info(f"[Report] Markdown → {md_path}")

    logger.info(f"\n[OK] Report generated in {out_path}/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate dissertation report artefacts")
    parser.add_argument("--results", type=str, default="results",
                        help="Path to results directory (default: results)")
    parser.add_argument("--out", type=str, default="results/report",
                        help="Output directory (default: results/report)")
    args = parser.parse_args()
    generate_report(results_dir=args.results, out_dir=args.out)


if __name__ == "__main__":
    main()
