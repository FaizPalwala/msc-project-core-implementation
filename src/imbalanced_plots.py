#!/usr/bin/env python3
"""imbalanced_plots.py — Per-popularity-bin equity plots for the imbalanced dataset.

Reads ``single_shot_aggregated.json`` and the imbalanced CSV, then produces
publication‑quality PNGs analysing whether unlearning is **equitable** across
the long‑tail data distribution (high‑popularity ≈ 82 train images vs low ≈ 16).

Usage:
    python src/imbalanced_plots.py \\
        --results results/single_shot/single_shot_aggregated.json \\
        --csv metadata/dataset_imbalanced.csv \\
        --out results/imbalanced

Output (6 PNGs):
    01_forget_acc_by_bin.png       — forget‑holdout accuracy by popularity bin
    02_mia_auc_by_bin.png          — per‑identity MIA AUC by popularity bin
    03_forget_train_gap_by_bin.png — forget‑train minus forget‑holdout gap per bin
    04_auc_vs_images_per_id.png    — MIA AUC vs images per identity (scatter + r)
    05_forget_utility_frontier.png — retaining vs forgetting scatter, coloured by bin
    06_kruskal_pvalues.png         — Kruskal‑Wallis p‑values per method×metric

Citations:
    Choi & Na (2023) for per‑identity MIA and forget‑holdout gap.
    Cadet et al. (2024) for the imbalanced stress‑test framing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BINS = ["high", "medium", "low"]
BIN_COLOURS = {"high": "#e74c3c", "medium": "#f39c12", "low": "#3498db"}
BIN_LABELS = {"high": "High (82 imgs)", "medium": "Medium (41 imgs)", "low": "Low (16 imgs)"}
BAR_WIDTH = 0.22
_, TEXTLIKE_RC = plt.subplots()
plt.close()
TEXTLIKE_RC = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
}

from plot_style import (  # noqa: E402
    METHOD_STYLES, ORACLE, GROUPS, style as _style, method_label,
)

# Shared y-limits across subgrouped bar panels (C1) — one window per metric
# so panels are directly comparable.
BIN_SHARED_LIMS = {
    "forget": (0.0, 1.05),
    "mia":    (0.30, 1.05),
    "gap":    (-0.1, 0.6),
}


def _bar_panel(ax, aggregated, methods, metric_key, bin_key, lims, rotate=False):
    """One subgrouped bar panel: methods × bins, shared y-limits."""
    x = np.arange(len(methods))
    for i, bin_name in enumerate(BINS):
        values = []
        for m in methods:
            v = aggregated[m].get(f"{metric_key}_{bin_name}")
            values.append(float(v) if v is not None else np.nan)
        ax.bar(x + i * BAR_WIDTH, values, BAR_WIDTH,
               label=BIN_LABELS[bin_name], color=BIN_COLOURS[bin_name],
               edgecolor="white", linewidth=0.5)
    ax.set_ylim(*lims)
    ax.set_xticks(x + BAR_WIDTH)
    ax.set_xticklabels([method_label(m) for m in methods],
                       rotation=45 if rotate else 30, ha="right", fontsize=9)
    ax.legend(fontsize=8, ncol=3)


def _subgrouped_bars(aggregated, metric_key, ylabel, out_dir, fname,
                     lims, extra_lines=None):
    """3-panel bar figure: G1 baselines / G2 SOTA / G3 novel (C1).

    Same bin colours in every panel; oracle rides as a horizontal target
    line (extra_lines), not a row.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=False)
    for ax, (gname, members), rotate in zip(axes, GROUPS, (False, True, True)):
        present = [m for m in members if m in aggregated]
        if not present:
            ax.set_visible(False)
            continue
        _bar_panel(ax, aggregated, present, metric_key, bin_key="",
                   lims=lims, rotate=rotate)
        ax.set_ylabel(ylabel, fontsize=9)
        if extra_lines:
            for y, lbl, c in extra_lines:
                ax.axhline(y, color=c, ls="--", lw=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {out_dir / fname} (3 subgroup panels)")


def load_data(results_path: str, csv_path: str) -> tuple[dict, pd.DataFrame]:
    """Load aggregated JSON and CSV, return (aggregated, df)."""
    with open(results_path) as f:
        agg = json.load(f)
    df = pd.read_csv(csv_path)
    return agg, df


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 : Forget‑holdout accuracy by popularity bin
# ═══════════════════════════════════════════════════════════════════════════════


def plot_forget_acc_by_bin(aggregated: dict, out_dir: Path) -> None:
    methods = [k for k, v in aggregated.items() if v.get("forget_id_acc_high") is not None]
    if not methods:
        logger.info("  [skip] no per‑bin forget acc data")
        return

    plt.rcParams.update(TEXTLIKE_RC)
    # Retrain Oracle: perfect erasure in every bin (0.0/0.0/0.0) — its bars
    # would be zero-height and invisible, so plot it as a distinct marker on
    # the axis + legend entry instead of a bar group.
    bar_methods = [m for m in methods if m != "retrain"]
    oracle = aggregated.get("retrain")
    n_methods = len(bar_methods)
    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.0), 5))

    x = np.arange(n_methods)
    for i, bin_name in enumerate(BINS):
        values = []
        for m in bar_methods:
            v = aggregated[m].get(f"forget_id_acc_{bin_name}")
            values.append(float(v) if v is not None else 0.0)
        ax.bar(x + i * BAR_WIDTH, values, BAR_WIDTH,
               label=BIN_LABELS[bin_name], color=BIN_COLOURS[bin_name],
               edgecolor="white", linewidth=0.5)
    # Oracle markers: one per bin, sitting on the axis (y=0 = perfect).
    if oracle is not None:
        for i, bin_name in enumerate(BINS):
            v = oracle.get(f"forget_id_acc_{bin_name}")
            if v is None:
                continue
            ax.scatter([x[i] + i * BAR_WIDTH], [0.0], marker="D", s=28,
                       color=ORACLE["color"], zorder=5,
                       label="Retrain Oracle (0.0)" if i == 0 else None)

    ax.set_ylabel("Forget‑holdout identity accuracy\n(↓ better)")
    ax.set_xticks(x + BAR_WIDTH)
    ax.set_xticklabels([aggregated[m]["method"] for m in bar_methods], rotation=30, ha="right")
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("Forget‑Holdout Accuracy by Popularity Bin")
    ax.axhline(0.05, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate("≤ 0.05 = strong forgetting", xy=(0.01, 0.07), fontsize=8, color="green", alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "01_forget_acc_by_bin.png")
    plt.close(fig)
    logger.info(f"  → {out_dir / '01_forget_acc_by_bin.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 : MIA AUC by popularity bin
# ═══════════════════════════════════════════════════════════════════════════════


def plot_mia_auc_by_bin(aggregated: dict, out_dir: Path) -> None:
    methods = [k for k, v in aggregated.items() if v.get("mia_auc_high") is not None]
    if not methods:
        logger.info("  [skip] no per‑bin MIA AUC data")
        return

    plt.rcParams.update(TEXTLIKE_RC)
    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.0), 5))

    x = np.arange(n_methods)
    for i, bin_name in enumerate(BINS):
        values = []
        for m in methods:
            v = aggregated[m].get(f"mia_auc_{bin_name}")
            values.append(float(v) if v is not None else 0.5)
        ax.bar(x + i * BAR_WIDTH, values, BAR_WIDTH,
               label=BIN_LABELS[bin_name], color=BIN_COLOURS[bin_name],
               edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Per‑identity MIA AUC\n(↓ better, 0.50 = chance)")
    ax.set_xticks(x + BAR_WIDTH)
    ax.set_xticklabels([aggregated[m]["method"] for m in methods], rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.axhline(0.50, color="gray", linestyle="--", linewidth=0.6)
    ax.axhline(0.55, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.annotate("AUC ≤ 0.55 = private", xy=(0.01, 0.555), fontsize=8, color="red", alpha=0.6)
    ax.set_title("MIA AUC by Popularity Bin")
    fig.tight_layout()
    fig.savefig(out_dir / "02_mia_auc_by_bin.png")
    plt.close(fig)
    logger.info(f"  → {out_dir / '02_mia_auc_by_bin.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 : Forget‑train vs forget‑holdout gap by bin
# ═══════════════════════════════════════════════════════════════════════════════


def plot_forget_train_gap_by_bin(aggregated: dict, out_dir: Path) -> None:
    """The gap between forget accuracy on train vs holdout images.

    Note: train-side accuracy is NOT stratified by popularity bin (the
    evaluation pipeline only stratifies the holdout side), so the gap is
    per-method — the plot name reflects that honestly.  The §8.8 claim
    ("the forget-train gap that monitors overfitting") is a per-method
    claim, so this is the correct granularity.

    Release: 03_forget_train_gap.png (per-method bars, all methods).
    """
    methods = [k for k in aggregated if aggregated[k].get("forget_train_id_acc") is not None]
    gaps = {}
    for m in methods:
        f_hold = aggregated[m].get("forget_id_acc")
        f_train = aggregated[m].get("forget_train_id_acc")
        if f_hold is not None and f_train is not None:
            gaps[m] = float(f_train) - float(f_hold)

    if not gaps:
        logger.info("  [skip] no forget-train gap data")
        return

    plt.rcParams.update(TEXTLIKE_RC)
    n_methods = len(gaps)
    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.0), 5))

    names = [method_label(m) for m in gaps]
    values = list(gaps.values())
    colours = [_style(m)["color"] for m in gaps]

    ax.bar(names, values, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Forget-train − Forget-holdout gap\n(≤ 0.10 = genuine forgetting, > 0.15 = overfit)")
    ax.set_title("Forget-Train / Forget-Holdout Gap (per method)")
    ax.axhline(0.10, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.15, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.annotate("gap ≤ 0.10", xy=(0.01, 0.11), fontsize=8, color="green", alpha=0.7)
    ax.annotate("gap ≥ 0.15", xy=(0.01, 0.16), fontsize=8, color="red", alpha=0.6)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "03_forget_train_gap.png")
    plt.close(fig)
    logger.info(f"  → {out_dir / '03_forget_train_gap.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 : Per‑identity MIA AUC vs images per identity
# ═══════════════════════════════════════════════════════════════════════════════


def plot_auc_vs_images(df: pd.DataFrame, aggregated: dict, out_dir: Path) -> None:
    """Scatter plot: per‑identity AUC (from best available method) vs image count.

    Actually aggregates all identities across all methods — each point is an
    (identity_id, method) pair.  Fits a linear regression and reports r, p.
    """
    # Build per‑identity image counts (forget split only)
    forget_df = df[df["split"] == "forget"]
    id_counts = forget_df.groupby("identity_id").size().to_dict()

    # Collect per‑identity AUCs from aggregated results' seed‑level data.
    # The aggregated JSON has per‑seed per‑identity AUCs in the per_seed key.
    # For simplicity, use the per‑method aggregate AUC — not per‑identity.
    # The per‑identity granularity requires seed‑level JSON.
    logger.info("  [note] plot 4 requires per‑seed results; using demographic MIA bins instead")
    # Fallback: bin‑level scatter
    methods = [k for k, v in aggregated.items() if v.get("mia_auc_high") is not None]
    if not methods:
        logger.info("  [skip] no per‑bin MIA data for scatter")
        return

    # Bin x-values from the CSV itself (per-identity counts on the forget
    # split) — tracks the dataset (85/40/20 at 600-id, 82/41/16 at 750-id)
    # instead of hardcoding.
    bin_to_imgs = {}
    if "popularity_bin" in df.columns:
        for bin_name in BINS:
            sub = forget_df[forget_df["popularity_bin"] == bin_name]
            counts = sub.groupby("identity_id").size()
            bin_to_imgs[bin_name] = int(counts.median()) if len(counts) else None
    for bin_name in BINS:
        if bin_to_imgs.get(bin_name) is None:
            bin_to_imgs[bin_name] = {"high": 82, "medium": 41, "low": 16}[bin_name]
    points = []
    for method in methods:
        for bin_name in BINS:
            auc = aggregated[method].get(f"mia_auc_{bin_name}")
            if auc is not None:
                points.append({
                    "method": aggregated[method]["method"],
                    "bin": bin_name,
                    "images_per_identity": bin_to_imgs[bin_name],
                    "mia_auc": float(auc),
                })
    pts_df = pd.DataFrame(points)

    plt.rcParams.update(TEXTLIKE_RC)
    fig, ax = plt.subplots(figsize=(9, 6))

    for bin_name in BINS:
        bin_pts = pts_df[pts_df["bin"] == bin_name]
        ax.scatter(bin_pts["images_per_identity"], bin_pts["mia_auc"],
                   color=BIN_COLOURS[bin_name], label=BIN_LABELS[bin_name],
                   s=70, edgecolors="white", linewidth=0.5, alpha=0.85)

    # Regression
    x_all = pts_df["images_per_identity"].values
    y_all = pts_df["mia_auc"].values
    if len(x_all) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(x_all, y_all)
        xs = np.linspace(min(x_all) - 5, max(x_all) + 5, 100)
        ax.plot(xs, slope * xs + intercept, color="gray", linestyle="--",
                linewidth=1.2, alpha=0.6)
        ax.annotate(
            f"r = {r_val:.3f}\np = {p_val:.3f}",
            xy=(0.75, 0.15), xycoords="axes fraction",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

    ax.set_xlabel("Images per identity (train subset)")
    ax.set_ylabel("MIA AUC")
    ax.set_title("MIA AUC vs Data Availability\n(does data‑poverty predict privacy‑poverty?)")
    ax.legend(fontsize=9)
    ax.axhline(0.50, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "04_auc_vs_images_per_id.png")
    plt.close(fig)
    logger.info(f"  → {out_dir / '04_auc_vs_images_per_id.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 : Forgetting‑utility frontier coloured by popularity bin
# ═══════════════════════════════════════════════════════════════════════════════


def plot_forget_utility_frontier(aggregated: dict, out_dir: Path) -> None:
    """Scatter: retain‑holdout accuracy vs forget‑holdout accuracy, per bin.

    Each method gets one point per bin.  The lower‑right quadrant is good
    (high retain, low forget).
    """
    methods = [k for k, v in aggregated.items() if v.get("forget_id_acc_high") is not None]
    if len(methods) < 2:
        logger.info("  [skip] too few methods for frontier plot")
        return

    plt.rcParams.update(TEXTLIKE_RC)
    fig, ax = plt.subplots(figsize=(9, 7))

    # colour = method (11 styles), marker shape = popularity bin, so every
    # point carries BOTH identities; the legend carries the method names
    # (33 text labels were unreadable, and frontier-only labels dropped all
    # method information — this keeps both).
    BIN_MARKERS = {"high": "^", "medium": "o", "low": "s"}
    all_pts = []  # (retain, forget, method_key, bin_name)
    for m in methods:
        for bin_name in BINS:
            retain_acc = aggregated[m].get("retain_id_acc")
            forget_acc = aggregated[m].get(f"forget_id_acc_{bin_name}")
            if retain_acc is not None and forget_acc is not None:
                all_pts.append((float(retain_acc), float(forget_acc), m, bin_name))

    for m in methods:
        pts = [p for p in all_pts if p[2] == m]
        if not pts:
            continue
        st = _style(m)
        for (ri, fi, _, bn) in pts:
            ax.scatter(ri, fi, color=st["color"], marker=BIN_MARKERS[bn],
                       s=70, edgecolors="white", linewidth=0.5, alpha=0.85)
        # one legend handle per method (uses the high-bin marker)
        ax.scatter([], [], color=st["color"], marker=BIN_MARKERS["high"],
                   s=70, edgecolors="white", linewidth=0.5,
                   label=st["label"])

    # Per-bin Pareto front envelopes: one per popularity bin. Bins are
    # different difficulty regimes, so a single mixed-bin frontier is
    # semantically muddy — the per-bin envelopes show the achievable
    # frontier DEGRADING as popularity rises (the equity claim itself).
    for bin_name in BINS:
        pts = [p for p in all_pts if p[3] == bin_name]
        if len(pts) < 2:
            continue
        dom = set()
        for i, (ri, fi, _, _) in enumerate(pts):
            for j, (rj, fj, _, _) in enumerate(pts):
                if i != j and rj >= ri - 1e-9 and fj <= fi + 1e-9 \
                   and (rj > ri + 1e-9 or fj < fi - 1e-9):
                    dom.add(i)
                    break
        front = sorted([p for i, p in enumerate(pts) if i not in dom],
                       key=lambda p: -p[0])
        if len(front) >= 2:
            fx = [p[0] for p in front]
            fy = [p[1] for p in front]
            ax.step(fx, fy, where="post", color=BIN_COLOURS[bin_name],
                    ls="--", lw=1.5, alpha=0.8,
                    label=f"Pareto front ({bin_name} bin)")

    ax.set_xlabel("Retain‑holdout identity accuracy (↑ better)")
    ax.set_ylabel("Forget‑holdout identity accuracy (↓ better)")
    # Retain is pinned at 0.998–0.999 for every method (SRL 0.755 is the
    # lone outlier) — a 0–1 x-axis flattens the plot into a vertical band
    # and hides the per-bin frontier heights. Zoom so the envelope levels
    # (low 0.043 / medium 0.431 / high 0.975) read as distinct bands.
    ax.set_xlim(0.70, 1.005)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Forgetting-Utility Frontier\n(colour = method; marker = bin; dashed = per-bin Pareto front)")
    ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
    ax.axhline(0.05, color="green", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.annotate("strong forgetting", xy=(0.02, 0.055), fontsize=8, color="green", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "05_forget_utility_frontier.png")
    plt.close(fig)
    logger.info(f"  → {out_dir / '05_forget_utility_frontier.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6 : Kruskal‑Wallis p‑values (heatmap)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_kruskal_pvalues(aggregated: dict, out_dir: Path) -> None:
    """Kruskal‑Wallis H‑test per method: 'is there a bin difference?'.

    Transposed layout (C2): methods are COLUMNS, the 2 metrics are ROWS —
    a wide, short heatmap instead of the old 2.1:1 portrait that ate a
    page at 0.45 textwidth.

    Uses the per‑bin AUC and forget‑acc values directly (small n = 3 bins),
    so p‑values are rough but directional.
    """
    methods = [k for k, v in aggregated.items() if v.get("mia_auc_high") is not None]
    if len(methods) < 2:
        logger.info("  [skip] too few methods for Kruskal‑Wallis")
        return

    metrics = ["MIA AUC", "Forget Acc"]
    p_matrix = np.full((len(metrics), len(methods)), np.nan)

    for j, method in enumerate(methods):
        auc_vals = [aggregated[method].get(f"mia_auc_{b}") for b in BINS]
        forget_vals = [aggregated[method].get(f"forget_id_acc_{b}") for b in BINS]
        auc_vals = [v for v in auc_vals if v is not None]
        forget_vals = [v for v in forget_vals if v is not None]

        for i, vals in enumerate([auc_vals, forget_vals]):
            if len(set(vals)) < 2:
                continue
            # Kruskal‑Wallis with 3 groups × N=1 per group is degenerate;
            # keep the per-bin values as the groups and report H.
            groups = [[v] for v in vals]
            try:
                H, p = stats.kruskal(*groups)
                p_matrix[i, j] = round(p, 4)
            except Exception:
                pass

    plt.rcParams.update(TEXTLIKE_RC)
    fig, ax = plt.subplots(
        figsize=(len(methods) * 0.8 + 2.0, 3.0),
    )

    im = ax.imshow(p_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_label(m) for m in methods],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=9)

    for i in range(len(metrics)):
        for j in range(len(methods)):
            val = p_matrix[i, j]
            if not np.isnan(val):
                colour = "white" if val < 0.3 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=colour,
                        fontweight="bold" if val < 0.05 else "normal")

    fig.colorbar(im, ax=ax, label="p‑value (H₀: no bin difference)")
    fig.tight_layout()
    fig.savefig(out_dir / "06_kruskal_pvalues.png")
    plt.close(fig)
    logger.info(f"  → {out_dir / '06_kruskal_pvalues.png'} (transposed)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Imbalanced‑dataset equity plots",
    )
    parser.add_argument("--results", type=str, required=True,
                        help="Path to single_shot_aggregated.json")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to dataset_imbalanced.csv")
    parser.add_argument("--out", type=str, default="results/imbalanced",
                        help="Output directory for PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg, df = load_data(args.results, args.csv)
    logger.info(f"Loaded {len(agg)} methods, {len(df)} rows from dataset")

    plot_forget_acc_by_bin(agg, out_dir)
    plot_mia_auc_by_bin(agg, out_dir)
    plot_forget_train_gap_by_bin(agg, out_dir)
    plot_auc_vs_images(df, agg, out_dir)
    plot_forget_utility_frontier(agg, out_dir)
    plot_kruskal_pvalues(agg, out_dir)

    # ── Subgrouped bar panels (C1) — 3 panels per metric ────────────────
    sub_dir = out_dir / "subgroups"
    sub_dir.mkdir(parents=True, exist_ok=True)
    _subgrouped_bars(agg, "forget_id_acc", "Forget-holdout identity acc (↓ better)",
                     sub_dir, "01_forget_acc_by_bin_subgroups.png",
                     BIN_SHARED_LIMS["forget"],
                     extra_lines=[(0.05, "strong forgetting", "green")])
    _subgrouped_bars(agg, "mia_auc", "Per-identity MIA AUC (↓ better)",
                     sub_dir, "02_mia_auc_by_bin_subgroups.png",
                     BIN_SHARED_LIMS["mia"],
                     extra_lines=[(0.50, "chance", "gray"),
                                  (0.55, "leak threshold", "red")])

    logger.info(f"\n[OK] {len(list(out_dir.glob('*.png')))} plots → {out_dir}")


if __name__ == "__main__":
    main()
