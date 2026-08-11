"""
stability.py — Stability Analysis & Visualisation (extended suite).

Generates publication-quality plots from iterative unlearning results:

Core (from original suite):
  01. Retain Accuracy vs. Iteration
  02. MIA AUC vs. Iteration
  03. Forget Advantage vs. Iteration
  04. Model Drift vs. Iteration
  05. Step Time vs. Iteration
  06. Pareto scatter: Retain Acc vs. MIA AUC
  07. Heatmap: Forget Advantage across methods × steps
  08. Radar chart: final-step multi-metric comparison
  09. Cumulative Time vs. Iteration

New:
  10. Per-identity forgetting signatures (sorted bar chart of per-ID MIA AUC)
  11. Demographic breakdown heatmap (rows=age groups/gender/popularity)
  12. Within-step boxplots (4 identities per step confidence distribution)
  13. Phase-space trajectory (forget loss vs retain loss, connected by steps)
  14. Total wall-clock time bar chart (sorted, cost-emphasis)

Ad-hoc subsetting: --methods and --steps CLI flags filter the data
before plotting, for quick focused comparisons of busy graphs.

All plots at 300 DPI, publication-ready.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "lines.linewidth": 2.0, "lines.markersize": 6,
})

METHOD_STYLES = {
    "no_unlearning": {"color": "#9e9e9e", "ls": ":",  "marker": "x", "label": "No-Op"},
    "ga":            {"color": "#e57373", "ls": "--", "marker": "s", "label": "GA"},
    "srl":           {"color": "#ffb74d", "ls": "--", "marker": "^", "label": "SRL"},
    "ft":            {"color": "#fff176", "ls": "--", "marker": "D", "label": "FT"},
    "ng_plus":       {"color": "#64b5f6", "ls": "-",  "marker": "o", "label": "NG+"},
    "msg":           {"color": "#4db6ac", "ls": "-",  "marker": "v", "label": "MSG"},
    "msg_kd":        {"color": "#81c784", "ls": "-",  "marker": "P", "label": "MSG-KD"},
    "ct":            {"color": "#ba68c8", "ls": "-",  "marker": "h", "label": "CT"},
    "adaptiforget":  {"color": "#ff7043", "ls": "-",  "marker": "*", "label": "AdaptiForget"},
    "budget_scaled": {"color": "#8d6e63", "ls": "-.", "marker": "X", "label": "Budget-Scaled"},
}
ORACLE_COLOR = "#1565c0"


def _style(method: str) -> dict:
    return METHOD_STYLES.get(method, {"color": "#555", "ls": "-",
                                       "marker": "o", "label": method})


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(combined_csv: str) -> pd.DataFrame:
    df = pd.read_csv(combined_csv)

    # ── Aggregated format (multi-seed μ ± σ) normalisation ──────────────
    # If columns look like "mia_mean_auc_mean"/"mia_mean_auc_std", rename
    # the _mean columns to their base metric name and keep _std alongside.
    # This lets all plotting functions work unchanged on aggregated data.
    mean_cols = [c for c in df.columns if c.endswith("_mean")
                 and not c.endswith("_std")]
    if mean_cols:
        rename = {}
        for c in mean_cols:
            base = c[:-len("_mean")]
            std_col = f"{base}_std"
            if std_col in df.columns:
                rename[c] = base  # keep base + base_std pairs
        if rename:
            df = df.rename(columns=rename)
            logger.info(f"  [Load] Aggregated format detected — normalised "
                        f"{len(rename)} mean/std column pairs")

    if "is_baseline" in df.columns:
        df = df[df["is_baseline"] != True]
    if "step" in df.columns:
        df["step"] = df["step"].astype(int)
    for col in ["retain_acc", "test_acc", "forget_acc",
                "retain_age_acc", "mia_mean_auc", "mia_max_auc",
                "forget_advantage", "model_drift",
                "step_time_s", "cumulative_time_s", "fraction_leaked"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _plot_vs_step(df, metric, ylabel, title, out_file,
                  target_line=None, target_label=None, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = sorted(df["method"].unique())
    std_col = f"{metric}_std"
    has_std = std_col in df.columns

    for method in methods:
        sub = df[df["method"] == method].sort_values("step")
        s = _style(method)
        ax.plot(sub["step"], sub[metric],
                color=s["color"], ls=s["ls"], marker=s["marker"],
                label=s["label"], alpha=0.9)
        # Error band from multi-seed std (μ ± σ)
        if has_std:
            lo = sub[metric] - sub[std_col]
            hi = sub[metric] + sub[std_col]
            ax.fill_between(sub["step"], lo, hi,
                            color=s["color"], alpha=0.12, lw=0)

    if has_std:
        ax.set_title(f"{title}\n(shaded = μ ± σ across seeds)",
                     fontweight="bold", pad=10)
    else:
        ax.set_title(title, fontweight="bold", pad=10)

    if target_line is not None:
        ax.axhline(target_line, color=ORACLE_COLOR, ls=":", lw=1.5,
                   label=target_label or f"Target ({target_line})", alpha=0.7)
    ax.set_xlabel("Forget step")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {Path(out_file).name}")


# ── Core plots (01–09) ────────────────────────────────────────────────────────


def plot_retain_acc(df, out_dir: Path):
    _plot_vs_step(df, "retain_acc", "Retain Identity Accuracy",
                  "Retain Accuracy vs. Iteration", out_dir / "01_retain_acc.png",
                  ylim=(0.0, 1.05))


def plot_mia_auc(df, out_dir: Path):
    _plot_vs_step(df, "mia_mean_auc", "MIA AUC (identity head)",
                  "MIA AUC vs. Iteration", out_dir / "02_mia_auc.png",
                  target_line=0.50, target_label="Perfect forgetting (0.50)",
                  ylim=(0.40, 1.05))


def plot_forget_advantage(df, out_dir: Path):
    _plot_vs_step(df, "forget_advantage", "Forget Advantage |AUC − 0.5|",
                  "Forget Advantage vs. Iteration", out_dir / "03_forget_adv.png",
                  target_line=0.0, target_label="Perfect forgetting",
                  ylim=(-0.02, 0.55))


def plot_model_drift(df, out_dir: Path):
    _plot_vs_step(df, "model_drift", "Weight L₂ Distance from Original",
                  "Model Drift vs. Iteration", out_dir / "04_model_drift.png")


def plot_step_time(df, out_dir: Path):
    _plot_vs_step(df, "step_time_s", "Step Time (s)",
                  "Unlearning Step Time vs. Iteration", out_dir / "05_step_time.png")


def pareto_frontier(
    x: np.ndarray,   # x = MIA AUC (lower is better)
    y: np.ndarray,   # y = retain accuracy (higher is better)
) -> np.ndarray:
    """Return indices of non-dominated points (Pareto-optimal set).

    A point dominates another if it is better (or equal) on both axes
    and strictly better on at least one.  Here: lower x is better,
    higher y is better.
    """
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # j dominates i if x_j <= x_i and y_j >= y_i (strict on one)
            if (x[j] <= x[i] and y[j] >= y[i]) and (x[j] < x[i] or y[j] > y[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def hypervolume(
    x: np.ndarray,   # lower is better
    y: np.ndarray,   # higher is better
    ref_x: float,
    ref_y: float,
) -> float:
    """Hypervolume of the Pareto-optimal set relative to a reference point.

    Reference point (ref_x, ref_y) is the WORST point (e.g. max MIA AUC,
    min retain acc).  Larger hypervolume = better trade-off frontier.
    """
    # Filter to points strictly better than the reference
    mask = (x < ref_x) & (y > ref_y)
    if mask.sum() == 0:
        return 0.0
    xs = x[mask]
    ys = y[mask]
    # Sort by x descending (process from worst-x to best-x)
    order = np.argsort(-xs)
    xs = xs[order]
    ys = ys[order]
    vol = 0.0
    prev_y = ref_y
    for xi, yi in zip(xs, ys):
        if yi > prev_y:
            vol += (ref_x - xi) * (yi - prev_y)
            prev_y = yi
    return float(vol)


def plot_pareto(df, out_dir: Path):
    """Pareto scatter + computed frontier line + hypervolume annotation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = sorted(df["method"].unique())

    frontier_x, frontier_y = [], []
    all_x, all_y = [], []

    for method in methods:
        sub = df[df["method"] == method].dropna(subset=["retain_acc", "mia_mean_auc"])
        if sub.empty:
            continue
        s = _style(method)
        x = sub["mia_mean_auc"].values
        y = sub["retain_acc"].values
        all_x.extend(x)
        all_y.extend(y)
        ax.scatter(x, y, c=sub["step"], cmap="Blues", vmin=1,
                   vmax=df["step"].max(), alpha=0.75,
                   edgecolors=s["color"], linewidths=1.5,
                   marker=s["marker"], s=60, label=s["label"])

        # Frontier for this method (its own steps)
        idx = pareto_frontier(x, y)
        frontier_x.extend(x[idx])
        frontier_y.extend(y[idx])

    # Global frontier across all points
    if all_x:
        gx = np.array(all_x)
        gy = np.array(all_y)
        g_idx = pareto_frontier(gx, gy)
        fx = np.sort(gx[g_idx])
        fy = gy[g_idx][np.argsort(gx[g_idx])]
        ax.plot(fx, fy, color="#222", lw=2.5, ls="-", alpha=0.8,
                label="Pareto frontier")

        # Hypervolume
        hv = hypervolume(gx, gy, ref_x=gx.max(), ref_y=gy.min())
        ax.text(0.02, 0.02, f"Hypervolume: {hv:.4f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.axvline(0.50, color=ORACLE_COLOR, ls=":", lw=1.5, alpha=0.7,
               label="Perfect MIA (0.50)")
    ax.set_xlabel("MIA AUC (← better forgetting)")
    ax.set_ylabel("Retain Accuracy (↑ better)")
    ax.set_title("Pareto Frontier: Utility vs. Forgetting (with hypervolume)",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    if ax.collections:
        plt.colorbar(ax.collections[0], ax=ax, label="Step")
    fig.tight_layout()
    fig.savefig(out_dir / "06_pareto.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 06_pareto.png (with frontier + hypervolume)")


def plot_heatmap(df, out_dir: Path):
    methods = sorted(df["method"].unique())
    steps = sorted(df["step"].unique())
    matrix = np.full((len(methods), len(steps)), np.nan)
    for i, m in enumerate(methods):
        for j, s in enumerate(steps):
            sub = df[(df["method"] == m) & (df["step"] == s)]
            if "forget_advantage" in sub.columns and len(sub):
                matrix[i, j] = sub["forget_advantage"].mean()
    fig, ax = plt.subplots(figsize=(max(10, len(steps)*0.6),
                                    max(4, len(methods)*0.7)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=0.5,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_style(m)["label"] for m in methods], fontsize=9)
    ax.set_xlabel("Forget step")
    ax.set_title("Forget Advantage Heatmap (↓ = better)", fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, label="|AUC − 0.5|")
    fig.tight_layout()
    fig.savefig(out_dir / "07_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 07_heatmap.png")


def plot_radar(df, out_dir: Path):
    metrics = ["retain_acc", "step_forget_quality", "forget_quality", "mia_quality", "speed"]
    max_step = df["step"].max()
    final = df[df["step"] == max_step].copy()
    # forget_quality: 1.0 when forget acc is low (good) — normalised on the
    # holdout forget accuracy (0 = still recognised, 1 = fully forgotten).
    if "forget_acc" in final.columns:
        final["forget_quality"] = (1.0 - final["forget_acc"].clip(0, 1)).fillna(0)
    # step_forget_quality: step-local version — the current step's identities.
    if "step_forget_acc" in final.columns:
        final["step_forget_quality"] = (1.0 - final["step_forget_acc"].clip(0, 1)).fillna(0)
    else:
        final["step_forget_quality"] = final.get("forget_quality", 0)
    if "mia_mean_auc" in final.columns:
        final["mia_quality"] = 1.0 - (final["mia_mean_auc"] - 0.5).abs().clip(0, 0.5) / 0.5
    t_max = final["step_time_s"].max() if "step_time_s" in final.columns else 1
    final["speed"] = 1.0 - (final["step_time_s"] / max(t_max, 1)).clip(0, 1) if "step_time_s" in final.columns else 0

    labels = ["Retain Acc", "Step-Forget", "Forget\nQuality", "MIA\nQuality", "Speed"]
    N = len(labels); angles = [n / N * 2 * math.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for method in [m for m in METHOD_STYLES if m in final["method"].values]:
        sub = final[final["method"] == method]
        if sub.empty: continue
        vals = [max(0, min(1, float(sub[m].mean()))) for m in metrics[:N]]
        vals += [vals[0]]
        s = _style(method)
        ax.plot(angles, vals, color=s["color"], lw=2, label=s["label"])
        ax.fill(angles, vals, color=s["color"], alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Final-Step Multi-Metric (outer=better)", fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "08_radar.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 08_radar.png")


def plot_cumulative_time(df, out_dir: Path):
    _plot_vs_step(df, "cumulative_time_s", "Cumulative Time (s)",
                  "Cumulative Time vs. Iteration", out_dir / "09_cumulative_time.png")


def plot_total_time_bar(df, out_dir: Path):
    """Total wall-clock per method as a sorted horizontal bar chart.

    Emphasises cost differences between methods — a key deployment
    feasibility metric.  Uses final cumulative_time_s per method.
    """
    methods = sorted(df["method"].unique())
    totals = []
    for method in methods:
        sub = df[df["method"] == method]
        if "cumulative_time_s" in sub.columns and len(sub):
            totals.append(sub["cumulative_time_s"].max())
        else:
            totals.append(0.0)

    order = np.argsort(totals)  # ascending
    sorted_methods = [METHOD_STYLES.get(methods[i], {"label": methods[i]})["label"]
                      for i in order]
    sorted_totals = [totals[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(methods) + 2)))
    colors = [_style(methods[i])["color"] for i in order]
    bars = ax.barh(sorted_methods, sorted_totals, color=colors, alpha=0.85)
    # Annotate values
    for bar, val in zip(bars, sorted_totals):
        ax.text(bar.get_width() + max(sorted_totals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val/60:.1f} min", va="center", fontsize=9)

    ax.set_xlabel("Total cumulative time (s)")
    ax.set_title("Total Unlearning Time by Method\n(lower = cheaper to deploy)",
                 fontweight="bold", pad=10)
    ax.set_xlim(0, max(sorted_totals) * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "14_total_time_bar.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 14_total_time_bar.png")


# ── New plots (10–13) ─────────────────────────────────────────────────────────


def plot_per_identity_signatures(
    per_id_csv: str | None,
    out_dir: Path,
) -> None:
    """Per-identity MIA AUC bar chart (sorted, one panel per method).

    Requires per-identity MIA output from single-shot evaluation.
    Falls back gracefully if file not found.
    """
    if per_id_csv is None or not Path(per_id_csv).exists():
        logger.info("  [SKIP] 10_identity_signatures — no per-identity data")
        return

    df = pd.read_csv(per_id_csv)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    for ax, (method, group) in zip(axes, df.groupby("method")):
        aucs = group.sort_values("mia_auc", ascending=False)
        colors = ["#e57373" if a > 0.55 else "#81c784" for a in aucs["mia_auc"]]
        ax.bar(range(len(aucs)), aucs["mia_auc"], color=colors, width=0.8)
        ax.axhline(0.50, color="grey", ls="--", lw=1, alpha=0.5)
        ax.set_title(_style(method)["label"], fontsize=10)
        ax.set_ylabel("MIA AUC")
        ax.set_ylim(0.40, 1.0)

    for ax in axes[len(df["method"].unique()):]:
        ax.set_visible(False)

    fig.suptitle("Per-Identity Forgetting Signatures\n(red = leaked > 0.55, green = forgotten)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "10_identity_signatures.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 10_identity_signatures.png")


def plot_demographic_heatmap(
    demog_csv: str | None,
    out_dir: Path,
) -> None:
    """Demographic-stratified MIA AUC heatmap.

    Requires demographic MIA output from single-shot evaluation.
    """
    if demog_csv is None or not Path(demog_csv).exists():
        logger.info("  [SKIP] 11_demographic_heatmap — no demographic data")
        return

    df = pd.read_csv(demog_csv)
    pivoted = df.pivot_table(
        index="demographic_group", columns="method",
        values="mia_auc", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pivoted.columns)*1.2),
                                    max(4, len(pivoted.index)*0.6)))
    im = ax.imshow(pivoted.values, cmap="RdYlGn_r", vmin=0.45, vmax=0.65,
                   aspect="auto")
    ax.set_xticks(range(len(pivoted.columns)))
    ax.set_xticklabels(pivoted.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivoted.index)))
    ax.set_yticklabels(pivoted.index, fontsize=9)
    ax.set_title("MIA AUC by Demographic Group\n(↓ = better forgetting)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="MIA AUC")
    fig.tight_layout()
    fig.savefig(out_dir / "11_demographic_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 11_demographic_heatmap.png")


def plot_phase_space(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Forget loss trajectory in phase space (if loss columns present).

    Plots forget loss progression across steps connected by arrows.
    """
    if "mia_mean_auc" not in df.columns or "retain_acc" not in df.columns:
        logger.info("  [SKIP] 12_phase_space — missing required columns")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    methods = sorted(df["method"].unique())

    for method in methods:
        sub = df[df["method"] == method].sort_values("step")
        if len(sub) < 2:
            continue
        s = _style(method)
        x = sub["mia_mean_auc"].values
        y = sub["retain_acc"].values
        # Points
        ax.scatter(x, y, color=s["color"], marker=s["marker"],
                   s=40, alpha=0.7, label=s["label"])
        # Arrows
        for i in range(len(x) - 1):
            ax.annotate("", xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="->", color=s["color"],
                                        lw=1.5, alpha=0.4))

    ax.axvline(0.50, color=ORACLE_COLOR, ls=":", lw=1.5, alpha=0.7, label="Perfect forgetting")
    ax.set_xlabel("MIA AUC →")
    ax.set_ylabel("Retain Accuracy →")
    ax.set_title("Unlearning Trajectory: Forgetting vs. Utility\n(arrows = sequential steps)",
                 fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.invert_xaxis()  # better forgetting = left
    fig.tight_layout()
    fig.savefig(out_dir / "12_phase_space.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 12_phase_space.png")


def plot_fraction_leaked(df: pd.DataFrame, out_dir: Path) -> None:
    """Fraction of leaked identities (AUC > 0.55) over steps."""
    if "fraction_leaked" not in df.columns:
        logger.info("  [SKIP] 13_fraction_leaked — column not found")
        return
    _plot_vs_step(df, "fraction_leaked", "Fraction Identities Leaked (AUC > 0.55)",
                  "Identity Leakage Rate vs. Iteration",
                  out_dir / "13_fraction_leaked.png",
                  target_line=0.0, target_label="Zero leakage", ylim=(-0.02, 1.05))


def plot_forget_train_gap(df: pd.DataFrame, out_dir: Path) -> None:
    """Step-local forget-train vs forget-holdout gap over steps.

    The gap is the overfitting-to-forgetting detector: if a method
    memorised the unlearning images, forget acc on the TRAIN images of
    the current step drops to ~0 while the HOLD-OUT acc stays high →
    large positive gap.  A gap ≈ 0 means genuine identity-level erasure.
    """
    if "step_forget_acc" not in df.columns or "step_forget_train_acc" not in df.columns:
        logger.info("  [SKIP] 15_forget_train_gap — step-local columns not found")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = sorted(df["method"].unique())
    for method in methods:
        sub = df[df["method"] == method].sort_values("step")
        s = _style(method)
        gap = sub["step_forget_train_acc"] - sub["step_forget_acc"]
        ax.plot(sub["step"], gap,
                color=s["color"], ls=s["ls"], marker=s["marker"],
                label=s["label"], alpha=0.9)
    ax.axhline(0.10, color="green", ls="--", lw=1.2, alpha=0.6,
               label="gap ≤ 0.10 = genuine forgetting")
    ax.axhline(0.15, color="red", ls="--", lw=1.2, alpha=0.5,
               label="gap ≥ 0.15 = overfit to unlearning images")
    ax.set_xlabel("Forget step")
    ax.set_ylabel("Step forget-train acc − step forget-holdout acc")
    ax.set_title("Forget-Train / Forget-Holdout Gap by Step\n(overfitting-to-forgetting detector)",
                 fontweight="bold", pad=10)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "15_forget_train_gap.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 15_forget_train_gap.png")


def plot_step_forget_acc(df: pd.DataFrame, out_dir: Path) -> None:
    """Step-local forget accuracy (this step's identities only).

    Unlike the cumulative forget_acc (which averages over ALL identities
    forgotten so far), this curve shows the current step's 4 identities.
    A method can't hide a failing step behind an improving cumulative
    average.
    """
    if "step_forget_acc" not in df.columns:
        logger.info("  [SKIP] 16_step_forget_acc — column not found")
        return
    _plot_vs_step(df, "step_forget_acc", "Step-Local Forget Identity Accuracy",
                  "Step-Local Forgetting vs. Iteration",
                  out_dir / "16_step_forget_acc.png",
                  target_line=0.0, target_label="Perfect step forgetting")


# ── Summary statistics ────────────────────────────────────────────────────────


def compute_summary_stats(df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["retain_acc", "forget_acc", "step_forget_acc", "step_forget_train_acc",
            "forget_advantage", "mia_mean_auc", "model_drift", "step_time_s"]
    rows = []
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        row = {"method": _style(method)["label"]}
        for c in cols:
            if c in sub.columns:
                row[f"{c}_mean"] = round(sub[c].mean(), 4)
                row[f"{c}_std"] = round(sub[c].std(), 4)
                final_step = sub["step"].max()
                row[f"{c}_final"] = round(
                    sub[sub["step"] == final_step][c].mean(), 4,
                )
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    path = out_dir / "stability_summary.csv"
    summary_df.to_csv(path, index=False)
    logger.info(f"\n  Summary stats → {path}")

    # Print
    cols_to_show = [c for c in ["method", "retain_acc_mean", "forget_advantage_mean",
                                 "mia_mean_auc_final"] if c in summary_df.columns]
    logger.info(f"\n{'='*70}")
    logger.info("  STABILITY SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(summary_df[cols_to_show].to_string(index=False))
    logger.info("=" * 70)


# ── Master runner ─────────────────────────────────────────────────────────────


def run_stability_analysis(
    combined_csv: str,
    out_dir: str = "results/iterative/plots",
    per_id_csv: str | None = None,
    demog_csv: str | None = None,
    methods: list[str] | None = None,
    steps: list[int] | None = None,
) -> None:
    """Generate stability plots.

    Args:
        combined_csv:  Path to iterative_combined.csv.
        out_dir:       Output directory for plots.
        per_id_csv:    Optional per-identity MIA CSV (single-shot output).
        demog_csv:     Optional demographic MIA CSV.
        methods:       Subset of methods to plot (None = all).
        steps:         Subset of steps to plot (None = all).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n[StabilityAnalysis] Loading {combined_csv}…")
    df = load_data(combined_csv)
    # Filter re-emergence rows ONLY if the column exists.  df.get("type","")
    # returns the SCALAR default when the column is absent, so
    # df[scalar_bool] → KeyError: True (the 7081731/7081733 crash).
    if "type" in df.columns:
        df = df[df["type"] != "re_emergence"]

    # ── Ad-hoc subset filters ───────────────────────────────────────────
    if methods:
        missing = [m for m in methods if m not in set(df["method"].unique())]
        if missing:
            logger.warning(f"  [WARN] Unknown methods (ignored): {missing}")
        df = df[df["method"].isin(methods)]
    if steps:
        df = df[df["step"].isin(steps)]

    if df.empty:
        logger.error("  No data after filtering — check --methods/--steps values")
        return

    logger.info(f"  Methods: {sorted(df['method'].unique())}")
    logger.info(f"  Steps:   {sorted(df['step'].unique())}")
    logger.info(f"  Rows:    {len(df)}")

    # Core
    plot_retain_acc(df, out_path)
    plot_mia_auc(df, out_path)
    plot_forget_advantage(df, out_path)
    plot_model_drift(df, out_path)
    plot_step_time(df, out_path)
    plot_pareto(df, out_path)
    plot_heatmap(df, out_path)
    plot_radar(df, out_path)
    plot_cumulative_time(df, out_path)
    plot_total_time_bar(df, out_path)

    # New
    plot_per_identity_signatures(per_id_csv, out_path)
    plot_demographic_heatmap(demog_csv, out_path)
    plot_phase_space(df, out_path)
    plot_fraction_leaked(df, out_path)

    # Step-local curves (forget_step flow)
    plot_step_forget_acc(df, out_path)
    plot_forget_train_gap(df, out_path)

    compute_summary_stats(df, out_path)
    logger.info(f"\n[OK] All plots → {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined",   type=str, required=True)
    parser.add_argument("--out",        type=str, default="results/iterative/plots")
    parser.add_argument("--per_id_csv", type=str, default=None)
    parser.add_argument("--demog_csv",  type=str, default=None)
    parser.add_argument("--methods",    type=str, nargs="*", default=None,
                        help="Plot only these methods (e.g. ga adaptiforget)")
    parser.add_argument("--steps",      type=int, nargs="*", default=None,
                        help="Plot only these steps (e.g. 1 5 10 15)")
    args = parser.parse_args()
    run_stability_analysis(
        args.combined, args.out,
        per_id_csv=args.per_id_csv,
        demog_csv=args.demog_csv,
        methods=args.methods or None,
        steps=args.steps or None,
    )
