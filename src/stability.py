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

# ── Style (shared module — see plot_style.py) ──────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "lines.linewidth": 2.0, "lines.markersize": 6,
})

from plot_style import (  # noqa: E402
    METHOD_STYLES, ORACLE, GROUPS, SURVIVORS, COLLAPSERS,
    style as _style, method_label, panel_order,
)

# ── Shared scale policy (B1) ────────────────────────────────────────────────
# Each metric gets ONE y-window used by ALL subgroup panels, so the panels
# are directly comparable instead of each auto-scaling into its own world.
# Log-scale metrics span orders of magnitude (drift 9.7 → 1.36e6): a linear
# axis would flatten the SOTA/novel panels entirely.
LOG_METRICS = {"model_drift", "step_time_s", "cumulative_time_s"}
SCALE_POLICY = {
    # metric: (shared (min, max), log?)
    "retain_acc":         ((0.0, 1.05),   False),
    # MIA AUC: 0.5 = chance, BELOW 0.5 = the attacker is worse than chance
    # at singling out forget members = the erasure signal (adaptiforget
    # 0.003, FT 0.0004, CT 0.012). A 0.40 floor clipped every erasing
    # method's curve/panel off-axis (F10-class bug — same as the per-identity
    # signatures plot). Full range is the honest window.
    "mia_mean_auc":       ((0.0, 1.05),   False),
    "mia_auc":            ((0.0, 1.05),   False),
    "forget_advantage":   ((-0.02, 0.55), False),
    "fraction_leaked":    ((-0.02, 1.05), False),
    "model_drift":        (None,          True),
    "step_time_s":        (None,          True),
    "cumulative_time_s":  (None,          True),
    "step_forget_acc":    ((0.0, 1.05),   False),
}
PARETO_LIM = (0.0, 1.0)


def scale_for(metric: str) -> tuple[tuple | None, bool]:
    """Return (shared (min,max) or None, log?) for a metric."""
    return SCALE_POLICY.get(metric, (None, False))


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


def _draw_vs_step(ax, df, metric, style_kwargs: dict | None = None):
    """Plotting body: one line + μ±σ band per method on an existing axes.

    style_kwargs: {lw, ms} overrides for compact/thesis panels (B3).
    """
    methods = panel_order(sorted(df["method"].unique()))
    std_col = f"{metric}_std"
    has_std = std_col in df.columns
    lw = style_kwargs.get("lw", 2.0) if style_kwargs else 2.0
    ms = style_kwargs.get("ms", 6) if style_kwargs else 6

    for method in methods:
        sub = df[df["method"] == method].sort_values("step")
        if sub.empty:
            continue
        s = _style(method)
        ax.plot(sub["step"], sub[metric],
                color=s["color"], ls=s["ls"], marker=s["marker"],
                label=s["label"], alpha=0.9, lw=lw, ms=ms)
        # Error band from multi-seed std (μ ± σ) — α 0.10
        if has_std:
            lo = sub[metric] - sub[std_col]
            hi = sub[metric] + sub[std_col]
            ax.fill_between(sub["step"], lo, hi,
                            color=s["color"], alpha=0.10, lw=0)
    return has_std


def _apply_scale(ax, metric):
    """Shared y-limits / log-axis per the SCALE_POLICY (B1)."""
    lims, log = scale_for(metric)
    if log:
        ax.set_yscale("log")
    if lims:
        ax.set_ylim(*lims)


def _plot_vs_step(df, metric, ylabel, title, out_file,
                  target_line=None, target_label=None, ylim=None,
                  style_kwargs: dict | None = None, show_title=True):
    """Full figure (release record): all methods, one axes, saved to out_file."""
    fig, ax = plt.subplots(figsize=(9, 5))
    has_std = _draw_vs_step(ax, df, metric, style_kwargs)

    if show_title:
        if has_std:
            ax.set_title(f"{title}\n(shaded = μ ± σ across seeds)",
                         fontweight="bold", pad=10)
        else:
            ax.set_title(title, fontweight="bold", pad=10)

    if target_line is not None:
        ax.axhline(target_line, color=ORACLE["color"], ls=ORACLE["ls"], lw=1.5,
                   label=target_label or f"Target ({target_line})", alpha=0.7)
    ax.set_xlabel("Forget step")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        _apply_scale(ax, metric)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {Path(out_file).name}")


def _plot_subgrouped(df, metric, ylabel, out_file,
                     target_line=None, target_label=None,
                     style_kwargs: dict | None = None,
                     tag_letters=("a", "b", "c")):
    """3 stacked panels — one per registry group (G1/G2/G3, B2).

    Each panel draws its group's methods + the oracle target line; all
    panels SHARE the metric's y-limits (B1) so they read side-by-side.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    all_steps = sorted(df["step"].unique())
    lw = style_kwargs.get("lw", 2.0) if style_kwargs else 2.0
    ms = style_kwargs.get("ms", 6) if style_kwargs else 6

    for ax, (gname, members), tag in zip(axes, GROUPS, tag_letters):
        sub = df[df["method"].isin(members)]
        if sub.empty:
            ax.set_visible(False)
            continue
        _draw_vs_step(ax, sub, metric,
                      {"lw": lw, "ms": ms} if style_kwargs else None)
        if target_line is not None:
            ax.axhline(target_line, color=ORACLE["color"], ls=ORACLE["ls"],
                       lw=1.5, label=target_label or f"Target ({target_line})",
                       alpha=0.7)
        _apply_scale(ax, metric)
        ax.text(-0.06, 1.02, f"({tag})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="best", fontsize=9, ncol=2)

    axes[-1].set_xlabel("Forget step")
    axes[-1].set_xticks(all_steps)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {Path(out_file).name} (3 subgroup panels)")


def _plot_pair(df, metric, ylabel, out_file,
               target_line=None, target_label=None, members=None):
    """Thesis-pair variant (B3): compact single panel, family colours,
    ~200 pt half-width — tuned for a two-figure LaTeX pair.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sub = df if members is None else df[df["method"].isin(members)]
    _draw_vs_step(ax, sub, metric, {"lw": 1.5, "ms": 5})
    if target_line is not None:
        ax.axhline(target_line, color=ORACLE["color"], ls=ORACLE["ls"], lw=1.5,
                   label=target_label or f"Target ({target_line})", alpha=0.7)
    ax.set_xlabel("Forget step")
    ax.set_ylabel(ylabel, fontsize=9)
    _apply_scale(ax, metric)
    ax.legend(loc="best", fontsize=8, ncol=2)   # compact legend
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {Path(out_file).name} (pair)")


def _vs_step_targets(metric: str) -> tuple:
    """(target_line, target_label) per metric, if any."""
    return {
        "mia_mean_auc": (0.50, "Perfect forgetting (0.50)"),
        "forget_advantage": (0.0, "Perfect forgetting"),
        "fraction_leaked": (0.0, "Zero leakage"),
        "step_forget_acc": (0.0, "Perfect step forgetting"),
    }.get(metric, (None, None))


# ── Core plots (01–09) ────────────────────────────────────────────────────────


def plot_retain_acc(df, out_dir: Path):
    _plot_vs_step(df, "retain_acc", "Retain Identity Accuracy",
                  "Retain Accuracy vs. Iteration", out_dir / "01_retain_acc.png",
                  ylim=(0.0, 1.05))


def plot_mia_auc(df, out_dir: Path):
    _plot_vs_step(df, "mia_mean_auc", "MIA AUC (identity head)",
                  "MIA AUC vs. Iteration", out_dir / "02_mia_auc.png",
                  target_line=0.50, target_label="Perfect forgetting (0.50)",
                  ylim=scale_for("mia_mean_auc")[0])


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
    """Pareto trajectory: per-method step path + global frontier envelope.

    Option C design: colour = method throughout (legend carries identity,
    no label storm); each method's 16 steps are CONNECTED as a trajectory
    line (the path through utility/forgetting space IS the story — FT's
    climb vs GA's divergence vs the collapse family's flatline); markers
    at steps {1, 5, 10, 15}; bold marker at the final step; the global
    Pareto front + hypervolume close the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = sorted(df["method"].unique())
    marker_steps = {1, 5, 10, 15}
    max_step = int(df["step"].max())

    all_x, all_y = [], []

    for method in methods:
        sub = df[df["method"] == method].dropna(subset=["retain_acc", "mia_mean_auc"])
        if sub.empty:
            continue
        s = _style(method)
        x = sub["mia_mean_auc"].values
        y = sub["retain_acc"].values
        steps = sub["step"].values
        all_x.extend(x)
        all_y.extend(y)

        # trajectory line through the steps (the path, not a scatter storm)
        ax.plot(x, y, color=s["color"], ls="-", lw=1.3, alpha=0.75,
                label=s["label"], zorder=2)
        # selected step markers
        for ms, mx, my in zip(steps, x, y):
            if int(ms) in marker_steps:
                ax.scatter(mx, my, color=s["color"], marker=s["marker"],
                           s=45, edgecolors="white", linewidths=0.5,
                           zorder=3)
        # bold final-step marker
        ax.scatter(x[-1], y[-1], color=s["color"], marker=s["marker"],
                   s=90, edgecolors="white", linewidths=1.0, zorder=4)

    # Global frontier across all points (the achievable envelope)
    if all_x:
        gx = np.array(all_x)
        gy = np.array(all_y)
        g_idx = pareto_frontier(gx, gy)
        fx = np.sort(gx[g_idx])
        fy = gy[g_idx][np.argsort(gx[g_idx])]
        ax.plot(fx, fy, color="#222", lw=2.0, ls="--", alpha=0.8,
                label="Pareto frontier", zorder=5)

        # Hypervolume — sits ABOVE the lower-left legend
        hv = hypervolume(gx, gy, ref_x=gx.max(), ref_y=gy.min())
        ax.text(0.02, 0.30, f"Hypervolume: {hv:.4f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    ax.axvline(0.50, color=ORACLE["color"], ls=ORACLE["ls"], lw=1.5, alpha=0.7,
               label="Perfect MIA (0.50)")
    ax.set_xlabel("MIA AUC (← better forgetting)")
    ax.set_ylabel("Retain Accuracy (↑ better)")
    ax.set_title(f"Pareto Trajectory: Utility vs. Forgetting over {max_step} steps\n"
                 "(lines = step path; bold marker = final step)",
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "06_pareto.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: 06_pareto.png (with frontier + hypervolume)")


def plot_heatmap(df, out_dir: Path):
    methods = sorted(df["method"].unique())
    steps = sorted(df["step"].unique())
    # DIRECTIONAL advantage: 0.5 − MIA_AUC. Positive = below-chance MIA
    # (attacker CANNOT find forget members = erased, green); negative =
    # above-chance (leaked, red). The old |AUC−0.5| rendered No-U (adv 0.02,
    # does nothing) green and FT (adv 0.499, actually erased) red — the
    # symmetric metric cannot distinguish "erased" from "leaked".
    matrix = np.full((len(methods), len(steps)), np.nan)
    for i, m in enumerate(methods):
        for j, s in enumerate(steps):
            sub = df[(df["method"] == m) & (df["step"] == s)]
            if "mia_mean_auc" in sub.columns and len(sub):
                matrix[i, j] = 0.5 - sub["mia_mean_auc"].mean()
    fig, ax = plt.subplots(figsize=(max(10, len(steps)*0.6),
                                    max(4, len(methods)*0.7)))
    # RdYlGn: low = red (leaked), high = green (erased); center at 0 so
    # chance (0.5 MIA → adv 0) renders yellow — No-U now sits neutral, not
    # falsely green.
    vmax = max(0.5, np.nanmax(np.abs(matrix), initial=0.0))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_style(m)["label"] for m in methods], fontsize=9)
    ax.set_xlabel("Forget step")
    ax.set_title("Forget Advantage Heatmap (green = erased, red = leaked)",
                 fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, label="0.5 − MIA AUC")
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


def plot_total_time_bar(df, out_dir: Path, oracle_time_s: float | None = None):
    """Total wall-clock per method as a sorted horizontal bar chart.

    Emphasises cost differences between methods — a key deployment
    feasibility metric.  Uses final cumulative_time_s per method.

    oracle_time_s (optional): single-shot retrain-oracle wall time — the
    'cheaper-than-retraining' threshold (Bourtoule et al. 2021). Drawn as a
    dashed reference line, NOT a bar: the oracle never ran the iterative
    protocol (order-independent by design, §9), so it has no cumulative
    trajectory — but its one-shot cost is exactly the deployment baseline
    unlearning must beat.
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

    # Log axis: bars span 98–952 s but the iterative-oracle estimate sits at
    # ~10,800 s (15× single retrain) — a linear axis that fits both would
    # crush the bars into the left 8%. Log keeps the method comparison and
    # the two reference lines readable.
    ax.set_xscale("log")
    # no_unlearning has total 0.0s — log10(0) = -inf would break the axis
    # floor. Floor one full DECADE below the smallest positive total: with
    # floor == min's decade (poisson min 100.6 → lo=100) the shortest bars
    # collapse into the left 1-3% and their labels stack on each other and
    # the y-axis names. One decade down (poisson → lo=10) keeps every bar
    # ≥ ~32% of the axis width.
    pos_totals = [t for t in sorted_totals if t > 0]
    lo = 10 ** (np.floor(np.log10(min(pos_totals))) - 1) if pos_totals else 1.0
    top = max((oracle_time_s or 0) * 16, max(sorted_totals))
    hi = top * 1.15
    ax.set_xlim(lo, hi)

    # Annotate values AFTER the bar tip in black (consistent style across
    # all bars — white-inside is invisible on short/pale bars). Offset in
    # log space (tip × 1.03) so labels never sit on decade ticks. The
    # 0.0 bar (No-U) has no width — place its label at the axis floor.
    for bar, val in zip(bars, sorted_totals):
        label = f"{val/60:.1f} min" if val > 0 else "0.0 min"
        xpos = bar.get_width() * 1.03 if val > 0 else 10 ** (np.log10(lo) + 0.02)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=9, color="#111111")

    ax.set_xlabel("Total cumulative time (s, log scale)")
    ax.set_title("Total Unlearning Time by Method\n(lower = cheaper to deploy)",
                 fontweight="bold", pad=10)
    if oracle_time_s is not None:
        n_steps = int(df["step"].max()) if "step" in df.columns else 15
        ax.axvline(oracle_time_s, color=ORACLE["color"], ls=ORACLE["ls"], lw=1.8,
                   alpha=0.8,
                   label=f"Single retrain (oracle): {oracle_time_s/60:.1f} min")
        # The oracle never ran iteratively (order-independent, §9) — but if
        # it had (retrain from scratch at each step), its cumulative time at
        # the 15-step horizon ≈ n_steps × one retrain. This is the honest
        # comparison against the bars, which are 15-step cumulative totals.
        est = n_steps * oracle_time_s
        ax.axvline(est, color=ORACLE["color"], ls=":", lw=1.8, alpha=0.7,
                   label=f"Iterative-oracle estimate ({n_steps}× retrain): "
                         f"{est/3600:.1f} h")
        ax.legend(loc="lower right", fontsize=9)
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
    # 9 methods only — No-Op (control) dropped; Retrain Oracle becomes a
    # dashed reference line, not a panel. Panel order follows G1→G2→G3.
    # CSV method labels carry config markers ("AdaptiForget ‡", "MSG-KD †",
    # "Retrain Oracle*") — strip them and match on the clean label prefix.
    def _key_from_label(label: str) -> str | None:
        # Strip config markers (‡ † *) AND the trailing space left behind
        # ("Budget-Scaled GA ‡" → "Budget-Scaled GA" — an unstripped compare
        # dropped the whole method → empty panel).
        clean = label.replace("‡", "").replace("†", "").replace("*", "").strip()
        for k, v in METHOD_STYLES.items():
            if clean == v["label"] or clean.split()[0] == v["label"]:
                return k
        if "Retrain" in clean:
            return "retrain"
        return None

    df["method_key"] = df["method"].map(_key_from_label)
    methods = [m for m in panel_order(list(METHOD_STYLES.keys())) if m != "no_unlearning"]
    df = df[df["method_key"].isin(methods + ["retrain"])]
    oracle_mean = df.loc[df["method_key"] == "retrain", "mia_auc"].mean() \
        if "retrain" in df["method_key"].values else None
    df = df[df["method_key"] != "retrain"]

    n_methods = len(methods)
    n_cols, n_rows = (3, 3) if n_methods <= 9 else (4, 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 14))
    axes = axes.flatten()

    for ax, method_key in zip(axes, methods):
        group = df[df["method_key"] == method_key]
        aucs = group.sort_values("mia_auc", ascending=False)
        colors = ["#e57373" if a > 0.55 else "#81c784" for a in aucs["mia_auc"]]
        ax.bar(range(len(aucs)), aucs["mia_auc"], color=colors, width=0.8)
        ax.axhline(0.50, color="grey", ls="--", lw=1, alpha=0.5)
        if oracle_mean is not None:
            ax.axhline(oracle_mean, color=ORACLE["color"], ls=ORACLE["ls"],
                       lw=1.2, alpha=0.7,
                       label=f"Retrain oracle ({oracle_mean:.3f})")
            ax.legend(loc="lower right", fontsize=7)
        ax.set_title(_style(method_key)["label"], fontsize=10)
        ax.set_ylabel("MIA AUC")
        # Linear 0–1 keeps the oracle reference and cross-panel comparability.
        # Stand-out erasures (FT/CT: bars ~1e-4..5e-3, invisible at this
        # scale) are highlighted by the per-panel leak summary in the top
        # corner — the same mechanism on every panel, so "0/75 leaked" reads
        # against "18/75 leaked" rather than being logged away.
        ax.set_ylim(0.0, 1.0)
        n_leaked = int((aucs["mia_auc"] > 0.55).sum())
        n_total = len(aucs)
        max_auc = float(aucs["mia_auc"].max())
        ax.text(0.02, 0.97,
                f"{n_leaked}/{n_total} leaked · max {max_auc:.4f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#555555", alpha=0.85))

    for ax in axes[len(methods):]:
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
    # Full 0–1 colour window: MIA AUC spans 0.0002 (erased) to 0.84 (leaked);
    # the old 0.45–0.65 window saturated 36/44 cells (F11-class bug).
    im = ax.imshow(pivoted.values, cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
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

    ax.axvline(0.50, color=ORACLE["color"], ls=ORACLE["ls"], lw=1.5, alpha=0.7, label="Perfect forgetting")
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
    # No-Unlearning's gap ≈ 0 because it performs NO unlearning (train and
    # holdout acc both stay ≈1.0) — visually identical to genuine erasure,
    # semantically the opposite. Annotate so the flat zero line can't be
    # misread as a good result.
    if "no_unlearning" in df["method"].values:
        ax.annotate("No-U: gap ≈ 0 trivially —\nno unlearning performed (acc stays ≈ 1.0)",
                    xy=(df["step"].max() * 0.5, 0.01), fontsize=8,
                    color="#111111", alpha=0.7, ha="center", va="bottom")
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
    pair: bool = False,
    oracle_time_s: float | None = None,
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

    # Core (release record — full-method files, unchanged names)
    plot_retain_acc(df, out_path)
    plot_mia_auc(df, out_path)
    plot_forget_advantage(df, out_path)
    plot_model_drift(df, out_path)
    plot_step_time(df, out_path)
    plot_pareto(df, out_path)
    plot_heatmap(df, out_path)
    plot_radar(df, out_path)
    plot_cumulative_time(df, out_path)
    plot_total_time_bar(df, out_path, oracle_time_s=oracle_time_s)

    # New
    plot_per_identity_signatures(per_id_csv, out_path)
    plot_demographic_heatmap(demog_csv, out_path)
    plot_phase_space(df, out_path)
    plot_fraction_leaked(df, out_path)

    # Step-local curves (forget_step flow)
    plot_step_forget_acc(df, out_path)
    plot_forget_train_gap(df, out_path)

    # ── Subgrouped panels (B4: plots/subgroups/<NN>_<name>_subgroups.png) ─
    sub_dir = out_path / "subgroups"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for metric, name, ylabel in [
        ("retain_acc", "01_retain_acc", "Retain Identity Accuracy"),
        ("mia_mean_auc", "02_mia_auc", "MIA AUC (identity head)"),
        ("forget_advantage", "03_forget_adv", "Forget Advantage |AUC − 0.5|"),
        ("model_drift", "04_model_drift", "Weight L₂ Distance from Original"),
        ("step_time_s", "05_step_time", "Step Time (s)"),
        ("cumulative_time_s", "09_cumulative_time", "Cumulative Time (s)"),
        ("fraction_leaked", "13_fraction_leaked", "Fraction Identities Leaked"),
        ("step_forget_acc", "16_step_forget_acc", "Step-Local Forget Identity Accuracy"),
    ]:
        tline, tlabel = _vs_step_targets(metric)
        _plot_subgrouped(df, metric, ylabel,
                         sub_dir / f"{name}_subgroups.png",
                         target_line=tline, target_label=tlabel)

    # ── Thesis pairs (B3) — written only with --pair ─────────────────────
    if pair:
        fig_dir = Path("src/figures") if Path("src").exists() else out_path
        fig_dir.mkdir(parents=True, exist_ok=True)
        pairs = [
            ("iter_01_retain_acc.png", "retain_acc",
             "Retain Identity Accuracy", ("poisson",)),
        ]
        for fname, metric, ylabel, _members in pairs:
            tline, tlabel = _vs_step_targets(metric)
            _plot_pair(df, metric, ylabel, fig_dir / fname,
                       target_line=tline, target_label=tlabel)

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
    parser.add_argument("--pair", action="store_true",
                        help="Also write compact thesis-pair figures to src/figures")
    parser.add_argument("--oracle_time_s", type=float, default=None,
                        help="Single-shot retrain-oracle wall time (s) — drawn "
                             "as the cheaper-than-retraining reference line in 14")
    args = parser.parse_args()
    run_stability_analysis(
        args.combined, args.out,
        per_id_csv=args.per_id_csv,
        demog_csv=args.demog_csv,
        methods=args.methods or None,
        steps=args.steps or None,
        pair=args.pair,
        oracle_time_s=args.oracle_time_s,
    )
