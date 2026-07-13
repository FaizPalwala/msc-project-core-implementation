"""
stability_analysis.py  —  Phase 5: Stability Analysis & Visualisation

Generates the full suite of iterative-unlearning plots:

  1. Retain Accuracy vs. Iteration          — utility stability
  2. MIA AUC vs. Iteration                  — forgetting quality over time
  3. Forget Advantage vs. Iteration         — target: flat near 0
  4. Model Drift (L2) vs. Iteration         — how far weights wander
  5. Step Time vs. Iteration                — computational cost
  6. Pareto scatter: Retain Acc vs. MIA AUC — utility/forgetting trade-off
  7. Heatmap: Forget Advantage across methods × iterations
  8. Radar chart: final-step multi-metric comparison

All plots saved as PNG (publication quality, 300 DPI).
Summary statistics CSV also produced.

Usage:
    python stability_analysis.py \
        --combined ../../results/iterative/iterative_combined.csv \
        --out      ../../results/iterative/plots
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "lines.linewidth":  2.0,
    "lines.markersize": 6,
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
    "adaptiformet":  {"color": "#ff7043", "ls": "-",  "marker": "*", "label": "AdaptiForget"},
}
ORACLE_COLOR = "#1565c0"


def _style(method: str) -> dict:
    return METHOD_STYLES.get(method, {"color": "#555", "ls": "-",
                                       "marker": "o", "label": method})


# ──────────────────────────────────────────────────────────────────────────────
# Load and prepare data
# ──────────────────────────────────────────────────────────────────────────────

def load_data(combined_csv: str) -> pd.DataFrame:
    df = pd.read_csv(combined_csv)
    df = df[df["is_baseline"] != True].copy()
    df["step"] = df["step"].astype(int)
    for col in ["retain_acc", "test_acc", "forget_acc",
                "mia_auc", "forget_advantage", "model_drift",
                "step_time_s", "cumulative_time_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Individual plots
# ──────────────────────────────────────────────────────────────────────────────

def _plot_metric_vs_step(df, metric, ylabel, title, out_file,
                         target_line=None, target_label=None,
                         ylim=None, invert=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = df["method"].unique()

    for method in sorted(methods):
        sub = df[df["method"] == method].sort_values("step")
        s   = _style(method)
        ax.plot(sub["step"], sub[metric],
                color=s["color"], ls=s["ls"], marker=s["marker"],
                label=s["label"], alpha=0.9)

    if target_line is not None:
        ax.axhline(target_line, color=ORACLE_COLOR, ls=":", lw=1.5,
                   label=target_label or f"Target ({target_line})", alpha=0.7)

    ax.set_xlabel("Forget step (cumulative identity deletions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_file.name}")


def plot_retain_acc(df, out_dir: Path):
    _plot_metric_vs_step(
        df, "retain_acc",
        ylabel="Retain Accuracy",
        title="Retain Accuracy vs. Iteration\n(utility stability under sequential deletions)",
        out_file=out_dir / "01_retain_acc_vs_step.png",
        ylim=(0.0, 1.05),
    )


def plot_mia_auc(df, out_dir: Path):
    _plot_metric_vs_step(
        df, "mia_auc",
        ylabel="MIA AUC (forget set)",
        title="MIA AUC vs. Iteration\n(forgetting quality — target: 0.50 = perfectly forgotten)",
        out_file=out_dir / "02_mia_auc_vs_step.png",
        target_line=0.50,
        target_label="Perfect forgetting (AUC=0.50)",
        ylim=(0.40, 1.05),
    )


def plot_forget_advantage(df, out_dir: Path):
    _plot_metric_vs_step(
        df, "forget_advantage",
        ylabel="Forget Advantage  |MIA AUC − 0.5|",
        title="Forget Advantage vs. Iteration\n(target: 0 = no information advantage for attacker)",
        out_file=out_dir / "03_forget_advantage_vs_step.png",
        target_line=0.0,
        target_label="Perfect forgetting",
        ylim=(-0.02, 0.55),
    )


def plot_model_drift(df, out_dir: Path):
    _plot_metric_vs_step(
        df, "model_drift",
        ylabel="Weight L₂ Distance from Original Model",
        title="Model Drift vs. Iteration\n(how far weights move from the original model)",
        out_file=out_dir / "04_model_drift_vs_step.png",
    )


def plot_step_time(df, out_dir: Path):
    _plot_metric_vs_step(
        df, "step_time_s",
        ylabel="Step wall-clock time (s)",
        title="Unlearning Step Time vs. Iteration\n(computational cost per deletion)",
        out_file=out_dir / "05_step_time_vs_step.png",
    )


def plot_pareto(df, out_dir: Path):
    """Scatter: Retain Acc vs. MIA AUC, one point per method per step."""
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = df["method"].unique()

    for method in sorted(methods):
        sub = df[df["method"] == method].dropna(subset=["retain_acc", "mia_auc"])
        s   = _style(method)
        sc  = ax.scatter(sub["mia_auc"], sub["retain_acc"],
                         c=sub["step"], cmap="Blues",
                         vmin=1, vmax=df["step"].max(),
                         alpha=0.75, edgecolors=s["color"],
                         linewidths=1.5, marker=s["marker"],
                         s=60, label=s["label"])

    # Ideal point
    ax.axvline(0.50, color=ORACLE_COLOR, ls=":", lw=1.5, alpha=0.7,
               label="Perfect MIA (0.50)")
    ax.set_xlabel("MIA AUC (← better forgetting at 0.50)")
    ax.set_ylabel("Retain Accuracy (↑ better utility)")
    ax.set_title("Pareto Frontier: Utility vs. Forgetting Quality\n"
                 "(colour = iteration step; ideal = top-left)",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.colorbar(sc, ax=ax, label="Iteration step")
    fig.tight_layout()
    fig.savefig(out_dir / "06_pareto_utility_forgetting.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06_pareto_utility_forgetting.png")


def plot_heatmap(df, out_dir: Path):
    """Heatmap of forget_advantage across methods × steps."""
    methods = sorted(df["method"].unique())
    steps   = sorted(df["step"].unique())

    matrix = np.full((len(methods), len(steps)), np.nan)
    for i, m in enumerate(methods):
        for j, s in enumerate(steps):
            sub = df[(df["method"] == m) & (df["step"] == s)]["forget_advantage"]
            if len(sub):
                matrix[i, j] = sub.mean()

    fig, ax = plt.subplots(figsize=(max(10, len(steps)*0.6),
                                    max(4, len(methods)*0.7)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=0.5,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_style(m)["label"] for m in methods], fontsize=9)
    ax.set_xlabel("Forget step")
    ax.set_title("Forget Advantage Heatmap (lower = better forgetting)",
                 fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, label="Forget Advantage |AUC−0.5|")
    fig.tight_layout()
    fig.savefig(out_dir / "07_heatmap_forget_advantage.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07_heatmap_forget_advantage.png")


def plot_radar(df, out_dir: Path):
    """Radar chart: final-step metrics for each method."""
    import math
    metrics  = ["retain_acc", "test_acc",
                "forget_advantage_inv",   # invert so high=good
                "mia_quality",            # 1 - |AUC-0.5|/0.5
                "speed_score"]            # 1 - norm(time)

    max_step = df["step"].max()
    final    = df[df["step"] == max_step].copy()
    final["forget_advantage_inv"] = 1.0 - final["forget_advantage"].clip(0, 0.5) / 0.5
    final["mia_quality"] = 1.0 - (final["mia_auc"] - 0.5).abs().clip(0, 0.5) / 0.5
    t_max = final["step_time_s"].max()
    final["speed_score"] = 1.0 - (final["step_time_s"] / max(t_max, 1)).clip(0, 1)

    labels = ["Retain Acc", "Test Acc",
              "Forget\nQuality", "MIA\nQuality", "Speed"]
    N     = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    methods_present = [m for m in METHOD_STYLES if m in final["method"].values]
    for method in methods_present:
        sub = final[final["method"] == method]
        if sub.empty:
            continue
        vals = [float(sub[m].mean()) for m in metrics]
        vals = [max(0, min(1, v)) for v in vals]
        vals += vals[:1]
        s = _style(method)
        ax.plot(angles, vals, color=s["color"], lw=2, label=s["label"])
        ax.fill(angles, vals, color=s["color"], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Final-Step Multi-Metric Comparison (normalised, outer=better)",
                 fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "08_radar_final_step.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08_radar_final_step.png")


def plot_cumulative_time(df, out_dir: Path):
    """Cumulative wall-clock time per method over steps."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("step")
        s   = _style(method)
        ax.plot(sub["step"], sub["cumulative_time_s"],
                color=s["color"], ls=s["ls"], marker=s["marker"],
                label=s["label"], alpha=0.9)
    ax.set_xlabel("Forget step")
    ax.set_ylabel("Cumulative time (s)")
    ax.set_title("Cumulative Unlearning Time\n(total compute per method over all steps)",
                 fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "09_cumulative_time.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_cumulative_time.png")


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame, out_dir: Path):
    """Per-method summary statistics over all steps."""
    cols = ["retain_acc", "test_acc", "forget_advantage", "mia_auc",
            "model_drift", "step_time_s"]
    rows = []
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        row = {"method": _style(method)["label"]}
        for c in cols:
            if c in sub.columns:
                row[f"{c}_mean"] = round(sub[c].mean(), 4)
                row[f"{c}_std"]  = round(sub[c].std(),  4)
                row[f"{c}_final"]= round(sub[sub["step"] == sub["step"].max()][c].mean(), 4)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    path = out_dir / "stability_summary_stats.csv"
    summary_df.to_csv(path, index=False)
    print(f"\n  Summary stats → {path}")

    # Print table
    print(f"\n{'='*90}")
    print("  STABILITY SUMMARY  (mean ± std over all steps, final step)")
    print(f"{'='*90}")
    print(f"{'Method':<14} | "
          f"{'Retain(μ±σ)':>14} {'Test(μ±σ)':>13} "
          f"{'F-Adv(μ±σ)':>13} {'MIA_AUC final':>14}")
    print("─" * 70)
    for _, r in summary_df.iterrows():
        print(f"{r['method']:<14} | "
              f"{r.get('retain_acc_mean',0):.4f}±{r.get('retain_acc_std',0):.4f}  "
              f"{r.get('test_acc_mean',0):.4f}±{r.get('test_acc_std',0):.4f}  "
              f"{r.get('forget_advantage_mean',0):.4f}±{r.get('forget_advantage_std',0):.4f}  "
              f"{r.get('mia_auc_final',0):.4f}")
    print("=" * 90)
    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# Master plot runner
# ──────────────────────────────────────────────────────────────────────────────

def run_stability_analysis(combined_csv: str, out_dir: str = "../results/iterative/plots"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[StabilityAnalysis] Loading {combined_csv}…")
    df = load_data(combined_csv)
    print(f"  Methods : {sorted(df['method'].unique())}")
    print(f"  Steps   : {sorted(df['step'].unique())}")
    print(f"  Rows    : {len(df)}")
    print(f"\n  Generating plots → {out_path}\n")

    plot_retain_acc(df, out_path)
    plot_mia_auc(df, out_path)
    plot_forget_advantage(df, out_path)
    plot_model_drift(df, out_path)
    plot_step_time(df, out_path)
    plot_pareto(df, out_path)
    plot_heatmap(df, out_path)
    plot_radar(df, out_path)
    plot_cumulative_time(df, out_path)
    compute_summary_stats(df, out_path)

    print(f"\n[OK] All plots saved to {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=str, required=True,
                        help="Path to iterative_combined.csv")
    parser.add_argument("--out",      type=str,
                        default="../results/iterative/plots")
    args = parser.parse_args()
    run_stability_analysis(args.combined, args.out)
