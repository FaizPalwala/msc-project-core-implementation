#!/usr/bin/env python3
"""plot_style.py — shared publication style for all evaluation figures.

Centralises method colours/linestyles/markers/labels so stability.py and
imbalanced_plots.py (and any future plotter) agree on the visual language:

- **Families** (behavioural, full-figure colour coding — all methods in one
  panel):
  *survivors* (green family: ft, msg, msg_kd, adaptiforget) vs *collapsers*
  (red/orange family: ga, ng_plus, ct, budget_scaled); no_unlearning is
  the neutral control.
- **Groups** (release/appendix panels): G1 baselines, G2 SOTA, G3 novel —
  the oracle rides in EVERY panel as a dashed reference, never as a row.
- Within a family each method keeps a distinct marker + linestyle so the
  figures survive grayscale/CVD printing (never hue alone).
"""
from __future__ import annotations

# ── Behavioural families (full-figure colour coding) ─────────────────────────
SURVIVORS  = {"ft", "msg", "msg_kd", "adaptiforget"}   # green family
COLLAPSERS = {"ga", "ng_plus", "ct", "budget_scaled"}   # red/orange family
NEUTRAL    = {"no_unlearning", "srl"}                   # grey / control-ish

# ── Registry groups (release/appendix panels) ──────────────────────────────
# oracle rides in EVERY panel as the dashed target_line — not a row here.
G1_BASELINES = ["no_unlearning", "ga", "srl", "ft"]
G2_SOTA      = ["ng_plus", "msg", "ct"]
G3_NOVEL     = ["msg_kd", "adaptiforget", "budget_scaled"]
GROUPS       = [("baselines", G1_BASELINES),
                ("sota",      G2_SOTA),
                ("novel",     G3_NOVEL)]

# ── Per-method style — one hex per method, family-consistent ───────────────
METHOD_STYLES = {
    "no_unlearning": {"color": "#111111", "ls": ":",  "marker": "x",
                      "label": "No-Unlearning", "family": "neutral"},
    "ga":            {"color": "#d32f2f", "ls": "--", "marker": "s",
                      "label": "GA", "family": "collapser"},
    "srl":           {"color": "#9e9e9e", "ls": "--", "marker": "^",
                      "label": "SRL", "family": "neutral"},
    "ft":            {"color": "#2e7d32", "ls": "--", "marker": "D",
                      "label": "FT", "family": "survivor"},
    "ng_plus":       {"color": "#e57373", "ls": "-",  "marker": "o",
                      "label": "NG+", "family": "collapser"},
    "msg":           {"color": "#43a047", "ls": "-",  "marker": "v",
                      "label": "MSG", "family": "survivor"},
    "msg_kd":        {"color": "#81c784", "ls": "-",  "marker": "P",
                      "label": "MSG-KD", "family": "survivor"},
    "ct":            {"color": "#ff9800", "ls": "-",  "marker": "h",
                      "label": "CT", "family": "collapser"},
    "adaptiforget":  {"color": "#a5d6a7", "ls": "-",  "marker": "*",
                      "label": "AdaptiForget", "family": "survivor"},
    "budget_scaled": {"color": "#ff7043", "ls": "-.", "marker": "X",
                      "label": "Budget-Scaled GA", "family": "collapser"},
}
ORACLE = {"color": "#111111", "ls": "--", "label": "Oracle (retrain)"}


def style(method: str) -> dict:
    """Return the style dict for a method (fallback: neutral grey)."""
    s = METHOD_STYLES.get(method, {
        "color": "#555555", "ls": "-", "marker": "o",
        "label": method, "family": "other",
    })
    # Never let a caller mutate the shared registry.
    return dict(s)


def method_label(method: str) -> str:
    return style(method)["label"]


def group_for(method: str) -> str | None:
    """Group name ('baselines'|'sota'|'novel') for a method, or None."""
    for name, members in GROUPS:
        if method in members:
            return name
    return None


def panel_order(methods: list[str]) -> list[str]:
    """Order methods for display: G1 → oracle → G2 → G3 → no-unlearning.

    The oracle is not a data row (it is the target_line), but if it appears
    in the data it sorts right after G1.
    """
    rank: dict[str, float] = {m: float(i) for i, g in enumerate(GROUPS)
                              for m in g[1]}
    # oracle slots between G1 and G2
    rank["retrain"] = len(G1_BASELINES) + 0.5
    rank["oracle"]  = len(G1_BASELINES) + 0.5
    return sorted(methods, key=lambda m: rank.get(m, 99.0))


def family_color(method: str) -> str:
    """Colour by behavioural family (full-figure colour coding)."""
    if method in SURVIVORS:
        return "#2e7d32"            # green
    if method in COLLAPSERS:
        return "#d32f2f"            # red
    return "#111111"                # neutral / control
