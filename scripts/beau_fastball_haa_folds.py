"""
Beau Elson — Fastball HAA Fold Analysis
Split fastballs into HAA terciles and show where each fold is most effective.
Run from the repo root:  python scripts/beau_fastball_haa_folds.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "personal_reports" / "beau_elson_fastball_haa_folds.png"

BG     = "#0E1117"
PANEL  = "#171D27"
PANEL2 = "#202838"
GRID   = "#344055"
TXT    = "#F7F2E8"
MUTED  = "#BFC7D5"
MAROON = "#990F14"
GOLD   = "#C8A45D"

FOLD_COLORS = ["#4FA3FF", "#35C46B", "#F59E0B"]   # low / mid / high HAA
FOLD_LABELS = ["Low HAA\n(Flattest)", "Mid HAA", "High HAA\n(Steepest)"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_beau() -> pd.DataFrame:
    keep = {
        "Pitcher", "PitcherTeam", "PitcherThrows", "TaggedPitchType", "PitchCall",
        "BatterSide", "HorzApprAngle", "VertApprAngle", "RelSide", "Extension",
        "PlateLocSide", "PlateLocHeight", "RelSpeed", "RelHeight", "InducedVertBreak",
        "SpinRate", "Balls", "Strikes", "Date",
    }
    parts: list[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in keep, low_memory=False)
        except Exception:
            continue
        if not {"Pitcher", "PitcherTeam"}.issubset(df.columns):
            continue
        mask = (
            df["PitcherTeam"].astype(str).str.upper().eq("FOR_RAM")
            & df["Pitcher"].astype(str).eq("Elson, Beau")
        )
        if mask.any():
            parts.append(df.loc[mask].copy())
    if not parts:
        raise SystemExit("No data found for Elson, Beau / FOR_RAM")

    df = pd.concat(parts, ignore_index=True)

    # Keep only fastballs
    pt = df.get("TaggedPitchType", pd.Series("", index=df.index)).fillna("").astype(str)
    fb_mask = pt.str.contains("Fastball", case=False) | pt.str.upper().isin({"FB", "FF"})
    df = df.loc[fb_mask].copy()

    for col in ["HorzApprAngle", "VertApprAngle", "RelSide", "RelHeight", "Extension",
                "PlateLocSide", "PlateLocHeight", "RelSpeed", "InducedVertBreak", "SpinRate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    call = df.get("PitchCall", pd.Series("", index=df.index)).fillna("").astype(str)
    df["Swing"]  = call.isin(["StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                               "InPlay", "InPlayNoOut", "InPlayOut"])
    df["Whiff"]  = call.eq("StrikeSwinging")
    df["CSW"]    = call.isin(["StrikeCalled", "StrikeSwinging"])
    df["Strike"] = call.isin(["StrikeCalled", "StrikeSwinging", "FoulBall",
                               "FoulBallNotFieldable", "FoulBallFieldable",
                               "InPlay", "InPlayNoOut", "InPlayOut"])
    df["InZone"] = (
        df["PlateLocSide"].between(-0.83, 0.83) &
        df["PlateLocHeight"].between(1.5, 3.5)
    )
    return df.dropna(subset=["HorzApprAngle"])


# ── Outcome metrics per group ─────────────────────────────────────────────────

def _metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {}
    swings = sub["Swing"].sum()
    return {
        "N":       n,
        "Velo":    sub["RelSpeed"].mean(),
        "HAA":     sub["HorzApprAngle"].mean(),
        "VAA":     sub["VertApprAngle"].mean() if "VertApprAngle" in sub.columns else np.nan,
        "Whiff%":  sub["Whiff"].sum() / swings * 100 if swings >= 1 else 0,
        "CSW%":    sub["CSW"].mean() * 100,
        "Strike%": sub["Strike"].mean() * 100,
        "Zone%":   sub["InZone"].mean() * 100,
    }


# ── Strike-zone heatmap helper ────────────────────────────────────────────────

def _zone_scatter(ax, sub: pd.DataFrame, side: str, color: str, title: str):
    """Scatter of pitches in the strike zone for a given batter side."""
    ax.set_facecolor(PANEL)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1.0, 4.0)
    ax.set_aspect("equal")
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # Strike zone box
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.83, 1.5), 1.66, 2.0,
        boxstyle="square,pad=0", edgecolor=GOLD, facecolor="none",
        linewidth=1.5, linestyle="--", zorder=2
    ))

    mask = sub["BatterSide"].astype(str).str.upper().str.startswith(side[0].upper())
    pts  = sub[mask].dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not pts.empty:
        ax.scatter(pts["PlateLocSide"], pts["PlateLocHeight"],
                   c=color, alpha=0.45, s=18, edgecolors="none", zorder=3)
        # Density center
        ax.scatter([pts["PlateLocSide"].mean()], [pts["PlateLocHeight"].mean()],
                   c=color, s=90, edgecolors="white", linewidths=1.4, zorder=5, marker="D")

    ax.axvline(0, color=GRID, linewidth=0.6, linestyle=":")
    ax.text(0.5, 1.02, f"vs {side.upper()}", transform=ax.transAxes,
            color=MUTED, fontsize=7.5, ha="center", va="bottom", fontweight="bold")
    ax.set_title(title, color=TXT, fontsize=8, fontweight="bold", pad=3)
    ax.set_xlabel("PlateLocSide (ft)", color=MUTED, fontsize=6.5)
    ax.set_ylabel("Height (ft)", color=MUTED, fontsize=6.5)


# ── Bar chart helper ──────────────────────────────────────────────────────────

def _metric_bars(ax, fold_metrics: list[dict], metric: str, label: str):
    vals   = [m.get(metric, 0) or 0 for m in fold_metrics]
    colors = FOLD_COLORS
    bars   = ax.bar(range(3), vals, color=colors, width=0.6, edgecolor=GRID, linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", color=TXT, fontsize=8, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f.split("\n")[0] for f in FOLD_LABELS], color=MUTED, fontsize=7.5)
    ax.set_ylabel(label, color=MUTED, fontsize=7.5)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.yaxis.set_tick_params(labelcolor=MUTED)
    ax.set_ylim(0, max(vals) * 1.22 + 1)


# ── Main build ────────────────────────────────────────────────────────────────

def build(df: pd.DataFrame) -> plt.Figure:
    # Assign HAA tercile folds
    df = df.copy()
    df["HAA_fold"] = pd.qcut(df["HorzApprAngle"], q=3, labels=[0, 1, 2]).astype(int)

    folds        = [df[df["HAA_fold"] == i] for i in range(3)]
    fold_metrics = [_metrics(f) for f in folds]

    # Fold HAA ranges for labels
    ranges = [
        f"{f['HorzApprAngle'].min():.1f}° to {f['HorzApprAngle'].max():.1f}°"
        for f in folds
    ]

    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(BG)

    # ── Header ────────────────────────────────────────────────────────────────
    ax_hdr = fig.add_axes([0, 0.93, 1, 0.07])
    ax_hdr.set_facecolor(MAROON); ax_hdr.axis("off")
    ax_hdr.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 0.06, transform=ax_hdr.transAxes,
        facecolor=GOLD, edgecolor="none", alpha=0.9
    ))
    ax_hdr.text(0.015, 0.72, "Elson, Beau  —  Fastball HAA Fold Analysis",
                color="#FFF7E8", fontsize=24, fontweight="bold",
                transform=ax_hdr.transAxes, va="center")
    ax_hdr.text(0.015, 0.20,
                "Fastballs split into three equal HAA terciles  ·  "
                "Low = flattest horizontal approach  ·  High = steepest",
                color=MAROON, fontsize=11, fontweight="bold",
                transform=ax_hdr.transAxes, va="center")

    # ── Layout: 3 folds across, each fold = location (LHH+RHH) + metrics ─────
    gs = gridspec.GridSpec(
        3, 7,
        figure=fig,
        left=0.04, right=0.98, top=0.91, bottom=0.06,
        hspace=0.48, wspace=0.35,
        width_ratios=[1, 1, 0.08, 1, 1, 0.08, 1],   # lhh, rhh, gap, lhh, rhh, gap, metrics col
    )

    # Simpler layout: 3 rows (one per fold), 4 cols per row
    gs2 = gridspec.GridSpec(
        3, 5,
        figure=fig,
        left=0.04, right=0.98, top=0.91, bottom=0.06,
        hspace=0.52, wspace=0.30,
        width_ratios=[1, 1, 0.06, 1, 2],
    )
    # Clear the first gs (used only to plan; we'll use gs2)
    # Actually just use gs2 directly.

    for row_i, (fold, fm, fc, fl, rng) in enumerate(
        zip(folds, fold_metrics, FOLD_COLORS, FOLD_LABELS, ranges)
    ):
        # Location vs LHH
        ax_lhh = fig.add_subplot(gs2[row_i, 0])
        _zone_scatter(ax_lhh, fold, "Left", fc,
                      f"{fl.split(chr(10))[0]}  ·  {rng}")

        # Location vs RHH
        ax_rhh = fig.add_subplot(gs2[row_i, 1])
        _zone_scatter(ax_rhh, fold, "Right", fc, "")

        # Fold summary metrics panel
        ax_stats = fig.add_subplot(gs2[row_i, 3])
        ax_stats.set_facecolor(PANEL2); ax_stats.axis("off")
        ax_stats.set_xlim(0, 1); ax_stats.set_ylim(0, 1)

        stat_rows = [
            ("Pitches",  f"{fm.get('N', 0)}"),
            ("Avg Velo", f"{fm.get('Velo', 0):.1f} mph"),
            ("Avg HAA",  f"{fm.get('HAA', 0):.2f}°"),
            ("Avg VAA",  f"{fm.get('VAA', float('nan')):.2f}°" if pd.notna(fm.get("VAA")) else "—"),
            ("Strike%",  f"{fm.get('Strike%', 0):.1f}%"),
            ("Zone%",    f"{fm.get('Zone%', 0):.1f}%"),
            ("CSW%",     f"{fm.get('CSW%', 0):.1f}%"),
            ("Whiff%",   f"{fm.get('Whiff%', 0):.1f}%"),
        ]
        ax_stats.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.98,
            boxstyle="round,pad=0.01", facecolor=PANEL, edgecolor=fc,
            linewidth=2.5
        ))
        ax_stats.text(0.5, 0.94, fl.replace("\n", "  ·  "),
                      ha="center", va="top", color=fc,
                      fontsize=11, fontweight="bold")
        ax_stats.text(0.5, 0.86, rng,
                      ha="center", va="top", color=MUTED, fontsize=8.5)
        for j, (lbl, val) in enumerate(stat_rows):
            y = 0.76 - j * 0.098
            ax_stats.text(0.12, y, lbl, va="center", color=MUTED, fontsize=9)
            ax_stats.text(0.88, y, val, va="center", ha="right", color=TXT,
                          fontsize=9, fontweight="bold")

    # ── Right column: comparison bar charts ───────────────────────────────────
    bar_gs = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=gs2[:, 4], hspace=0.55
    )
    metrics_to_plot = [
        ("Whiff%",   "Whiff %"),
        ("CSW%",     "CSW %"),
        ("Strike%",  "Strike %"),
        ("Zone%",    "Zone %"),
    ]
    for k, (metric, label) in enumerate(metrics_to_plot):
        ax_bar = fig.add_subplot(bar_gs[k])
        _metric_bars(ax_bar, fold_metrics, metric, label)
        ax_bar.set_title(label, color=TXT, fontsize=9, fontweight="bold", pad=3)

    # Legend
    handles = [mpatches.Patch(color=c, label=f"{l.split(chr(10))[0]}  {r}")
               for c, l, r in zip(FOLD_COLORS, FOLD_LABELS, ranges)]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               facecolor=PANEL, edgecolor=GRID, labelcolor=TXT,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))

    return fig


if __name__ == "__main__":
    print("Loading Beau Elson fastball data…")
    df = _load_beau()
    print(f"  {len(df)} fastballs found.")

    if df.empty:
        print("No data — exiting.")
    else:
        fig = build(df)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"Saved → {OUT_PATH}")
