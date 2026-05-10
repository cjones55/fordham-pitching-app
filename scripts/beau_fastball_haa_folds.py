"""
Beau Elson — Fastball HAA Fold Analysis (vs RHH only)
Splits fastballs into HAA terciles and explains where each fold locates
relative to the 1B / 3B sides and how effective each is vs right-handed hitters.

Run from the repo root:  python scripts/beau_fastball_haa_folds.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "personal_reports" / "beau_elson_fastball_haa_folds.png"

BG     = "#0E1117"
PANEL  = "#171D27"
PANEL2 = "#1C2435"
GRID   = "#2E3D55"
TXT    = "#F7F2E8"
MUTED  = "#9BAABF"
MAROON = "#8C1515"
GOLD   = "#C8A45D"

# Low / Mid / High HAA fold colours
FOLD_COLORS  = ["#4FA3FF", "#35C46B", "#F04444"]
# Strategic labels per fold (for LHP vs RHH)
FOLD_NAMES   = ["Low HAA", "Mid HAA", "High HAA"]
FOLD_LOCS    = ["Outside (1B side)", "Middle", "Inside (3B side)"]
FOLD_READS   = [
    "Ball stays 1B-side / away from RHH\n→ Most whiffs, highest CSW",
    "Ball works middle of plate\n→ Moderate outcomes",
    "Ball runs inside (3B-side) to RHH\n→ Harder to generate whiffs",
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_beau_rhh() -> pd.DataFrame:
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

    # Fastballs only
    pt      = df.get("TaggedPitchType", pd.Series("", index=df.index)).fillna("").astype(str)
    fb_mask = pt.str.contains("Fastball", case=False) | pt.str.upper().isin({"FB", "FF"})
    df      = df.loc[fb_mask].copy()

    # RHH only
    df = df[df.get("BatterSide", pd.Series("", index=df.index)).astype(str)
              .str.upper().str.startswith("R")].copy()

    for col in ["HorzApprAngle", "VertApprAngle", "RelSide", "RelHeight", "Extension",
                "PlateLocSide", "PlateLocHeight", "RelSpeed", "InducedVertBreak", "SpinRate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    call         = df.get("PitchCall", pd.Series("", index=df.index)).fillna("").astype(str)
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


def _metrics(sub: pd.DataFrame) -> dict:
    n      = len(sub)
    swings = sub["Swing"].sum()
    return {
        "N":       n,
        "Velo":    sub["RelSpeed"].mean(),
        "HAA":     sub["HorzApprAngle"].mean(),
        "VAA":     sub["VertApprAngle"].mean() if "VertApprAngle" in sub.columns else np.nan,
        "IVB":     sub["InducedVertBreak"].mean() if "InducedVertBreak" in sub.columns else np.nan,
        "RelSide": sub["RelSide"].mean() if "RelSide" in sub.columns else np.nan,
        "Whiff%":  sub["Whiff"].sum() / swings * 100 if swings >= 1 else 0.0,
        "CSW%":    sub["CSW"].mean() * 100,
        "Strike%": sub["Strike"].mean() * 100,
        "Zone%":   sub["InZone"].mean() * 100,
        "LocSide": sub["PlateLocSide"].mean(),
    }


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_zone(ax, sub: pd.DataFrame, color: str, fold_name: str, haa_range: str):
    """Strike-zone scatter with 1B / 3B side labels."""
    ax.set_facecolor(PANEL)
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(0.95, 4.1)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=6.5)

    # Strike zone
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.83, 1.5), 1.66, 2.0,
        boxstyle="square,pad=0", edgecolor=GOLD,
        facecolor="none", linewidth=1.8, linestyle="--", zorder=3
    ))

    pts = sub.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if not pts.empty:
        ax.scatter(pts["PlateLocSide"], pts["PlateLocHeight"],
                   c=color, alpha=0.40, s=16, edgecolors="none", zorder=4)
        mx, my = pts["PlateLocSide"].mean(), pts["PlateLocHeight"].mean()
        ax.scatter([mx], [my], c=color, s=110, edgecolors="white",
                   linewidths=1.6, zorder=6, marker="D")
        ax.annotate(f"avg\n{mx:+.2f}", (mx, my),
                    xytext=(mx + (0.45 if mx < 0 else -0.45), my + 0.45),
                    fontsize=6.5, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9))

    # Side labels
    ax.text(-1.60, 0.98, "← 1B SIDE\n(Outside)", color="#4FA3FF",
            fontsize=6.5, fontweight="bold", va="bottom", ha="left")
    ax.text(1.60, 0.98, "3B SIDE →\n(Inside)", color="#F04444",
            fontsize=6.5, fontweight="bold", va="bottom", ha="right")

    ax.axvline(0, color=GRID, linewidth=0.7, linestyle=":", zorder=2)
    ax.set_xlabel("PlateLocSide (ft)  ←1B  |  3B→", color=MUTED, fontsize=6.5)
    ax.set_ylabel("Height (ft)", color=MUTED, fontsize=6.5)
    ax.set_title(f"{fold_name}  ·  {haa_range}", color=color,
                 fontsize=9, fontweight="bold", pad=4)


def _bar_compare(ax, fold_metrics: list[dict], metric: str, title: str, fmt="{:.1f}"):
    vals   = [m.get(metric, 0) or 0 for m in fold_metrics]
    bars   = ax.bar(range(3), vals, color=FOLD_COLORS,
                    width=0.55, edgecolor=GRID, linewidth=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                fmt.format(v), ha="center", va="bottom",
                color=TXT, fontsize=9, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(FOLD_NAMES, color=MUTED, fontsize=8)
    ax.set_facecolor(PANEL2)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.set_ylim(0, max(vals) * 1.30 + 1)
    ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=5)
    ax.yaxis.set_tick_params(labelcolor=MUTED)
    ax.set_ylabel("%", color=MUTED, fontsize=8)


# ── Main build ────────────────────────────────────────────────────────────────

def build(df: pd.DataFrame) -> plt.Figure:
    df = df.copy()
    df["HAA_fold"] = pd.qcut(df["HorzApprAngle"], q=3, labels=[0, 1, 2]).astype(int)

    folds        = [df[df["HAA_fold"] == i] for i in range(3)]
    fold_metrics = [_metrics(f) for f in folds]
    haa_ranges   = [
        f"HAA  {f['HorzApprAngle'].min():.1f}° – {f['HorzApprAngle'].max():.1f}°"
        for f in folds
    ]

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)

    # ── Header ────────────────────────────────────────────────────────────────
    ax_hdr = fig.add_axes([0, 0.928, 1, 0.072])
    ax_hdr.set_facecolor(MAROON); ax_hdr.axis("off")
    ax_hdr.add_patch(plt.Rectangle((0, 0), 1, 0.07, transform=ax_hdr.transAxes,
                                   facecolor=GOLD, edgecolor="none", alpha=0.9))
    ax_hdr.text(0.012, 0.73,
                "Elson, Beau  (LHP)  —  Fastball HAA Fold Analysis  ·  vs RHH Only",
                color="#FFF7E8", fontsize=22, fontweight="bold",
                transform=ax_hdr.transAxes, va="center")
    ax_hdr.text(0.012, 0.20,
                "Fastballs split into 3 equal HAA terciles  ·  "
                "Positive HAA = ball arriving from 3B side (inside to RHH)  ·  "
                "Negative/low HAA = ball arrives from 1B side (outside to RHH)  ·  "
                "Low HAA most effective",
                color=MAROON, fontsize=10, fontweight="bold",
                transform=ax_hdr.transAxes, va="center")

    # ── Layout ────────────────────────────────────────────────────────────────
    # Top 2/3: one column per fold (zone scatter + stat card)
    # Bottom 1/3: comparison bar charts across folds
    top_gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.03, right=0.98, top=0.915, bottom=0.38,
        wspace=0.22,
    )
    bot_gs = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.05, right=0.98, top=0.32, bottom=0.05,
        wspace=0.30,
    )

    # ── Top: one panel per fold ───────────────────────────────────────────────
    for col_i, (fold, fm, fc, fn, fr, fl, fread) in enumerate(
        zip(folds, fold_metrics, FOLD_COLORS, FOLD_NAMES,
            haa_ranges, FOLD_LOCS, FOLD_READS)
    ):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=top_gs[col_i],
            height_ratios=[2.5, 1], hspace=0.12
        )

        # Zone scatter
        ax_zone = fig.add_subplot(inner[0])
        _draw_zone(ax_zone, fold, fc, fn, fr)

        # Stat card below the zone
        ax_card = fig.add_subplot(inner[1])
        ax_card.set_facecolor(PANEL2); ax_card.axis("off")
        ax_card.set_xlim(0, 1); ax_card.set_ylim(0, 1)

        ax_card.add_patch(mpatches.FancyBboxPatch(
            (0.01, 0.02), 0.98, 0.96,
            boxstyle="round,pad=0.01",
            facecolor=PANEL, edgecolor=fc, linewidth=2.2
        ))

        # Location label banner
        ax_card.add_patch(plt.Rectangle(
            (0.01, 0.78), 0.98, 0.20, facecolor=fc, alpha=0.18
        ))
        ax_card.text(0.50, 0.875, fl, ha="center", va="center",
                     color=fc, fontsize=10, fontweight="bold")

        rel = fm.get("RelSide")
        rel_str = f"{rel:+.2f} ft" if rel is not None and not np.isnan(rel) else "—"
        stat_rows = [
            ("N (vs RHH)",   f"{fm.get('N', 0)}"),
            ("Avg HAA",      f"{fm.get('HAA', 0):.2f}°"),
            ("Avg Velo",     f"{fm.get('Velo', 0):.1f} mph"),
            ("Avg Rel Side", rel_str),
            ("Avg Location", f"{fm.get('LocSide', 0):+.2f} ft"),
            ("Whiff%",       f"{fm.get('Whiff%', 0):.1f}%"),
            ("CSW%",         f"{fm.get('CSW%', 0):.1f}%"),
            ("Zone%",        f"{fm.get('Zone%', 0):.1f}%"),
        ]
        for j, (lbl, val) in enumerate(stat_rows):
            y = 0.72 - j * 0.092
            ax_card.text(0.06, y, lbl, va="center",
                         color=MUTED, fontsize=8.2)
            ax_card.text(0.94, y, val, va="center", ha="right",
                         color=TXT, fontsize=8.2, fontweight="bold")

        # Strategic read
        ax_card.text(0.50, 0.025, fread, ha="center", va="bottom",
                     color=GOLD, fontsize=7.5, fontstyle="italic",
                     wrap=True)

    # ── Bottom: comparison bars ───────────────────────────────────────────────
    metrics_plot = [
        ("Whiff%",   "Whiff %"),
        ("CSW%",     "CSW %"),
        ("Strike%",  "Strike %"),
        ("Zone%",    "Zone %"),
    ]
    for k, (metric, title) in enumerate(metrics_plot):
        ax_bar = fig.add_subplot(bot_gs[k])
        _bar_compare(ax_bar, fold_metrics, metric, title)

    # ── Legend strip ─────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=c, label=f"{n}  ·  {l}  ·  {r}")
        for c, n, l, r in zip(FOLD_COLORS, FOLD_NAMES, FOLD_LOCS, haa_ranges)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               facecolor=PANEL, edgecolor=GRID, labelcolor=TXT,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.003), framealpha=0.9)

    # Side labels in bottom margin
    fig.text(0.03, 0.345, "← 1B Side / Outside to RHH", color="#4FA3FF",
             fontsize=9, fontweight="bold", va="top")
    fig.text(0.72, 0.345, "3B Side / Inside to RHH →", color="#F04444",
             fontsize=9, fontweight="bold", va="top")

    return fig


if __name__ == "__main__":
    print("Loading Beau Elson fastball data (RHH only)…")
    df = _load_beau_rhh()
    print(f"  {len(df)} fastballs vs RHH loaded.")
    fig = build(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")
