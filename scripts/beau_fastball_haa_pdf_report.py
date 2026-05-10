"""
beau_fastball_haa_pdf_report.py
================================
Generates an enhanced, standalone multi-page PDF for Beau Elson's
fastball Horizontal Approach Angle (HAA) analysis.

Fully self-contained — no imports from other local scripts.
Output: personal_reports/beau_elson_fastball_haa_detailed_report.pdf
"""

import os
import glob
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# REPO ROOT & PATHS
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "data")
OUT_DIR   = os.path.join(REPO_ROOT, "personal_reports")
OUT_PDF   = os.path.join(OUT_DIR, "beau_elson_fastball_haa_detailed_report.pdf")
FOLDS_PNG = os.path.join(OUT_DIR, "beau_elson_fastball_haa_folds.png")

LOGO_PATHS = [
    os.path.join(REPO_ROOT, "national_pitchingplus_app", "team_logos", "FOR_RAM.png"),
    os.path.join(REPO_ROOT, "static", "rams.png"),
]

# ---------------------------------------------------------------------------
# COLOR SCHEME
# ---------------------------------------------------------------------------
BG     = "#0E1117"
PANEL  = "#171D27"
PANEL2 = "#202838"
GRID   = "#344055"
TXT    = "#F7F2E8"
MUTED  = "#9BAABF"
MAROON = "#8C1515"
GOLD   = "#C8A45D"
FOLD_COLORS = ["#4FA3FF", "#35C46B", "#F04444"]   # Low / Mid / High
WHITE  = "#FFF7E8"

PAGE_W, PAGE_H = 11, 8.5   # landscape

# ---------------------------------------------------------------------------
# HELPER — figure background
# ---------------------------------------------------------------------------
def make_fig(facecolor=BG):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=facecolor)
    return fig


def dark_axes(ax, facecolor=PANEL):
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TXT)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.6)


def draw_rounded_rect(ax, x, y, w, h, radius=0.02, facecolor=PANEL2,
                      edgecolor=GRID, linewidth=1.0, transform=None):
    tr = transform if transform is not None else ax.transAxes
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, transform=tr, clip_on=False,
        zorder=3,
    )
    ax.add_patch(box)
    return box


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
FASTBALL_TAGS = {"Fastball", "FourSeamFastBall"}

def load_data():
    """Load all CSV files, return (all_beau_fb, rhh_fb_with_haa_fold)."""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  [WARN] Could not read {os.path.basename(f)}: {e}")

    df = pd.concat(dfs, ignore_index=True)

    # Filter Beau Elson fastballs
    beau_mask = df["Pitcher"].str.contains("Elson", na=False)
    fb_mask   = df["TaggedPitchType"].isin(FASTBALL_TAGS)
    all_fb    = df[beau_mask & fb_mask].copy()

    # vs RHH with valid HAA
    rhh = all_fb[
        (all_fb["BatterSide"] == "Right") &
        all_fb["HorzApprAngle"].notna()
    ].copy()

    if len(rhh) < 10:
        raise ValueError(f"Only {len(rhh)} RHH fastball rows with HAA — check data.")

    # HAA tercile folds
    rhh["haa_fold"] = pd.qcut(rhh["HorzApprAngle"], q=3, labels=["Low", "Mid", "High"])

    return all_fb, rhh


# ---------------------------------------------------------------------------
# METRIC HELPERS
# ---------------------------------------------------------------------------
def whiff_pct(g):
    swings  = g["PitchCall"].isin(["StrikeSwinging", "FoulBallNotFieldable",
                                    "FoulBallFieldable", "InPlay"])
    whiffs  = g["PitchCall"] == "StrikeSwinging"
    return whiffs.sum() / swings.sum() * 100 if swings.sum() > 0 else 0.0


def csw_pct(g):
    csw = g["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])
    return csw.sum() / len(g) * 100 if len(g) > 0 else 0.0


def zone_pct(g):
    in_zone = (
        g["PlateLocSide"].between(-0.831, 0.831) &
        g["PlateLocHeight"].between(1.5, 3.5)
    )
    return in_zone.sum() / len(g) * 100 if len(g) > 0 else 0.0


def build_fold_stats(rhh):
    rows = []
    for fold, color in zip(["Low", "Mid", "High"], FOLD_COLORS):
        g = rhh[rhh["haa_fold"] == fold]
        rows.append({
            "fold":       fold,
            "color":      color,
            "n":          len(g),
            "avg_haa":    g["HorzApprAngle"].mean(),
            "avg_rside":  g["RelSide"].mean(),
            "avg_plside": g["PlateLocSide"].mean(),
            "whiff":      whiff_pct(g),
            "csw":        csw_pct(g),
            "zone":       zone_pct(g),
            "data":       g,
        })
    return rows


# ---------------------------------------------------------------------------
# LOGO LOADER
# ---------------------------------------------------------------------------
def load_logo():
    for p in LOGO_PATHS:
        if os.path.exists(p):
            try:
                img = plt.imread(p)
                return img
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# SHARED HEADER / FOOTER
# ---------------------------------------------------------------------------
def draw_header_bar(fig, title_left="Beau Elson · LHP",
                    title_right="Fordham Baseball Analytics · 2026",
                    bar_height=0.10):
    """Draw a maroon header bar across the top of fig (figure coordinates)."""
    ax_h = fig.add_axes([0, 1 - bar_height, 1, bar_height])
    ax_h.set_facecolor(MAROON)
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, 1)
    ax_h.axis("off")

    ax_h.text(0.015, 0.50, title_left,
              color=WHITE, fontsize=16, fontweight="bold",
              va="center", ha="left", transform=ax_h.transAxes)
    ax_h.text(0.985, 0.50, title_right,
              color=GOLD, fontsize=10, va="center", ha="right",
              transform=ax_h.transAxes)

    logo = load_logo()
    if logo is not None:
        ax_logo = fig.add_axes([0.88, 1 - bar_height * 0.95, 0.09, bar_height * 0.90])
        ax_logo.imshow(logo)
        ax_logo.axis("off")

    return ax_h


def draw_footer(fig, text="Generated from TrackMan data · Fordham Baseball Analytics · 2026"):
    ax_f = fig.add_axes([0, 0, 1, 0.035])
    ax_f.set_facecolor(PANEL2)
    ax_f.set_xlim(0, 1); ax_f.set_ylim(0, 1)
    ax_f.axis("off")
    ax_f.text(0.5, 0.50, text, color=MUTED, fontsize=7,
              va="center", ha="center", transform=ax_f.transAxes,
              style="italic")


# ---------------------------------------------------------------------------
# PAGE 1 — Cover + Executive Summary
# ---------------------------------------------------------------------------
def page_cover(pdf, fold_stats, rhh):
    fig = make_fig()

    # --- header ---
    draw_header_bar(fig,
                    title_left="Beau Elson · LHP  |  Fastball HAA Analysis",
                    title_right="Fordham Baseball Analytics · 2026",
                    bar_height=0.11)

    draw_footer(fig)

    # main content area: y=0.04 to y=0.89
    CONTENT_TOP    = 0.89
    CONTENT_BOTTOM = 0.04
    CONTENT_LEFT   = 0.02
    CONTENT_RIGHT  = 0.98

    # ---- overall stat tiles (3 tiles across top of content) ----
    overall_whiff = whiff_pct(rhh)
    overall_csw   = csw_pct(rhh)
    overall_zone  = zone_pct(rhh)

    tile_labels = ["Whiff%  (vs RHH)", "CSW%  (vs RHH)", "Zone%  (vs RHH)"]
    tile_values = [f"{overall_whiff:.1f}%", f"{overall_csw:.1f}%", f"{overall_zone:.1f}%"]
    tile_colors = [GOLD, GOLD, GOLD]

    tile_y_top  = CONTENT_TOP
    tile_height = 0.16
    tile_y_bot  = tile_y_top - tile_height
    tile_w      = 0.18
    tile_gap    = 0.02
    tile_x_start = CONTENT_LEFT + 0.01

    ax_tile = fig.add_axes([0, 0, 1, 1])
    ax_tile.set_xlim(0, 1); ax_tile.set_ylim(0, 1)
    ax_tile.axis("off")

    for i, (lbl, val, col) in enumerate(zip(tile_labels, tile_values, tile_colors)):
        tx = tile_x_start + i * (tile_w + tile_gap)
        draw_rounded_rect(ax_tile, tx, tile_y_bot, tile_w, tile_height,
                          radius=0.015, facecolor=PANEL2, edgecolor=GRID,
                          linewidth=1.2, transform=ax_tile.transAxes)
        ax_tile.text(tx + tile_w / 2, tile_y_bot + tile_height * 0.70, val,
                     color=col, fontsize=22, fontweight="bold",
                     ha="center", va="center", transform=ax_tile.transAxes)
        ax_tile.text(tx + tile_w / 2, tile_y_bot + tile_height * 0.22, lbl,
                     color=MUTED, fontsize=8, ha="center", va="center",
                     transform=ax_tile.transAxes)

    # N label
    ax_tile.text(tile_x_start + 3 * (tile_w + tile_gap) + 0.01,
                 tile_y_bot + tile_height * 0.50,
                 f"n = {len(rhh)} pitches\nvs RHH",
                 color=MUTED, fontsize=8, va="center", ha="left",
                 transform=ax_tile.transAxes)

    # ---- HAA fold mini-bar chart (right side of tiles row) ----
    bar_ax = fig.add_axes([0.68, tile_y_bot, 0.29, tile_height + 0.02])
    bar_ax.set_facecolor(PANEL2)
    for sp in bar_ax.spines.values(): sp.set_edgecolor(GRID)
    bar_ax.tick_params(colors=MUTED, labelsize=7.5)

    folds  = [s["fold"] for s in fold_stats]
    whiffs = [s["whiff"] for s in fold_stats]
    cols   = [s["color"] for s in fold_stats]

    bars = bar_ax.bar(folds, whiffs, color=cols, width=0.5, zorder=3)
    bar_ax.set_facecolor(PANEL2)
    bar_ax.set_ylim(0, max(whiffs) * 1.35)
    bar_ax.set_ylabel("Whiff%", color=MUTED, fontsize=7)
    bar_ax.set_title("Whiff% by HAA Fold", color=TXT, fontsize=8, pad=3)
    bar_ax.grid(axis="y", color=GRID, linewidth=0.4, alpha=0.6)
    bar_ax.set_facecolor(PANEL2)
    for bar, val in zip(bars, whiffs):
        bar_ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{val:.1f}%",
                    ha="center", va="bottom", color=WHITE, fontsize=8, fontweight="bold")

    # ---- Key Finding panel ----
    kf_top = tile_y_bot - 0.025
    kf_h   = 0.29
    kf_bot = kf_top - kf_h
    kf_x   = CONTENT_LEFT
    kf_w   = 0.55

    draw_rounded_rect(ax_tile, kf_x, kf_bot, kf_w, kf_h,
                      radius=0.015, facecolor=PANEL, edgecolor=MAROON,
                      linewidth=1.8, transform=ax_tile.transAxes)

    ax_tile.text(kf_x + 0.012, kf_bot + kf_h - 0.028,
                 "KEY FINDING",
                 color=MAROON, fontsize=9, fontweight="bold",
                 va="top", ha="left", transform=ax_tile.transAxes)

    finding_lines = [
        "Low HAA (≈1.08°) generates  25.9% Whiff%  — 3× the rate of High HAA (8.5%).",
        "",
        "When Beau's fastball exits the hand at a lower horizontal approach angle,",
        "it arrives from the 1B side and finishes  outside  to right-handed hitters.",
        "That deceptive approach angle makes it extremely difficult to square up,",
        "driving the dramatic difference in swing-and-miss rate across folds.",
        "",
        "HAA and PlateLocSide share a  0.90 correlation — the approach angle",
        "dictates where the ball ends up, not the other way around.",
    ]

    for i, line in enumerate(finding_lines):
        color = WHITE if line.strip() else MUTED
        fontsize = 8.5
        ax_tile.text(kf_x + 0.012,
                     kf_bot + kf_h - 0.065 - i * 0.028,
                     line, color=color, fontsize=fontsize,
                     va="top", ha="left", transform=ax_tile.transAxes)

    # ---- Fold color legend inside key finding panel ----
    legend_y = kf_bot + 0.022
    for i, (s, col) in enumerate(zip(fold_stats, FOLD_COLORS)):
        lx = kf_x + 0.012 + i * 0.16
        ax_tile.add_patch(Rectangle((lx, legend_y), 0.015, 0.018,
                                    facecolor=col, transform=ax_tile.transAxes,
                                    zorder=5, clip_on=False))
        ax_tile.text(lx + 0.020, legend_y + 0.009,
                     f"{s['fold']} HAA  ({s['whiff']:.1f}% Whiff)",
                     color=MUTED, fontsize=7.5, va="center",
                     transform=ax_tile.transAxes)

    # ---- Target Release Point panel ----
    tp_x   = kf_x + kf_w + 0.02
    tp_w   = CONTENT_RIGHT - tp_x
    tp_h   = kf_h
    tp_bot = kf_bot

    draw_rounded_rect(ax_tile, tp_x, tp_bot, tp_w, tp_h,
                      radius=0.015, facecolor=PANEL, edgecolor=GOLD,
                      linewidth=1.8, transform=ax_tile.transAxes)

    ax_tile.text(tp_x + 0.012, tp_bot + tp_h - 0.028,
                 "TARGET RELEASE POINT",
                 color=GOLD, fontsize=9, fontweight="bold",
                 va="top", ha="left", transform=ax_tile.transAxes)

    overall_rel = rhh["RelSide"].mean()
    low_rel     = fold_stats[0]["avg_rside"]    # Low HAA fold
    delta_rel   = low_rel - overall_rel          # +0.10 ft ≈ 1.2 in

    target_lines = [
        f"Overall mean RelSide:      {overall_rel:.2f} ft",
        f"Low-HAA fold RelSide:      {low_rel:.2f} ft",
        f"Required shift:              Δ = {delta_rel:+.2f} ft  (~{abs(delta_rel)*12:.1f} in toward 1B/glove side)",
        "",
        "For a LHP, moving the release point ~1.2 inches",
        "toward the 1B / glove side naturally lowers HAA.",
        "The ball arrives from the 1B side and stays",
        "outside to RHH — the most effective approach angle.",
        "",
        "Expected outcome:",
        "  Whiff% from ~14.2% (overall) → 25.9% range",
        "",
        "Guardrail: Zone% drops to 41.5% in Low-HAA fold.",
        "Pair with location discipline or pitch-to-contact mix.",
    ]

    for i, line in enumerate(target_lines):
        is_highlight = "Required shift" in line or "Expected outcome" in line
        color = GOLD if is_highlight else WHITE if line.strip() else MUTED
        weight = "bold" if is_highlight else "normal"
        ax_tile.text(tp_x + 0.012,
                     tp_bot + tp_h - 0.065 - i * 0.028,
                     line, color=color, fontsize=8.2, fontweight=weight,
                     va="top", ha="left", transform=ax_tile.transAxes)

    # ---- Bottom fold summary strip ----
    strip_y = kf_bot - 0.025
    strip_h = 0.085
    strip_bot = strip_y - strip_h

    draw_rounded_rect(ax_tile, CONTENT_LEFT, strip_bot,
                      CONTENT_RIGHT - CONTENT_LEFT, strip_h,
                      radius=0.01, facecolor=PANEL2, edgecolor=GRID,
                      linewidth=0.8, transform=ax_tile.transAxes)

    cols_w = (CONTENT_RIGHT - CONTENT_LEFT) / len(fold_stats)
    headers = ["Fold", "N", "Avg HAA", "Avg RelSide", "Avg PlateLocSide",
               "Whiff%", "CSW%", "Zone%"]
    col_xs  = [0.00, 0.06, 0.13, 0.22, 0.35, 0.50, 0.62, 0.74]

    for xi, hdr in zip(col_xs, headers):
        ax_tile.text(CONTENT_LEFT + xi, strip_bot + strip_h * 0.82,
                     hdr, color=GOLD, fontsize=7, fontweight="bold",
                     va="center", transform=ax_tile.transAxes)

    for ri, s in enumerate(fold_stats):
        row_y = strip_bot + strip_h * 0.82 - (ri + 1) * (strip_h * 0.28)
        vals = [
            s["fold"],
            str(s["n"]),
            f"{s['avg_haa']:.2f}°",
            f"{s['avg_rside']:.2f} ft",
            f"{s['avg_plside']:+.2f} ft",
            f"{s['whiff']:.1f}%",
            f"{s['csw']:.1f}%",
            f"{s['zone']:.1f}%",
        ]
        for xi, val in zip(col_xs, vals):
            ax_tile.text(CONTENT_LEFT + xi, row_y, val,
                         color=s["color"] if xi == 0.00 else WHITE,
                         fontsize=7.5, va="center",
                         fontweight="bold" if xi == 0.00 else "normal",
                         transform=ax_tile.transAxes)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Page 1: Cover + Executive Summary")


# ---------------------------------------------------------------------------
# PAGE 2 — HAA Folds PNG embed
# ---------------------------------------------------------------------------
def page_folds_png(pdf):
    fig = make_fig()
    draw_header_bar(fig,
                    title_left="Beau Elson · HAA Fold Analysis",
                    title_right="Fordham Baseball Analytics · 2026",
                    bar_height=0.09)
    draw_footer(fig)

    ax_img = fig.add_axes([0.01, 0.07, 0.98, 0.83])
    ax_img.axis("off")
    ax_img.set_facecolor(BG)

    if os.path.exists(FOLDS_PNG):
        img = plt.imread(FOLDS_PNG)
        ax_img.imshow(img, aspect="auto")
        note = ("Three HAA tercile folds (Low / Mid / High) vs RHH  |  "
                "Low fold: HAA ≈ 1.08° → 25.9% Whiff%  ·  "
                "Mid fold: HAA ≈ 2.12° → 11.4% Whiff%  ·  "
                "High fold: HAA ≈ 3.09° → 8.5% Whiff%")
    else:
        ax_img.text(0.5, 0.5, "Folds PNG not found:\n" + FOLDS_PNG,
                    color=MUTED, fontsize=12, ha="center", va="center",
                    transform=ax_img.transAxes)
        note = "See personal_reports/beau_elson_fastball_haa_folds.png for fold visualization."

    # footer note
    ax_note = fig.add_axes([0.01, 0.035, 0.98, 0.04])
    ax_note.axis("off")
    ax_note.set_facecolor(PANEL2)
    ax_note.text(0.5, 0.5, note, color=MUTED, fontsize=7.5,
                 ha="center", va="center", style="italic",
                 transform=ax_note.transAxes)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Page 2: HAA Folds PNG")


# ---------------------------------------------------------------------------
# STRIKE ZONE HELPER
# ---------------------------------------------------------------------------
def draw_strike_zone(ax):
    """Draw a standard strike zone rectangle."""
    zone = Rectangle((-0.831, 1.5), 1.662, 2.0,
                     edgecolor=GOLD, facecolor="none",
                     linewidth=1.5, zorder=5)
    ax.add_patch(zone)
    # Draw inner thirds
    for x in [-0.277, 0.277]:
        ax.axvline(x, color=GRID, linewidth=0.5, alpha=0.5, zorder=4)
    for y in [2.167, 2.833]:
        ax.axhline(y, color=GRID, linewidth=0.5, alpha=0.5, zorder=4)


# ---------------------------------------------------------------------------
# PAGE 3 — Zone scatter plots + comparison table
# ---------------------------------------------------------------------------
def page_scatter_table(pdf, fold_stats, rhh):
    fig = make_fig()
    draw_header_bar(fig,
                    title_left="Beau Elson · Zone Location by HAA Fold  (vs RHH)",
                    title_right="Fordham Baseball Analytics · 2026",
                    bar_height=0.09)
    draw_footer(fig)

    gs = GridSpec(2, 3, figure=fig,
                  left=0.04, right=0.98,
                  top=0.87, bottom=0.37,
                  wspace=0.28, hspace=0.35)

    x_lim = (-2.5, 2.5)
    y_lim = (0.2, 5.2)

    for col_i, s in enumerate(fold_stats):
        ax = fig.add_subplot(gs[0, col_i])
        dark_axes(ax, facecolor=PANEL)

        g = s["data"]
        valid = g.dropna(subset=["PlateLocSide", "PlateLocHeight"])

        # scatter
        ax.scatter(valid["PlateLocSide"], valid["PlateLocHeight"],
                   c=s["color"], alpha=0.35, s=18, zorder=3, edgecolors="none")

        # mean diamond
        mx = valid["PlateLocSide"].mean()
        my = valid["PlateLocHeight"].mean()
        ax.scatter([mx], [my], marker="D", s=80, c=GOLD,
                   zorder=10, linewidths=0.8, edgecolors=WHITE)

        draw_strike_zone(ax)

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xlabel("PlateLocSide (ft)", fontsize=7, color=MUTED)
        ax.set_ylabel("PlateLocHeight (ft)", fontsize=7, color=MUTED)
        ax.set_title(f"{s['fold']} HAA  (n={s['n']})\nHAA ≈ {s['avg_haa']:.2f}°",
                     color=s["color"], fontsize=9, fontweight="bold")

        # 1B / 3B side labels
        ax.text(x_lim[0] + 0.08, y_lim[1] - 0.15, "← 1B",
                color=MUTED, fontsize=7, va="top")
        ax.text(x_lim[1] - 0.08, y_lim[1] - 0.15, "3B →",
                color=MUTED, fontsize=7, va="top", ha="right")

        # whiff annotation
        ax.text(0.50, 0.04,
                f"Whiff: {s['whiff']:.1f}%  |  CSW: {s['csw']:.1f}%  |  Zone: {s['zone']:.1f}%",
                color=WHITE, fontsize=7, ha="center", va="bottom",
                transform=ax.transAxes, fontweight="bold",
                bbox=dict(facecolor=PANEL2, edgecolor=s["color"],
                          linewidth=0.8, boxstyle="round,pad=0.3"))

        # Gold border for Low HAA (optimal)
        if s["fold"] == "Low":
            for spine in ax.spines.values():
                spine.set_edgecolor(GOLD)
                spine.set_linewidth(2.0)

    # --- axis label explanation ---
    ax_lbl = fig.add_axes([0.04, 0.85, 0.92, 0.02])
    ax_lbl.axis("off")
    ax_lbl.set_facecolor(BG)
    ax_lbl.text(0.5, 0.5,
                "← 1B Side (Outside to RHH)   |   PlateLocSide   |   3B Side (Inside to RHH) →",
                color=MUTED, fontsize=8, ha="center", va="center",
                transform=ax_lbl.transAxes)

    # ---- Comparison table ----
    table_top    = 0.34
    table_height = 0.22
    table_ax = fig.add_axes([0.04, table_top - table_height, 0.92, table_height + 0.01])
    table_ax.set_facecolor(PANEL2)
    table_ax.set_xlim(0, 1); table_ax.set_ylim(0, 1)
    table_ax.axis("off")

    table_ax.text(0.5, 0.96,
                  "Fold Comparison Table  (Fastball vs RHH)",
                  color=GOLD, fontsize=9, fontweight="bold",
                  ha="center", va="top", transform=table_ax.transAxes)

    col_names = ["Fold", "N", "Avg HAA", "RelSide (ft)", "PlateLocSide (ft)",
                 "Whiff%", "CSW%", "Zone%"]
    col_xs    = [0.04, 0.13, 0.22, 0.33, 0.46, 0.62, 0.73, 0.84]

    for xi, hdr in zip(col_xs, col_names):
        table_ax.text(xi, 0.82, hdr, color=GOLD, fontsize=8,
                      fontweight="bold", va="top", transform=table_ax.transAxes)

    # divider line
    table_ax.axhline(0.79, color=GRID, linewidth=0.8)

    for ri, s in enumerate(fold_stats):
        row_y = 0.62 - ri * 0.22
        vals  = [
            s["fold"],
            str(s["n"]),
            f"{s['avg_haa']:.2f}°",
            f"{s['avg_rside']:.2f}",
            f"{s['avg_plside']:+.2f}",
            f"{s['whiff']:.1f}%",
            f"{s['csw']:.1f}%",
            f"{s['zone']:.1f}%",
        ]
        for xi, val in zip(col_xs, vals):
            is_name = xi == col_xs[0]
            table_ax.text(xi, row_y, val,
                          color=s["color"] if is_name else WHITE,
                          fontsize=8.5, fontweight="bold" if is_name else "normal",
                          va="center", transform=table_ax.transAxes)

    # Gold "optimal" box around Low row
    draw_rounded_rect(table_ax, 0.01, 0.58, 0.97, 0.18,
                      radius=0.02, facecolor="none", edgecolor=GOLD,
                      linewidth=1.5, transform=table_ax.transAxes)
    table_ax.text(0.99, 0.67, "← OPTIMAL",
                  color=GOLD, fontsize=7.5, fontweight="bold",
                  va="center", ha="right", transform=table_ax.transAxes)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Page 3: Zone Scatter + Comparison Table")


# ---------------------------------------------------------------------------
# 9-ZONE HEATMAP HELPER
# ---------------------------------------------------------------------------
def build_9zone_heatmap(rhh):
    """Return a 3×3 array of Whiff% values (row=height, col=side)."""
    g = rhh.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    g["side_bin"]   = pd.cut(g["PlateLocSide"],
                              bins=[-np.inf, -0.277, 0.277, np.inf],
                              labels=["1B-side", "Middle", "3B-side"])
    g["height_bin"] = pd.cut(g["PlateLocHeight"],
                              bins=[-np.inf, 2.167, 2.833, np.inf],
                              labels=["Low", "Mid", "Up"])

    heatmap = np.full((3, 3), np.nan)
    side_order   = ["1B-side", "Middle", "3B-side"]
    height_order = ["Up", "Mid", "Low"]   # top row = Up

    for ri, ht in enumerate(height_order):
        for ci, sd in enumerate(side_order):
            cell = g[(g["height_bin"] == ht) & (g["side_bin"] == sd)]
            if len(cell) >= 3:
                heatmap[ri, ci] = whiff_pct(cell)

    return heatmap


def draw_9zone_heatmap(ax, heatmap):
    """Draw a 3×3 heatmap of Whiff% on ax."""
    cmap = LinearSegmentedColormap.from_list(
        "bwr_dark", ["#1A3A6B", "#3A7ABF", "#222222", "#B33030", "#6B0000"]
    )
    vmin, vmax = 0, 40

    for ri in range(3):
        for ci in range(3):
            val = heatmap[ri, ci]
            if np.isnan(val):
                color = GRID
                txt   = "n<3"
                txtc  = MUTED
            else:
                norm  = (val - vmin) / (vmax - vmin)
                color = cmap(norm)
                txt   = f"{val:.1f}%"
                txtc  = WHITE if norm < 0.5 else "#111111"

            rect = Rectangle((ci, 2 - ri), 1, 1,
                              facecolor=color, edgecolor=BG, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(ci + 0.5, 2 - ri + 0.5, txt,
                    color=WHITE, fontsize=8.5, ha="center", va="center",
                    fontweight="bold")

    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["1B Side\n(Outside)", "Middle", "3B Side\n(Inside)"],
                       color=MUTED, fontsize=7.5)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Low", "Mid", "High"], color=MUTED, fontsize=7.5)
    ax.set_title("9-Zone Whiff%  (All vs-RHH Fastballs)", color=TXT,
                 fontsize=9, fontweight="bold", pad=5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor(BG)

    # Strike-zone outline
    zone_rect = Rectangle((0, 0.5), 3, 2,
                           edgecolor=GOLD, facecolor="none",
                           linewidth=1.8, zorder=10)
    ax.add_patch(zone_rect)


# ---------------------------------------------------------------------------
# PAGE 4 — Coaching Summary + Rubber Recommendation + 9-Zone Heatmap
# ---------------------------------------------------------------------------
def page_coaching(pdf, fold_stats, rhh):
    fig = make_fig()
    draw_header_bar(fig,
                    title_left="Beau Elson · Coaching Summary & Rubber Recommendation",
                    title_right="Fordham Baseball Analytics · 2026",
                    bar_height=0.09)
    draw_footer(fig)

    overall_rel = rhh["RelSide"].mean()
    low_rel     = fold_stats[0]["avg_rside"]
    delta_rel   = low_rel - overall_rel

    main_ax = fig.add_axes([0, 0, 1, 1])
    main_ax.set_xlim(0, 1); main_ax.set_ylim(0, 1)
    main_ax.axis("off")
    main_ax.set_facecolor(BG)

    CONTENT_TOP  = 0.87
    CONTENT_LEFT = 0.02
    CONTENT_RIGHT = 0.97

    # ---- Left column: Rubber / Release Recommendation ----
    rubber_x = CONTENT_LEFT
    rubber_w  = 0.45
    rubber_h  = 0.54
    rubber_bot = CONTENT_TOP - rubber_h

    draw_rounded_rect(main_ax, rubber_x, rubber_bot, rubber_w, rubber_h,
                      radius=0.015, facecolor=PANEL, edgecolor=MAROON,
                      linewidth=1.8, transform=main_ax.transAxes)

    main_ax.text(rubber_x + 0.015, rubber_bot + rubber_h - 0.025,
                 "RUBBER / RELEASE RECOMMENDATION",
                 color=MAROON, fontsize=9, fontweight="bold",
                 va="top", transform=main_ax.transAxes)

    rubber_items = [
        ("Current mean RelSide:", f"{overall_rel:.2f} ft", WHITE),
        ("Target RelSide (Low HAA fold):", f"{low_rel:.2f} ft", GOLD),
        ("Required shift:",
         f"Δ = {delta_rel:+.2f} ft  ≈ {abs(delta_rel)*12:.1f} in toward 1B/glove side",
         GOLD),
        ("", "", ""),
        ("Expected outcome:",
         f"Whiff% from ~{whiff_pct(rhh):.1f}% (overall) into 25.9% range",
         "#35C46B"),
        ("", "", ""),
        ("Mechanism:", "", MUTED),
        ("  For a LHP, moving release toward the 1B/glove side", "", WHITE),
        ("  lowers HAA → ball arrives from 1B side → stays", "", WHITE),
        ("  outside vs RHH → most effective approach angle.", "", WHITE),
        ("", "", ""),
        ("Guardrail:", "", MAROON),
        ("  Zone% drops to 41.5% in Low-HAA fold (vs 73.9% Mid).", "", WHITE),
        ("  Must pair with pitch-to-contact mix or sharp location", "", WHITE),
        ("  discipline to maintain acceptable in-zone rates.", "", WHITE),
    ]

    for i, (label, value, color) in enumerate(rubber_items):
        y_pos = rubber_bot + rubber_h - 0.07 - i * 0.033
        if label and value:
            main_ax.text(rubber_x + 0.015, y_pos, label,
                         color=MUTED, fontsize=7.8, va="top",
                         transform=main_ax.transAxes)
            main_ax.text(rubber_x + rubber_w - 0.015, y_pos, value,
                         color=color, fontsize=8, va="top", ha="right",
                         fontweight="bold", transform=main_ax.transAxes)
        elif label:
            main_ax.text(rubber_x + 0.015, y_pos, label,
                         color=color, fontsize=7.8, va="top",
                         transform=main_ax.transAxes)

    # ---- 9-Zone Heatmap (right column, top) ----
    heatmap_data = build_9zone_heatmap(rhh)
    hm_ax = fig.add_axes([0.52, 0.50, 0.44, 0.37])
    dark_axes(hm_ax, facecolor=BG)
    draw_9zone_heatmap(hm_ax, heatmap_data)

    # ---- Strategic Read per Fold (bottom) ----
    strat_top = rubber_bot - 0.025
    strat_h   = 0.30
    strat_bot = strat_top - strat_h

    draw_rounded_rect(main_ax, CONTENT_LEFT, strat_bot,
                      CONTENT_RIGHT - CONTENT_LEFT, strat_h,
                      radius=0.015, facecolor=PANEL, edgecolor=GRID,
                      linewidth=1.2, transform=main_ax.transAxes)

    main_ax.text(CONTENT_LEFT + 0.015, strat_bot + strat_h - 0.022,
                 "STRATEGIC READ PER FOLD",
                 color=TXT, fontsize=9, fontweight="bold",
                 va="top", transform=main_ax.transAxes)

    strategies = [
        {
            "fold": "LOW HAA (≈1.08°) — OPTIMAL",
            "color": FOLD_COLORS[0],
            "bullets": [
                "Ball arrives from 1B side, finishing outside to RHH.",
                "25.9% Whiff% — 3× the rate of High-HAA zone.",
                "Zone% is 41.5%; effective even when missing slightly off plate.",
                "Primary use case: put-away pitch / two-strike count; must spot carefully.",
            ],
        },
        {
            "fold": "MID HAA (≈2.12°) — NEUTRAL",
            "color": FOLD_COLORS[1],
            "bullets": [
                "Ball centers in the zone (PlateLocSide ≈ –0.05 ft). High Zone% (73.9%).",
                "Moderate swing-and-miss (11.4%). Good for early counts / strike-one sequencing.",
                "Effective when establishing zone presence before going to Low-HAA for whiff.",
            ],
        },
        {
            "fold": "HIGH HAA (≈3.09°) — AVOID FOR WHIFFS",
            "color": FOLD_COLORS[2],
            "bullets": [
                "Ball runs inside to RHH (PlateLocSide +0.61 ft). Harder to miss.",
                "Only 8.5% Whiff% — hitters can track and barrel the pitch.",
                "Zone% (56%) is moderate. Risk of hard contact on middle-in fastballs.",
                "Use sparingly; best only as a complementary location for pitch mix variety.",
            ],
        },
    ]

    col_w   = (CONTENT_RIGHT - CONTENT_LEFT - 0.03) / 3
    for si, strat in enumerate(strategies):
        col_x = CONTENT_LEFT + 0.015 + si * (col_w + 0.01)

        main_ax.text(col_x, strat_bot + strat_h - 0.055, strat["fold"],
                     color=strat["color"], fontsize=7.8, fontweight="bold",
                     va="top", transform=main_ax.transAxes)

        for bi, bullet in enumerate(strat["bullets"]):
            bx = col_x
            by = strat_bot + strat_h - 0.090 - bi * 0.052

            # wrap long bullets
            wrapped = textwrap.wrap(bullet, width=40)
            for li, wline in enumerate(wrapped):
                prefix = "• " if li == 0 else "  "
                main_ax.text(bx, by - li * 0.024,
                             prefix + wline,
                             color=WHITE if bi == 0 else MUTED,
                             fontsize=7.2, va="top",
                             transform=main_ax.transAxes)

    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Page 4: Coaching Summary + 9-Zone Heatmap")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Beau Elson Fastball HAA — PDF Report Generator")
    print("=" * 60)

    # ---- Load data ----
    print("\nLoading data from:", DATA_DIR)
    try:
        all_fb, rhh = load_data()
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        raise

    print(f"  Fastballs (all batters): {len(all_fb)}")
    print(f"  Fastballs vs RHH (with HAA): {len(rhh)}")

    # ---- Build fold stats ----
    fold_stats = build_fold_stats(rhh)

    print("\nFold Stats (vs RHH, pd.qcut HAA terciles):")
    print(f"  {'Fold':<6} {'N':>4} {'HAA':>7} {'RelSide':>9} {'PlateSide':>10} "
          f"{'Whiff%':>7} {'CSW%':>6} {'Zone%':>6}")
    for s in fold_stats:
        print(f"  {s['fold']:<6} {s['n']:>4} {s['avg_haa']:>7.2f}° "
              f"{s['avg_rside']:>9.2f}ft {s['avg_plside']:>10.2f}ft "
              f"{s['whiff']:>7.1f}% {s['csw']:>6.1f}% {s['zone']:>6.1f}%")

    # ---- Correlation ----
    corr = rhh["HorzApprAngle"].corr(rhh["PlateLocSide"])
    print(f"\n  HAA ↔ PlateLocSide correlation: {corr:.2f}")

    # ---- Generate PDF ----
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nGenerating PDF: {OUT_PDF}")

    with PdfPages(OUT_PDF) as pdf:
        page_cover(pdf, fold_stats, rhh)
        page_folds_png(pdf)
        page_scatter_table(pdf, fold_stats, rhh)
        page_coaching(pdf, fold_stats, rhh)

    size_kb = os.path.getsize(OUT_PDF) / 1024
    print(f"\nDone! PDF saved → {OUT_PDF}")
    print(f"  File size: {size_kb:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
