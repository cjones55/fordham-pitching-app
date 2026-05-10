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


def swing_pct(g):
    swings = g["PitchCall"].isin(["StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                                   "FoulBallFieldable", "InPlay", "InPlayNoOut", "InPlayOut"])
    return swings.sum() / len(g) * 100 if len(g) > 0 else 0.0


def csw_pct(g):
    csw = g["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])
    return csw.sum() / len(g) * 100 if len(g) > 0 else 0.0


def zone_pct(g):
    in_zone = (
        g["PlateLocSide"].between(-0.831, 0.831) &
        g["PlateLocHeight"].between(1.5, 3.5)
    )
    return in_zone.sum() / len(g) * 100 if len(g) > 0 else 0.0


def zone_take_pct(g):
    """% of pitches IN the zone that the batter took."""
    in_zone = (
        g["PlateLocSide"].between(-0.831, 0.831) &
        g["PlateLocHeight"].between(1.5, 3.5)
    )
    zone_pitches = g[in_zone]
    if len(zone_pitches) == 0:
        return 0.0
    took = zone_pitches["PitchCall"].isin(["StrikeCalled", "BallCalled", "Ball", "HitByPitch"])
    return took.sum() / len(zone_pitches) * 100


_SWING_CALLS = {"StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
                "FoulBallFieldable", "InPlay", "InPlayNoOut", "InPlayOut"}
_CONTACT_CALLS = {"FoulBall", "FoulBallNotFieldable", "FoulBallFieldable",
                  "InPlay", "InPlayNoOut", "InPlayOut"}


def _in_zone(g):
    return g["PlateLocSide"].between(-0.831, 0.831) & g["PlateLocHeight"].between(1.5, 3.5)


def zswing_pct(g):
    """Swing% on in-zone pitches."""
    z = g[_in_zone(g)]
    return z["PitchCall"].isin(_SWING_CALLS).sum() / len(z) * 100 if len(z) > 0 else 0.0


def oswing_pct(g):
    """Swing% on out-of-zone pitches (chase rate)."""
    o = g[~_in_zone(g)]
    return o["PitchCall"].isin(_SWING_CALLS).sum() / len(o) * 100 if len(o) > 0 else 0.0


def zcontact_pct(g):
    """Contact% on in-zone swings."""
    z = g[_in_zone(g)]
    swings = z["PitchCall"].isin(_SWING_CALLS).sum()
    contact = z["PitchCall"].isin(_CONTACT_CALLS).sum()
    return contact / swings * 100 if swings > 0 else 0.0


def ocontact_pct(g):
    """Contact% on out-of-zone swings."""
    o = g[~_in_zone(g)]
    swings = o["PitchCall"].isin(_SWING_CALLS).sum()
    contact = o["PitchCall"].isin(_CONTACT_CALLS).sum()
    return contact / swings * 100 if swings > 0 else 0.0


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
            "swing":      swing_pct(g),
            "csw":        csw_pct(g),
            "zone":       zone_pct(g),
            "zone_take":  zone_take_pct(g),
            "zswing":     zswing_pct(g),
            "oswing":     oswing_pct(g),
            "zcontact":   zcontact_pct(g),
            "ocontact":   ocontact_pct(g),
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

    draw_header_bar(fig,
                    title_left="Beau Elson · LHP  |  Fastball HAA Analysis",
                    title_right="Fordham Baseball Analytics · 2026",
                    bar_height=0.10)
    draw_footer(fig)

    # content area: y=0.05 to y=0.88, x=0.02 to 0.98
    CONTENT_TOP   = 0.88
    CONTENT_LEFT  = 0.02
    CONTENT_RIGHT = 0.98

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ------------------------------------------------------------------ #
    # 1. STAT TILES ROW  (y=0.72 to 0.88, 5 tiles)
    # ------------------------------------------------------------------ #
    overall_swing = swing_pct(rhh)
    overall_whiff = whiff_pct(rhh)
    overall_csw   = csw_pct(rhh)
    overall_zone  = zone_pct(rhh)

    tile_labels = [
        f"n = {len(rhh)}",
        "Swing%",
        "Whiff%",
        "CSW%",
        "Zone%",
    ]
    tile_values = [
        "Pitches",
        f"{overall_swing:.1f}%",
        f"{overall_whiff:.1f}%",
        f"{overall_csw:.1f}%",
        f"{overall_zone:.1f}%",
    ]
    tile_val_colors = [MUTED, GOLD, GOLD, GOLD, GOLD]

    tile_top = CONTENT_TOP
    tile_h   = 0.14
    tile_bot = tile_top - tile_h
    tile_w   = 0.17
    tile_gap = 0.015
    # centre the 5 tiles across full width
    total_tile_w = 5 * tile_w + 4 * tile_gap
    tile_x0 = (1.0 - total_tile_w) / 2

    for i, (lbl, val, vcol) in enumerate(zip(tile_labels, tile_values, tile_val_colors)):
        tx = tile_x0 + i * (tile_w + tile_gap)
        draw_rounded_rect(ax, tx, tile_bot, tile_w, tile_h,
                          radius=0.012, facecolor=PANEL2, edgecolor=GRID,
                          linewidth=1.2, transform=ax.transAxes)
        ax.text(tx + tile_w / 2, tile_bot + tile_h * 0.68, val,
                color=vcol, fontsize=20, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(tx + tile_w / 2, tile_bot + tile_h * 0.20, lbl,
                color=MUTED, fontsize=8, ha="center", va="center",
                transform=ax.transAxes)

    # ------------------------------------------------------------------ #
    # 2. TWO PANELS SIDE BY SIDE  (y=0.06 to 0.68)
    # ------------------------------------------------------------------ #
    panel_top = tile_bot - 0.02
    panel_bot = 0.06
    panel_h   = panel_top - panel_bot

    # ---- LEFT: Key Finding ----
    kf_x = CONTENT_LEFT
    kf_w = 0.54
    draw_rounded_rect(ax, kf_x, panel_bot, kf_w, panel_h,
                      radius=0.015, facecolor=PANEL, edgecolor=MAROON,
                      linewidth=1.8, transform=ax.transAxes)

    ax.text(kf_x + 0.012, panel_bot + panel_h - 0.022,
            "KEY FINDING",
            color=MAROON, fontsize=9, fontweight="bold",
            va="top", ha="left", transform=ax.transAxes)

    # Big headline
    ax.text(kf_x + 0.012, panel_bot + panel_h - 0.055,
            "Low HAA = 3× more whiffs than High HAA",
            color=GOLD, fontsize=13, fontweight="bold",
            va="top", ha="left", transform=ax.transAxes)

    body_lines = [
        "Low HAA (≈1.08°) generates 25.9% Whiff% — 3× the rate of High HAA (8.5%).",
        "",
        "When Beau’s fastball exits the hand at a lower horizontal approach",
        "angle, it arrives from the 1B side and finishes outside to RHH.",
        "That deceptive angle makes it extremely difficult to square up,",
        "driving the dramatic difference in swing-and-miss across folds.",
        "",
        "HAA and PlateLocSide share a 0.90 correlation — the approach",
        "angle dictates where the ball ends up, not the other way around.",
    ]
    line_spacing = 0.054
    body_y0 = panel_bot + panel_h - 0.105
    for i, line in enumerate(body_lines):
        color = WHITE if line.strip() else MUTED
        ax.text(kf_x + 0.012, body_y0 - i * line_spacing,
                line, color=color, fontsize=9,
                va="top", ha="left", transform=ax.transAxes)

    # Color legend at bottom of left panel
    legend_y = panel_bot + 0.030
    for i, (s, col) in enumerate(zip(fold_stats, FOLD_COLORS)):
        lx = kf_x + 0.012 + i * 0.165
        ax.add_patch(Rectangle((lx, legend_y), 0.014, 0.016,
                                facecolor=col, transform=ax.transAxes,
                                zorder=5, clip_on=False))
        ax.text(lx + 0.018, legend_y + 0.008,
                f"{s['fold']} HAA  ({s['whiff']:.1f}% Whiff)",
                color=MUTED, fontsize=7.5, va="center",
                transform=ax.transAxes)

    # ---- RIGHT: Target Release Point ----
    tp_x  = kf_x + kf_w + 0.02
    tp_w  = CONTENT_RIGHT - tp_x
    draw_rounded_rect(ax, tp_x, panel_bot, tp_w, panel_h,
                      radius=0.015, facecolor=PANEL, edgecolor=GOLD,
                      linewidth=1.8, transform=ax.transAxes)

    ax.text(tp_x + 0.012, panel_bot + panel_h - 0.022,
            "TARGET RELEASE POINT",
            color=GOLD, fontsize=9, fontweight="bold",
            va="top", ha="left", transform=ax.transAxes)

    overall_rel = rhh["RelSide"].mean()
    low_rel     = fold_stats[0]["avg_rside"]
    delta_rel   = low_rel - overall_rel

    rp_items = [
        ("Current RelSide:",     f"{overall_rel:.2f} ft",   WHITE),
        ("Target RelSide:",      f"{low_rel:.2f} ft",        GOLD),
        ("Delta:",
         f"Δ = {delta_rel:+.2f} ft  (≈{abs(delta_rel)*12:.1f} in toward 1B/glove side)",
         GOLD),
    ]
    rp_y0 = panel_bot + panel_h - 0.060
    rp_ls = 0.055
    for i, (lbl, val, vcol) in enumerate(rp_items):
        y = rp_y0 - i * rp_ls
        ax.text(tp_x + 0.012, y, lbl,
                color=MUTED, fontsize=9, va="top", transform=ax.transAxes)
        ax.text(tp_x + tp_w - 0.012, y, val,
                color=vcol, fontsize=9, va="top", ha="right",
                fontweight="bold", transform=ax.transAxes)

    sep1_y = rp_y0 - len(rp_items) * rp_ls - 0.010
    ax.plot([tp_x + 0.012, tp_x + tp_w - 0.012], [sep1_y, sep1_y],
            color=GRID, linewidth=0.8, transform=ax.transAxes)

    ax.text(tp_x + 0.012, sep1_y - 0.014,
            f"Expected outcome:  Whiff% → ~25.9% (from {overall_whiff:.1f}% overall)",
            color="#35C46B", fontsize=9, va="top", fontweight="bold",
            transform=ax.transAxes)

    mech_lines = [
        "For a LHP, shifting release toward the 1B / glove side",
        "lowers HAA → ball arrives from 1B side → stays outside",
        "vs RHH — the most effective approach angle.",
    ]
    mech_y0 = sep1_y - 0.075
    for i, line in enumerate(mech_lines):
        ax.text(tp_x + 0.012, mech_y0 - i * rp_ls,
                line, color=WHITE, fontsize=9, va="top",
                transform=ax.transAxes)

    sep2_y = mech_y0 - len(mech_lines) * rp_ls - 0.010
    ax.plot([tp_x + 0.012, tp_x + tp_w - 0.012], [sep2_y, sep2_y],
            color=GRID, linewidth=0.8, transform=ax.transAxes)

    guard_lines = [
        "Guardrail: Zone% drops to 41.5% in Low-HAA fold.",
        "Pair with location discipline or pitch-to-contact mix.",
    ]
    guard_y0 = sep2_y - 0.014
    for i, line in enumerate(guard_lines):
        ax.text(tp_x + 0.012, guard_y0 - i * rp_ls,
                line, color=MUTED, fontsize=9, va="top",
                transform=ax.transAxes)

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

    # ------------------------------------------------------------------ #
    # 1. SCATTER PLOTS  (y=0.38 to 0.89, three columns)
    # ------------------------------------------------------------------ #
    gs = GridSpec(1, 3, left=0.04, right=0.97,
                  top=0.89, bottom=0.38, wspace=0.30)

    x_lim = (-2.0, 2.0)
    y_lim = (0.5, 5.0)

    for col_i, s in enumerate(fold_stats):
        ax = fig.add_subplot(gs[0, col_i])
        dark_axes(ax, facecolor=PANEL)

        g = s["data"]
        valid = g.dropna(subset=["PlateLocSide", "PlateLocHeight"])

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
        ax.set_xlabel(
            "← 1B Side (Outside)    PlateLocSide (ft)    3B Side (Inside) →",
            fontsize=7.5, color=MUTED)
        ax.set_ylabel("PlateLocHeight (ft)", fontsize=7, color=MUTED)
        ax.set_title(f"{s['fold']} HAA  (n={s['n']})\nHAA ≈ {s['avg_haa']:.2f}°",
                     color=s["color"], fontsize=9, fontweight="bold")

        # 1B / 3B corner labels
        ax.text(x_lim[0] + 0.06, y_lim[1] - 0.12, "← 1B",
                color=MUTED, fontsize=8, va="top")
        ax.text(x_lim[1] - 0.06, y_lim[1] - 0.12, "3B →",
                color=MUTED, fontsize=8, va="top", ha="right")

        # Single Whiff% chip at bottom
        whiff_color = GOLD if s["fold"] == "Low" else WHITE
        ax.text(0.50, 0.04,
                f"{s['whiff']:.1f}% Whiff",
                color=whiff_color, fontsize=9, ha="center", va="bottom",
                transform=ax.transAxes, fontweight="bold",
                bbox=dict(facecolor=PANEL2, edgecolor=s["color"],
                          linewidth=0.8, boxstyle="round,pad=0.25"))

        # Gold spine for Low HAA (optimal)
        if s["fold"] == "Low":
            for spine in ax.spines.values():
                spine.set_edgecolor(GOLD)
                spine.set_linewidth(2.0)

    # ------------------------------------------------------------------ #
    # 2. COMPARISON TABLE  (y=0.04 to 0.34)
    # ------------------------------------------------------------------ #
    table_ax = fig.add_axes([0.04, 0.04, 0.92, 0.30])
    table_ax.set_facecolor(PANEL2)
    table_ax.set_xlim(0, 1); table_ax.set_ylim(0, 1)
    table_ax.axis("off")

    table_ax.text(0.5, 0.95,
                  "Fold Comparison — Fastball vs RHH",
                  color=GOLD, fontsize=9, fontweight="bold",
                  ha="center", va="top", transform=table_ax.transAxes)

    col_names = ["Fold", "N", "HAA", "Swing%", "Whiff%", "CSW%",
                 "Zone%", "ZTake%", "ZSwing%", "OSwing%", "ZCon%", "OCon%"]
    col_xs    = [0.01, 0.08, 0.15, 0.23, 0.31, 0.39,
                 0.47, 0.55, 0.63, 0.72, 0.81, 0.90]

    # Header row
    for xi, hdr in zip(col_xs, col_names):
        table_ax.text(xi, 0.88, hdr, color=GOLD, fontsize=7.5,
                      fontweight="bold", va="top", transform=table_ax.transAxes)

    table_ax.axhline(0.83, color=GRID, linewidth=0.8)

    # Data rows — well-separated at 0.66, 0.48, 0.30
    row_ys = [0.66, 0.48, 0.30]
    for ri, (s, row_y) in enumerate(zip(fold_stats, row_ys)):
        vals = [
            s["fold"],
            str(s["n"]),
            f"{s['avg_haa']:.2f}°",
            f"{s['swing']:.1f}%",
            f"{s['whiff']:.1f}%",
            f"{s['csw']:.1f}%",
            f"{s['zone']:.1f}%",
            f"{s['zone_take']:.1f}%",
            f"{s['zswing']:.1f}%",
            f"{s['oswing']:.1f}%",
            f"{s['zcontact']:.1f}%",
            f"{s['ocontact']:.1f}%",
        ]
        for xi, val in zip(col_xs, vals):
            is_name = xi == col_xs[0]
            table_ax.text(xi, row_y, val,
                          color=s["color"] if is_name else WHITE,
                          fontsize=8.5, fontweight="bold" if is_name else "normal",
                          va="center", transform=table_ax.transAxes)

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

    # Outer border fitting exactly around all 9 cells
    zone_rect = Rectangle((0, 0), 3, 3,
                           edgecolor=GOLD, facecolor="none",
                           linewidth=2.5, zorder=10)
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

    CONTENT_LEFT  = 0.02
    CONTENT_RIGHT = 0.97

    # ------------------------------------------------------------------ #
    # TOP HALF (y=0.47 to 0.89)
    # ------------------------------------------------------------------ #
    top_bot = 0.47
    top_top = 0.89

    # ---- Left column: Rubber / Release Recommendation ----
    rubber_x   = CONTENT_LEFT
    rubber_w   = 0.48
    rubber_h   = top_top - top_bot
    rubber_bot = top_bot

    draw_rounded_rect(main_ax, rubber_x, rubber_bot, rubber_w, rubber_h,
                      radius=0.015, facecolor=PANEL, edgecolor=MAROON,
                      linewidth=1.8, transform=main_ax.transAxes)

    main_ax.text(rubber_x + 0.015, rubber_bot + rubber_h - 0.022,
                 "RUBBER / RELEASE RECOMMENDATION",
                 color=MAROON, fontsize=9, fontweight="bold",
                 va="top", transform=main_ax.transAxes)

    # 3 metric rows
    ls = 0.055
    metrics = [
        ("Current RelSide:",           f"{overall_rel:.2f} ft",  WHITE),
        ("Target RelSide:",             f"{low_rel:.2f} ft",       GOLD),
        ("Required Shift:",
         f"Δ = {delta_rel:+.2f} ft  ≈ {abs(delta_rel)*12:.1f} in toward 1B/glove side",
         GOLD),
    ]
    m_y0 = rubber_bot + rubber_h - 0.060
    for i, (lbl, val, vcol) in enumerate(metrics):
        y = m_y0 - i * ls
        main_ax.text(rubber_x + 0.015, y, lbl,
                     color=MUTED, fontsize=8.5, va="top",
                     transform=main_ax.transAxes)
        main_ax.text(rubber_x + rubber_w - 0.015, y, val,
                     color=vcol, fontsize=8.5, va="top", ha="right",
                     fontweight="bold", transform=main_ax.transAxes)

    sep1_y = m_y0 - len(metrics) * ls - 0.012
    main_ax.plot([rubber_x + 0.012, rubber_x + rubber_w - 0.012],
                 [sep1_y, sep1_y],
                 color=GRID, linewidth=0.8, transform=main_ax.transAxes)

    main_ax.text(rubber_x + 0.015, sep1_y - 0.012,
                 f"Expected Outcome:  Whiff% → ~25.9%  (from {whiff_pct(rhh):.1f}% overall)",
                 color="#35C46B", fontsize=8.5, va="top", fontweight="bold",
                 transform=main_ax.transAxes)

    sep2_y = sep1_y - ls - 0.012
    main_ax.plot([rubber_x + 0.012, rubber_x + rubber_w - 0.012],
                 [sep2_y, sep2_y],
                 color=GRID, linewidth=0.8, transform=main_ax.transAxes)

    mech_lines = [
        "For a LHP, shifting release toward the 1B / glove side",
        "lowers HAA → ball arrives from 1B side → stays outside",
        "vs RHH — the most effective approach angle.",
    ]
    mech_y0 = sep2_y - 0.012
    for i, line in enumerate(mech_lines):
        main_ax.text(rubber_x + 0.015, mech_y0 - i * ls,
                     line, color=WHITE, fontsize=8.5, va="top",
                     transform=main_ax.transAxes)

    sep3_y = mech_y0 - len(mech_lines) * ls - 0.012
    main_ax.plot([rubber_x + 0.012, rubber_x + rubber_w - 0.012],
                 [sep3_y, sep3_y],
                 color=GRID, linewidth=0.8, transform=main_ax.transAxes)

    guard_lines = [
        "Guardrail: Zone% drops to 41.5% in Low-HAA fold (vs 73.9% Mid).",
        "Pair with pitch-to-contact mix or sharp location discipline.",
    ]
    guard_y0 = sep3_y - 0.012
    for i, line in enumerate(guard_lines):
        main_ax.text(rubber_x + 0.015, guard_y0 - i * ls,
                     line, color=MUTED, fontsize=8.5, va="top",
                     transform=main_ax.transAxes)

    # ---- Right column: 9-Zone Heatmap ----
    heatmap_data = build_9zone_heatmap(rhh)
    hm_ax = fig.add_axes([0.53, top_bot, 0.44, top_top - top_bot])
    dark_axes(hm_ax, facecolor=BG)
    draw_9zone_heatmap(hm_ax, heatmap_data)

    # ------------------------------------------------------------------ #
    # BOTTOM HALF (y=0.05 to 0.42): STRATEGIC READ PER FOLD
    # ------------------------------------------------------------------ #
    strat_bot = 0.05
    strat_top = 0.42
    strat_h   = strat_top - strat_bot

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
            "fold": "LOW HAA (≈1.08°)",
            "color": FOLD_COLORS[0],
            "bullets": [
                "Ball arrives from 1B side, finishing outside to RHH.",
                "25.9% Whiff% — 3× the rate of High-HAA zone.",
                "Zone% is 41.5%; effective even when missing slightly off plate.",
                "Primary use: put-away pitch / two-strike count; must spot carefully.",
            ],
        },
        {
            "fold": "MID HAA (≈2.12°) — NEUTRAL",
            "color": FOLD_COLORS[1],
            "bullets": [
                "Ball centers in zone (PlateLocSide ≈ –0.05 ft). High Zone% (73.9%).",
                "Moderate swing-and-miss (11.4%). Good for early counts / strike-one.",
                "Use to establish zone presence before going to Low-HAA for whiff.",
            ],
        },
        {
            "fold": "HIGH HAA (≈3.09°) — AVOID FOR WHIFFS",
            "color": FOLD_COLORS[2],
            "bullets": [
                "Ball runs inside to RHH (PlateLocSide +0.61 ft). Harder to miss.",
                "Only 8.5% Whiff% — hitters can track and barrel the pitch.",
                "Zone% (56%) moderate. Risk of hard contact on middle-in fastballs.",
                "Use sparingly as a complementary location for pitch-mix variety.",
            ],
        },
    ]

    col_w = (CONTENT_RIGHT - CONTENT_LEFT - 0.03) / 3
    bullet_ls = 0.038  # tighter linespacing for bullets

    for si, strat in enumerate(strategies):
        col_x = CONTENT_LEFT + 0.015 + si * (col_w + 0.01)

        # Fold name header
        main_ax.text(col_x, strat_bot + strat_h - 0.050, strat["fold"],
                     color=strat["color"], fontsize=8.5, fontweight="bold",
                     va="top", transform=main_ax.transAxes)

        # Bullets — 4 bullets × ~2 wrapped lines each at 0.038 spacing fits easily
        bullet_y0 = strat_bot + strat_h - 0.090
        cur_y = bullet_y0
        for bi, bullet in enumerate(strat["bullets"]):
            wrapped = textwrap.wrap(bullet, width=42)
            for li, wline in enumerate(wrapped):
                prefix = "• " if li == 0 else "  "
                main_ax.text(col_x, cur_y,
                             prefix + wline,
                             color=WHITE if bi == 0 else MUTED,
                             fontsize=7.5, va="top",
                             transform=main_ax.transAxes)
                cur_y -= bullet_ls
            cur_y -= 0.006  # small gap between bullets

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
    print(f"  File size: {size_kb:.1f} KB")

    # ---- Render page PNGs for Streamlit inline display ----
    pages_dir = os.path.join(OUT_DIR, "beau_elson_fastball_haa_detailed_report")
    os.makedirs(pages_dir, exist_ok=True)
    for old in glob.glob(os.path.join(pages_dir, "*.png")):
        os.remove(old)

    class _PngSaver:
        def __init__(self):
            self.n = 0
        def savefig(self, fig, **kwargs):
            self.n += 1
            fig.savefig(os.path.join(pages_dir, f"page_{self.n}.png"),
                        dpi=150, bbox_inches="tight", facecolor=BG)

    saver = _PngSaver()
    page_cover(saver, fold_stats, rhh)
    page_folds_png(saver)
    page_scatter_table(saver, fold_stats, rhh)
    page_coaching(saver, fold_stats, rhh)
    print(f"  {saver.n} page PNGs → {pages_dir}")

    print(f"\nDone! PDF saved → {OUT_PDF}")
    print("=" * 60)


if __name__ == "__main__":
    main()
