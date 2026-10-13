"""Build the 2027 position-player depth charts vs LHP and vs RHP.

Lineups were set from 2026 game splits plus the fall Pins vs Maroons intersquads, and
confirmed roster/positions from the staff. Output goes to personal_reports/,
which the Private Reports page renders.

    python scripts/depth_chart_lineups_2027.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Wedge, Circle
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "personal_reports"
LOGO = ROOT / "assets" / "rams.png"

BG = "#100c0b"
GOLD = "#c9a25e"
MAROON = "#8b1717"
CARD = "#1a1512"
GRASS = "#1f3a10"
DIRT = "#45491a"
WHITE = "#f5f0e6"
MUTED = "#b8ae9c"
DIM = "#7d7365"

W, H = 2000, 1900

# Field card centers (x, y) in pixel space, y down.
POS_XY = {
    "LF": (270, 290), "CF": (1000, 210), "RF": (1730, 290),
    "SS": (745, 640), "2B": (1255, 640),
    "3B": (455, 912), "1B": (1545, 912),
    "C": (1000, 1440), "DH": (250, 1440),
}

# (hand, starter, backups) per position; batting order lists (pos, name, hand, split line)
LINEUPS = {
    "RHP": {
        "positions": {
            "C": ("Young, C.", ["Chavez, C.", "Velit, L."]),
            "1B": ("Rubin, J.", ["Dieguez, M.", "Ford, T."]),
            "2B": ("Reina, B.", ["Caron, M.", "Beaudreau, B."]),
            "SS": ("Grabau, A.", ["Beaudreau, B.", "Maryott, G."]),
            "3B": ("Beaudreau, B.", ["Ford, T.", "Forney, C."]),
            "LF": ("Maryott, G.", ["Dean, M.", "Reina, B."]),
            "CF": ("Markey, T.", ["Caron, M.", "Dean, M."]),
            "RF": ("Kaufman, T.", ["Dean, M.", "Palmer, O."]),
            "DH": ("Ford, T.", ["Hanawalt, C.", "Dieguez, M."]),
        },
        "order": [
            ("2B", "Reina, B.", "L", "1.008 OPS · 22 PA"),
            ("SS", "Grabau, A.", "L", ".850 OPS · 174 PA"),
            ("RF", "Kaufman, T.", "R", "1.125 OPS · 24 PA"),
            ("3B", "Beaudreau, B.", "L", "1.119 OPS · 21 PA"),
            ("1B", "Rubin, J.", "R", ".963 OPS · 52 PA"),
            ("LF", "Maryott, G.", "R", ".922 OPS · 18 PA"),
            ("DH", "Ford, T.", "R", ".741 OPS · 91 PA"),
            ("C", "Young, C.", "R", ".791 OPS · 97 PA"),
            ("CF", "Markey, T.", "L", ".729 OPS · 155 PA"),
        ],
    },
    "LHP": {
        "positions": {
            "C": ("Young, C.", ["Chavez, C.", "Velit, L."]),
            "1B": ("Palmer, O.", ["Rubin, J.", "Crang, M."]),
            "2B": ("Maryott, G.", ["Caron, M.", "Dieguez, M."]),
            "SS": ("Grabau, A.", ["Beaudreau, B.", "Maryott, G."]),
            "3B": ("Beaudreau, B.", ["Crang, M.", "Ford, T."]),
            "LF": ("Wansaw, M.", ["Dean, M.", "Reina, B."]),
            "CF": ("Caron, M.", ["Markey, T.", "Dean, M."]),
            "RF": ("Kaufman, T.", ["Wansaw, M.", "Palmer, O."]),
            "DH": ("Rubin, J.", ["Ford, T.", "Hanawalt, C."]),
        },
        "order": [
            ("CF", "Caron, M.", "S", "1.490 OPS · 13 PA"),
            ("3B", "Beaudreau, B.", "L", "1.500 OPS · 12 PA"),
            ("2B", "Maryott, G.", "R", "1.857 OPS · 7 PA"),
            ("RF", "Kaufman, T.", "R", "1.250 OPS · 8 PA"),
            ("LF", "Wansaw, M.", "R", "1.722 OPS · 9 PA"),
            ("1B", "Palmer, O.", "L", "1.597 OPS · 11 PA"),
            ("DH", "Rubin, J.", "R", ".716 OPS · 19 PA"),
            ("C", "Young, C.", "R", ".649 OPS · 39 PA"),
            ("SS", "Grabau, A.", "L", ".546 OPS · everyday"),
        ],
    },
}


def _box(ax, x, y, w, h, face, edge, lw=3, r=14, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=face, ec=edge, lw=lw, zorder=z))


def _pos_card(ax, pos, starter, backups, slot):
    cx, cy = POS_XY[pos]
    w, h = 290, 165
    x, y = cx - w / 2, cy - 62
    _box(ax, x, y, w, h, CARD, GOLD)
    _box(ax, x + 6, y + 6, w - 12, 38, MAROON, MAROON, lw=0, r=6, z=4)
    ax.text(cx, y + 26, pos, ha="center", va="center", color=GOLD, fontsize=15, weight="bold", zorder=5)
    label = f"{slot}. {starter}" if slot else starter
    ax.text(cx, y + 66, label, ha="center", va="center", color=WHITE, fontsize=15, weight="bold", zorder=5)
    for i, b in enumerate(backups):
        ax.text(cx, y + 100 + i * 30, f"↳  {b}", ha="center", va="center",
                color=MUTED if i == 0 else DIM, fontsize=12, style="normal" if i == 0 else "italic", zorder=5)


def build(hand: str) -> Path:
    cfg = LINEUPS[hand]
    slots = {pos: i + 1 for i, (pos, *_rest) in enumerate(cfg["order"])}

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    # Header
    ax.text(W / 2, 45, "FORDHAM BASEBALL", ha="center", va="center", color=GOLD, fontsize=34, weight="bold")
    ax.text(W / 2, 100, f"2 0 2 7   L I N E U P   V S   {' '.join(hand)}",
            ha="center", va="center", color="#cbbfa9", fontsize=16, weight="bold")
    ax.plot([75, W - 75], [128, 128], color="#6b5a3a", lw=2)
    if LOGO.exists():
        img = mpimg.imread(str(LOGO))
        for lx in (115, W - 195):
            ax.imshow(img, extent=(lx, lx + 80, 105, 25), zorder=6)

    # Field
    ax.add_patch(Wedge((1000, 1350), 1230, 225, 315, fc=GRASS, ec="none", zorder=1, transform=ax.transData))
    ax.add_patch(Polygon([(1000, 460), (1415, 875), (1000, 1290), (585, 875)], closed=True,
                         fc=DIRT, ec=WHITE, lw=3, zorder=2))
    ax.add_patch(Circle((1000, 900), 40, fc="#86651a", ec=WHITE, lw=2, zorder=2))
    for dx in (-1, 1):
        ax.plot([1000, 1000 + dx * 1000], [1350, 350], ls="--", color="#777", lw=2, zorder=1)

    for pos, (starter, backups) in cfg["positions"].items():
        _pos_card(ax, pos, starter, backups, slots.get(pos))

    # Batting order panel
    px, py, pw, ph = 1290, 1180, 640, 520
    _box(ax, px, py, pw, ph, CARD, GOLD)
    _box(ax, px + 6, py + 6, pw - 12, 44, MAROON, MAROON, lw=0, r=6, z=4)
    ax.text(px + pw / 2, py + 29, f"BATTING ORDER VS {hand}", ha="center", va="center",
            color=GOLD, fontsize=15, weight="bold", zorder=5)
    for i, (pos, name, bats, note) in enumerate(cfg["order"]):
        yy = py + 85 + i * 49
        ax.text(px + 30, yy, f"{i + 1}", ha="center", va="center", color=GOLD, fontsize=15, weight="bold", zorder=5)
        ax.text(px + 60, yy, f"{name} ({bats})", ha="left", va="center", color=WHITE, fontsize=14, weight="bold", zorder=5)
        ax.text(px + 330, yy, pos, ha="center", va="center", color=GOLD, fontsize=13, weight="bold", zorder=5)
        ax.text(px + pw - 20, yy, note, ha="right", va="center", color=MUTED, fontsize=12, zorder=5)

    ax.plot([75, W - 75], [1745, 1745], color="#3d3426", lw=2)
    ax.text(W / 2, 1785, f"OPS vs {hand}: 2026 season + fall Pins vs Maroons intersquads combined",
            ha="center", va="center", color=DIM, fontsize=13, style="italic")
    ax.text(W / 2, 1850, "STARTER   ↳ BACKUP   ↳ DEPTH  ·  FORDHAM BASEBALL — INTERNAL USE ONLY",
            ha="center", va="center", color="#5a5145", fontsize=12)

    out = OUT_DIR / f"2027_Outlook_lineup_vs_{hand}.png"
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for hand in ("LHP", "RHP"):
        print(build(hand))
