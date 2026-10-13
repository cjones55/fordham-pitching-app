"""Build the 2027 pitching depth chart (personal_reports/2027_Outlook_pitching_staff.png).

Roles confirmed by the staff in October 2026, using the fall Pins vs Maroons
intersquads for the open rotation spots.

    python scripts/depth_chart_pitching_2027.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "personal_reports" / "2027_Outlook_pitching_staff.png"
LOGO = ROOT / "assets" / "rams.png"

BG = "#100c0b"
GOLD = "#c9a25e"
CARD = "#1a1512"
WHITE = "#f5f0e6"
DIM = "#7d7365"

W, H = 2000, 1420

SECTIONS = [
    ("WEEKEND ROTATION", "#8b1717", 3, [
        ("FRIDAY", "#6b0f0f", "Hanawalt, Chase"),
        ("SATURDAY", "#00349a", "Dowd, Aidan"),
        ("SUNDAY", "#2f4a1a", "Rodarte, Jordan"),
    ]),
    ("MIDWEEK / CONFERENCE", "#2f4a1a", 2, [
        ("MIDWEEK", "#2f4a1a", "Gaskin, Brendan"),
        ("CONFERENCE", "#5a3500", "Howard, Austin"),
    ]),
    ("HIGH LEVERAGE BULLPEN", "#5a0a0d", 4, [
        ("CLOSER", "#7a0000", "Cawley, Declan"),
        ("SETUP", "#5a3200", "Kapica, Andrew"),
        ("7th INN", "#1a3a5a", "Vieira, William"),
        ("HI-K", "#1a4a1a", "Smith, Koen"),
    ]),
    ("MIDDLE RELIEF  /  SWING", "#3d3d1a", 4, [
        ("SWING", "#5a3500", "Chavez, Caden"),
        ("MIDDLE", "#454512", "Murray, Alec"),
        ("MIDDLE", "#454512", "Redick, James"),
        ("MIDDLE", "#454512", "Sullivan, Maximus"),
    ]),
    ("DEPTH", "#3b1a1d", 4, [
        ("LONG", "#2a2a2a", "Osterhus, Eric"),
        ("DEPTH", "#2a2a2a", "O'Brien-Gonzalez, A."),
        ("LHP", "#2a2a2a", "Curran, Tommy"),
        ("DEPTH", "#2a2a2a", "Swain, Carson"),
    ]),
]


def _box(ax, x, y, w, h, face, edge, lw=3, r=12, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=face, ec=edge, lw=lw, zorder=z))


def build() -> Path:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    _box(ax, 12, 12, W - 24, 140, "#8b1717", "#8b1717", lw=0, r=0, z=1)
    ax.plot([12, W - 12], [152, 152], color=GOLD, lw=8, zorder=2)
    ax.text(W / 2, 75, "FORDHAM BASEBALL", ha="center", va="center", color=GOLD, fontsize=34, weight="bold", zorder=3)
    ax.text(W / 2, 130, "2 0 2 7   P I T C H I N G   D E P T H   C H A R T",
            ha="center", va="center", color="#e8d8bd", fontsize=14, weight="bold", zorder=3)
    if LOGO.exists():
        img = mpimg.imread(str(LOGO))
        for lx in (50, W - 150):
            ax.imshow(img, extent=(lx, lx + 100, 130, 30), zorder=4)

    y = 175
    for title, color, cols, cards in SECTIONS:
        _box(ax, 37, y, W - 74, 70, color, color, lw=0, r=10)
        ax.text(W / 2, y + 32, title, ha="center", va="center", color=GOLD, fontsize=15, weight="bold", zorder=5)
        ax.plot([55, W - 55], [y + 58, y + 58], color=GOLD, alpha=0.6, lw=4, zorder=4)
        y += 82
        gap = 14
        cw = (W - 74 - gap * (cols - 1)) / cols
        x0 = 37 + (W - 74 - (cw * len(cards) + gap * (len(cards) - 1))) / 2
        ch = 150 if cols <= 3 else 125
        for i, (label, lc, name) in enumerate(cards):
            x = x0 + i * (cw + gap)
            _box(ax, x, y, cw, ch, CARD, GOLD)
            _box(ax, x + 7, y + 7, cw - 14, 50, lc, lc, lw=0, r=6, z=4)
            ax.text(x + cw / 2, y + 32, label, ha="center", va="center", color=WHITE,
                    fontsize=13 if cols <= 3 else 11, weight="bold", zorder=5)
            ax.text(x + cw / 2, y + 57 + (ch - 57) / 2, name, ha="center", va="center", color=WHITE,
                    fontsize=17 if cols <= 3 else 14, weight="bold", zorder=5)
        y += ch + 18

    ax.text(W / 2, H - 35, "FORDHAM BASEBALL  ·  INTERNAL USE ONLY  ·  2027 SEASON PROJECTION",
            ha="center", va="center", color="#6b6255", fontsize=12)
    fig.savefig(OUT, dpi=100, facecolor=BG)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(build())
