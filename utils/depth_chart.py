"""2027 depth chart: stored data, validation and the field / staff graphics.

The chart lives in depth_chart/depth_chart_2027.json so staff edits from the
Depth Chart page survive redeploys (the page pushes the file back to GitHub).
"""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Wedge, Circle
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "depth_chart" / "depth_chart_2027.json"
LOGO = ROOT / "assets" / "rams.png"

POSITIONS = ["C", "1B", "2B", "SS", "3B", "LF", "CF", "RF", "DH"]
HANDS = ["RHP", "LHP"]
PITCHING_SECTIONS = [
    "WEEKEND ROTATION",
    "MIDWEEK / CONFERENCE",
    "HIGH LEVERAGE BULLPEN",
    "MIDDLE RELIEF  /  SWING",
    "DEPTH",
    "ADDITIONAL DEPTH",
]

BG = "#100c0b"
GOLD = "#c9a25e"
MAROON = "#8b1717"
CARD = "#1a1512"
GRASS = "#1f3a10"
DIRT = "#45491a"
WHITE = "#f5f0e6"
MUTED = "#b8ae9c"
DIM = "#7d7365"

SECTION_COLORS = {
    "WEEKEND ROTATION": "#8b1717",
    "MIDWEEK / CONFERENCE": "#2f4a1a",
    "HIGH LEVERAGE BULLPEN": "#5a0a0d",
    "MIDDLE RELIEF  /  SWING": "#3d3d1a",
    "DEPTH": "#3b1a1d",
    "ADDITIONAL DEPTH": "#3b1a1d",
}
ROLE_COLORS = {
    "FRIDAY": "#6b0f0f", "SATURDAY": "#00349a", "SUNDAY": "#2f4a1a",
    "MIDWEEK": "#2f4a1a", "CONFERENCE": "#5a3500",
    "CLOSER": "#7a0000", "SETUP": "#5a3200", "7TH INN": "#1a3a5a", "HI-K": "#1a4a1a",
    "SWING": "#5a3500", "MIDDLE": "#454512", "PROJ": "#1a3a1a",
}

POS_XY = {
    "LF": (270, 290), "CF": (1000, 210), "RF": (1730, 290),
    "SS": (745, 640), "2B": (1255, 640),
    "3B": (455, 912), "1B": (1545, 912),
    "C": (1000, 1440), "DH": (250, 1440),
}


# ── Storage ──────────────────────────────────────────────────────────────────

def load_chart(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text())


def chart_to_json(chart: dict) -> str:
    return json.dumps(chart, indent=2, ensure_ascii=False) + "\n"


def save_chart(chart: dict, path: Path = DATA_PATH) -> bytes:
    content = chart_to_json(chart).encode()
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return content


def bats_lookup(chart: dict) -> dict:
    return {h["name"]: h.get("bats", "") for h in chart["roster"]["hitters"]}


def validate_lineup(rows: list) -> list:
    """Return a list of problems with one lineup (empty when it is usable)."""
    problems = []
    orders = [r.get("order") for r in rows]
    if sorted(o for o in orders if o is not None) != list(range(1, 10)) or len(rows) != 9:
        problems.append("Batting order must use each spot 1 through 9 exactly once.")
    positions = [r.get("pos") for r in rows]
    missing = [p for p in POSITIONS if p not in positions]
    if missing:
        problems.append(f"Every position needs a starter (missing: {', '.join(missing)}).")
    players = [r.get("player") for r in rows if r.get("player")]
    dupes = sorted({p for p in players if players.count(p) > 1})
    if dupes:
        problems.append(f"Listed as a starter twice: {', '.join(dupes)}.")
    if len(players) < len(rows):
        problems.append("Every lineup spot needs a player.")
    return problems


# ── Graphics ─────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, face, edge, lw=3, r=14, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=face, ec=edge, lw=lw, zorder=z))


def _canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    return fig, ax


def _logos(ax, xs, extent_y, size):
    if LOGO.exists():
        img = mpimg.imread(str(LOGO))
        for lx in xs:
            ax.imshow(img, extent=(lx, lx + size, extent_y[0], extent_y[1]), zorder=6)


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


def lineup_figure(chart: dict, hand: str):
    W, H = 2000, 1900
    rows = sorted(chart["lineups"][hand], key=lambda r: r.get("order") or 99)
    bats = bats_lookup(chart)
    fig, ax = _canvas(W, H)

    ax.text(W / 2, 45, "FORDHAM BASEBALL", ha="center", va="center", color=GOLD, fontsize=34, weight="bold")
    ax.text(W / 2, 100, f"2 0 2 7   L I N E U P   V S   {' '.join(hand)}",
            ha="center", va="center", color="#cbbfa9", fontsize=16, weight="bold")
    ax.plot([75, W - 75], [128, 128], color="#6b5a3a", lw=2)
    _logos(ax, (115, W - 195), (105, 25), 80)

    ax.add_patch(Wedge((1000, 1350), 1230, 225, 315, fc=GRASS, ec="none", zorder=1))
    ax.add_patch(Polygon([(1000, 460), (1415, 875), (1000, 1290), (585, 875)], closed=True,
                         fc=DIRT, ec=WHITE, lw=3, zorder=2))
    ax.add_patch(Circle((1000, 900), 40, fc="#86651a", ec=WHITE, lw=2, zorder=2))
    for dx in (-1, 1):
        ax.plot([1000, 1000 + dx * 1000], [1350, 350], ls="--", color="#777", lw=2, zorder=1)

    for r in rows:
        if r.get("pos") in POS_XY and r.get("player"):
            backups = [b for b in (r.get("backup1"), r.get("backup2")) if b]
            _pos_card(ax, r["pos"], r["player"], backups, r.get("order"))

    px, py, pw, ph = 1290, 1180, 640, 520
    _box(ax, px, py, pw, ph, CARD, GOLD)
    _box(ax, px + 6, py + 6, pw - 12, 44, MAROON, MAROON, lw=0, r=6, z=4)
    ax.text(px + pw / 2, py + 29, f"BATTING ORDER VS {hand}", ha="center", va="center",
            color=GOLD, fontsize=15, weight="bold", zorder=5)
    for i, r in enumerate(rows[:9]):
        yy = py + 85 + i * 49
        name = r.get("player") or "—"
        b = bats.get(name, "")
        ax.text(px + 30, yy, f"{r.get('order') or i + 1}", ha="center", va="center", color=GOLD,
                fontsize=15, weight="bold", zorder=5)
        ax.text(px + 60, yy, f"{name} ({b})" if b else name, ha="left", va="center", color=WHITE,
                fontsize=14, weight="bold", zorder=5)
        ax.text(px + 330, yy, r.get("pos") or "", ha="center", va="center", color=GOLD,
                fontsize=13, weight="bold", zorder=5)
        ax.text(px + pw - 20, yy, r.get("note") or "", ha="right", va="center", color=MUTED,
                fontsize=12, zorder=5)

    ax.plot([75, W - 75], [1745, 1745], color="#3d3426", lw=2)
    ax.text(W / 2, 1785, f"OPS vs {hand}: 2026 season + fall Pins vs Maroons intersquads combined",
            ha="center", va="center", color=DIM, fontsize=13, style="italic")
    ax.text(W / 2, 1850, "STARTER   ↳ BACKUP   ↳ DEPTH  ·  FORDHAM BASEBALL — INTERNAL USE ONLY",
            ha="center", va="center", color="#5a5145", fontsize=12)
    return fig


def _pitching_sections(chart: dict) -> list:
    """Group pitching rows into (section, cards) in display order."""
    by_section = {}
    for r in chart["pitching"]:
        if r.get("pitcher"):
            by_section.setdefault(r.get("section") or "DEPTH", []).append(r)
    ordered = [s for s in PITCHING_SECTIONS if s in by_section]
    ordered += [s for s in by_section if s not in PITCHING_SECTIONS]
    return [(s, by_section[s]) for s in ordered]


def pitching_figure(chart: dict):
    W = 2000
    sections = _pitching_sections(chart)
    layout = []
    for title, cards in sections:
        wide = title in ("WEEKEND ROTATION", "MIDWEEK / CONFERENCE")
        cols = max(1, min(len(cards), 3)) if wide else 4
        for start in range(0, len(cards), cols):
            layout.append((title if start == 0 else None, cols, cards[start:start + cols]))
    H = 175 + sum((82 if t else 0) + (150 if c <= 3 else 125) + 18 for t, c, _ in layout) + 70
    fig, ax = _canvas(W, H)

    _box(ax, 12, 12, W - 24, 140, "#8b1717", "#8b1717", lw=0, r=0, z=1)
    ax.plot([12, W - 12], [152, 152], color=GOLD, lw=8, zorder=2)
    ax.text(W / 2, 75, "FORDHAM BASEBALL", ha="center", va="center", color=GOLD, fontsize=34, weight="bold", zorder=3)
    ax.text(W / 2, 130, "2 0 2 7   P I T C H I N G   D E P T H   C H A R T",
            ha="center", va="center", color="#e8d8bd", fontsize=14, weight="bold", zorder=3)
    _logos(ax, (50, W - 150), (130, 30), 100)

    y = 175
    for title, cols, cards in layout:
        if title:
            color = SECTION_COLORS.get(title, "#3b1a1d")
            _box(ax, 37, y, W - 74, 70, color, color, lw=0, r=10)
            ax.text(W / 2, y + 32, title, ha="center", va="center", color=GOLD, fontsize=15, weight="bold", zorder=5)
            ax.plot([55, W - 55], [y + 58, y + 58], color=GOLD, alpha=0.6, lw=4, zorder=4)
            y += 82
        gap = 14
        cw = (W - 74 - gap * (cols - 1)) / cols
        x0 = 37 + (W - 74 - (cw * len(cards) + gap * (len(cards) - 1))) / 2
        ch = 150 if cols <= 3 else 125
        for i, r in enumerate(cards):
            label = (r.get("role") or "").upper()
            lc = ROLE_COLORS.get(label, "#2a2a2a")
            x = x0 + i * (cw + gap)
            _box(ax, x, y, cw, ch, CARD, GOLD, r=12)
            _box(ax, x + 7, y + 7, cw - 14, 50, lc, lc, lw=0, r=6, z=4)
            ax.text(x + cw / 2, y + 32, label, ha="center", va="center", color=WHITE,
                    fontsize=13 if cols <= 3 else 11, weight="bold", zorder=5)
            ax.text(x + cw / 2, y + 57 + (ch - 57) / 2, r["pitcher"], ha="center", va="center", color=WHITE,
                    fontsize=17 if cols <= 3 else 14, weight="bold", zorder=5)
        y += ch + 18

    ax.text(W / 2, H - 35, "FORDHAM BASEBALL  ·  INTERNAL USE ONLY  ·  2027 SEASON PROJECTION",
            ha="center", va="center", color="#6b6255", fontsize=12)
    return fig


def figure_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=BG)
    plt.close(fig)
    return buf.getvalue()
