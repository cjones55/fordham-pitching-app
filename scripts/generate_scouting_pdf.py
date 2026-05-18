#!/usr/bin/env python3
"""
Generate a multi-player hitter scouting PDF using the Fordham app's report engine.
Usage:  python scripts/generate_scouting_pdf.py
Output: scripts/scouting_report_<date>.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ── Mock streamlit before importing app so decorators are no-ops ──────────────
def _cache_passthrough(func=None, **kwargs):
    if func is not None:
        return func
    return lambda f: f

class _FakeCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass

def _columns(spec, **kw):
    n = len(spec) if hasattr(spec, "__len__") else int(spec)
    return [_FakeCtx() for _ in range(n)]

_st = MagicMock()
_st.cache_data      = _cache_passthrough
_st.cache_resource  = _cache_passthrough
_st.session_state   = {}
_st.columns         = _columns
_st.secrets         = MagicMock()
_st.secrets.__getitem__ = lambda self, k: {}
_st.secrets.get     = lambda k, d=None: d
sys.modules["streamlit"] = _st

# ── Add app root to path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from datetime import date

# ── Import functions from the Fordham app ────────────────────────────────────
import app as _app

# ── Player list ───────────────────────────────────────────────────────────────
# Format: (Last, First, BatterTeam code)
PLAYERS = [
    ("Britt",           "Sean",     "STJ_RED"),
    ("Gell",            "Scott",    "SBU_SEA"),
    ("Ruiz",            "Justin",   "HIG_PAN"),
    ("DiPrima",         "Chris",    "YAL_BUL"),
    ("Flores",          "Nick",     "VCU_RAM"),
    ("Maggy",           "Ryan",     "WES_MOU"),
    ("Menzel",          "Evan",     "UCO_HUS"),
    ("Balcer",          "Jack",     "QUI_BOB"),
    ("Breton",          "Mason",    "LON_ISL22"),
    ("Collins",         "Gavin",    "PEN_QUA"),
    ("Ducatelli",       "Antonio",  "CCU_BLD"),
    ("Irizarry",        "Julian",   "PIT_PAN"),
    ("Rice",            "Trent",    "MIC_SPA"),
    ("Francuzenko",     "Nicholas", "TOW_TIG"),
]

PARQUET_PARTS = [
    ROOT / "scouting_data_1.parquet",
    ROOT / "scouting_data_2.parquet",
]

# ── Team name / color helpers (mirror app) ────────────────────────────────────
def _team_label(code: str) -> str:
    return _app.TEAM_NAMES.get(code, code)

# ── Load full parquet once, filter per player ─────────────────────────────────
print("Loading scouting parquet…")
chunks = []
for p in PARQUET_PARTS:
    if p.exists():
        chunks.append(pd.read_parquet(p))
if not chunks:
    print("ERROR: No scouting parquet found.")
    sys.exit(1)
raw = pd.concat(chunks, ignore_index=True)
print(f"  {len(raw):,} rows loaded")

# Run standard cleaning
raw = _app._coerce_scouting_numeric(raw)
raw = _app.add_perceived_velocity(raw)

def load_player(last: str, first: str, team: str) -> tuple[pd.DataFrame, str]:
    """Return (filtered_df, exact_batter_name). df is empty if not found."""
    canonical = f"{last}, {first}"
    mask_team  = raw["BatterTeam"] == team
    mask_name  = raw["Batter"].str.contains(last, case=False, na=False)
    subset = raw[mask_team & mask_name].copy()
    if subset.empty:
        return pd.DataFrame(), canonical
    # Pick the closest name match
    names = subset["Batter"].unique()
    exact = [n for n in names if first.lower() in n.lower()]
    chosen = exact[0] if exact else names[0]
    return subset[subset["Batter"] == chosen].copy(), chosen

# ── Build PDF ─────────────────────────────────────────────────────────────────
out_path = ROOT / "scripts" / f"scouting_report_{date.today()}.pdf"
print(f"\nGenerating PDF → {out_path}")

skipped = []
included = []

with PdfPages(str(out_path)) as pdf:
    # Cover page
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    fig.text(0.5, 0.62, "Hitter Scouting Report",
             ha="center", va="center", color="#FFF7E8",
             fontsize=36, fontweight="bold")
    fig.text(0.5, 0.54, f"Fordham Baseball  ·  {date.today().strftime('%B %d, %Y')}",
             ha="center", va="center", color="#CDBFAF", fontsize=16)
    fig.text(0.5, 0.46, f"{len(PLAYERS)} players scouted",
             ha="center", va="center", color="#8C7B5E", fontsize=13)
    # Player index
    lines = []
    for i, (last, first, team) in enumerate(PLAYERS, 1):
        lines.append(f"{i:>2}.  {first} {last}  —  {_team_label(team)}")
    fig.text(0.5, 0.38, "\n".join(lines),
             ha="center", va="top", color="#CDBFAF", fontsize=9,
             fontfamily="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    for last, first, team in PLAYERS:
        hdf, batter_name = load_player(last, first, team)
        if hdf.empty:
            print(f"  ✗ {first} {last} ({team}) — no data, skipping")
            skipped.append(f"{first} {last} ({_team_label(team)})")
            continue
        team_label = _team_label(team)
        print(f"  ✓ {batter_name} — {team_label}  ({len(hdf)} pitches)")
        try:
            _app._append_hitter_scouting_pages(pdf, hdf, batter_name, team_label, team_code=team)
            included.append(batter_name)
        except Exception as e:
            print(f"    ERROR generating pages: {e}")
            skipped.append(f"{batter_name} (render error)")

print(f"\nDone — {len(included)} players included, {len(skipped)} skipped")
if skipped:
    print("Skipped (no data):")
    for s in skipped:
        print(f"  • {s}")
print(f"\nSaved: {out_path}")
