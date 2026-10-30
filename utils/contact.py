#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contact+ — a grade for the quality of contact a pitcher allows.

Scaled like Stuff+/Loc+: 100 = D1 average, every 10 points is one standard
deviation among D1 pitchers, higher = weaker contact allowed.

Per ball in play we look at five things, compare each pitcher's rate to D1
(as a z-score against the spread of D1 pitchers), and blend them:

    Avg EV        30%   lower is better
    HardHit%      30%   EV >= 95, lower is better
    xwOBAcon      20%   from the app's xBA model (EV, LA, spray), lower is better
    Barrel%       10%   EV >= 92 and LA 16-36, lower is better
    GB%           10%   LA < 10, higher is better

EV and HardHit% get the most weight because they are the most repeatable
(split-half r ~0.40 at 40 BIP vs ~0.14 for xwOBAcon) and the best predictors of a
pitcher's contact quality on his other balls in play. The blended score is
regressed toward average by BIP / (BIP + K), K ~70 BIP, so a 10-BIP pitcher
can't post a 140.

Baselines live in models/contact_league.json, built from the D1 scouting data by
scripts/build_contact_league.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BIP_CALLS = ["InPlay", "InPlayNoOut", "InPlayOut", "InPlayRun"]
HARD_HIT_EV = 95
BARREL_EV_MIN = 92
BARREL_LA_MIN = 16
BARREL_LA_MAX = 36
GROUND_BALL_LA = 10
WOBA_WEIGHTS = {1: 0.888, 2: 1.271, 3: 1.616, 4: 2.101}

# component -> (weight, sign). sign = -1 means a lower raw value is better.
CONTACT_WEIGHTS = {
    "AvgEV":    (0.30, -1),
    "HardHit%": (0.30, -1),
    "xwOBAcon": (0.20, -1),
    "Barrel%":  (0.10, -1),
    "GB%":      (0.10, +1),
}

CONTACT_LEAGUE_PATH = Path(__file__).resolve().parent.parent / "models" / "contact_league.json"


def load_contact_league(path: Path = CONTACT_LEAGUE_PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _first_numeric(df: pd.DataFrame, *names) -> pd.Series:
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=df.index)


def batted_balls(df: pd.DataFrame, xba_model=None) -> pd.DataFrame:
    """Tracked balls in play with the per-ball contact flags.

    Uses an existing xwOBA column when present, otherwise scores xwOBAcon with
    xba_model (the app's EV/LA/spray classifier) if one is passed.
    """
    if df is None or df.empty or "PitchCall" not in df.columns:
        return pd.DataFrame()

    ev = _first_numeric(df, "EV", "ExitSpeed")
    la = _first_numeric(df, "LA", "Angle")
    is_bip = df["PitchCall"].astype(str).isin(BIP_CALLS)
    mask = is_bip & ev.between(40, 130) & la.between(-90, 90)

    bip = df.loc[mask].copy()
    bip["_ev"] = ev[mask]
    bip["_la"] = la[mask]
    bip["_hh"] = bip["_ev"].ge(HARD_HIT_EV)
    bip["_barrel"] = bip["_ev"].ge(BARREL_EV_MIN) & bip["_la"].between(BARREL_LA_MIN, BARREL_LA_MAX)
    bip["_gb"] = bip["_la"].lt(GROUND_BALL_LA)

    if "xwOBA" in bip.columns and pd.to_numeric(bip["xwOBA"], errors="coerce").notna().any():
        bip["_xwoba"] = pd.to_numeric(bip["xwOBA"], errors="coerce")
    elif xba_model is not None and not bip.empty:
        direction = _first_numeric(bip, "Direction").fillna(0)
        X = np.column_stack([bip["_ev"], bip["_la"], direction]).astype(np.float32)
        p = xba_model.predict_proba(X)
        bip["_xwoba"] = sum(p[:, k] * w for k, w in WOBA_WEIGHTS.items())
    else:
        bip["_xwoba"] = np.nan
    return bip


def contact_components(bip: pd.DataFrame, group_cols) -> pd.DataFrame:
    """Raw contact-against stats per group (BIP, AvgEV, HardHit%, Barrel%, GB%, xwOBAcon)."""
    if bip.empty:
        return pd.DataFrame(columns=list(group_cols) + ["BIP"] + list(CONTACT_WEIGHTS))
    out = bip.groupby(list(group_cols)).agg(
        BIP=("_ev", "size"),
        AvgEV=("_ev", "mean"),
        HH=("_hh", "mean"),
        Barrel=("_barrel", "mean"),
        GB=("_gb", "mean"),
        xwOBAcon=("_xwoba", "mean"),
    ).reset_index()
    for src, dst in [("HH", "HardHit%"), ("Barrel", "Barrel%"), ("GB", "GB%")]:
        out[dst] = out.pop(src) * 100
    return out


def score_contact_plus(comp: pd.DataFrame, league: dict) -> pd.Series:
    """Contact+ for each row of a contact_components table."""
    if comp.empty or not league:
        return pd.Series(np.nan, index=comp.index, dtype=float)

    raw = pd.Series(0.0, index=comp.index)
    total_w = pd.Series(0.0, index=comp.index)
    for col, (w, sign) in CONTACT_WEIGHTS.items():
        base = league["components"][col]
        z = sign * (pd.to_numeric(comp[col], errors="coerce") - base["mean"]) / base["std"]
        ok = z.notna()
        raw[ok] += w * z[ok]
        total_w[ok] += w
    # Renormalize if a component (e.g. xwOBAcon without the model) is missing.
    raw = raw.where(total_w > 0) / total_w.where(total_w > 0)

    n = pd.to_numeric(comp["BIP"], errors="coerce").fillna(0)
    shrunk = raw * n / (n + league["shrink_k"])
    return 100 + 10 * shrunk / league["shrunk_std"]


def contact_plus_table(df: pd.DataFrame, group_cols, league: dict, xba_model=None, min_bip: int = 1) -> pd.DataFrame:
    """One-call helper: raw contact stats + Contact+ per group."""
    comp = contact_components(batted_balls(df, xba_model), group_cols)
    comp["Contact+"] = score_contact_plus(comp, league)
    return comp[comp["BIP"] >= min_bip].reset_index(drop=True)


def contact_grade(value: float) -> str:
    """A-F letter for a Contact+ score (10 points = 1 SD)."""
    if value is None or pd.isna(value):
        return "—"
    for cut, letter in [(120, "A+"), (115, "A"), (110, "A-"), (105, "B+"), (102, "B"),
                        (98, "B-"), (95, "C+"), (90, "C"), (85, "C-"), (80, "D")]:
        if value >= cut:
            return letter
    return "F"
