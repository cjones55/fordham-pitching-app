#!/usr/bin/env python3
"""
Build the D1 baselines for Contact+ from the scouting TrackMan parquet files.

  - component means: per-ball D1 averages (so 100 = an average D1 ball in play)
  - component stds:  spread of pitcher-level rates among pitchers with >= MIN_BIP
  - shrink_k:        BIP at which half a pitcher's score is his own, from the
                     split-half reliability of the blended score
  - shrunk_std:      spread of the regressed score, so Contact+ has SD 10

Output: models/contact_league.json  (read by utils/contact.py)

Usage: python scripts/build_contact_league.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils.contact import CONTACT_WEIGHTS, batted_balls, contact_components

MIN_BIP = 40
OUT_PATH = ROOT / "models" / "contact_league.json"
COLS = ["Date", "GameID", "PAofInning", "PitchofPA", "Pitcher", "PitcherTeam", "Inning",
        "PitchCall", "ExitSpeed", "Angle", "Direction"]


def blended(comp: pd.DataFrame, components: dict) -> pd.Series:
    return sum(
        w * sign * (comp[col] - components[col]["mean"]) / components[col]["std"]
        for col, (w, sign) in CONTACT_WEIGHTS.items()
    )


def main():
    parts = sorted(ROOT.glob("scouting_data_*.parquet"))
    if not parts:
        sys.exit("No scouting_data_*.parquet files found.")
    df = pd.concat([pd.read_parquet(p, columns=COLS) for p in parts], ignore_index=True)
    df = df.drop_duplicates()
    print(f"Loaded {len(df):,} pitches from {len(parts)} file(s)")

    xba_model = joblib.load(ROOT / "models" / "xba_lgbm.pkl")
    bip = batted_balls(df, xba_model)
    print(f"Tracked balls in play: {len(bip):,}")

    per_ball = {
        "AvgEV": bip["_ev"].mean(),
        "HardHit%": bip["_hh"].mean() * 100,
        "xwOBAcon": bip["_xwoba"].mean(),
        "Barrel%": bip["_barrel"].mean() * 100,
        "GB%": bip["_gb"].mean() * 100,
    }
    pitchers = contact_components(bip, ["Pitcher", "PitcherTeam"])
    qualified = pitchers[pitchers["BIP"] >= MIN_BIP]
    components = {
        col: {"mean": float(per_ball[col]), "std": float(qualified[col].std())}
        for col in CONTACT_WEIGHTS
    }

    # Split-half reliability of the blended score -> regression constant.
    rng = np.random.default_rng(0)
    shuffled = bip.iloc[rng.permutation(len(bip))].copy()
    shuffled["_half"] = shuffled.groupby(["Pitcher", "PitcherTeam"]).cumcount() % 2
    halves = contact_components(shuffled, ["Pitcher", "PitcherTeam", "_half"])
    halves["raw"] = blended(halves, components)
    wide = halves.pivot_table(index=["Pitcher", "PitcherTeam"], columns="_half", values=["raw", "BIP"])
    wide = wide[(wide[("BIP", 0)] >= MIN_BIP / 2) & (wide[("BIP", 1)] >= MIN_BIP / 2)]
    r = float(wide[("raw", 0)].corr(wide[("raw", 1)]))
    half_n = float(wide[[("BIP", 0), ("BIP", 1)]].mean(axis=1).median())
    shrink_k = half_n * (1 - r) / r

    raw = blended(qualified, components)
    shrunk = raw * qualified["BIP"] / (qualified["BIP"] + shrink_k)
    shrunk_std = float(shrunk.std())

    league = {
        "source": [p.name for p in parts],
        "n_bip": int(len(bip)),
        "n_pitchers_qualified": int(len(qualified)),
        "min_bip": MIN_BIP,
        "weights": {c: w for c, (w, _) in CONTACT_WEIGHTS.items()},
        "components": components,
        "split_half_r": r,
        "split_half_median_bip": half_n,
        "shrink_k": float(shrink_k),
        "shrunk_std": shrunk_std,
    }
    OUT_PATH.write_text(json.dumps(league, indent=2) + "\n")

    check = 100 + 10 * shrunk / shrunk_std
    print(json.dumps({k: v for k, v in league.items() if k != "source"}, indent=2))
    print(f"Qualified D1 Contact+: mean {check.mean():.1f}, sd {check.std():.1f}, "
          f"p10 {check.quantile(.1):.1f}, p90 {check.quantile(.9):.1f}")
    print(f"Saved {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
