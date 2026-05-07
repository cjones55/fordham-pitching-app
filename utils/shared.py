#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PITCH_MAP = {
    "Fastball": "FB", "FourSeamFastBall": "FB", "FourSeamFastball": "FB",
    "4-Seam": "FB", "FF": "FB", "FA": "FB", "FB": "FB",
    "Sinker": "SI", "SI": "SI", "SNK": "SI", "SU": "SI",
    "Cutter": "FC", "FC": "FC", "CT": "FC",
    "Slider": "SL", "SL": "SL",
    "Curveball": "CU", "CurveBall": "CU", "CB": "CU", "CU": "CU",
    "ChangeUp": "CH", "Changeup": "CH", "CH": "CH",
    "Sweeper": "SW", "SW": "SW",
    "Splitter": "SP", "Split-Finger": "SP", "SplitFinger": "SP", "FS": "SP",
}

# Pitcher-specific pitch type overrides applied after standard mapping.
# Key: (pitcher_name_as_in_trackman, from_abbr)  →  Value: to_abbr
PITCHER_PITCH_OVERRIDES = {
    ("Stewart, Robbie", "FB"): "SI",
    ("Stewart, Robbie", "SW"): "SL",
    ("Stewart, Robbie", "CU"): "SL",
    ("Berg, Aric",      "SW"): "SL",
    ("Kapica, Andrew",  "SI"): "FB",
    ("Kapica, Andrew",  "FC"): "SL",
    ("Murray, Alec",    "CU"): "SL",
    ("Murray, Alec",    "SW"): "SL",
    ("Murray, Alec",    "CH"): "FC",
    ("Egan, Ryan",      "CU"): "SL",
    ("Chavez, Caden",   "FC"): "SL",
    ("Dowd, Aidan",     "SW"): "SL",   # no sweeper in arsenal
    ("Dowd, Aidan",     "SI"): "FB",   # sinkers at FB velocity/shape
    ("Dowd, Aidan",     "SP"): "FB",   # 1 splitter is a fastball
}

# Conditional overrides: remap from_abbr → to_abbr only when a numeric column
# satisfies the threshold. Tuple: (pitcher, from_abbr, to_abbr, column, op, threshold)
# Supported ops: ">=", ">", "<=", "<"
PITCHER_PITCH_CONDITIONAL_OVERRIDES = [
    ("Hanawalt, Chase", "SL", "FC", "IVB", ">=", -6),
    ("Hanawalt, Chase", "SL", "CU", "IVB", "<",  -6),
    ("Hanawalt, Chase", "CH", "FC", "IVB", "<",  -3),   # CHs with deep negative IVB are cutters
    ("Berg, Aric",      "FB", "CU", "Velo", "<",  75),   # 1 pitch at 71.7 mph is a curveball
    ("Berg, Aric",      "SL", "CU", "IVB",  "<",  -5),  # deep-breaking sliders are curveballs
    ("Berg, Aric",      "CU", "SL", "IVB",  ">=", -5),  # shallow curveballs are sliders
    ("Elson, Beau",     "CH", "CU", "IVB",  "<", -10),  # 1 CH at IVB -13.4 is a curveball
    ("Dowd, Aidan",     "CH", "SL", "IVB",  "<", -3),   # deep-negative-IVB CH → slider
    ("Dowd, Aidan",     "CH", "FC", "HB",   "<",  0),   # gloveside CH (HB<0) → cutter
    ("Dowd, Aidan",     "FC", "FB", "IVB",  ">=", 10),  # 9 high-ride "cutters" (IVB≥10, ~85.5 mph) → FB
    ("Dowd, Aidan",     "SL", "CH", "HB",   ">",  5),   # 1 arm-side "slider" (HB+18.7, IVB+10) → CH
    ("Egan, Ryan",      "SI", "FB", "Velo", ">=", 76),  # normal-velo SI → FB
    ("Egan, Ryan",      "SI", "SL", "Velo", "<",  76),  # 73 mph SI → SL
    ("Egan, Ryan",      "SL", "FB", "Velo", ">",  85),  # 87 mph tagged SL → FB (must run before SL→CH)
    ("Egan, Ryan",      "SL", "CH", "HB",   ">",   8),  # arm-side SL → CH
    ("Vieira, William", "CH", "CU", "IVB",  "<", -4),   # 1 CH at IVB -5.4 is a curveball
]

RENAME_MAP = {
    "RelSpeed": "Velo",
    "InducedVertBreak": "IVB",
    "HorzBreak": "HB",
    "SpinRate": "Spin",
    "RelHeight": "RelH",
    "RelSide": "RelS",
    "Extension": "Ext",
    "VertApprAngle": "VAA",
    "HorzApprAngle": "HAA",
    "PlateLocSide": "PlateLocSide",
    "PlateLocHeight": "PlateLocHeight",
}

STUFF_FEATURES = ["Velo","IVB","HB","Spin","RelH","RelS","Ext","VAA","HAA"]


def load_models(models_dir: Path = Path("models")):
    stuff_model = joblib.load(models_dir / "stuff_lgbm_model.pkl")
    stuff_league = joblib.load(models_dir / "stuff_lgbm_league.pkl")
    loc_model = joblib.load(models_dir / "location_lgbm_model.pkl")
    loc_league = joblib.load(models_dir / "location_lgbm_league.pkl")
    return stuff_model, stuff_league, loc_model, loc_league


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP)

    # -------------------------------
    # FIX: Pitcher column fallback
    # -------------------------------
    if "Pitcher" in df.columns:
        df["Pitcher"] = df["Pitcher"].astype(str).str.strip()
    elif "PitcherId" in df.columns:
        df["Pitcher"] = df["PitcherId"].astype(str)
    else:
        raise ValueError("CSV missing both Pitcher and PitcherId columns.")

    if "PitcherTeam" in df.columns:
        df["PitcherTeam"] = df["PitcherTeam"].astype(str).str.strip()
    else:
        raise ValueError("CSV missing PitcherTeam column.")

    df["pitch_abbr"] = df["TaggedPitchType"].map(PITCH_MAP)
    df["pitch_abbr"] = df["pitch_abbr"].fillna(
        df["TaggedPitchType"].astype(str).str[:2].str.upper()
    )

    for (pitcher, from_abbr), to_abbr in PITCHER_PITCH_OVERRIDES.items():
        mask = (df["Pitcher"].str.strip() == pitcher) & (df["pitch_abbr"] == from_abbr)
        df.loc[mask, "pitch_abbr"] = to_abbr

    _ops = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b, "<": lambda a, b: a < b}
    for pitcher, from_abbr, to_abbr, col, op, thresh in PITCHER_PITCH_CONDITIONAL_OVERRIDES:
        vals = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
        base = (df["Pitcher"].str.strip() == pitcher) & (df["pitch_abbr"] == from_abbr)
        df.loc[base & _ops[op](vals, thresh), "pitch_abbr"] = to_abbr

    df = df[~df["pitch_abbr"].isin(["UN", "TW", "OT"])].reset_index(drop=True)

    return df


def filter_fordham(df: pd.DataFrame) -> pd.DataFrame:
    fordham_team = df["PitcherTeam"].mode()[0]
    return df[df["PitcherTeam"] == fordham_team].copy()


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_LHH"] = df["BatterSide"].eq("Left")
    df["is_RHH"] = df["BatterSide"].eq("Right")

    df["is_csw"] = df["PitchCall"].isin(["StrikeCalled","StrikeSwinging"])
    df["is_whiff"] = df["PitchCall"].eq("StrikeSwinging")

    df["is_swing"] = df["PitchCall"].isin([
        "StrikeSwinging","FoulBall","FoulBallNotFieldable",
        "InPlay","InPlayNoOut","InPlayOut"
    ])

    df["is_strike"] = df["PitchCall"].isin([
        "StrikeCalled","StrikeSwinging","FoulBall","FoulBallNotFieldable",
        "InPlay","InPlayNoOut","InPlayOut"
    ])

    df["in_zone"] = (
        df["PlateLocSide"].between(-0.83, 0.83) &
        df["PlateLocHeight"].between(1.5, 3.5)
    )
    return df


def compute_stuffplus(df: pd.DataFrame, stuff_model, stuff_league) -> pd.DataFrame:
    mu = stuff_league["mean"]
    sigma = stuff_league["std"] if stuff_league["std"] > 0 else 1.0

    Xs = df[STUFF_FEATURES].fillna(0)

    df["stuff_prob"] = stuff_model.predict_proba(Xs)[:, 1]
    df["Stuff+"] = 100 + 50 * ((df["stuff_prob"] - mu) / sigma)
    return df


def compute_locationplus(df: pd.DataFrame, loc_model, loc_league) -> pd.DataFrame:
    mu = loc_league["mean"]
    sigma = loc_league["std"] if loc_league["std"] > 0 else 1.0

    df["Balls"] = pd.to_numeric(df.get("Balls", 0), errors="coerce").fillna(0).astype(int)
    df["Strikes"] = pd.to_numeric(df.get("Strikes", 0), errors="coerce").fillna(0).astype(int)

    df["pitch_abbr_code"] = df["pitch_abbr"].astype("category").cat.codes

    # --------------------------------------------------------
    # FIX: zone fallback must be a Series, not a scalar
    # --------------------------------------------------------
    if "zone" in df.columns:
        df["zone"] = pd.to_numeric(df["zone"], errors="coerce").fillna(0)
    else:
        df["zone"] = 0  # creates a full-length Series automatically

    loc_X = pd.DataFrame({
        "PlateLocSide": df["PlateLocSide"],
        "PlateLocHeight": df["PlateLocHeight"],
        "zone": df["zone"],
        "Balls": df["Balls"],
        "Strikes": df["Strikes"],
        "pitch_abbr": df["pitch_abbr_code"],
    })

    # --------------------------------------------------------
    # FIX: LightGBM _n_classes bug
    # --------------------------------------------------------
    if hasattr(loc_model, "_n_classes") and loc_model._n_classes is None:
        loc_model._n_classes = 1

    df["loc_pred"] = loc_model.predict(loc_X)
    df["Loc+"] = 100 + 50 * ((df["loc_pred"] - mu) / sigma)
    return df
