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
    "Sweeper": "SW", "SW": "SW"
}

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
    df["zone"] = pd.to_numeric(df.get("zone", 0), errors="coerce").fillna(0)

    loc_X = pd.DataFrame({
        "PlateLocSide": df["PlateLocSide"],
        "PlateLocHeight": df["PlateLocHeight"],
        "zone": df["zone"],
        "Balls": df["Balls"],
        "Strikes": df["Strikes"],
        "pitch_abbr": df["pitch_abbr_code"],
    })

    # -------------------------------
    # FIX: LightGBM _n_classes bug
    # -------------------------------
    if hasattr(loc_model, "_n_classes") and loc_model._n_classes is None:
        loc_model._n_classes = 1

    df["loc_pred"] = loc_model.predict(loc_X)
    df["Loc+"] = 100 + 50 * ((df["loc_pred"] - mu) / sigma)
    return df
