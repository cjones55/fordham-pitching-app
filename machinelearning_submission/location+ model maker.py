#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm
from lightgbm import LGBMRegressor

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path.home() / "Desktop" / "college_raw_data"
MODELS_DIR = Path("models_location_plus_lgbm")

RENAME_MAP = {
    "PlateLocSide": "plate_x",
    "PlateLocHeight": "plate_z",
    "RelSpeed": "Velo",
    "InducedVertBreak": "IVB",
    "HorzBreak": "HB",
    "SpinRate": "Spin",
    "RelHeight": "RelH",
    "RelSide": "RelS",
    "Extension": "Ext",
    "VertApprAngle": "VAA",
    "HorzApprAngle": "HAA",
}

LOCATION_FEATURES = [
    "plate_x",
    "plate_z",
    "zone",
    "balls",
    "strikes",
    "pitch_abbr"
]

# ============================================================
# HELPERS
# ============================================================

def is_pitching_file(df: pd.DataFrame) -> bool:
    return "PitchCall" in df.columns and "PlateLocSide" in df.columns

def load_all_pitches(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    csv_files = list(base_dir.rglob("*.csv"))
    all_rows = []

    print(f"\n📥 Loading TrackMan CSVs ({len(csv_files)} files)...")
    for p in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not is_pitching_file(df):
            continue
        df["source_file"] = str(p)
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No valid pitching CSVs found.")

    df = pd.concat(all_rows, ignore_index=True)
    print(f"Loaded {len(df):,} pitches.")
    return df

# ============================================================
# RUN VALUE ENGINE
# ============================================================

RUN_VALUE_MAP = {
    "Single": 0.47,
    "Double": 0.78,
    "Triple": 1.09,
    "HomeRun": 1.40,
    "InPlayOut": -0.27,
    "InPlayNoOut": 0.00,
    "Strikeout": -0.29,
    "Walk": 0.33,
    "HitByPitch": 0.33,
}

def compute_run_value(row):
    pr = str(row.get("PlayResult", ""))
    return RUN_VALUE_MAP.get(pr, 0.0)

# ============================================================
# PREP
# ============================================================

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    print("\n⚙️  Preparing features and run value target...")

    # Rename columns
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # Pitch type normalization
    pitch_map = {
        "Fastball": "FB",
        "FourSeamFastBall": "FB",
        "4-Seam": "FB",
        "FF": "FB",
        "Sinker": "SI",
        "Cutter": "FC",
        "Slider": "SL",
        "Sweeper": "SW",
        "Curveball": "CU",
        "ChangeUp": "CH",
        "Changeup": "CH"
    }

    df["pitch_abbr"] = df["TaggedPitchType"].map(pitch_map)
    df["pitch_abbr"] = df["pitch_abbr"].fillna(
        df["TaggedPitchType"].astype(str).str[:2].str.upper()
    )

    # Count (ensure numeric)
    df["balls"] = pd.to_numeric(df.get("Balls", 0), errors="coerce").fillna(0).astype(int)
    df["strikes"] = pd.to_numeric(df.get("Strikes", 0), errors="coerce").fillna(0).astype(int)

    # Zone fallback
    if "zone" not in df.columns:
        df["zone"] = 0

    # Run value target
    df["run_value"] = df.apply(compute_run_value, axis=1)

    # Drop rows missing location features
    df = df.dropna(subset=["plate_x", "plate_z"])

    # Encode pitch_abbr as category codes (numeric for LightGBM)
    df["pitch_abbr"] = df["pitch_abbr"].astype("category").cat.codes

    return df

# ============================================================
# TRAIN LOCATION+ MODEL
# ============================================================

def train_location_model(df: pd.DataFrame, models_dir: Path = MODELS_DIR):
    models_dir.mkdir(exist_ok=True)

    X = df[LOCATION_FEATURES]
    y = df["run_value"]

    print("\n⚡ Training LightGBM Location+ model (run value)...")

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    joblib.dump(model, models_dir / "location_lgbm_model.pkl")

    # League distribution of predicted RV
    pred = model.predict(X)
    mu = pred.mean()
    sigma = pred.std() if pred.std() > 0 else 1.0
    league = {"mean": float(mu), "std": float(sigma)}
    joblib.dump(league, models_dir / "location_lgbm_league.pkl")

    print(f"League mean RV: {mu:.3f}, std: {sigma:.3f}")

# ============================================================
# GRADE PITCHES (Location+ PER ROW)
# ============================================================

def grade_pitches(df: pd.DataFrame, models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    print("\n📊 Generating Location+ for each pitch...")

    model = joblib.load(models_dir / "location_lgbm_model.pkl")
    league = joblib.load(models_dir / "location_lgbm_league.pkl")

    X = df[LOCATION_FEATURES]
    pred = model.predict(X)

    mu = league["mean"]
    sigma = league["std"] if league["std"] > 0 else 1.0

    df = df.copy()
    df["LocationPlus"] = 100 + 50 * ((pred - mu) / sigma)

    return df

# ============================================================
# MAIN
# ============================================================

def main():
    df = load_all_pitches()
    df = prepare(df)

    train_location_model(df)

    graded = grade_pitches(df)

    print("\nSample graded pitches:")
    cols_show = [
        "PitchCall","plate_x","plate_z","zone",
        "balls","strikes","pitch_abbr","run_value","LocationPlus"
    ]
    cols_show = [c for c in cols_show if c in graded.columns]
    print(graded[cols_show].head(20))

    # Fix object columns before saving
    for col in graded.columns:
        if graded[col].dtype == "object":
            graded[col] = graded[col].astype("string")

    out_path = Path("graded_pitches_with_locationplus_lgbm.parquet")
    graded.to_parquet(out_path, index=False)
    print(f"\nSaved graded pitches with LocationPlus to: {out_path}")

if __name__ == "__main__":
    main()
