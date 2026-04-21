#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import joblib
from pathlib import Path
from lightgbm import LGBMClassifier
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path.home() / "Desktop" / "college_raw_data"
MODELS_DIR = Path("models_stuff_single_pitch_lgbm")

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
}

STUFF_FEATURES = [
    "Velo","IVB","HB","Spin",
    "RelH","RelS","Ext",
    "VAA","HAA"
]


# ============================================================
# LOADING
# ============================================================

def is_pitching_file(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in RENAME_MAP.keys()) and "PitchCall" in df.columns

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
# PREP / TARGET
# ============================================================

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    print("\n⚙️  Preparing features and CSW target...")

    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # CSW target: called or swinging strike
    df["CSW"] = df["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).astype(int)

    # Keep only rows where we have stuff features
    df = df.dropna(subset=[c for c in STUFF_FEATURES if c in df.columns])

    return df


# ============================================================
# TRAIN GLOBAL STUFF MODEL (LIGHTGBM)
# ============================================================

def train_stuff_model(df: pd.DataFrame, models_dir: Path = MODELS_DIR):
    models_dir.mkdir(exist_ok=True)

    X = df[STUFF_FEATURES]
    y = df["CSW"]

    print("\n⚡ Training LightGBM Stuff model (CSW probability)...")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    joblib.dump(model, models_dir / "stuff_lgbm_model.pkl")

    # League distribution of predicted CSW probability
    p = model.predict_proba(X)[:, 1]
    mu = p.mean()
    sigma = p.std() if p.std() > 0 else 1.0
    league = {"mean": float(mu), "std": float(sigma)}
    joblib.dump(league, models_dir / "stuff_lgbm_league.pkl")

    print(f"League mean CSW prob: {mu:.3f}, std: {sigma:.3f}")


# ============================================================
# GRADE SINGLE PITCHES (Stuff+ PER ROW)
# ============================================================

def grade_pitches(df: pd.DataFrame, models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    print("\n📊 Generating Stuff+ for each pitch...")

    model = joblib.load(models_dir / "stuff_lgbm_model.pkl")
    league = joblib.load(models_dir / "stuff_lgbm_league.pkl")

    X = df[STUFF_FEATURES]
    p = model.predict_proba(X)[:, 1]

    mu = league["mean"]
    sigma = league["std"] if league["std"] > 0 else 1.0

    df = df.copy()
    df["StuffPlus"] = 100 + 50 * ((p - mu) / sigma)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_all_pitches()
    df = prepare(df)

    train_stuff_model(df)

    graded = grade_pitches(df)

    print("\nSample graded pitches:")
    cols_show = ["PitchCall","Velo","IVB","HB","Spin","RelH","RelS","Ext","VAA","HAA","StuffPlus"]
    cols_show = [c for c in cols_show if c in graded.columns]
    print(graded[cols_show].head(20))

    # 🔧 Fix mixed-type object columns before saving
    for col in graded.columns:
        if graded[col].dtype == "object":
            graded[col] = graded[col].astype("string")

    out_path = Path("graded_pitches_with_stuffplus_lgbm.parquet")
    graded.to_parquet(out_path, index=False)
    print(f"\nSaved graded pitches with StuffPlus to: {out_path}")

if __name__ == "__main__":
    main()
