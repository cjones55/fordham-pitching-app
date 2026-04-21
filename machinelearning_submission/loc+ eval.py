#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:34:10 2026

@author: chrisjones
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ============================================================
# CORRECT PATHS (based on your folder structure)
# ============================================================

MODEL_PATH = "/Users/chrisjones/Desktop/models_location_plus_lgbm/location_lgbm_model.pkl"
LEAGUE_PATH = "/Users/chrisjones/Desktop/models_location_plus_lgbm/location_lgbm_league.pkl"
DATA_PATH = "/Users/chrisjones/Desktop/models_location_plus_lgbm/graded_pitches_with_locationplus_lgbm.parquet"

print("\n📥 Loading Location+ model and data...")
model = joblib.load(MODEL_PATH)
league = joblib.load(LEAGUE_PATH)
df = pd.read_parquet(DATA_PATH)

# ============================================================
# FEATURES + TARGET
# ============================================================

X = df[["plate_x","plate_z","zone","balls","strikes","pitch_abbr"]]
y = df["run_value"]

# ============================================================
# PREDICTIONS
# ============================================================

pred = model.predict(X)

# ============================================================
# METRICS
# ============================================================

rmse = mean_squared_error(y, pred, squared=False)
mae = mean_absolute_error(y, pred)
r2 = r2_score(y, pred)

# Baseline: always predict league mean RV
baseline_pred = np.full_like(y, league["mean"])
baseline_rmse = mean_squared_error(y, baseline_pred, squared=False)
baseline_mae = mean_absolute_error(y, baseline_pred)

print("\n=== Location+ Model Evaluation ===")
print(f"RMSE:              {rmse:.4f}")
print(f"MAE:               {mae:.4f}")
print(f"R²:                {r2:.4f}")
print(f"Baseline RMSE:     {baseline_rmse:.4f}")
print(f"Baseline MAE:      {baseline_mae:.4f}")

print("\nDone evaluating Location+ model.")
