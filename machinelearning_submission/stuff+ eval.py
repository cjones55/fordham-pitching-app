#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)

# ============================================================
# CORRECT PATHS (based on your screenshot)
# ============================================================

MODEL_PATH = "/Users/chrisjones/Desktop/models_stuff_single_pitch_lgbm/stuff_lgbm_model.pkl"
LEAGUE_PATH = "/Users/chrisjones/Desktop/models_stuff_single_pitch_lgbm/stuff_lgbm_league.pkl"
DATA_PATH = "/Users/chrisjones/Desktop/models_stuff_single_pitch_lgbm/graded_pitches_with_stuffplus_lgbm.parquet"

print("\n📥 Loading Stuff+ model and data...")
model = joblib.load(MODEL_PATH)
league = joblib.load(LEAGUE_PATH)
df = pd.read_parquet(DATA_PATH)

# Features + target
X = df[["Velo","IVB","HB","Spin","RelH","RelS","Ext","VAA","HAA"]]
y = df["CSW"]

# ============================================================
# PREDICTIONS
# ============================================================

probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)

# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(y, preds)
prec = precision_score(y, preds)
rec = recall_score(y, preds)
f1 = f1_score(y, preds)
auc = roc_auc_score(y, probs)
ll = log_loss(y, probs)

print("\n=== Stuff+ Model Evaluation ===")
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"ROC–AUC:       {auc:.4f}")
print(f"Log Loss:      {ll:.4f}")

print("\nDone evaluating Stuff+ model.")
