#!/usr/bin/env python3
"""
Train xBA, xSLG, and xwOBA models on 2026 college TrackMan BIP data.

Methodology (Baseball Savant-style):
  - LightGBM multi-class classifier on every tracked ball in play
  - Features: ExitSpeed, LaunchAngle, SprayDirection
  - Classes: Out(0), Single(1), Double(2), Triple(3), HomeRun(4)
  - xBA   = P(1B) + P(2B) + P(3B) + P(HR)
  - xSLG  = P(1B)×1 + P(2B)×2 + P(3B)×3 + P(HR)×4
  - xwOBA = P(1B)×w1B + P(2B)×w2B + P(3B)×w3B + P(HR)×wHR
             (using standard linear weights, normalized to OBP scale)

Outputs:
  models/xba_lgbm.pkl       – trained LGBMClassifier
  models/xba_league.pkl     – league stats and calibration info
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
import warnings
warnings.filterwarnings("ignore")

# ── Outcome mapping ──────────────────────────────────────────────────────────
OUTCOME_MAP = {
    "Out":            0,
    "FieldersChoice": 0,   # batter out, runner advances — treat as out
    "Error":          0,   # fielding error; contact quality drove the result
    "Sacrifice":      0,   # intentional; remove below instead
    "Single":         1,
    "Double":         2,
    "Triple":         3,
    "HomeRun":        4,
}
CLASS_NAMES = ["Out", "Single", "Double", "Triple", "HomeRun"]

# Linear weights for xwOBA (2024 MLB standard, normalized to OBP scale)
# Source: FanGraphs guts page
WOBA_WEIGHTS = {1: 0.888, 2: 1.271, 3: 1.616, 4: 2.101}  # 1B, 2B, 3B, HR

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading scouting parquets…")
parts = [ROOT / f"scouting_data_{i}.parquet" for i in [1, 2]]
df = pd.concat([pd.read_parquet(p) for p in parts if p.exists()], ignore_index=True)
print(f"  {len(df):,} total pitches loaded")

# Numeric coerce
for col, src in [("EV", "ExitSpeed"), ("LA", "Angle"), ("Dir", "Direction")]:
    df[col] = pd.to_numeric(df.get(src, df.get(col)), errors="coerce")

# ── Filter to usable BIP ─────────────────────────────────────────────────────
bip = df[
    df["EV"].notna() &
    df["LA"].notna() &
    df["Dir"].notna() &
    df["PlayResult"].isin(OUTCOME_MAP.keys())
].copy()

# Remove sacrifice bunts (intentional contact, not quality-driven)
bip = bip[bip["PlayResult"] != "Sacrifice"]

# EV sanity bounds (exclude mis-tracked or popup-off-handle extremes)
bip = bip[(bip["EV"] >= 40) & (bip["EV"] <= 130)]
# LA bounds
bip = bip[(bip["LA"] >= -90) & (bip["LA"] <= 90)]

bip["outcome"] = bip["PlayResult"].map(OUTCOME_MAP)
print(f"  {len(bip):,} usable BIP after filtering")

print("\nOutcome distribution:")
for cls, name in enumerate(CLASS_NAMES):
    n = (bip["outcome"] == cls).sum()
    print(f"  {name:<10} {n:>8,}  ({n/len(bip)*100:.1f}%)")

# ── Features ─────────────────────────────────────────────────────────────────
FEATURES = ["EV", "LA", "Dir"]

# Add squared/interaction terms that help LightGBM (optional but helps)
# LightGBM handles these naturally via splits, so raw features are enough
X = bip[FEATURES].values.astype(np.float32)
y = bip["outcome"].values.astype(np.int32)

# ── Train / Val / Test split ─────────────────────────────────────────────────
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv
)

print(f"\nSplit: {len(X_train):,} train | {len(X_val):,} val | {len(X_test):,} test")

# ── Train LightGBM ───────────────────────────────────────────────────────────
# No class weights — we want calibrated probabilities, not class-balanced accuracy.
# The model learns from the natural distribution so predicted P(hit) ≈ actual BA.
print("\nTraining LightGBM multi-class model…")

model = LGBMClassifier(
    objective        = "multiclass",
    num_class        = 5,
    n_estimators     = 2000,
    learning_rate    = 0.03,
    max_depth        = 7,
    num_leaves       = 63,
    min_child_samples= 30,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 1.0,
    reg_alpha        = 0.05,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1,
)

model.fit(
    X_train, y_train,
    eval_set  = [(X_val, y_val)],
    callbacks = [
        lgb.early_stopping(stopping_rounds=60, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)

best_iter = model.best_iteration_
print(f"  Best iteration: {best_iter}")

# ── Evaluate on test set ─────────────────────────────────────────────────────
probs_test = model.predict_proba(X_test)   # shape (N, 5)

# xBA, xSLG, xwOBA per pitch
xba_test   = probs_test[:, 1] + probs_test[:, 2] + probs_test[:, 3] + probs_test[:, 4]
xslg_test  = (probs_test[:, 1]*1 + probs_test[:, 2]*2 +
              probs_test[:, 3]*3 + probs_test[:, 4]*4)
xwoba_test = (probs_test[:, 1]*WOBA_WEIGHTS[1] + probs_test[:, 2]*WOBA_WEIGHTS[2] +
              probs_test[:, 3]*WOBA_WEIGHTS[3] + probs_test[:, 4]*WOBA_WEIGHTS[4])

# Actual stats on test set
is_hit  = (y_test > 0).astype(float)
bases   = y_test.astype(float)
woba_act = np.where(y_test == 0, 0.0,
           np.where(y_test == 1, WOBA_WEIGHTS[1],
           np.where(y_test == 2, WOBA_WEIGHTS[2],
           np.where(y_test == 3, WOBA_WEIGHTS[3], WOBA_WEIGHTS[4]))))

actual_ba   = is_hit.mean()
actual_slg  = bases.mean()
actual_woba = woba_act.mean()
pred_xba    = xba_test.mean()
pred_xslg   = xslg_test.mean()
pred_xwoba  = xwoba_test.mean()

print("\n=== TEST SET RESULTS ===")
print(f"  Actual BA   : {actual_ba:.4f}   Predicted xBA   : {pred_xba:.4f}   Δ={pred_xba-actual_ba:+.4f}")
print(f"  Actual SLG  : {actual_slg:.4f}   Predicted xSLG  : {pred_xslg:.4f}   Δ={pred_xslg-actual_slg:+.4f}")
print(f"  Actual wOBA : {actual_woba:.4f}   Predicted xwOBA : {pred_xwoba:.4f}   Δ={pred_xwoba-actual_woba:+.4f}")

# Log-loss
ll = log_loss(y_test, probs_test)
print(f"  Log-loss    : {ll:.4f}")

# Per-class calibration
print("\nPer-class calibration (actual vs predicted probability):")
for cls, name in enumerate(CLASS_NAMES):
    actual_rate = (y_test == cls).mean()
    pred_rate   = probs_test[:, cls].mean()
    print(f"  {name:<10}  actual={actual_rate:.4f}  predicted={pred_rate:.4f}  Δ={pred_rate-actual_rate:+.4f}")

# ── Sanity check: HR probability heat map (EV/LA) ───────────────────────────
print("\nHR probability sanity check (EV vs LA):")
ev_vals = [85, 90, 95, 100, 105, 110]
la_vals = [10, 20, 25, 30, 35]
header = "       " + "".join(f"  LA={la:2d}°" for la in la_vals)
print(header)
for ev in ev_vals:
    row = f"EV={ev}  "
    for la in la_vals:
        test_x = np.array([[ev, la, 0.0]], dtype=np.float32)  # center field
        p = model.predict_proba(test_x)[0]
        row += f"  {p[4]*100:5.1f}%"
    print(row)

print("\nxBA sanity check (EV=90, center field):")
for la in [-20, -10, 0, 10, 20, 25, 30, 35]:
    test_x = np.array([[90.0, float(la), 0.0]], dtype=np.float32)
    p = model.predict_proba(test_x)[0]
    xba_v = p[1]+p[2]+p[3]+p[4]
    print(f"  LA={la:4d}°  xBA={xba_v:.3f}  [Out={p[0]:.3f} 1B={p[1]:.3f} 2B={p[2]:.3f} HR={p[4]:.3f}]")

# ── Compute league averages on full BIP dataset ───────────────────────────────
print("\nComputing league averages on full dataset…")
probs_all  = model.predict_proba(X)
xba_all    = (probs_all[:, 1] + probs_all[:, 2] + probs_all[:, 3] + probs_all[:, 4])
xslg_all   = (probs_all[:, 1]*1 + probs_all[:, 2]*2 +
               probs_all[:, 3]*3 + probs_all[:, 4]*4)
xwoba_all  = (probs_all[:, 1]*WOBA_WEIGHTS[1] + probs_all[:, 2]*WOBA_WEIGHTS[2] +
               probs_all[:, 3]*WOBA_WEIGHTS[3] + probs_all[:, 4]*WOBA_WEIGHTS[4])

league = {
    "mean_xba":    float(xba_all.mean()),
    "std_xba":     float(xba_all.std()),
    "mean_xslg":   float(xslg_all.mean()),
    "std_xslg":    float(xslg_all.std()),
    "mean_xwoba":  float(xwoba_all.mean()),
    "std_xwoba":   float(xwoba_all.std()),
    "woba_weights": WOBA_WEIGHTS,
    "classes":     CLASS_NAMES,
    "features":    FEATURES,
    "n_bip":       int(len(bip)),
    "xba_pcts":    [float(np.percentile(xba_all,  p)) for p in [10,25,50,75,90]],
    "xslg_pcts":   [float(np.percentile(xslg_all, p)) for p in [10,25,50,75,90]],
    "xwoba_pcts":  [float(np.percentile(xwoba_all,p)) for p in [10,25,50,75,90]],
}

print(f"  League xBA  : {league['mean_xba']:.4f}  ± {league['std_xba']:.4f}")
print(f"  League xSLG : {league['mean_xslg']:.4f}  ± {league['std_xslg']:.4f}")
print(f"  League xwOBA: {league['mean_xwoba']:.4f}  ± {league['std_xwoba']:.4f}")
print(f"  xBA  pcts [10/25/50/75/90]: {[f'{v:.3f}' for v in league['xba_pcts']]}")
print(f"  xSLG pcts [10/25/50/75/90]: {[f'{v:.3f}' for v in league['xslg_pcts']]}")
print(f"  xwOBA pcts [10/25/50/75/90]: {[f'{v:.3f}' for v in league['xwoba_pcts']]}")

# ── Save ──────────────────────────────────────────────────────────────────────
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model,  MODEL_DIR / "xba_lgbm.pkl")
joblib.dump(league, MODEL_DIR / "xba_league.pkl")

print(f"\n✓  Saved models/xba_lgbm.pkl  ({(MODEL_DIR/'xba_lgbm.pkl').stat().st_size//1024} KB)")
print(f"✓  Saved models/xba_league.pkl")
print("\nDone.")
