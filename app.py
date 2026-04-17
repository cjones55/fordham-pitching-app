#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PATHS / IMPORTS
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from utils.shared import (
    load_models, basic_clean, filter_fordham,
    add_flags, compute_stuffplus, compute_locationplus
)
from utils.plotting import postgame_or_season_card

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fordham Pitching Analyzer",
    page_icon="⚾",
    layout="wide"
)

PASSWORD = "Baseball_1"

# ------------------------------------------------------------
# PASSWORD GATE
# ------------------------------------------------------------
def check_password():
    st.sidebar.title("Login")
    pw = st.sidebar.text_input("Enter password", type="password")
    if pw == PASSWORD:
        return True
    elif pw:
        st.sidebar.error("Incorrect password")
    return False

# ------------------------------------------------------------
# LOAD ALL CSVs FROM data/ (HARDENED)
# ------------------------------------------------------------
def load_all_data():
    DATA_DIR = ROOT / "data"
    csvs = list(DATA_DIR.glob("*.csv"))

    if not csvs:
        return pd.DataFrame()

    dfs = []
    for f in csvs:
        try:
            tmp = pd.read_csv(f, encoding="latin1", sep=None, engine="python")

            # Must contain at least one pitcher identifier
            if "Pitcher" not in tmp.columns and "PitcherId" not in tmp.columns:
                continue

            dfs.append(tmp)
        except:
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

# ------------------------------------------------------------
# SAFE WRAPPER FOR CLEANING + MODEL PIPELINE
# ------------------------------------------------------------
def prepare_data():
    df = load_all_data()
    if df.empty:
        return pd.DataFrame()

    try:
        df = basic_clean(df)
    except Exception:
        return pd.DataFrame()

    try:
        df = filter_fordham(df)
    except Exception:
        return pd.DataFrame()

    try:
        df = add_flags(df)
    except Exception:
        return pd.DataFrame()

    try:
        stuff_model, stuff_league, loc_model, loc_league = load_models()
        df = compute_stuffplus(df, stuff_model, stuff_league)
        df = compute_locationplus(df, loc_model, loc_league)
    except Exception:
        return pd.DataFrame()

    return df

# ------------------------------------------------------------
# PAGE 1 — POSTGAME SUMMARY
# ------------------------------------------------------------
def postgame_page():
    st.title("Postgame Summary – Stuff+ & Location+")

    df = prepare_data()

    if df.empty or "Pitcher" not in df.columns:
        st.error("No valid pitcher data found in data/ folder.")
        return

    pitchers = sorted(df["Pitcher"].unique())
    if len(pitchers) == 0:
        st.error("No pitchers found after cleaning.")
        return

    pitcher = st.selectbox("Select pitcher", pitchers, key="pg_pitcher")

    pdf = df[df["Pitcher"] == pitcher].copy()

    total_pitches = len(pdf)
    whiffs = pdf["is_whiff"].sum()
    walks = pdf["KorBB"].eq("Walk").sum()
    strikeouts = pdf["KorBB"].eq("Strikeout").sum()
    hbp = pdf["PitchCall"].eq("HitByPitch").sum()
    hits = pdf["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()
    hr = pdf["PlayResult"].eq("HomeRun").sum()
    outs_on_play = pdf["OutsOnPlay"].sum() if "OutsOnPlay" in pdf.columns else 0
    total_outs = outs_on_play + strikeouts
    ip = total_outs // 3 + (total_outs % 3) / 10 if total_outs else 0.0
    strike_pct = round(pdf["is_strike"].mean() * 100, 1)

    title = f"{pitcher} – Postgame Summary"
    summary = (
        f"Pitches: {total_pitches}  IP: {ip:.1f}  H: {hits}  "
        f"BB: {walks}  K: {strikeouts}  HR: {hr}  HBP: {hbp}  "
        f"Whiffs: {whiffs}  Strike%: {strike_pct}%"
    )

    logo_path = ROOT / "assets" / "rams.png"
    fig = postgame_or_season_card(pdf, title, summary, logo_path)

    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
    buf.seek(0)

    st.download_button(
        "Download PNG",
        buf,
        file_name=f"{pitcher.replace(',','')}_Postgame_Summary.png",
        mime="image/png",
        key="pg_dl"
    )

# ------------------------------------------------------------
# PAGE 2 — SEASON SUMMARY
# ------------------------------------------------------------
def season_page():
    st.title("Season Summary – Stuff+ & Location+")

    df = prepare_data()

    if df.empty or "Pitcher" not in df.columns:
        st.error("No valid pitcher data found.")
        return

    pitchers = sorted(df["Pitcher"].unique())
    if len(pitchers) == 0:
        st.error("No pitchers found after cleaning.")
        return

    pitcher = st.selectbox("Select pitcher", pitchers, key="season_pitcher")

    pdf = df[df["Pitcher"] == pitcher].copy()

    total_pitches = len(pdf)
    whiffs = pdf["is_whiff"].sum()
    walks = pdf["KorBB"].eq("Walk").sum()
    strikeouts = pdf["KorBB"].eq("Strikeout").sum()
    hbp = pdf["PitchCall"].eq("HitByPitch").sum()
    hits = pdf["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()
    hr = pdf["PlayResult"].eq("HomeRun").sum()
    outs_on_play = pdf["OutsOnPlay"].sum() if "OutsOnPlay" in pdf.columns else 0
    total_outs = outs_on_play + strikeouts
    ip = total_outs // 3 + (total_outs % 3) / 10 if total_outs else 0.0
    strike_pct = round(pdf["is_strike"].mean() * 100, 1)

    title = f"{pitcher} – 2026 Season Summary"
    summary = (
        f"Pitches: {total_pitches}  IP: {ip:.1f}  H: {hits}  "
        f"BB: {walks}  K: {strikeouts}  HR: {hr}  HBP: {hbp}  "
        f"Whiffs: {whiffs}  Strike%: {strike_pct}%"
    )

    logo_path = ROOT / "assets" / "rams.png"
    fig = postgame_or_season_card(pdf, title, summary, logo_path)

    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
    buf.seek(0)

    st.download_button(
        "Download PNG",
        buf,
        file_name=f"{pitcher.replace(',','')}_2026_Season_Summary.png",
        mime="image/png",
        key="season_dl"
    )

# ------------------------------------------------------------
# PAGE 3 — STUFF+ LEADERBOARD
# ------------------------------------------------------------
def stuff_leaderboard_page():
    st.title("Stuff+ Leaderboard")

    df = prepare_data()
    if df.empty:
        st.error("No valid data found.")
        return

    agg = df.groupby("Pitcher").agg(
        Stuff_plus=("Stuff+", "mean"),
        N=("Stuff+", "count")
    ).reset_index()

    min_pitches = st.slider("Minimum pitches", 10, 200, 25, 5, key="stuff_min")
    agg = agg[agg["N"] >= min_pitches].sort_values("Stuff_plus", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 13))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text(
        0.5, 1.1, "Fordham Baseball — Total Stuff+",
        color="#A00000", fontsize=30, fontweight="bold",
        ha="center", va="top"
    )

    y_start = 0.95
    y_step = 0.052

    for i, row in enumerate(agg.itertuples()):
        y = y_start - i * y_step
        ax.text(0.12, y, row.Pitcher, color="#F5F0E1", fontsize=19)
        ax.text(0.88, y, f"{round(row.Stuff_plus,1)}",
                color="#A00000", fontsize=19, ha="right")

    st.pyplot(fig)

# ------------------------------------------------------------
# PAGE 4 — LOCATION+ LEADERBOARD
# ------------------------------------------------------------
def location_leaderboard_page():
    st.title("Location+ Leaderboard")

    df = prepare_data()
    if df.empty:
        st.error("No valid data found.")
        return

    agg = df.groupby("Pitcher").agg(
        Loc_plus=("Loc+", "mean"),
        N=("Loc+", "count")
    ).reset_index()

    min_pitches = st.slider("Minimum pitches", 10, 200, 25, 5, key="loc_min")
    agg = agg[agg["N"] >= min_pitches].sort_values("Loc_plus", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 13))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text(
        0.5, 1.1, "Fordham Baseball — Total Location+",
        color="#A00000", fontsize=30, fontweight="bold",
        ha="center", va="top"
    )

    y_start = 0.95
    y_step = 0.052

    for i, row in enumerate(agg.itertuples()):
        y = y_start - i * y_step
        ax.text(0.12, y, row.Pitcher, color="#F5F0E1", fontsize=19)
        ax.text(0.88, y, f"{round(row.Loc_plus,1)}",
                color="#A00000", fontsize=19, ha="right")

    st.pyplot(fig)

# ------------------------------------------------------------
# PAGE 5 — PITCH-TYPE GRIDS
# ------------------------------------------------------------
def pitchtype_grids_page():
    st.title("Pitch-type Grids – Location+")

    df = prepare_data()
    if df.empty:
        st.error("No valid data found.")
        return

    if "pitch_abbr" not in df.columns:
        st.error("pitch_abbr missing — check data.")
        return

    agg = df.groupby(["Pitcher","pitch_abbr"]).agg(
        Loc_plus=("Loc+", "mean"),
        N=("Loc+", "count")
    ).reset_index()

    min_pitches = st.slider("Minimum pitches per pitch type", 5, 50, 10, 5, key="pt_min")
    agg = agg[agg["N"] >= min_pitches]

    pitch_types = sorted(agg["pitch_abbr"].unique())

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.patch.set_facecolor("#1e1e1e")
    axes = axes.flatten()

    pitch_types_extended = pitch_types + ["__LOGO__", "__EMPTY__"]
    pitch_types_extended = pitch_types_extended[:9]

    for ax, pitch in zip(axes, pitch_types_extended):
        ax.set_facecolor("#1e1e1e")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        for s in ax.spines.values():
            s.set_visible(False)

        if pitch == "__LOGO__":
            logo_path = ROOT / "assets" / "rams.png"
            if logo_path.exists():
                img = mpimg.imread(logo_path)
                ax.imshow(img)
            ax.axis("off")
            continue

        if pitch == "__EMPTY__":
            ax.axis("off")
            continue

        sub = agg[agg["pitch_abbr"] == pitch].sort_values("Loc_plus", ascending=False).head(10)

        ax.text(0.05, 0.96, f"{pitch} – Top 10 Loc+",
                color="#A00000", fontsize=14, fontweight="bold", va="top")

        y_start = 0.76
        y_step = 0.10

        for i, row in enumerate(sub.itertuples()):
            y = y_start - i * y_step
            ax.text(0.05, y, row.Pitcher, color="white", fontsize=14)
            ax.text(0.95, y, f"{round(row.Loc_plus,1)}",
                    color="white", fontsize=14, ha="right")

    st.pyplot(fig)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    st.markdown(
        "<h1 style='text-align:center; color:#A00000;'>Fordham Baseball – Pitching Analytics</h1>",
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Postgame Summary",
        "Season Summary",
        "Stuff+ Leaderboard",
        "Location+ Leaderboard",
        "Pitch-Type Grids"
    ])

    with tab1:
        postgame_page()
    with tab2:
        season_page()
    with tab3:
        stuff_leaderboard_page()
    with tab4:
        location_leaderboard_page()
    with tab5:
        pitchtype_grids_page()

# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if check_password():
    main()
else:
    st.stop()
