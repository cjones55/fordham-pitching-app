#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:37:51 2026

@author: chrisjones
"""

import streamlit as st
import pandas as pd
from io import BytesIO
from pathlib import Path

from utils.shared import (
    load_models, basic_clean, filter_fordham,
    add_flags, compute_stuffplus, compute_locationplus
)
from utils.plotting import postgame_or_season_card

st.set_page_config(page_title="Season Summary", layout="wide")

st.title("Season Summary – Stuff+ & Location+")

uploads = st.file_uploader(
    "Upload all 2026 TrackMan CSVs (multi-select)",
    type=["csv"],
    accept_multiple_files=True
)

if uploads:
    dfs = []
    for f in uploads:
        try:
            tmp = pd.read_csv(f, encoding="latin1", sep=None, engine="python")
            dfs.append(tmp)
        except Exception:
            continue

    if not dfs:
        st.error("No valid CSVs parsed.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)
    stuff_model, stuff_league, loc_model, loc_league = load_models()

    df = basic_clean(df)
    df = filter_fordham(df)
    df = add_flags(df)
    df = compute_stuffplus(df, stuff_model, stuff_league)
    df = compute_locationplus(df, loc_model, loc_league)

    pitchers = sorted(df["Pitcher"].unique())
    pitcher = st.selectbox("Select pitcher", pitchers)

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

    logo_path = Path("assets") / "rams.png"
    fig = postgame_or_season_card(pdf, title, summary, logo_path)

    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
    buf.seek(0)

    st.download_button(
        label="Download PNG",
        data=buf,
        file_name=f"{pitcher.replace(',','')}_2026_Season_Summary.png",
        mime="image/png"
    )
else:
    st.info("Upload all 2026 CSVs to generate season summaries.")
