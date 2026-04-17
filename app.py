#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------
# PATHS / IMPORTS
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from utils.shared import (
    load_models, basic_clean, add_flags,
    compute_stuffplus, compute_locationplus
)

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
# LOAD RAW CSVs
# ------------------------------------------------------------
def load_all_raw():
    DATA_DIR = ROOT / "data"
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        return []

    valid_raw = []
    for f in csvs:
        try:
            df = pd.read_csv(f, encoding="latin1", sep=None, engine="python")
            if "Pitcher" in df.columns:
                valid_raw.append(df)
        except:
            continue

    return valid_raw

# ------------------------------------------------------------
# FULL PIPELINE
# ------------------------------------------------------------
def prepare_data():
    raw_files = load_all_raw()
    if not raw_files:
        return pd.DataFrame()

    processed = []

    for raw in raw_files:
        try:
            df = basic_clean(raw)
            df = add_flags(df)

            stuff_model, stuff_league, loc_model, loc_league = load_models()
            df = compute_stuffplus(df, stuff_model, stuff_league)
            df = compute_locationplus(df, loc_model, loc_league)

            processed.append(df)
        except:
            continue

    if not processed:
        return pd.DataFrame()

    return pd.concat(processed, ignore_index=True)

# ------------------------------------------------------------
# FORDHAM FILTER (FOR_RAM)
# ------------------------------------------------------------
def filter_fordham_only(df):
    if "PitcherTeam" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["PitcherTeam"].astype(str).str.upper() == "FOR_RAM"].copy()

# ------------------------------------------------------------
# SAFE PITCHER LIST
# ------------------------------------------------------------
def get_pitcher_list(df):
    if df.empty or "Pitcher" not in df.columns:
        return []
    return sorted([p for p in df["Pitcher"].unique() if isinstance(p, str) and p.strip() != ""])

# ------------------------------------------------------------
# OPPONENT DETECTION (FOR_RAM → BatterTeam)
# ------------------------------------------------------------
def detect_opponent(pdf):
    if "BatterTeam" not in pdf.columns:
        return "Opponent"

    teams = pdf["BatterTeam"].dropna().unique()
    if len(teams) == 1:
        return teams[0]

    return pdf["BatterTeam"].mode().iloc[0]

# ------------------------------------------------------------
# MLB-STYLE POSTGAME FIGURE (DARK GRAY + PITCH-COLORED TABLE)
# ------------------------------------------------------------
def build_postgame_figure(pdf, pitcher, game_date, opponent):
    BACKGROUND = "#2A2A2A"
    HEADER_MAROON = "#A00000"

    pitch_colors = {
        "FB": "#1f77b4",
        "SI": "#17becf",
        "FC": "#ff7f0e",
        "SL": "#d62728",
        "CU": "#9467bd",
        "CH": "#2ca02c",
        "SW": "#8c564b"
    }

    # -----------------------------
    # GAME TOTALS
    # -----------------------------
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

    # -----------------------------
    # SPLITS
    # -----------------------------
    LHH_pdf = pdf[pdf["is_LHH"]]
    RHH_pdf = pdf[pdf["is_RHH"]]

    stuff_LHH = round(LHH_pdf["Stuff+"].mean(), 1) if len(LHH_pdf) else np.nan
    stuff_RHH = round(RHH_pdf["Stuff+"].mean(), 1) if len(RHH_pdf) else np.nan

    loc_LHH = round(LHH_pdf["Loc+"].mean(), 1) if len(LHH_pdf) else np.nan
    loc_RHH = round(RHH_pdf["Loc+"].mean(), 1) if len(RHH_pdf) else np.nan

    # -----------------------------
    # AGG TABLE
    # -----------------------------
    if "pitch_abbr" not in pdf.columns:
        pdf["pitch_abbr"] = "UNK"

    agg = pdf.groupby("pitch_abbr").agg(
        N=("PitchCall","count"),
        Velo=("Velo","mean"),
        IVB=("IVB","mean"),
        HB=("HB","mean"),
        Spin=("Spin","mean"),
        Stuff_plus=("Stuff+","mean"),
        Loc_plus=("Loc+","mean"),
        CSW=("is_csw","sum"),
        Whiffs=("is_whiff","sum"),
        Swings=("is_swing","sum"),
        Strikes=("is_strike","sum"),
        InZone=("in_zone","sum")
    ).reset_index()

    agg = agg.rename(columns={"pitch_abbr": "Pitch", "Stuff_plus": "Stuff+", "Loc_plus": "Loc+"})

    total_N = agg["N"].sum()
    agg["Usage%"] = (agg["N"] / total_N * 100).round(1)
    agg["CSW%"] = (agg["CSW"] / agg["N"] * 100).round(1)
    agg["Whiff%"] = np.where(agg["Swings"] > 0, (agg["Whiffs"] / agg["Swings"] * 100).round(1), 0.0)
    agg["Strike%"] = (agg["Strikes"] / agg["N"] * 100).round(1)
    agg["Zone%"] = (agg["InZone"] / agg["N"] * 100).round(1)

    # -----------------------------
    # FIGURE
    # -----------------------------
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(BACKGROUND)

    # LOGO
    logo_path = ROOT / "assets" / "rams.png"
    if logo_path.exists():
        logo_img = mpimg.imread(logo_path)
        fig.figimage(logo_img, xo=40, yo=fig.bbox.ymax + 300, zorder=50, alpha=1.0)

    # TITLE + SUMMARY
    title = f"{pitcher} – Fordham vs {opponent}"
    summary = (
        f"IP: {ip:.1f}  H: {hits}  R: {hits}  ER: {hits}  "
        f"BB: {walks}  K: {strikeouts}  HR: {hr}  HBP: {hbp}  "
        f"Whiffs: {whiffs}  Strike%: {strike_pct}%  "
        f"Stf+LHH: {stuff_LHH}  Stf+RHH: {stuff_RHH}  "
        f"Loc+LHH: {loc_LHH}  Loc+RHH: {loc_RHH}"
    )

    fig.suptitle(title, fontsize=28, fontweight="bold", color=HEADER_MAROON, y=0.97)
    plt.text(0.5, 0.93, summary, ha="center", va="center", color="white", fontsize=15)

    # -----------------------------
    # MOVEMENT PLOT
    # -----------------------------
    def style_axes(ax):
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=2)
    style_axes(ax1)
    ax1.set_facecolor(BACKGROUND)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax1.scatter(row["HB"], row["IVB"], s=40, color=c, edgecolor="white", linewidth=0.5)

    centroids = pdf.groupby("pitch_abbr")[["HB", "IVB"]].mean().reset_index()
    for _, row in centroids.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax1.scatter(row["HB"], row["IVB"], s=250, color=c, edgecolor="white", linewidth=1.5)
        ax1.text(row["HB"], row["IVB"], row["pitch_abbr"], color="white", fontsize=10, weight="bold", ha="center")

    ax1.set_title("Movement", color="white")

    # -----------------------------
    # LOCATION PLOTS (LHH / RHH)
    # -----------------------------
    def draw_zone(ax):
        ax.set_facecolor(BACKGROUND)
        style_axes(ax)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        zone_x = [-0.83, 0.83, 0.83, -0.83, -0.83]
        zone_y = [1.5, 1.5, 3.5, 3.5, 1.5]
        ax.plot(zone_x, zone_y, color="white", linewidth=2.5)

    axL = plt.subplot2grid((5, 4), (0, 1), rowspan=2)
    draw_zone(axL)
    LHH = pdf[pdf["BatterSide"] == "Left"]
    for _, row in LHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axL.scatter(row["PlateLocSide"], row["PlateLocHeight"], s=85, color=c, edgecolor="white")
    axL.set_title("LHH", color="white")

    axR = plt.subplot2grid((5, 4), (0, 2), rowspan=2)
    draw_zone(axR)
    RHH = pdf[pdf["BatterSide"] == "Right"]
    for _, row in RHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axR.scatter(row["PlateLocSide"], row["PlateLocHeight"], s=85, color=c, edgecolor="white")
    axR.set_title("RHH", color="white")

    # -----------------------------
    # RELEASE PLOT
    # -----------------------------
    axRel = plt.subplot2grid((5, 4), (0, 3), rowspan=2)
    style_axes(axRel)
    axRel.set_facecolor(BACKGROUND)
    axRel.set_xlim(-4, 4)
    axRel.set_ylim(3, 7)
    axRel.set_aspect("equal")

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axRel.scatter(row["RelS"], row["RelH"], s=25, color=c, edgecolor="white")

    axRel.set_title("Release", color="white")

    # -----------------------------
    # WIDE TABLE (PITCH-COLORED ROWS)
    # -----------------------------
    axT = plt.subplot2grid((5, 4), (2, 0), colspan=4, rowspan=2)
    axT.axis("off")

    table_df = agg[[
        "Pitch","N","Usage%","Velo","IVB","HB",
        "Spin","Stuff+","Loc+","CSW%","Whiff%","Strike%","Zone%"
    ]].round(2)

    tbl = axT.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_MAROON)
            cell.set_text_props(color="white", weight="bold")
        else:
            pitch = table_df.iloc[r - 1]["Pitch"]
            bg = pitch_colors.get(pitch, BACKGROUND)
            cell.set_facecolor(bg)
            cell.set_text_props(color="white", weight="bold")

    # -----------------------------
    # FOOTER
    # -----------------------------
    axFooter = plt.subplot2grid((5, 4), (4, 0), colspan=4)
    axFooter.axis("off")
    axFooter.text(
        0.98, 0.15, f"Game Date: {game_date}",
        ha="right", va="center",
        fontsize=12, color="white"
    )

    return fig

# ------------------------------------------------------------
# PAGE 1 — POSTGAME SUMMARY (Pitcher → Game Selector)
# ------------------------------------------------------------
def postgame_page():
    st.title("Postgame Summary – MLB Style")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    pitchers = get_pitcher_list(df)
    pitcher = st.selectbox("Select pitcher", pitchers, key="pg_pitcher")

    pdf = df[df["Pitcher"] == pitcher].copy()

    if "Date" not in pdf.columns or "BatterTeam" not in pdf.columns:
        st.error("Missing Date or BatterTeam columns.")
        return

    games = (
        pdf.groupby(["Date", "BatterTeam"])
           .size()
           .reset_index()[["Date", "BatterTeam"]]
    )

    games["label"] = games["Date"].astype(str) + " vs " + games["BatterTeam"]
    selected_game = st.selectbox("Select Game", games["label"], key="pg_game")

    g_date, g_opp = selected_game.split(" vs ")

    g_pdf = pdf[
        (pdf["Date"].astype(str) == g_date) &
        (pdf["BatterTeam"] == g_opp)
    ].copy()

    if g_pdf.empty:
        st.error("No data found for that game.")
        return

    fig = build_postgame_figure(g_pdf, pitcher, g_date, g_opp)
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
    buf.seek(0)

    st.download_button(
        "Download PNG",
        buf,
        file_name=f"{pitcher.replace(',','')}_{g_date}_Postgame.png",
        mime="image/png",
        key="pg_dl"
    )

# ------------------------------------------------------------
# PAGE 2 — SEASON SUMMARY
# ------------------------------------------------------------
def season_page():
    st.title("Season Summary – Stuff+ & Location+")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    pitchers = get_pitcher_list(df)
    pitcher = st.selectbox("Select pitcher", pitchers, key="season_pitcher")

    pdf = df[df["Pitcher"] == pitcher].copy()

    fig = build_postgame_figure(pdf, pitcher, "Season Totals", "Season")
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
    buf.seek(0)

    st.download_button(
        "Download PNG",
        buf,
        file_name=f"{pitcher.replace(',','')}_Season_Summary.png",
        mime="image/png",
        key="season_dl"
    )

# ------------------------------------------------------------
# PAGE 3 — STUFF+ LEADERBOARD
# ------------------------------------------------------------
def stuff_leaderboard_page():
    st.title("Stuff+ Leaderboard")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    agg = df.groupby("Pitcher").agg(
        Stuff_plus=("Stuff+", "mean"),
        N=("Stuff+", "count")
    ).reset_index()

    min_pitches = st.slider("Minimum pitches", 10, 200, 25, 5, key="stuff_min")
    agg = agg[agg["N"] >= min_pitches].sort_values("Stuff_plus", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 13))
    fig.patch.set_facecolor("#2A2A2A")
    ax.set_facecolor("#2A2A2A")
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
        ax.text(0.12, y, row.Pitcher, color="white", fontsize=19)
        ax.text(0.88, y, f"{round(row.Stuff_plus,1)}",
                color="#A00000", fontsize=19, ha="right")

    st.pyplot(fig)

# ------------------------------------------------------------
# PAGE 4 — LOCATION+ LEADERBOARD
# ------------------------------------------------------------
def location_leaderboard_page():
    st.title("Location+ Leaderboard")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    agg = df.groupby("Pitcher").agg(
        Loc_plus=("Loc+", "mean"),
        N=("Loc+", "count")
    ).reset_index()

    min_pitches = st.slider("Minimum pitches", 10, 200, 25, 5, key="loc_min")
    agg = agg[agg["N"] >= min_pitches].sort_values("Loc_plus", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 13))
    fig.patch.set_facecolor("#2A2A2A")
    ax.set_facecolor("#2A2A2A")
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
        ax.text(0.12, y, row.Pitcher, color="white", fontsize=19)
        ax.text(0.88, y, f"{round(row.Loc_plus,1)}",
                color="#A00000", fontsize=19, ha="right")

    st.pyplot(fig)

# ------------------------------------------------------------
# PAGE 5 — PITCH-TYPE GRIDS
# ------------------------------------------------------------
def pitchtype_grids_page():
    st.title("Pitch-type Grids – Location+")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    if "pitch_abbr" not in df.columns:
        st.error("pitch_abbr column missing — check data.")
        return

    agg = df.groupby(["Pitcher","pitch_abbr"]).agg(
        Loc_plus=("Loc+", "mean"),
        N=("Loc+", "count")
    ).reset_index()

    min_pitches = st.slider("Minimum pitches per pitch type", 5, 50, 10, 5, key="pt_min")
    agg = agg[agg["N"] >= min_pitches]

    pitch_types = sorted(agg["pitch_abbr"].unique())

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.patch.set_facecolor("#2A2A2A")
    axes = axes.flatten()

    pitch_types_extended = pitch_types + ["__LOGO__", "__EMPTY__"]
    pitch_types_extended = pitch_types_extended[:9]

    for ax, pitch in zip(axes, pitch_types_extended):
        ax.set_facecolor("#2A2A2A")
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
