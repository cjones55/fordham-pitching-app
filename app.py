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
# GLOBAL TOP-LEFT LOGO (safe version)
# ------------------------------------------------------------
import base64

try:
    logo_path = ROOT / "static" / "rams.png"
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
            .top-left-logo {{
                position: fixed;
                top: 57px;
                left: 12px;
                width: 110px;
                z-index: 99999;
            }}
        </style>

        <img src="data:image/png;base64,{logo_b64}" class="top-left-logo">
        """,
        unsafe_allow_html=True
    )

except Exception as e:
    st.write("Logo failed to load:", e)

# ------------------------------------------------------------
# LOAD SEASON PITCHING STATS (from CSV)
# ------------------------------------------------------------
pitching_df = pd.read_csv(ROOT / "data" / "pitching_stats.csv")


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

def prepare_data():
    raw_files = load_all_raw()
    if not raw_files:
        return pd.DataFrame()

    processed = []

    for raw in raw_files:
        try:
            df = basic_clean(raw)

            # ⭐ ALWAYS ensure a Pitcher column exists
            if "Pitcher" not in df.columns:
                if "Player" in df.columns:
                    df = df.rename(columns={"Player": "Pitcher"})
                elif "PitcherName" in df.columns:
                    df = df.rename(columns={"PitcherName": "Pitcher"})
                else:
                    # If no pitcher column exists at all, skip this file
                    continue

            df = add_flags(df)

            stuff_model, stuff_league, loc_model, loc_league = load_models()
            df = compute_stuffplus(df, stuff_model, stuff_league)
            df = compute_locationplus(df, loc_model, loc_league)

            processed.append(df)

        except Exception as e:
            # Skip broken files
            continue

    if not processed:
        return pd.DataFrame()

    return pd.concat(processed, ignore_index=True)


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



def draw_home_plate(ax):
    plate_x = [-0.83, 0.83, 0.83, 0, -0.83, -0.83]
    plate_y = [0, 0, 0.17, 0.34, 0.17, 0]
    ax.plot(plate_x, plate_y, color="white", linewidth=2)
    ax.fill(plate_x, plate_y, color="white", alpha=0.10)


def build_postgame_figure(pdf, pitcher, game_date, opponent):
    import matplotlib.gridspec as gridspec

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
    fig = plt.figure(figsize=(24, 11))
    fig.patch.set_facecolor(BACKGROUND)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.80, bottom=0.06, wspace=0.25, hspace=0.35)

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        height_ratios=[2.2, 1.0, 1.0],
        width_ratios=[3.0, 1.7, 1.7, 1.2]
    )

    # -----------------------------
    # LOGO
    # -----------------------------
    logo_path = ROOT / "assets" / "rams.png"
    if logo_path.exists():
        logo_img = mpimg.imread(logo_path)
        fig.figimage(logo_img, xo=40, yo=int(fig.bbox.ymax * 1.5), zorder=50, alpha=1.0)

    # -----------------------------
    # TITLE + SUMMARY
    # -----------------------------
    title = f"{pitcher} – Fordham vs {opponent}"
    summary = (
        f"IP: {ip:.1f}  H: {hits}  R: {hits}  ER: {hits}  "
        f"BB: {walks}  K: {strikeouts}  HR: {hr}  HBP: {hbp}  "
        f"Whiffs: {whiffs}  Strike%: {strike_pct}%  "
        f"Stf+LHH: {stuff_LHH}  Stf+RHH: {stuff_RHH}  "
        f"Loc+LHH: {loc_LHH}  Loc+RHH: {loc_RHH}"
    )

    fig.text(0.5, 0.96, title, ha="center", va="center",
             fontsize=28, fontweight="bold", color=HEADER_MAROON)
    fig.text(0.5, 0.91, summary, ha="center", va="center",
             fontsize=15, color="white")

    # -----------------------------
    # MOVEMENT
    # -----------------------------
    ax_move = fig.add_subplot(gs[0, 0])
    ax_move.set_facecolor(BACKGROUND)
    ax_move.set_aspect('equal', adjustable='box')

    ax_move.set_xlim(-25, 25)
    ax_move.set_ylim(-25, 25)

    throws = pdf["PitcherThrows"].iloc[0] if "PitcherThrows" in pdf.columns else "Right"

    if throws.upper().startswith("R"):
        arm_xmin, arm_xmax = 0, 25
        glove_xmin, glove_xmax = -25, 0
    else:
        arm_xmin, arm_xmax = -25, 0
        glove_xmin, glove_xmax = 0, 25

    arm_color   = (0.10, 0.30, 0.60, 0.10)
    glove_color = (0.60, 0.10, 0.10, 0.10)

    ax_move.axvspan(arm_xmin, arm_xmax, facecolor=arm_color, zorder=0)
    ax_move.axvspan(glove_xmin, glove_xmax, facecolor=glove_color, zorder=0)

    ax_move.axhline(0, color="white", linestyle=":", linewidth=1.4)
    ax_move.axvline(0, color="white", linestyle=":", linewidth=1.4)

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_move.scatter(row["HB"], row["IVB"], s=55, color=c, edgecolor="white", linewidth=0.5)

    centroids = pdf.groupby("pitch_abbr")[["HB", "IVB"]].mean().reset_index()
    for _, row in centroids.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_move.scatter(row["HB"], row["IVB"], s=330, color=c, edgecolor="white", linewidth=1.5)
        ax_move.text(row["HB"], row["IVB"], row["pitch_abbr"],
                     color="white", fontsize=15, weight="bold", ha="center")

    ax_move.set_title("Movement", color="white", fontsize=18, weight="bold")
    ax_move.tick_params(colors="white")
    for spine in ax_move.spines.values():
        spine.set_color("white")

    # -----------------------------
    # LHH (with HOME PLATE)
    # -----------------------------
    ax_lhh = fig.add_subplot(gs[0, 1])
    ax_lhh.set_facecolor(BACKGROUND)
    ax_lhh.set_title("LHH", color="white", fontsize=16, weight="bold")
    ax_lhh.set_aspect(1.6)

    ax_lhh.set_xlim(-2.5, 2.5)
    ax_lhh.set_ylim(0, 5)

    zone_x = [-0.83, 0.83, 0.83, -0.83, -0.83]
    zone_y = [1.5, 1.5, 3.5, 3.5, 1.5]
    ax_lhh.plot(zone_x, zone_y, color="white", linewidth=2.5)

    draw_home_plate(ax_lhh)   # ⭐ ADDED

    LHH = pdf[pdf["BatterSide"] == "Left"]
    for _, row in LHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_lhh.scatter(row["PlateLocSide"], row["PlateLocHeight"],
                       s=140, color=c, edgecolor="white", linewidth=0.6)

    ax_lhh.tick_params(colors="white", labelsize=12)
    for spine in ax_lhh.spines.values():
        spine.set_color("white")

    # -----------------------------
    # RHH (with HOME PLATE)
    # -----------------------------
    ax_rhh = fig.add_subplot(gs[0, 2])
    ax_rhh.set_facecolor(BACKGROUND)
    ax_rhh.set_title("RHH", color="white", fontsize=16, weight="bold")
    ax_rhh.set_aspect(1.6)

    ax_rhh.set_xlim(-2.5, 2.5)
    ax_rhh.set_ylim(0, 5)
    ax_rhh.plot(zone_x, zone_y, color="white", linewidth=2.5)

    draw_home_plate(ax_rhh)   # ⭐ ADDED

    RHH = pdf[pdf["BatterSide"] == "Right"]
    for _, row in RHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_rhh.scatter(row["PlateLocSide"], row["PlateLocHeight"],
                       s=140, color=c, edgecolor="white", linewidth=0.6)

    ax_rhh.tick_params(colors="white", labelsize=12)
    for spine in ax_rhh.spines.values():
        spine.set_color("white")

    # -----------------------------
    # RELEASE
    # -----------------------------
    ax_rel = fig.add_subplot(gs[0, 3])
    ax_rel.set_facecolor(BACKGROUND)
    ax_rel.set_title("Release", color="white", fontsize=16, weight="bold")

    ax_rel.set_aspect(1.4)

    ax_rel.set_xlim(-3.2, 3.2)
    ax_rel.set_ylim(3.2, 6.8)

    ax_rel.axhline(np.mean(pdf["RelH"]), color="white", linestyle=":", linewidth=1.4)
    ax_rel.axvline(np.mean(pdf["RelS"]), color="white", linestyle=":", linewidth=1.4)

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_rel.scatter(row["RelS"], row["RelH"], s=40, color=c, edgecolor="white", linewidth=0.6)

    ax_rel.tick_params(colors="white", labelsize=12)
    for spine in ax_rel.spines.values():
        spine.set_color("white")

    # -----------------------------
    # TABLE
    # -----------------------------
    ax_table = fig.add_subplot(gs[1:, :])
    ax_table.axis("off")

    table_df = agg[[
        "Pitch","N","Usage%","Velo","IVB","HB",
        "Spin","Stuff+","Loc+","CSW%","Whiff%","Strike%","Zone%"
    ]].round(2)

    tbl = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0.08, 1, 0.92]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(18)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(0.042)
        cell.set_width(0.072)

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
    fig.text(
        0.98, 0.03,
        f"Game Date: {game_date}",
        ha="right", va="center",
        fontsize=12, color="white"
    )

    return fig




# ------------------------------------------------------------
# PAGE 1 — POSTGAME SUMMARY (Pitcher → Game Selector)
# ------------------------------------------------------------
def postgame_page():
    st.title("Postgame Summary")

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
# PAGE 5 — PITCH-TYPE GRIDS (SEPARATE STUFF+ AND LOC+ WITH COLOR CODING)
# ------------------------------------------------------------
def pitchtype_grids_page():
    st.title("Pitch-type Grids – Stuff+ and Location+")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    if "pitch_abbr" not in df.columns:
        st.error("pitch_abbr column missing — check data.")
        return

    # -----------------------------
    # COLORS
    # -----------------------------
    pitch_colors = {
        "FB": "#1f77b4",
        "SI": "#17becf",
        "FC": "#ff7f0e",
        "SL": "#d62728",
        "CU": "#9467bd",
        "CH": "#2ca02c",
        "SW": "#8c564b"
    }

    LHH_COLOR = "#4da6ff"
    RHH_COLOR = "#ff6666"

    # -----------------------------
    # SPLIT BY LHH / RHH
    # -----------------------------
    df_LHH = df[df["is_LHH"]]
    df_RHH = df[df["is_RHH"]]

    # -----------------------------
    # AGGREGATE ALL METRICS
    # -----------------------------
    agg = df.groupby(["Pitcher","pitch_abbr"]).agg(
        Stuff_plus=("Stuff+", "mean"),
        Loc_plus=("Loc+", "mean"),
        N=("Loc+", "count")
    ).reset_index()

    agg_LHH = df_LHH.groupby(["Pitcher","pitch_abbr"]).agg(
        Stuff_plus_LHH=("Stuff+", "mean"),
        Loc_plus_LHH=("Loc+", "mean")
    ).reset_index()

    agg_RHH = df_RHH.groupby(["Pitcher","pitch_abbr"]).agg(
        Stuff_plus_RHH=("Stuff+", "mean"),
        Loc_plus_RHH=("Loc+", "mean")
    ).reset_index()

    agg = (
        agg
        .merge(agg_LHH, on=["Pitcher","pitch_abbr"], how="left")
        .merge(agg_RHH, on=["Pitcher","pitch_abbr"], how="left")
    )

    # -----------------------------
    # FILTER BY MINIMUM PITCHES
    # -----------------------------
    min_pitches = st.slider("Minimum pitches per pitch type", 5, 50, 10, 5, key="pt_min")
    agg = agg[agg["N"] >= min_pitches]

    pitch_types = sorted(agg["pitch_abbr"].unique())

    # ============================================================
    # 1️⃣ STUFF+ GRID (2×3)
    # ============================================================
    st.subheader("Stuff+ Leaderboards")

    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 22))
    fig1.patch.set_facecolor("#2A2A2A")
    axes1 = axes1.flatten()

    pitch_types_extended = pitch_types + ["__LOGO__", "__EMPTY__", "__EMPTY__"]
    pitch_types_extended = pitch_types_extended[:6]

    for ax, pitch in zip(axes1, pitch_types_extended):
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

        sub = agg[agg["pitch_abbr"] == pitch].sort_values("Stuff_plus", ascending=False).head(10)

        ax.text(0.05, 0.96, f"{pitch} – Top 10 Stuff+",
                color=pitch_colors.get(pitch, "#A00000"),
                fontsize=18, fontweight="bold", va="top")

        y_start = 0.87
        y_step = 0.095

        for i, row in enumerate(sub.itertuples()):
            y = y_start - i * y_step

            # Pitcher name (white)
            ax.text(0.02, y, row.Pitcher, color="white", fontsize=14, weight="bold")

            # Stuff+ (pitch color)
            ax.text(0.60, y, f"St+: {round(row.Stuff_plus,1)}",
                    color=pitch_colors.get(pitch, "white"), fontsize=14)

            # LHH (blue)
            ax.text(0.60, y - 0.03, f"LHH: {round(row.Stuff_plus_LHH or 0,1)}",
                    color=LHH_COLOR, fontsize=12)

            # RHH (red)
            ax.text(0.60, y - 0.06, f"RHH: {round(row.Stuff_plus_RHH or 0,1)}",
                    color=RHH_COLOR, fontsize=12)

    st.pyplot(fig1)

    # ============================================================
    # 2️⃣ LOC+ GRID (2×3)
    # ============================================================
    st.subheader("Location+ Leaderboards")

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 22))
    fig2.patch.set_facecolor("#2A2A2A")
    axes2 = axes2.flatten()

    for ax, pitch in zip(axes2, pitch_types_extended):
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
                color=pitch_colors.get(pitch, "#A00000"),
                fontsize=18, fontweight="bold", va="top")

        y_start = 0.87
        y_step = 0.095

        for i, row in enumerate(sub.itertuples()):
            y = y_start - i * y_step

            ax.text(0.02, y, row.Pitcher, color="white", fontsize=14, weight="bold")

            ax.text(0.60, y, f"Loc+: {round(row.Loc_plus,1)}",
                    color=pitch_colors.get(pitch, "white"), fontsize=14)

            ax.text(0.60, y - 0.03, f"LHH: {round(row.Loc_plus_LHH or 0,1)}",
                    color=LHH_COLOR, fontsize=12)

            ax.text(0.60, y - 0.06, f"RHH: {round(row.Loc_plus_RHH or 0,1)}",
                    color=RHH_COLOR, fontsize=12)

    st.pyplot(fig2)
    
########
#PAGE 6
#########

def pitcher_profile_page():
    st.header("🎯 Pitcher Profile")

    # -----------------------------
    # LOAD PITCH-BY-PITCH DATA
    # -----------------------------
    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    full_df = df.copy()

    # -----------------------------
    # SELECT PITCHER
    # -----------------------------
    pitcher_list = sorted(full_df["Pitcher"].unique())
    pitcher = st.selectbox("Select Pitcher", pitcher_list)

    # -----------------------------
    # SEASON SUMMARY (from CSV pitching_df)
    # -----------------------------
    season_row = pitching_df[pitching_df["Player"] == pitcher]

    if season_row.empty:
        st.warning("No season stats found for this pitcher in the season CSV.")
    else:
        row = season_row.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ERA", f"{row['ERA']:.2f}")
        col2.metric("IP", f"{row['IP']:.1f}")
        col3.metric("W-L", row["W-L"])
        col4.metric("Opp BA", f"{row['BA']:.3f}")

        col1, col2, col3, col4 = st.columns(4)
        WHIP = (row["BB"] + row["H"]) / row["IP"] if row["IP"] > 0 else np.nan
        col1.metric("WHIP", f"{WHIP:.2f}")
        col2.metric("K%", f"{100 * row['SO'] / (row['SO'] + row['BB'] + 1e-6):.1f}%")
        col3.metric("BB%", f"{100 * row['BB'] / (row['SO'] + row['BB'] + 1e-6):.1f}%")
        col4.metric("HR Allowed", int(row["HR"]))

    st.markdown("---")

    # -----------------------------
    # GAME LOG
    # -----------------------------
    st.subheader("📘 Game Log")

    games_df = (
        full_df.groupby(["game_id","GameDate","Opponent","Pitcher"])
        .size()
        .reset_index(name="Pitches")
    )

    pitcher_games = games_df[games_df["Pitcher"] == pitcher].copy()
    pitcher_games["label"] = (
        pitcher_games["GameDate"].astype(str) + " vs " +
        pitcher_games["Opponent"]
    )

    st.dataframe(
        pitcher_games[["GameDate","Opponent","Pitches"]],
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # GAME REPORT GENERATOR
    # -----------------------------
    st.subheader("📄 Generate Game Report")

    selected_game = st.selectbox(
        "Select a game",
        pitcher_games["label"]
    )

    if selected_game:
        g = pitcher_games[pitcher_games["label"] == selected_game].iloc[0]

        game_pdf = full_df[full_df["game_id"] == g["game_id"]]

        fig = build_postgame_figure(
            pdf=game_pdf,
            pitcher=pitcher,
            game_date=g["GameDate"],
            opponent=g["Opponent"]
        )

        st.pyplot(fig)

        pdf_bytes = figure_to_pdf_bytes(fig)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{pitcher}_{g['GameDate']}_{g['Opponent']}.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    # -----------------------------
    # TRENDLINES
    # -----------------------------
    st.subheader("📈 Season Trends")

    pitcher_df = full_df[full_df["Pitcher"] == pitcher].copy()

    trend_df = (
        pitcher_df.groupby("GameDate")
        .agg({"Stuff+":"mean","Loc+":"mean","is_strike":"mean"})
        .reset_index()
    )

    trend_df["Strike%"] = 100 * trend_df["is_strike"]

    st.line_chart(
        trend_df.set_index("GameDate")[["Stuff+","Loc+","Strike%"]],
        height=300
    )

    st.markdown("---")

    # -----------------------------
    # PITCH MIX EVOLUTION
    # -----------------------------
    st.subheader("🎛️ Pitch Mix Over Time")

    mix_df = (
        pitcher_df.groupby(["GameDate","pitch_abbr"])
        .size()
        .reset_index(name="N")
    )

    mix_df["Usage%"] = mix_df.groupby("GameDate")["N"].transform(lambda x: 100 * x / x.sum())

    mix_pivot = mix_df.pivot(index="GameDate", columns="pitch_abbr", values="Usage%").fillna(0)

    st.area_chart(mix_pivot, height=300)

    st.markdown("---")

    # -----------------------------
    # RELEASE DRIFT
    # -----------------------------
    st.subheader("🎯 Release Drift")

    st.scatter_chart(
        pitcher_df,
        x="RelS",
        y="RelH",
        color="pitch_abbr",
        size=50,
        height=300
    )

    st.markdown("---")

    # -----------------------------
    # MOVEMENT CLUSTERS
    # -----------------------------
    st.subheader("🌀 Movement Clusters")

    st.scatter_chart(
        pitcher_df,
        x="HB",
        y="IVB",
        color="pitch_abbr",
        size=50,
        height=300
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    st.markdown(
        "<h1 style='text-align:center; color:#A00000;'>Fordham Baseball – Pitching Analytics</h1>",
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Postgame Summary",
        "Season Summary",
        "Stuff+ Leaderboard",
        "Location+ Leaderboard",
        "Pitch-Type Grids",
        "Pitcher Profile"
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

    with tab6:
        pitcher_profile_page()   # ⭐ NEW TAB FUNCTION


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if check_password():
    main()
else:
    st.stop()
