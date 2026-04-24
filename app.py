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
import pandas as pd
import io
from matplotlib.backends.backend_pdf import PdfPages

def figure_to_pdf_bytes(fig):
    """
    Convert a Matplotlib figure to PDF bytes for download.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(ttl=1, show_spinner=False)
def load_pitching_stats():
    df = pd.read_csv(
        "teamstat/pitching_stats.csv",   # ← 🔥 NEW LOCATION
        dtype=str,
        keep_default_na=False
    )

    # Convert numeric columns manually
    df["ERA"] = df["ERA"].astype(float)
    df["H"] = df["H"].astype(int)
    df["ER"] = df["ER"].astype(int)
    df["BB"] = df["BB"].astype(int)
    df["SO"] = df["SO"].astype(int)
    df["HR"] = df["HR"].astype(int)

    # BA like ".241" → 0.241
    df["BA"] = df["BA"].astype(float)

    return df



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

def umpire_scorecard_page():
    st.header("Umpire Scorecard")

    data_dir = Path("data")
    game_files = sorted(list(data_dir.glob("*.csv")))

    if not game_files:
        st.error("No TrackMan CSVs found in /data")
        return

    selected_game = st.selectbox(
        "Select a TrackMan Game CSV",
        game_files,
        format_func=lambda x: x.name
    )

    if st.button("Generate Scorecard"):
        out_path = generate_umpire_scorecard(selected_game)
        st.image(str(out_path), caption="Umpire Scorecard", use_column_width=True)


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
# LOAD RAW CSVs (ignore season summary CSV)
# ------------------------------------------------------------
def load_all_raw():
    DATA_DIR = ROOT / "data"
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        return []

    valid_raw = []
    for f in csvs:
        try:
            # Skip the season summary CSV
            if f.name.lower() == "pitching_stats.csv":
                continue

            df = pd.read_csv(f, encoding="latin1", sep=None, engine="python")

            # Only accept pitch-by-pitch files
            if "Pitcher" in df.columns:
                valid_raw.append(df)

        except:
            continue

    return valid_raw

def ip_to_innings(ip_raw):
    """
    Convert baseball IP notation (e.g., 35.1, 35.2) to true innings:
    35.1 -> 35 + 1/3
    35.2 -> 35 + 2/3
    """
    ip_float = float(ip_raw)
    whole = int(ip_float)
    frac_tenths = round((ip_float - whole) * 10)

    if frac_tenths == 1:
        return whole + 1.0 / 3.0
    elif frac_tenths == 2:
        return whole + 2.0 / 3.0
    else:
        return float(whole)


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
        color="#FFFFFF", fontsize=30, fontweight="bold",
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
        color="#FFFFFF", fontsize=30, fontweight="bold",
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
    
# ------------------------------------------------------------
# PAGE 6 — PITCHER PROFILE (SEASON STATS + SIMPLE TUNNELING)
# ------------------------------------------------------------

# -----------------------------
# Convert baseball IP notation to true innings
# -----------------------------
def ip_to_innings(ip_raw):
    ip_str = str(ip_raw).strip()
    if "." not in ip_str:
        return float(ip_str)
    whole, frac = ip_str.split(".")
    whole = int(whole)
    if frac == "1":
        return whole + 1/3
    elif frac == "2":
        return whole + 2/3
    else:
        return float(whole)

# -----------------------------
# Normalize names so CSV + TrackMan match
# -----------------------------
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.replace(",", " ").replace("-", " ").upper().strip()
    parts = [p for p in name.split() if p]
    return " ".join(sorted(parts))   # ensures “CAWLEY DECLAN” matches “DECLAN CAWLEY”

# -----------------------------
# Load pitching stats from teamstat/
# -----------------------------
@st.cache_data(ttl=1, show_spinner=False)
def load_pitching_stats():
    df = pd.read_csv(
        "teamstat/pitching_stats.csv",
        dtype=str,
        keep_default_na=False
    )
    df["ERA"] = df["ERA"].astype(float)
    df["H"] = df["H"].astype(int)
    df["ER"] = df["ER"].astype(int)
    df["BB"] = df["BB"].astype(int)
    df["SO"] = df["SO"].astype(int)
    df["HR"] = df["HR"].astype(int)
    df["BA"] = df["BA"].astype(float)
    return df

# -----------------------------
# Movement Clusters Figure
# -----------------------------
def build_movement_figure(pitcher_df):
    df = pitcher_df.dropna(subset=["HB", "IVB"])
    fig, ax = plt.subplots(figsize=(6, 6))

    x_min, x_max = df["HB"].min() - 2, df["HB"].max() + 2
    y_min, y_max = df["IVB"].min() - 2, df["IVB"].max() + 2

    ax.axvspan(0, x_max, color="#d9f2ff", alpha=0.6)
    ax.axvspan(x_min, 0, color="#ffe0e0", alpha=0.6)

    for pitch, sub in df.groupby("pitch_abbr"):
        ax.scatter(sub["HB"], sub["IVB"], label=pitch, s=40, alpha=0.85)

    ax.axhline(0, color="white", linewidth=2)
    ax.axvline(0, color="white", linewidth=2)

    ax.set_xlabel("Horizontal Break (HB)")
    ax.set_ylabel("Induced Vertical Break (IVB)")
    ax.set_title("Movement Clusters")
    ax.legend(loc="best", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)

    return fig

# -----------------------------
# Release Drift Figure
# -----------------------------
def build_release_figure(pitcher_df):
    df = pitcher_df.dropna(subset=["RelS", "RelH"])
    fig, ax = plt.subplots(figsize=(6, 6))

    x_min, x_max = df["RelS"].min() - 0.5, df["RelS"].max() + 0.5
    y_min, y_max = df["RelH"].min() - 0.5, df["RelH"].max() + 0.5

    for pitch, sub in df.groupby("pitch_abbr"):
        ax.scatter(sub["RelS"], sub["RelH"], label=pitch, s=40, alpha=0.85)

    ax.axhline(0, color="white", linewidth=2)
    ax.axvline(0, color="white", linewidth=2)

    ax.set_xlabel("Release Side (RelS)")
    ax.set_ylabel("Release Height (RelH)")
    ax.set_title("Release Drift")
    ax.legend(loc="best", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)

    return fig

# -----------------------------
# SIMPLE TUNNELING (NO ARM ANGLE)
# -----------------------------
def build_tunneling_figure(pitcher_df):
    df = pitcher_df.dropna(subset=["RelS", "RelH", "HB", "IVB"]).copy()
    fig, ax = plt.subplots(figsize=(6, 6))

    if df.empty:
        ax.text(0.5, 0.5, "No valid tunneling data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Release points
    ax.scatter(
        df["RelS"], df["RelH"],
        s=80, alpha=0.9, color="#4fa3ff",
        label="Release Points", edgecolor="black"
    )

    # Movement endpoints
    ax.scatter(
        df["HB"], df["IVB"],
        s=80, alpha=0.9, color="#ff7f7f",
        label="Movement Endpoints", edgecolor="black"
    )

    ax.axhline(0, color="white", linewidth=2)
    ax.axvline(0, color="white", linewidth=2)

    ax.set_xlabel("Release Side / HB")
    ax.set_ylabel("Release Height / IVB")
    ax.set_title("Pitch Tunneling (Release → Movement)")
    ax.legend(loc="best", fontsize=8)

    ax.set_aspect("equal", adjustable="box")

    all_x = list(df["RelS"]) + list(df["HB"])
    all_y = list(df["RelH"]) + list(df["IVB"])

    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

    ax.grid(True, alpha=0.25)

    return fig

# -----------------------------
# MAIN PAGE 6 FUNCTION
# -----------------------------
def pitcher_profile_page():
    st.header("🎯 Pitcher Profile")

    df = prepare_data()
    df = filter_fordham_only(df)
    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    df["GameDate"] = pd.to_datetime(df.get("GameDate", df.get("Date")), errors="coerce")
    df["Opponent"] = df.get("Opponent", df.get("BatterTeam"))
    df["game_id"] = (
        df["GameDate"].dt.strftime("%Y%m%d").fillna("00000000")
        + "_"
        + df.get("PitcherTeam", "UNK").astype(str)
        + "_vs_"
        + df["Opponent"].astype(str)
    )

    full_df = df.copy()

    pitching_df = load_pitching_stats()

    pitchers = get_pitcher_list(full_df)
    pitcher = st.selectbox("Select Pitcher", pitchers)

    # 🔥 FIX: normalize both sides
    pitcher_norm = normalize_name(pitcher)
    pitching_df["name_norm"] = pitching_df["Pitcher"].apply(normalize_name)

    season_row = pitching_df[pitching_df["name_norm"] == pitcher_norm]

    # -----------------------------
    # SEASON SUMMARY
    # -----------------------------
    if not season_row.empty:
        row = season_row.iloc[0]

        ip = ip_to_innings(row["IP"])
        h = int(row["H"])
        bb = int(row["BB"])
        so = int(row["SO"])
        era = float(row["ERA"])
        hr_val = int(row["HR"])
        ba = float(row["BA"])
        wl_val = row["W-L"]

        whip = (bb + h) / ip
        k9 = (so * 9.0) / ip
        bb9 = (bb * 9.0) / ip

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ERA", f"{era:.2f}")
        col2.metric("IP", f"{ip:.1f}")
        col3.metric("W-L", wl_val)
        col4.metric("Opp BA", f"{ba:.3f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("WHIP", f"{whip:.2f}")
        col2.metric("K/9", f"{k9:.2f}")
        col3.metric("BB/9", f"{bb9:.2f}")
        col4.metric("HR Allowed", hr_val)

    st.markdown("---")

    # -----------------------------
    # GAME LOG
    # -----------------------------
    st.subheader("📘 Game Log")

    games_df = (
        full_df.groupby(["game_id", "GameDate", "Opponent", "Pitcher"])
        .size()
        .reset_index(name="Pitches")
    )

    pitcher_games = games_df[games_df["Pitcher"] == pitcher].copy()

    pitcher_games["label"] = (
        pitcher_games["GameDate"].dt.strftime("%Y-%m-%d") + " vs " + pitcher_games["Opponent"]
    )

    st.dataframe(
        pitcher_games[["GameDate", "Opponent", "Pitches"]],
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # GAME REPORT
    # -----------------------------
    st.subheader("📄 Generate Game Report")

    selected_game = st.selectbox("Select a game", pitcher_games["label"])

    if selected_game:
        g = pitcher_games[pitcher_games["label"] == selected_game].iloc[0]

        game_pdf = full_df[
            (full_df["game_id"] == g["game_id"]) &
            (full_df["Pitcher"] == pitcher)
        ]

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
            file_name=f"{pitcher}_{g['GameDate'].date()}_{g['Opponent']}.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    # -----------------------------
    # TRENDS
    # -----------------------------
    st.subheader("📈 Season Trends")

    pitcher_df = full_df[full_df["Pitcher"] == pitcher].copy()

    trend_df = (
        pitcher_df.groupby("GameDate")
        .agg({"Stuff+": "mean", "Loc+": "mean", "is_strike": "mean"})
        .reset_index()
    )

    trend_df["Strike%"] = 100 * trend_df["is_strike"]

    st.line_chart(
        trend_df.set_index("GameDate")[["Stuff+", "Loc+", "Strike%"]],
        height=300
    )

    st.markdown("---")

    # -----------------------------
    # RELEASE DRIFT
    # -----------------------------
    st.subheader("🎯 Release Drift")
    st.pyplot(build_release_figure(pitcher_df))

    st.markdown("---")

    # -----------------------------
    # MOVEMENT CLUSTERS
    # -----------------------------
    st.subheader("🌀 Movement Clusters")
    st.pyplot(build_movement_figure(pitcher_df))

    st.markdown("---")

    # -----------------------------
    # TUNNELING
    # -----------------------------
    st.subheader("🎯 Pitch Tunneling Visualization")
    st.pyplot(build_tunneling_figure(pitcher_df))


def generate_umpire_scorecard(csv_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path

    # Load CSV
    df = pd.read_csv(csv_path, encoding="latin1", sep=None, engine="python")

    fordham_team = df["HomeTeam"].iloc[0]
    opponent_team = df["AwayTeam"].iloc[0]
    game_date = pd.to_datetime(df["Date"].iloc[0]).strftime("%B %d, %Y")

    # Strike zone constants
    ZONE_LEFT, ZONE_RIGHT = -0.83, 0.83
    ZONE_BOTTOM, ZONE_TOP = 1.5, 3.5
    TOUCH_MARGIN = 0.15
    HEADER_MAROON = "#A00000"

    # Zone logic
    def in_zone(row):
        x = row["PlateLocSide"]
        y = row["PlateLocHeight"]
        in_main = ZONE_LEFT <= x <= ZONE_RIGHT and ZONE_BOTTOM <= y <= ZONE_TOP
        touching = (
            (ZONE_LEFT - TOUCH_MARGIN <= x <= ZONE_RIGHT + TOUCH_MARGIN) and
            (ZONE_BOTTOM - TOUCH_MARGIN <= y <= ZONE_TOP + TOUCH_MARGIN)
        )
        return in_main or touching

    df["InZone"] = df.apply(in_zone, axis=1)

    # Called pitches only
    called_df = df[df["PitchCall"].isin(["StrikeCalled", "BallCalled"])].copy()

    # Correct / incorrect
    called_df["Correct"] = (
        (called_df["PitchCall"] == "StrikeCalled") & (called_df["InZone"]) |
        (called_df["PitchCall"] == "BallCalled") & (~called_df["InZone"])
    )

    # Favor team
    def favor_team(row):
        if row["Correct"]:
            return "None"
        if row["PitchCall"] == "StrikeCalled":
            return row["PitcherTeam"]
        else:
            return row["BatterTeam"]

    called_df["FavoredTeam"] = called_df.apply(favor_team, axis=1)

    # Metrics
    overall_accuracy = round(called_df["Correct"].mean() * 100, 1)
    strike_accuracy = round(called_df[called_df["PitchCall"] == "StrikeCalled"]["Correct"].mean() * 100, 1)
    ball_accuracy = round(called_df[called_df["PitchCall"] == "BallCalled"]["Correct"].mean() * 100, 1)
    favor_counts = called_df["FavoredTeam"].value_counts()

    # Missed calls
    missed = called_df[~called_df["Correct"]][[
        "Inning", "PitchCall", "PlateLocSide", "PlateLocHeight",
        "Pitcher", "Batter", "PitcherTeam", "BatterTeam"
    ]]

    # Figure
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#1e1e1e")

    # Logo
    logo_path = Path("assets/rams.png")
    if logo_path.exists():
        logo_img = mpimg.imread(logo_path)
        fig.figimage(logo_img, xo=40, yo=fig.bbox.ymax - 300, zorder=50)

    # Title
    fig.suptitle(
        f"Umpire Scorecard – Fordham vs {opponent_team}",
        fontsize=30, fontweight="bold", color=HEADER_MAROON, y=0.97
    )

    # Metrics box
    axM = plt.subplot2grid((4, 4), (0, 2), colspan=2)
    axM.axis("off")
    metrics_text = (
        f"Overall Accuracy: {overall_accuracy}%\n"
        f"Called Strike Accuracy: {strike_accuracy}%\n"
        f"Called Ball Accuracy: {ball_accuracy}%\n\n"
        f"Favor – {fordham_team}: {favor_counts.get(fordham_team, 0)}\n"
        f"Favor – {opponent_team}: {favor_counts.get(opponent_team, 0)}"
    )
    axM.text(0, 0.9, "Umpire Metrics", fontsize=20, color="white", weight="bold")
    axM.text(0, 0.45, metrics_text, fontsize=16, color="white", va="top")

    # Strike zone plot
    axZ = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    axZ.set_facecolor("#1e1e1e")
    axZ.set_xlim(-2.5, 2.5)
    axZ.set_ylim(0, 5)
    axZ.set_aspect("equal")

    # Main zone
    axZ.plot(
        [ZONE_LEFT, ZONE_RIGHT, ZONE_RIGHT, ZONE_LEFT, ZONE_LEFT],
        [ZONE_BOTTOM, ZONE_BOTTOM, ZONE_TOP, ZONE_TOP, ZONE_BOTTOM],
        color="white", linewidth=2.5
    )

    # Touch zone (visual buffer)
    axZ.plot(
        [ZONE_LEFT - TOUCH_MARGIN, ZONE_RIGHT + TOUCH_MARGIN,
         ZONE_RIGHT + TOUCH_MARGIN, ZONE_LEFT - TOUCH_MARGIN,
         ZONE_LEFT - TOUCH_MARGIN],
        [ZONE_BOTTOM - TOUCH_MARGIN, ZONE_BOTTOM - TOUCH_MARGIN,
         ZONE_TOP + TOUCH_MARGIN, ZONE_TOP + TOUCH_MARGIN,
         ZONE_BOTTOM - TOUCH_MARGIN],
        color="white", linestyle="--", linewidth=1.2, alpha=0.4
    )

    # Home plate — moved to bottom of graphic
    plate_top = 0.25
    plate_bottom = 0.05
    home_x = [-0.85, 0.85, 0.55, 0.0, -0.55]
    home_y = [plate_bottom, plate_bottom, plate_top, plate_top + 0.12, plate_top]
    axZ.fill(home_x, home_y, facecolor="white", edgecolor="black", linewidth=2, zorder=5)

    # Plot pitches (smaller markers)
    for _, row in called_df.iterrows():
        if row["Correct"]:
            color, marker, size = "lime", "o", 45
        else:
            if row["PitchCall"] == "StrikeCalled" and not row["InZone"]:
                color, marker, size = "orange", "X", 75
            else:
                color, marker, size = "red", "o", 75

        axZ.scatter(
            row["PlateLocSide"],
            row["PlateLocHeight"],
            s=size, color=color, marker=marker,
            edgecolor="white", linewidth=0.9
        )

    axZ.set_title(
        "Green = Correct • Orange X = Bad Strike • Red = Bad Ball",
        color="white"
    )

    # Missed calls table
    axT = plt.subplot2grid((4, 4), (2, 0), colspan=4, rowspan=2)
    axT.axis("off")

    if len(missed) > 0:
        tbl = axT.table(
            cellText=missed.values,
            colLabels=missed.columns,
            loc="center",
            cellLoc="center",
            bbox=[0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_height(0.06)
            cell.set_width(0.12)
            if r == 0:
                cell.set_facecolor(HEADER_MAROON)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#1e1e1e")
                cell.set_text_props(color="white")
    else:
        axT.text(0.5, 0.5, "No Missed Calls", ha="center", va="center", color="white", fontsize=20)

    # Footer
    plt.text(
        0.99, 0.02,
        f"Game Date: {game_date}",
        ha="right", va="center",
        fontsize=14, color="white",
        transform=fig.transFigure
    )

    # Save
    output_dir = Path("output/umpire_scorecards")
    output_dir.mkdir(parents=True, exist_ok=True)

    out = output_dir / f"UmpireScorecard_{game_date.replace(' ', '_')}.png"
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close()

    return out


# ------------------------------------------------------------
# ------------------------------------------------------------
# TAB 8 — CONTACT QUALITY LEADERBOARD (FULLY FIXED)
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# SHARED PA ENGINE (IDENTICAL TO HITTER/PITCHER TABS)
# ============================================================

def get_pa_endings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pa_keys = [c for c in ["Date","Inning","PAofInning"] if c in df.columns]

    if "PitchofPA" in df.columns and len(pa_keys) >= 2:
        df = df.sort_values(pa_keys + ["PitchofPA"])
        return df.groupby(pa_keys).tail(1)

    fallback = [c for c in ["Date","Inning","Batter"] if c in df.columns]
    if "PitchNo" in df.columns and len(fallback) >= 2:
        df = df.sort_values(fallback + ["PitchNo"])
        return df.groupby(fallback).tail(1)

    return df


# ============================================================
# UNIVERSAL wOBA ENGINE (IDENTICAL TO OTHER TABS)
# ============================================================

def compute_woba(hdf: pd.DataFrame) -> float:
    if hdf.empty:
        return 0.0

    pa = get_pa_endings(hdf)

    wBB  = 0.69
    wHBP = 0.72
    w1B  = 0.88
    w2B  = 1.247
    w3B  = 1.578
    wHR  = 2.031

    BB  = (pa["KorBB"] == "Walk").sum()
    HBP = (pa["PitchCall"] == "HitByPitch").sum()
    _1B = (pa["PlayResult"] == "Single").sum()
    _2B = (pa["PlayResult"] == "Double").sum()
    _3B = (pa["PlayResult"] == "Triple").sum()
    HR  = (pa["PlayResult"] == "HomeRun").sum()
    SF  = (pa["PlayResult"] == "Sacrifice").sum()

    PA = len(pa)
    AB = PA - BB - HBP - SF

    numerator = (
        wBB*BB + wHBP*HBP + w1B*_1B + w2B*_2B + w3B*_3B + wHR*HR
    )
    denominator = AB + BB + HBP + SF

    return numerator / denominator if denominator > 0 else 0.0


def compute_league_woba(df: pd.DataFrame) -> float:
    return 0.320


def compute_wrc_plus(player_woba: float, league_woba: float = 0.290) -> float:
    wOBAScale = 1.15
    return round(((player_woba - league_woba) / wOBAScale) * 100 + 100, 0)


# ============================================================
# CONTACT QUALITY FLAGS (EV/LA ONLY — NO wOBA HERE)
# ============================================================

def add_contact_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]
    if "LA" not in df.columns and "Angle" in df.columns:
        df["LA"] = df["Angle"]

    df.loc[df["EV"] > 118, "EV"] = np.nan

    foul = ["Foul","FoulBall","FoulBallFieldable","FoulBallNotFieldable"]
    df.loc[df["PlayResult"].isin(foul), ["EV","LA"]] = np.nan

    bunts = [
        "Bunt","BuntGroundout","BuntPopOut","BuntLineOut",
        "SacrificeBunt","BuntFoul","BuntFoulTip"
    ]
    df.loc[df["TaggedHitType"].isin(bunts), ["EV","LA"]] = np.nan

    df["hard_hit"] = (df["EV"] >= 90).astype(int)
    df["barrel"] = ((df["EV"] >= 98) & df["LA"].between(26,30)).astype(int)
    df["sweet_spot"] = df["LA"].between(8,32).astype(int)

    df["is_swing"] = df["is_swing"].fillna(0).astype(int)
    df["is_whiff"] = df["is_whiff"].fillna(0).astype(int)
    df["is_chase"] = (
        (df["is_swing"] == 1) &
        (df["is_whiff"] == 1) &
        (~df["in_zone"].fillna(0).astype(bool))
    ).astype(int)

    return df


# ============================================================
# SUMMARY TABLE (NOW USING TRUE PA-BASED wOBA)
# ============================================================

def summarize_contact_quality(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return pd.DataFrame()

    lgwOBA = compute_league_woba(df)

    rows = []

    for name, g in df.groupby(group_col):
        if g.empty:
            continue

        pa_end = get_pa_endings(g)
        if pa_end.empty:
            continue

        PA = len(pa_end)
        BB = (pa_end["KorBB"] == "Walk").sum()
        K  = (pa_end["KorBB"] == "Strikeout").sum()

        player_woba = compute_woba(g)
        player_wrc_plus = compute_wrc_plus(player_woba, lgwOBA)

        swings = g["is_swing"].sum()
        whiffs = g["is_whiff"].sum()
        chases = g["is_chase"].sum()

        bip = g.dropna(subset=["EV","LA"])
        hard = bip["hard_hit"].mean() if not bip.empty else np.nan
        barrel = bip["barrel"].mean() if not bip.empty else np.nan
        sweet = bip["sweet_spot"].mean() if not bip.empty else np.nan
        avg_ev = bip["EV"].mean() if not bip.empty else np.nan
        max_ev = bip["EV"].max() if not bip.empty else np.nan
        avg_la = bip["LA"].mean() if not bip.empty else np.nan

        rows.append({
            group_col: name,
            "PA": PA,
            "BB": BB,
            "K": K,
            "wOBA": round(player_woba,3),
            "wRC+": player_wrc_plus,
            "Swings": swings,
            "Whiffs": whiffs,
            "Chases": chases,
            "HardHit%": round(hard*100,1) if hard==hard else np.nan,
            "Barrel%": round(barrel*100,1) if barrel==barrel else np.nan,
            "SweetSpot%": round(sweet*100,1) if sweet==sweet else np.nan,
            "AvgEV": round(avg_ev,1) if avg_ev==avg_ev else np.nan,
            "MaxEV": round(max_ev,1) if max_ev==max_ev else np.nan,
            "AvgLA": round(avg_la,1) if avg_la==avg_la else np.nan,
            "BB%": round(BB/PA*100,1) if PA>0 else 0,
            "K%": round(K/PA*100,1) if PA>0 else 0,
            "Whiff%": round(whiffs/swings*100,1) if swings>0 else 0,
            "Chase%": round(chases/swings*100,1) if swings>0 else 0,
        })

    return pd.DataFrame(rows)


# ============================================================
# PAGE
# ============================================================

def contact_quality_leaderboard_page(all_pitches_df: pd.DataFrame):
    st.markdown("## 🔥 Contact Quality Leaderboard")

    df = all_pitches_df.copy()

    df = df.rename(columns={
        "ExitSpeed":"EV",
        "Angle":"LA",
        "Direction":"Spray"
    })

    df = add_contact_quality(df)

    teams = sorted(set(
        df.get("BatterTeam", pd.Series()).dropna().unique().tolist() +
        df.get("PitcherTeam", pd.Series()).dropna().unique().tolist()
    ))

    if not teams:
        st.warning("No team info found.")
        return

    default = "FOR_RAM" if "FOR_RAM" in teams else teams[0]
    team = st.selectbox("Select Team", teams, index=teams.index(default))

    mode = st.radio("View:", ["Hitters","Pitchers"], horizontal=True)

    if mode == "Hitters":
        sub = df[df["BatterTeam"] == team]
        summary = summarize_contact_quality(sub, "Batter")
        summary = summary.sort_values("wRC+", ascending=False)
        st.dataframe(summary, use_container_width=True)

    else:
        sub = df[df["PitcherTeam"] == team]
        summary = summarize_contact_quality(sub, "Pitcher")
        summary = summary.sort_values("HardHit%", ascending=True)
        st.dataframe(summary, use_container_width=True)



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PA-LEVEL ENGINE (USED BY BOTH TABS)
# ============================================================

def get_pa_endings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the final pitch of each PA.
    Uses Date, Inning, PAofInning, PitchofPA when available.
    Falls back to last pitch per (Inning, Batter) if needed.
    """
    df = df.copy()

    # Best case: full PA keys exist
    pa_keys = []
    for c in ["Date", "Inning", "PAofInning"]:
        if c in df.columns:
            pa_keys.append(c)

    if "PitchofPA" in df.columns and len(pa_keys) >= 2:
        df = df.sort_values(pa_keys + ["PitchofPA"])
        pa_end = df.groupby(pa_keys).tail(1)
        return pa_end

    # Fallback: use Inning + Batter + PitchNo if PitchofPA missing
    fallback_keys = []
    for c in ["Date", "Inning", "Batter"]:
        if c in df.columns:
            fallback_keys.append(c)

    if "PitchNo" in df.columns and len(fallback_keys) >= 2:
        df = df.sort_values(fallback_keys + ["PitchNo"])
        pa_end = df.groupby(fallback_keys).tail(1)
        return pa_end

    # Last resort: return original (won't be perfect but avoids crash)
    return df


def compute_woba(hdf: pd.DataFrame) -> float:
    """
    True wOBA per PA using only PA-ending pitches.
    Same function is used for hitters and pitchers.
    """
    if hdf.empty:
        return 0.0

    pa = get_pa_endings(hdf)

    # Weights
    wBB  = 0.69
    wHBP = 0.72
    w1B  = 0.88
    w2B  = 1.247
    w3B  = 1.578
    wHR  = 2.031

    # Events
    BB  = (pa["KorBB"] == "Walk").sum() if "KorBB" in pa.columns else 0
    HBP = (pa["PitchCall"] == "HitByPitch").sum() if "PitchCall" in pa.columns else 0
    _1B = (pa["PlayResult"] == "Single").sum() if "PlayResult" in pa.columns else 0
    _2B = (pa["PlayResult"] == "Double").sum() if "PlayResult" in pa.columns else 0
    _3B = (pa["PlayResult"] == "Triple").sum() if "PlayResult" in pa.columns else 0
    HR  = (pa["PlayResult"] == "HomeRun").sum() if "PlayResult" in pa.columns else 0
    SF  = (pa["PlayResult"] == "Sacrifice").sum() if "PlayResult" in pa.columns else 0

    PA = len(pa)
    AB = PA - BB - HBP - SF

    numerator = (
        wBB * BB +
        wHBP * HBP +
        w1B * _1B +
        w2B * _2B +
        w3B * _3B +
        wHR * HR
    )
    denominator = AB + BB + HBP + SF

    return float(numerator / denominator) if denominator > 0 else 0.0


def compute_league_woba(df: pd.DataFrame) -> float:
    """
    Fixed league wOBA so both tabs scale identically.
    """
    return 0.290


def compute_wrc_plus(player_woba: float, league_woba: float = 0.320) -> int:
    """
    wRC+ scaled off fixed league wOBA.
    """
    wOBAScale = 1.15
    return int(round(((player_woba - league_woba) / wOBAScale) * 100 + 100))


# ============================================================
# SHARED NORMALIZATION + CONTACT QUALITY
# ============================================================

def normalize_hitter_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Batter
    if "Batter" not in df.columns:
        for alt in ["BatterName", "Hitter", "BatterId"]:
            if alt in df.columns:
                df["Batter"] = df[alt]
                break

    # Pitch type
    if "pitch_abbr" not in df.columns:
        if "TaggedPitchType" in df.columns:
            df["pitch_abbr"] = df["TaggedPitchType"]
        elif "AutoPitchType" in df.columns:
            df["pitch_abbr"] = df["AutoPitchType"]
        else:
            df["pitch_abbr"] = "UNK"

    # Count
    if "Count" not in df.columns and "Balls" in df.columns and "Strikes" in df.columns:
        df["Count"] = df["Balls"].astype(str) + "-" + df["Strikes"].astype(str)

    # Zone location
    if "PlateLocSide" not in df.columns:
        df["PlateLocSide"] = np.nan
    if "PlateLocHeight" not in df.columns:
        df["PlateLocHeight"] = np.nan

    # EV / LA
    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]
    if "LA" not in df.columns and "Angle" in df.columns:
        df["LA"] = df["Angle"]

    return df


def add_contact_quality_local(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]
    if "LA" not in df.columns and "Angle" in df.columns:
        df["LA"] = df["Angle"]

    df["hard_hit"] = np.where(df["EV"].fillna(0) >= 95, 1, 0)
    df["barrel"] = np.where(
        (df["EV"].fillna(0) >= 98) &
        (df["LA"].fillna(0).between(26, 30)),
        1, 0
    )
    df["sweet_spot"] = np.where(
        df["LA"].fillna(0).between(8, 32),
        1, 0
    )

    swing_calls = ["StrikeSwinging", "FoulBall", "InPlay"]
    df["is_swing"] = df.get("PitchCall", "").isin(swing_calls).astype(int)
    df["is_whiff"] = (df.get("PitchCall", "") == "StrikeSwinging").astype(int)

    df["is_chase"] = 0
    if "Count" in df.columns:
        chase_counts = ["0-2", "1-2", "2-2"]
        df.loc[df["Count"].isin(chase_counts) & (df["is_swing"] == 1), "is_chase"] = 1

    # woba_value for grids/heatmaps (per pitch, not PA)
    df["woba_value"] = 0.0
    wBB  = 0.69
    wHBP = 0.72
    w1B  = 0.88
    w2B  = 1.247
    w3B  = 1.578
    wHR  = 2.031

    if "KorBB" in df.columns:
        df.loc[df["KorBB"] == "Walk", "woba_value"] = wBB
    if "PitchCall" in df.columns:
        df.loc[df["PitchCall"] == "HitByPitch", "woba_value"] = wHBP
    if "PlayResult" in df.columns:
        df.loc[df["PlayResult"] == "Single", "woba_value"] = w1B
        df.loc[df["PlayResult"] == "Double", "woba_value"] = w2B
        df.loc[df["PlayResult"] == "Triple", "woba_value"] = w3B
        df.loc[df["PlayResult"] == "HomeRun", "woba_value"] = wHR

    return df


# ============================================================
# HITTER HELPERS
# ============================================================

def compute_hitter_card(hdf: pd.DataFrame, lgwOBA: float) -> dict:
    card = {}

    pa_end = get_pa_endings(hdf)

    card["PA"] = len(pa_end)

    card["BB"] = (pa_end.get("KorBB", "") == "Walk").sum()
    card["K"] = (pa_end.get("KorBB", "") == "Strikeout").sum()
    card["HBP"] = (pa_end.get("PitchCall", "") == "HitByPitch").sum()

    card["1B"] = (pa_end.get("PlayResult", "") == "Single").sum()
    card["2B"] = (pa_end.get("PlayResult", "") == "Double").sum()
    card["3B"] = (pa_end.get("PlayResult", "") == "Triple").sum()
    card["HR"] = (pa_end.get("PlayResult", "") == "HomeRun").sum()
    card["H"] = card["1B"] + card["2B"] + card["3B"] + card["HR"]

    SF = (pa_end.get("PlayResult", "") == "Sacrifice").sum()
    card["AB"] = card["PA"] - card["BB"] - card["HBP"] - SF

    card["BB%"] = round(card["BB"] / card["PA"] * 100, 1) if card["PA"] else 0.0
    card["K%"] = round(card["K"] / card["PA"] * 100, 1) if card["PA"] else 0.0

    player_woba = compute_woba(hdf)
    card["wOBA"] = round(player_woba, 3)
    card["wRC+"] = compute_wrc_plus(player_woba, lgwOBA)

    bip = hdf.dropna(subset=["EV", "LA"]) if {"EV", "LA"}.issubset(hdf.columns) else pd.DataFrame()
    if not bip.empty:
        card["HardHit%"] = round(bip["hard_hit"].mean() * 100, 1)
        card["Barrel%"] = round(bip["barrel"].mean() * 100, 1)
        card["SweetSpot%"] = round(bip["sweet_spot"].mean() * 100, 1)
        card["AvgEV"] = round(bip["EV"].mean(), 1)
        card["MaxEV"] = round(bip["EV"].max(), 1)
        card["AvgLA"] = round(bip["LA"].mean(), 1)
    else:
        card["HardHit%"] = card["Barrel%"] = card["SweetSpot%"] = np.nan
        card["AvgEV"] = card["MaxEV"] = card["AvgLA"] = np.nan

    swings = hdf["is_swing"].sum() if "is_swing" in hdf.columns else 0
    total_pitches = len(hdf)
    card["Swing%"] = round(swings / total_pitches * 100, 1) if total_pitches else 0.0
    card["Whiff%"] = round(hdf["is_whiff"].sum() / swings * 100, 1) if swings and "is_whiff" in hdf.columns else 0.0
    card["Chase%"] = round(hdf["is_chase"].sum() / swings * 100, 1) if swings and "is_chase" in hdf.columns else 0.0

    return card


def count_effectiveness(hdf: pd.DataFrame) -> pd.DataFrame:
    if "Count" not in hdf.columns:
        return pd.DataFrame()

    agg = hdf.groupby("Count").agg(
        N=("Count", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Chases=("is_chase", "sum")
    ).reset_index()

    bip = hdf.dropna(subset=["EV", "LA"]) if {"EV", "LA"}.issubset(hdf.columns) else pd.DataFrame()
    if not bip.empty:
        bip_agg = bip.groupby("Count").agg(
            HardHit=("hard_hit", "mean"),
            AvgEV=("EV", "mean"),
            AvgLA=("LA", "mean")
        ).reset_index()
        agg = agg.merge(bip_agg, on="Count", how="left")
    else:
        agg["HardHit"] = np.nan
        agg["AvgEV"] = np.nan
        agg["AvgLA"] = np.nan

    pa_df = hdf[hdf["woba_value"] > 0] if "woba_value" in hdf.columns else pd.DataFrame()
    if not pa_df.empty:
        woba_agg = pa_df.groupby("Count")["woba_value"].mean().reset_index(name="wOBA")
        agg = agg.merge(woba_agg, on="Count", how="left")
    else:
        agg["wOBA"] = np.nan

    agg["Swing%"] = (agg["Swings"] / agg["N"] * 100).round(1)
    agg["Whiff%"] = np.where(agg["Swings"] > 0, agg["Whiffs"] / agg["Swings"] * 100, 0).round(1)
    agg["Chase%"] = np.where(agg["Swings"] > 0, agg["Chases"] / agg["Swings"] * 100, 0).round(1)
    agg["HardHit%"] = (agg["HardHit"] * 100).round(1)
    agg["AvgEV"] = agg["AvgEV"].round(1)
    agg["AvgLA"] = agg["AvgLA"].round(1)
    agg["wOBA"] = agg["wOBA"].round(3)

    return agg.sort_values("Count")


def count_pitchtype_effectiveness(hdf: pd.DataFrame) -> pd.DataFrame:
    if "pitch_abbr" not in hdf.columns or "Count" not in hdf.columns:
        return pd.DataFrame()

    agg = hdf.groupby(["Count", "pitch_abbr"]).agg(
        N=("pitch_abbr", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Chases=("is_chase", "sum")
    ).reset_index()

    bip = hdf.dropna(subset=["EV", "LA"]) if {"EV", "LA"}.issubset(hdf.columns) else pd.DataFrame()
    if not bip.empty:
        bip_agg = bip.groupby(["Count", "pitch_abbr"]).agg(
            HardHit=("hard_hit", "mean"),
            AvgEV=("EV", "mean"),
            AvgLA=("LA", "mean")
        ).reset_index()
        agg = agg.merge(bip_agg, on=["Count", "pitch_abbr"], how="left")
    else:
        agg["HardHit"] = np.nan
        agg["AvgEV"] = np.nan
        agg["AvgLA"] = np.nan

    pa_df = hdf[hdf["woba_value"] > 0] if "woba_value" in hdf.columns else pd.DataFrame()
    if not pa_df.empty:
        woba_agg = pa_df.groupby(["Count", "pitch_abbr"])["woba_value"].mean().reset_index(name="wOBA")
        agg = agg.merge(woba_agg, on=["Count", "pitch_abbr"], how="left")
    else:
        agg["wOBA"] = np.nan

    agg["Swing%"] = (agg["Swings"] / agg["N"] * 100).round(1)
    agg["Whiff%"] = np.where(agg["Swings"] > 0, agg["Whiffs"] / agg["Swings"] * 100, 0).round(1)
    agg["Chase%"] = np.where(agg["Swings"] > 0, agg["Chases"] / agg["Swings"] * 100, 0).round(1)
    agg["HardHit%"] = (agg["HardHit"] * 100).round(1)
    agg["AvgEV"] = agg["AvgEV"].round(1)
    agg["AvgLA"] = agg["AvgLA"].round(1)
    agg["wOBA"] = agg["wOBA"].round(3)

    return agg.sort_values(["Count", "pitch_abbr"])


def hitter_splits(hdf: pd.DataFrame) -> pd.DataFrame:
    if "PitcherThrows" not in hdf.columns:
        return pd.DataFrame()

    hdf = hdf.copy()
    hdf["PitcherSide"] = np.where(
        hdf["PitcherThrows"].astype(str).str.upper().str.startswith("L"),
        "LHP", "RHP"
    )

    rows = []
    for side in ["LHP", "RHP"]:
        sub = hdf[hdf["PitcherSide"] == side]
        if sub.empty:
            continue

        pa_df = sub[sub["woba_value"] > 0] if "woba_value" in sub.columns else pd.DataFrame()
        swings = sub["is_swing"].sum() if "is_swing" in sub.columns else 0

        out = {
            "Split": side,
            "PA": len(pa_df),
            "wOBA": round(pa_df["woba_value"].mean(), 3) if len(pa_df) else np.nan,
            "Swing%": round(swings / len(sub) * 100, 1) if len(sub) else 0,
            "Whiff%": round(sub["is_whiff"].sum() / swings * 100, 1) if swings and "is_whiff" in sub.columns else 0,
            "Chase%": round(sub["is_chase"].sum() / swings * 100, 1) if swings and "is_chase" in sub.columns else 0,
        }

        bip = sub.dropna(subset=["EV"]) if "EV" in sub.columns else pd.DataFrame()
        out["HardHit%"] = round(bip["hard_hit"].mean() * 100, 1) if not bip.empty else np.nan
        out["AvgEV"] = round(bip["EV"].mean(), 1) if not bip.empty else np.nan

        rows.append(out)

    return pd.DataFrame(rows)


def make_zone_heatmap(hdf: pd.DataFrame, metric: str, title: str):
    if not {"PlateLocSide", "PlateLocHeight"}.issubset(hdf.columns):
        return None

    df = hdf.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    if df.empty:
        return None

    x_bins = np.linspace(-1.5, 1.5, 4)
    y_bins = np.linspace(1.0, 4.0, 4)

    df["x_bin"] = pd.cut(df["PlateLocSide"], bins=x_bins, labels=[0, 1, 2])
    df["y_bin"] = pd.cut(df["PlateLocHeight"], bins=y_bins, labels=[0, 1, 2])
    df = df.dropna(subset=["x_bin", "y_bin"])

    if metric == "Swing%":
        num = df.groupby(["y_bin", "x_bin"])["is_swing"].sum()
        den = df.groupby(["y_bin", "x_bin"])["is_swing"].count()
        grid = (num / den * 100).unstack().values

    elif metric == "Whiff%":
        swings = df.groupby(["y_bin", "x_bin"])["is_swing"].sum()
        whiffs = df.groupby(["y_bin", "x_bin"])["is_whiff"].sum()
        grid = np.where(swings.values > 0, whiffs.values / swings.values * 100, 0).reshape(3, 3)

    elif metric == "HardHit%":
        bip = df.dropna(subset=["EV"])
        if bip.empty:
            return None
        num = bip.groupby(["y_bin", "x_bin"])["hard_hit"].mean()
        grid = (num * 100).unstack().values

    elif metric == "wOBA":
        pa_df = df[df["woba_value"] > 0] if "woba_value" in df.columns else pd.DataFrame()
        if pa_df.empty:
            return None
        grid = pa_df.groupby(["y_bin", "x_bin"])["woba_value"].mean().unstack().values

    else:
        return None

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(grid, origin="lower", cmap="viridis")

    for i in range(3):
        for j in range(3):
            val = grid[i, j]
            txt = "" if np.isnan(val) else (f"{val:.1f}" if metric != "wOBA" else f"{val:.3f}")
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=10)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["In", "Mid", "Away"])
    ax.set_yticklabels(["Low", "Mid", "High"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig


def hitter_sequencing(hdf: pd.DataFrame) -> pd.DataFrame:
    df = hdf.copy()

    sort_cols = []
    for c in ["Date", "Inning", "PAofInning", "PitchofPA", "PitchNo"]:
        if c in df.columns:
            sort_cols.append(c)

    if sort_cols:
        df = df.sort_values(sort_cols)

    df["prev_pitch"] = df["pitch_abbr"].shift(1)
    df["same_pa"] = df["Batter"].eq(df["Batter"].shift(1))
    df = df[df["same_pa"]]

    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["prev_pitch", "pitch_abbr"]).agg(
        N=("pitch_abbr", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Chases=("is_chase", "sum"),
        wOBA=("woba_value", "mean"),
        HardHit=("hard_hit", "mean")
    ).reset_index()

    agg["Swing%"] = (agg["Swings"] / agg["N"] * 100).round(1)
    agg["Whiff%"] = np.where(agg["Swings"] > 0, agg["Whiffs"] / agg["Swings"] * 100, 0).round(1)
    agg["Chase%"] = np.where(agg["Swings"] > 0, agg["Chases"] / agg["Swings"] * 100, 0).round(1)
    agg["HardHit%"] = (agg["HardHit"] * 100).round(1)
    agg["wOBA"] = agg["wOBA"].round(3)

    return agg.sort_values(["prev_pitch", "pitch_abbr"])


# ============================================================
# PITCHER DEVELOPMENT & SEQUENCING TAB
# ============================================================

def sequencing_page(all_pitches_df: pd.DataFrame):
    st.markdown("## 🔧 Pitcher Development & Sequencing")

    df = all_pitches_df.copy()
    df = filter_fordham_only(df)  # assumes you already have this helper

    if df.empty:
        st.warning("No FOR_RAM pitcher data available.")
        return

    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]

    needed = [
        "Pitcher", "pitch_abbr", "Count", "Balls", "Strikes",
        "is_swing", "is_whiff", "in_zone", "EV", "LA",
        "PlayResult", "KorBB", "RelH", "RelS", "HB", "IVB",
        "BatterSide", "Date", "Inning", "PitchNumber"
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df["EV"] = pd.to_numeric(df["EV"], errors="coerce")

    if df["Balls"].notna().any() and df["Strikes"].notna().any():
        df["Count"] = df["Balls"].astype(str) + "-" + df["Strikes"].astype(str)
    else:
        df["Count"] = "Unknown"

    df["Count"] = df["Count"].replace({"nan-nan": "Unknown", "None-None": "Unknown"})

    if "is_chase" not in df.columns:
        in_zone_bool = df["in_zone"].fillna(0).astype(bool)
        df["is_swing"] = df["is_swing"].fillna(0).astype(int)
        df["is_whiff"] = df["is_whiff"].fillna(0).astype(int)
        df["is_chase"] = (
            (df["is_swing"] == 1) &
            (df["is_whiff"] == 1) &
            (~in_zone_bool)
        ).astype(int)

    pitchers = get_pitcher_list(df)  # assumes you already have this helper
    if not pitchers:
        st.warning("No FOR_RAM pitcher data available.")
        return

    pitcher = st.selectbox("Select Pitcher", pitchers, key="seq_pitcher_select")

    pdf = df[df["Pitcher"] == pitcher].copy()
    if pdf.empty:
        st.warning("No data for this pitcher.")
        return

    foul_labels = [
        "FoulBallNotFieldable",
        "FoulBallFieldable",
        "FoulBall",
        "Foul"
    ]
    bip_mask = pdf["EV"].notna() & ~pdf["PlayResult"].isin(foul_labels)
    bip = pdf[bip_mask].copy()

    # SECTION 1 — ARSENAL OVERVIEW
    st.markdown("### 🎯 Arsenal Overview")

    arsenal = pdf.groupby("pitch_abbr").agg(
        Usage=("pitch_abbr", "count"),
        Whiff=("is_whiff", "mean"),
        Chase=("is_chase", "mean"),
        InZone=("in_zone", "mean")
    )

    if not bip.empty:
        bb_agg = bip.groupby("pitch_abbr").agg(
            AvgEV=("EV", "mean"),
            HardHit=("EV", lambda x: (x >= 90).mean())
        )
        arsenal = arsenal.join(bb_agg, how="left")
    else:
        arsenal["AvgEV"] = 0.0
        arsenal["HardHit"] = 0.0

    arsenal["Usage%"] = (arsenal["Usage"] / arsenal["Usage"].sum() * 100).round(1)
    arsenal["Whiff%"] = (arsenal["Whiff"] * 100).round(1)
    arsenal["Chase%"] = (arsenal["Chase"] * 100).round(1)
    arsenal["InZone%"] = (arsenal["InZone"] * 100).round(1)
    arsenal["HardHit%"] = (arsenal["HardHit"].fillna(0) * 100).round(1)
    arsenal["AvgEV"] = arsenal["AvgEV"].fillna(0).round(1)

    st.dataframe(
        arsenal[["Usage%", "Whiff%", "Chase%", "InZone%", "HardHit%", "AvgEV"]],
        use_container_width=True
    )

    # SECTION 2 — COUNT-BASED EFFECTIVENESS
    st.markdown("### 📊 Count-Based Effectiveness")

    count_grid = pdf.groupby(["Count", "pitch_abbr"]).agg(
        N=("pitch_abbr", "count"),
        Whiff=("is_whiff", "mean"),
        Chase=("is_chase", "mean"),
        Zone=("in_zone", "mean"),
        CSW=("is_csw", "mean") if "is_csw" in pdf.columns else ("is_whiff", "mean"),
        K=("KorBB", lambda x: (x == "Strikeout").sum())
    ).reset_index()

    if not bip.empty:
        bb_count = bip.groupby(["Count", "pitch_abbr"]).agg(
            AvgEV=("EV", "mean"),
            HardHit=("EV", lambda x: (x >= 90).mean())
        ).reset_index()

        count_grid = count_grid.merge(
            bb_count,
            on=["Count", "pitch_abbr"],
            how="left"
        )
    else:
        count_grid["AvgEV"] = 0.0
        count_grid["HardHit"] = 0.0

    count_grid["Whiff%"] = (count_grid["Whiff"] * 100).round(1)
    count_grid["Chase%"] = (count_grid["Chase"] * 100).round(1)
    count_grid["Zone%"] = (count_grid["Zone"] * 100).round(1)
    count_grid["CSW%"] = (count_grid["CSW"] * 100).round(1)
    count_grid["HardHit%"] = (count_grid["HardHit"].fillna(0) * 100).round(1)
    count_grid["AvgEV"] = count_grid["AvgEV"].fillna(0).round(1)
    count_grid["K%"] = (count_grid["K"] / count_grid["N"] * 100).round(1)

    st.dataframe(
        count_grid[
            [
                "Count", "pitch_abbr", "N",
                "Whiff%", "Chase%", "Zone%", "CSW%", "K%",
                "HardHit%", "AvgEV"
            ]
        ],
        use_container_width=True
    )

    # SECTION 3 — RELEASE CONSISTENCY
    st.markdown("### 🎯 Release Consistency")

    rel = pdf.groupby("pitch_abbr").agg(
        RelH_std=("RelH", "std"),
        RelS_std=("RelS", "std")
    ).round(3)

    st.dataframe(rel, use_container_width=True)

    # SECTION 4 — PITCH-TO-PITCH SEQUENCING
    st.markdown("### 🔁 Pitch-to-Pitch Sequencing")

    sort_cols = [c for c in ["Date", "Inning", "PitchNumber"] if c in pdf.columns]
    if sort_cols:
        pdf = pdf.sort_values(sort_cols)

    pdf["PrevPitch"] = pdf["pitch_abbr"].shift(1)
    pdf["PrevPitcher"] = pdf["Pitcher"].shift(1)

    seq = pdf[pdf["PrevPitcher"] == pitcher].copy()

    seq_stats = seq.groupby(["PrevPitch", "pitch_abbr"]).agg(
        N=("pitch_abbr", "count"),
        Whiff=("is_whiff", "mean")
    ).reset_index()

    seq_bip = seq[seq["EV"].notna() & ~seq["PlayResult"].isin(foul_labels)].copy()
    if not seq_bip.empty:
        seq_bb = seq_bip.groupby(["PrevPitch", "pitch_abbr"]).agg(
            HardHit=("EV", lambda x: (x >= 90).mean())
        ).reset_index()
        seq_stats = seq_stats.merge(seq_bb, on=["PrevPitch", "pitch_abbr"], how="left")
    else:
        seq_stats["HardHit"] = 0.0

    seq_stats["Whiff%"] = (seq_stats["Whiff"] * 100).round(1)
    seq_stats["HardHit%"] = (seq_stats["HardHit"].fillna(0) * 100).round(1)

    st.dataframe(
        seq_stats[["PrevPitch", "pitch_abbr", "N", "Whiff%", "HardHit%"]],
        use_container_width=True
    )

    # SECTION 5 — LHH vs RHH SPLITS
    st.markdown("### ⚖️ LHH vs RHH Splits")

    splits = pdf.groupby(["BatterSide", "pitch_abbr"]).agg(
        Whiff=("is_whiff", "mean")
    ).reset_index()

    if not bip.empty:
        bb_splits = bip.groupby(["BatterSide", "pitch_abbr"]).agg(
            AvgEV=("EV", "mean"),
            HardHit=("EV", lambda x: (x >= 90).mean())
        ).reset_index()
        splits = splits.merge(bb_splits, on=["BatterSide", "pitch_abbr"], how="left")
    else:
        splits["AvgEV"] = 0.0
        splits["HardHit"] = 0.0

    splits["Whiff%"] = (splits["Whiff"] * 100).round(1)
    splits["HardHit%"] = (splits["HardHit"].fillna(0) * 100).round(1)
    splits["AvgEV"] = splits["AvgEV"].fillna(0).round(1)

    st.dataframe(
        splits[["BatterSide", "pitch_abbr", "Whiff%", "HardHit%", "AvgEV"]],
        use_container_width=True
    )

    # SECTION 6 — SMART DEVELOPMENT RECOMMENDATIONS
    st.markdown("### 🧠 Development Recommendations")

    recs = []

    for pitch in arsenal.index:
        usage = arsenal.loc[pitch, "Usage%"]
        whiff = arsenal.loc[pitch, "Whiff%"]
        hardhit = arsenal.loc[pitch, "HardHit%"]

        if usage < 5:
            continue

        if whiff >= 35 and hardhit <= 20:
            recs.append(
                f"Increase **{pitch}** usage — elite Whiff% ({whiff}) with low damage ({hardhit} HardHit%)."
            )

        if hardhit >= 40 and whiff <= 20:
            recs.append(
                f"Reduce **{pitch}** usage — high HardHit% ({hardhit}) with limited swing/miss ({whiff} Whiff%)."
            )

    seq_good = seq_stats[seq_stats["N"] >= 10].sort_values("Whiff%", ascending=False)
    if not seq_good.empty:
        best = seq_good.iloc[0]
        recs.append(
            f"Best sequencing pair: **{best['PrevPitch']} → {best['pitch_abbr']}** "
            f"(Whiff% {best['Whiff%']}, N={int(best['N'])})."
        )

    if not rel.empty:
        rel_mean = rel.mean()
        for pitch in rel.index:
            if (
                (not np.isnan(rel_mean["RelH_std"]) and rel.loc[pitch, "RelH_std"] > rel_mean["RelH_std"] * 1.5) or
                (not np.isnan(rel_mean["RelS_std"]) and rel.loc[pitch, "RelS_std"] > rel_mean["RelS_std"] * 1.5)
            ):
                recs.append(
                    f"Improve release consistency on **{pitch}** — large variance in release metrics."
                )

    if not recs:
        st.success("No major issues detected — arsenal is well optimized.")
    else:
        for r in recs:
            st.markdown(f"- {r}")


# ============================================================
# HITTER DEVELOPMENT & APPROACH TAB
# ============================================================

def hitter_development_page(all_pitches_df: pd.DataFrame):

    st.title("🧠 Hitter Development & Approach")

    df = normalize_hitter_columns(all_pitches_df)
    df = add_contact_quality_local(df)

    if "BatterTeam" in df.columns:
        df = df[df["BatterTeam"].astype(str).str.upper() == "FOR_RAM"]

    if df.empty:
        st.error("No FOR_RAM hitters found.")
        return

    hitters = sorted(df["Batter"].dropna().unique())
    hitter = st.selectbox("Select Hitter", hitters)

    hdf = df[df["Batter"] == hitter].copy()
    if hdf.empty:
        st.warning("No data for this hitter.")
        return

    lgwOBA = compute_league_woba(df)

    # HITTER CARD
    st.subheader("📇 Hitter Card")

    card = compute_hitter_card(hdf, lgwOBA)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("PA", card["PA"])
        st.metric("AB", card["AB"])
        st.metric("H", card["H"])
        st.metric("HR", card["HR"])

    with c2:
        st.metric("BB%", f"{card['BB%']}%")
        st.metric("K%", f"{card['K%']}%")
        st.metric("Swing%", f"{card['Swing%']}%")
        st.metric("Chase%", f"{card['Chase%']}%")

    with c3:
        st.metric("wOBA", f"{card['wOBA']:.3f}")
        st.metric("wRC+", f"{card['wRC+']}")
        st.metric("Whiff%", f"{card['Whiff%']}%")

    with c4:
        st.metric("HardHit%", f"{card['HardHit%']}%")
        st.metric("Barrel%", f"{card['Barrel%']}%")
        st.metric("Avg EV", f"{card['AvgEV']}")
        st.metric("Max EV", f"{card['MaxEV']}")

    # COUNT-BASED EFFECTIVENESS
    st.subheader("📊 Count-Based Effectiveness")
    count_df = count_effectiveness(hdf)
    st.dataframe(count_df, use_container_width=True)

    # COUNT × PITCH TYPE EFFECTIVENESS
    st.subheader("🎯 Count × Pitch Type Effectiveness")
    cpt_df = count_pitchtype_effectiveness(hdf)
    st.dataframe(cpt_df, use_container_width=True)

    # SPLITS VS LHP / RHP
    st.subheader("⚖️ Splits vs LHP / RHP")
    splits_df = hitter_splits(hdf)
    if splits_df.empty:
        st.info("No pitcher handedness data available.")
    else:
        st.dataframe(splits_df, use_container_width=True)

    # ZONE HEATMAPS
    st.subheader("🎯 Zone Heatmaps")

    colA, colB = st.columns(2)

    with colA:
        fig1 = make_zone_heatmap(hdf, "Swing%", "Swing% Heatmap")
        if fig1:
            st.pyplot(fig1)

        fig2 = make_zone_heatmap(hdf, "Whiff%", "Whiff% Heatmap")
        if fig2:
            st.pyplot(fig2)

    with colB:
        fig3 = make_zone_heatmap(hdf, "HardHit%", "HardHit% Heatmap")
        if fig3:
            st.pyplot(fig3)

        fig4 = make_zone_heatmap(hdf, "wOBA", "wOBA Heatmap")
        if fig4:
            st.pyplot(fig4)

    # SEQUENCING
    st.subheader("🔁 Pitch-to-Pitch Sequencing (Hitter Reaction)")

    seq_df = hitter_sequencing(hdf)

    if seq_df.empty:
        st.info("Not enough sequencing data for this hitter.")
        return

    st.dataframe(seq_df, use_container_width=True)

    damage = seq_df.sort_values("wOBA", ascending=False).head(1)
    whiff = seq_df.sort_values("Whiff%", ascending=False).head(1)

    best_row = damage.iloc[0]
    worst_row = whiff.iloc[0]

    st.markdown(
        f"**🔥 Best damage sequence:** {best_row['prev_pitch']} → {best_row['pitch_abbr']} "
        f"(wOBA {best_row['wOBA']:.3f}, HardHit% {best_row['HardHit%']}%, N={int(best_row['N'])})"
    )

    st.markdown(
        f"**❄️ Toughest sequence:** {worst_row['prev_pitch']} → {worst_row['pitch_abbr']} "
        f"(Whiff% {worst_row['Whiff%']}%, N={int(worst_row['N'])})"
    )



# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    st.markdown(
        "<h1 style='text-align:center; color:#FFFFFF;'>Fordham Baseball – Advanced Analytics</h1>",
        unsafe_allow_html=True
    )

    # Load all processed pitch-by-pitch data ONCE
    all_pitches_df = prepare_data()

    # ------------------------------------------------------------
    # TOP-LEVEL TABS
    # ------------------------------------------------------------
    tab_summaries, tab_leaders, tab_tools = st.tabs([
        "Summaries",
        "Leaderboards",
        "Tools"
    ])

    # ------------------------------------------------------------
    # SUMMARIES TAB
    # ------------------------------------------------------------
    with tab_summaries:
        sub1, sub2, sub3 = st.tabs([
            "Postgame Summary",
            "Season Summary",
            "Pitcher Profile"
        ])

        with sub1:
            postgame_page()

        with sub2:
            season_page()

        with sub3:
            pitcher_profile_page()

    # ------------------------------------------------------------
    # LEADERBOARDS TAB
    # ------------------------------------------------------------
    with tab_leaders:
        sub4, sub5, sub6, sub7 = st.tabs([
            "Stuff+",
            "Location+",
            "Pitch-Type Grids",
            "Contact Quality"
        ])

        with sub4:
            stuff_leaderboard_page()

        with sub5:
            location_leaderboard_page()

        with sub6:
            pitchtype_grids_page()

        with sub7:
            contact_quality_leaderboard_page(all_pitches_df)

    # ------------------------------------------------------------
    # TOOLS TAB
    # ------------------------------------------------------------
    with tab_tools:
        sub8, sub9, sub10 = st.tabs([
            "Pitcher Development & Sequencing",
            "Umpire Scorecard",
            "Hitter Development & Approach"   # ⭐ NEW TAB 10
        ])

        with sub8:
            sequencing_page(all_pitches_df)

        with sub9:
            umpire_scorecard_page()

        with sub10:
            hitter_development_page(all_pitches_df)   # ⭐ NEW FUNCTION


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if check_password():
    main()
else:
    st.stop()
