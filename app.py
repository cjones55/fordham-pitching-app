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
    # FIGURE (NO constrained_layout)
    # -----------------------------
    fig = plt.figure(figsize=(24, 11))
    fig.patch.set_facecolor(BACKGROUND)

    # Reserve top 20% for title + summary only
    fig.subplots_adjust(left=0.05, right=0.98, top=0.80, bottom=0.06, wspace=0.25, hspace=0.35)

    # GridSpec lives strictly below 0.80
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        height_ratios=[1.0, 1.1, 1.1],
        width_ratios=[2.0, 1.0, 1.0, 0.8]
    )

    # -----------------------------
    # LOGO
    # -----------------------------
    logo_path = ROOT / "assets" / "rams.png"
    if logo_path.exists():
        logo_img = mpimg.imread(logo_path)
        # place logo inside the reserved top band
        fig.figimage(logo_img, xo=40, yo=int(fig.bbox.ymax * 1.5), zorder=50, alpha=1.0)

    # -----------------------------
    # TITLE + SUMMARY (SAFE BAND)
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
    # MOVEMENT (SQUARE + SHADING + ZERO LINES)
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
        ax_move.scatter(row["HB"], row["IVB"], s=40, color=c, edgecolor="white", linewidth=0.5)

    centroids = pdf.groupby("pitch_abbr")[["HB", "IVB"]].mean().reset_index()
    for _, row in centroids.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_move.scatter(row["HB"], row["IVB"], s=250, color=c, edgecolor="white", linewidth=1.5)
        ax_move.text(row["HB"], row["IVB"], row["pitch_abbr"],
                     color="white", fontsize=15, weight="bold", ha="center")

    ax_move.set_title("Movement", color="white")
    ax_move.tick_params(colors="white")
    for spine in ax_move.spines.values():
        spine.set_color("white")

      # -----------------------------
    # LHH ZONE (proper strike-zone scaling)
    # -----------------------------
    ax_lhh = fig.add_subplot(gs[0, 1])
    ax_lhh.set_facecolor(BACKGROUND)
    ax_lhh.set_title("LHH", color="white", fontsize=16, weight="bold")

    # Real strike zone aspect ratio (taller than wide)
    ax_lhh.set_aspect(1.6)

    ax_lhh.set_xlim(-2.5, 2.5)
    ax_lhh.set_ylim(0, 5)

    zone_x = [-0.83, 0.83, 0.83, -0.83, -0.83]
    zone_y = [1.5, 1.5, 3.5, 3.5, 1.5]
    ax_lhh.plot(zone_x, zone_y, color="white", linewidth=2.5)

    LHH = pdf[pdf["BatterSide"] == "Left"]
    for _, row in LHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_lhh.scatter(
            row["PlateLocSide"],
            row["PlateLocHeight"],
            s=110,
            color=c,
            edgecolor="white",
            linewidth=0.6
        )

    ax_lhh.tick_params(colors="white", labelsize=12)
    for spine in ax_lhh.spines.values():
        spine.set_color("white")


    # -----------------------------
    # RHH ZONE (proper strike-zone scaling)
    # -----------------------------
    ax_rhh = fig.add_subplot(gs[0, 2])
    ax_rhh.set_facecolor(BACKGROUND)
    ax_rhh.set_title("RHH", color="white", fontsize=16, weight="bold")

    # Real strike zone aspect ratio
    ax_rhh.set_aspect(1.6)

    ax_rhh.set_xlim(-2.5, 2.5)
    ax_rhh.set_ylim(0, 5)
    ax_rhh.plot(zone_x, zone_y, color="white", linewidth=2.5)

    RHH = pdf[pdf["BatterSide"] == "Right"]
    for _, row in RHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_rhh.scatter(
            row["PlateLocSide"],
            row["PlateLocHeight"],
            s=110,
            color=c,
            edgecolor="white",
            linewidth=0.6
        )

    ax_rhh.tick_params(colors="white", labelsize=12)
    for spine in ax_rhh.spines.values():
        spine.set_color("white")


    # -----------------------------
    # RELEASE
    # -----------------------------
    ax_rel = fig.add_subplot(gs[0, 3])
    ax_rel.set_facecolor(BACKGROUND)
    ax_rel.set_title("Release", color="white")
    ax_rel.set_xlim(-4, 4)
    ax_rel.set_ylim(3, 7)

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_rel.scatter(row["RelS"], row["RelH"], s=25, color=c, edgecolor="white")

    # -----------------------------
    # TABLE (COMPACT, LOWER)
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
        bbox=[0, 0.08, 1, 0.92]   # shorter + lower
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)

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

    # ============================================================
    # 2️⃣ LOC+ GRID (2×3)
    # ============================================================
    st.subheader("Location+ Leaderboards")

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 22))   # ⭐ TALLER FIGURE
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
                color="#A00000", fontsize=18, fontweight="bold", va="top")

        # ⭐ SAME SPACING FIX
        y_start = 0.87
        y_step = 0.095

        for i, row in enumerate(sub.itertuples()):
            y = y_start - i * y_step

            ax.text(0.02, y, row.Pitcher, color="white", fontsize=14, weight="bold")

            ax.text(0.60, y, f"Loc+: {round(row.Loc_plus,1)}", color="white", fontsize=14)
            ax.text(0.60, y - 0.03, f"LHH: {round(row.Loc_plus_LHH or 0,1)}",
                    color="white", fontsize=12)
            ax.text(0.60, y - 0.06, f"RHH: {round(row.Loc_plus_RHH or 0,1)}",
                    color="white", fontsize=12)

    st.pyplot(fig2)



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
