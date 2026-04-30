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
import io
import base64
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
        "teamstat/pitching_stats.csv",
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
    page_icon="F",
    layout="wide"
)

PASSWORD = "Baseball_1"
FORDHAM_MAROON = "#8C1515"
FORDHAM_MAROON_DARK = "#5E0F0F"
FORDHAM_GOLD = "#C7A45D"
FORDHAM_CHARCOAL = "#202124"
FORDHAM_PANEL = "#F7F4EF"
FORDHAM_BORDER = "#E3D8C7"

# ------------------------------------------------------------
# VISUAL THEME
# ------------------------------------------------------------
def get_logo_b64():
    for logo_path in [ROOT / "static" / "rams.png", ROOT / "assets" / "rams.png"]:
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            continue
    return ""


def inject_fordham_theme(show_logo=True):
    logo_b64 = get_logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" class="top-left-logo">'
        if logo_b64 and show_logo else ""
    )
    st.markdown(
        f"""
        <style>
            :root {{
                --fordham-maroon: {FORDHAM_MAROON};
                --fordham-maroon-dark: {FORDHAM_MAROON_DARK};
                --fordham-gold: {FORDHAM_GOLD};
                --fordham-panel: #171717;
                --fordham-panel-soft: #211C1A;
                --fordham-border: rgba(199, 164, 93, 0.28);
                --fordham-charcoal: {FORDHAM_CHARCOAL};
                --fordham-text: #F8EFE2;
                --fordham-muted: #CDBFAF;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(140, 21, 21, 0.34), transparent 34rem),
                    linear-gradient(180deg, #100D0C 0px, #171312 44%, #0F0E0D 100%);
                color: var(--fordham-text);
            }}

            section[data-testid="stSidebar"] {{
                display: none;
            }}

            button[kind="header"] {{
                display: none;
            }}

            .block-container {{
                padding-top: 1.7rem;
                padding-bottom: 4rem;
                max-width: 1480px;
            }}

            h1, h2, h3 {{
                letter-spacing: 0;
                color: var(--fordham-text);
            }}

            h1 {{
                font-weight: 800;
            }}

            h2, h3 {{
                font-weight: 760;
            }}

            div[data-testid="stMarkdownContainer"] h3 {{
                padding-top: 0.4rem;
            }}

            .top-left-logo {{
                position: fixed;
                top: 50px;
                left: 18px;
                width: 92px;
                z-index: 99999;
                filter: drop-shadow(0 8px 18px rgba(0,0,0,0.25));
            }}

            .fordham-hero {{
                margin: 0.35rem 0 1.35rem 0;
                padding: 1.15rem 1.35rem;
                border-radius: 10px;
                background:
                    linear-gradient(135deg, rgba(94,15,15,0.98), rgba(36,20,18,0.96)),
                    var(--fordham-maroon);
                border: 1px solid rgba(199,164,93,0.56);
                box-shadow: 0 16px 42px rgba(0, 0, 0, 0.34);
            }}

            .fordham-hero h1 {{
                margin: 0;
                color: #fff9ee;
                font-size: 2.05rem;
                line-height: 1.12;
            }}

            .fordham-hero p {{
                margin: 0.35rem 0 0 0;
                color: #f0dcc0;
                font-size: 0.98rem;
            }}

            .login-shell {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding-top: 8vh;
                margin-bottom: 1rem;
            }}

            .login-card {{
                width: min(520px, 92vw);
                padding: 2.1rem 2.2rem 1.8rem 2.2rem;
                border-radius: 14px;
                background: rgba(23, 19, 18, 0.98);
                border: 1px solid rgba(199,164,93,0.62);
                box-shadow: 0 22px 70px rgba(0, 0, 0, 0.42);
                text-align: center;
            }}

            .login-card img {{
                width: 94px;
                margin-bottom: 0.75rem;
                filter: drop-shadow(0 8px 16px rgba(0,0,0,0.16));
            }}

            .login-card h1 {{
                color: #FFF7E8;
                margin: 0;
                font-size: 1.75rem;
            }}

            .login-card p {{
                color: var(--fordham-muted);
                margin: 0.5rem 0 0 0;
            }}

            div[data-testid="stMetric"] {{
                background: linear-gradient(180deg, #24201E, #171514);
                border: 1px solid var(--fordham-border);
                border-left: 5px solid var(--fordham-gold);
                border-radius: 10px;
                padding: 0.8rem 0.9rem;
                box-shadow: 0 10px 26px rgba(0, 0, 0, 0.24);
            }}

            div[data-testid="stMetricLabel"] p {{
                color: var(--fordham-muted);
                font-weight: 720;
            }}

            div[data-testid="stMetricValue"] {{
                color: #FFF8E9;
                font-weight: 820;
            }}

            div[data-testid="stTabs"] button {{
                border-radius: 999px;
                padding: 0.5rem 0.9rem;
                color: #F3E4D0;
            }}

            div[data-testid="stTabs"] button[aria-selected="true"] {{
                background: var(--fordham-maroon);
                color: white;
                border: 1px solid rgba(199,164,93,0.62);
            }}

            div[data-testid="stDataFrame"] {{
                border: 1px solid var(--fordham-border);
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
            }}

            div[data-testid="stSelectbox"] label,
            div[data-testid="stRadio"] label,
            div[data-testid="stTextInput"] label {{
                color: #F7E9D0;
                font-weight: 750;
            }}

            div[data-baseweb="select"] > div,
            div[data-testid="stTextInput"] input {{
                background-color: #211C1A;
                color: #FFF8E9;
                border-color: rgba(199,164,93,0.38);
            }}

            div[role="radiogroup"] label {{
                background: #211C1A;
                border: 1px solid rgba(199,164,93,0.22);
                border-radius: 999px;
                padding: 0.35rem 0.65rem;
                margin-right: 0.35rem;
            }}

            div[role="radiogroup"] label:has(input:checked) {{
                background: var(--fordham-maroon);
                border-color: var(--fordham-gold);
            }}

            .stButton > button,
            .stDownloadButton > button {{
                background: var(--fordham-maroon);
                color: #fff8e9;
                border: 1px solid rgba(199,164,93,0.7);
                border-radius: 8px;
                font-weight: 760;
                box-shadow: 0 8px 18px rgba(94,15,15,0.16);
            }}

            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                background: var(--fordham-maroon-dark);
                color: #ffffff;
                border-color: var(--fordham-gold);
            }}

            hr {{
                border-color: rgba(199, 164, 93, 0.20);
                margin: 1.25rem 0;
            }}

            .stAlert {{
                border-radius: 10px;
            }}

            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stMarkdownContainer"] li,
            div[data-testid="stMarkdownContainer"] span {{
                color: var(--fordham-text);
            }}

            .app-section-label {{
                color: var(--fordham-gold);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 800;
                margin-bottom: 0.45rem;
            }}

            .nav-panel {{
                background: linear-gradient(180deg, rgba(33,28,26,0.92), rgba(20,17,16,0.92));
                border: 1px solid rgba(199,164,93,0.24);
                border-radius: 12px;
                padding: 0.85rem 1rem 0.65rem 1rem;
                margin-bottom: 1.15rem;
                box-shadow: 0 12px 28px rgba(0,0,0,0.22);
            }}
        </style>

        {logo_html}
        """,
        unsafe_allow_html=True
    )


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "axes.edgecolor": "#3a3028",
    "axes.linewidth": 1.0,
    "figure.dpi": 125,
    "savefig.dpi": 160,
    "grid.color": "#D8CCB8",
    "grid.alpha": 0.28,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#E3D8C7",
})


inject_fordham_theme(show_logo=False)


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

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
    if st.session_state.get("authenticated"):
        return True

    inject_fordham_theme(show_logo=False)
    logo_b64 = get_logo_b64()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Fordham Rams">' if logo_b64 else ""

    st.markdown(
        f"""
        <div class="login-shell">
            <div class="login-card">
                {logo_html}
                <h1>Fordham Baseball Analytics</h1>
                <p>Enter the team password to open the pitching and hitter development dashboard.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    _, center, _ = st.columns([1.15, 1, 1.15])
    with center:
        pw = st.text_input("Team password", type="password", placeholder="Enter password")
        if st.button("Open Dashboard", use_container_width=True):
            if pw == PASSWORD:
                st.session_state["authenticated"] = True
                rerun_app()
            else:
                st.error("Incorrect password. Check capitalization and try again.")

    if pw == PASSWORD:
        st.session_state["authenticated"] = True
        rerun_app()

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


def style_fordham_axes(ax, title=None, dark=False):
    if dark:
        ax.set_facecolor("#25201D")
        ax.tick_params(colors="#F7E9D0", labelsize=9)
        ax.xaxis.label.set_color("#F7E9D0")
        ax.yaxis.label.set_color("#F7E9D0")
        for spine in ax.spines.values():
            spine.set_color("#C7A45D")
        ax.grid(True, color="#C7A45D", alpha=0.18, linewidth=0.8)
        if title:
            ax.set_title(title, color="#FFF7E8", fontsize=14, fontweight="bold", pad=12)
    else:
        ax.set_facecolor("#FFFDF8")
        ax.tick_params(colors="#302820", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#D6C7AE")
        ax.grid(True, color="#D9CCB8", alpha=0.38, linewidth=0.8)
        if title:
            ax.set_title(title, color=FORDHAM_MAROON_DARK, fontsize=14, fontweight="bold", pad=12)


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
        Ext=("Ext","mean"),
        RelH=("RelH","mean"),
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
    if not logo_path.exists():
        logo_path = ROOT / "static" / "rams.png"
    if logo_path.exists():
        logo_img = mpimg.imread(logo_path)
        logo_ax = fig.add_axes([0.035, 0.84, 0.105, 0.125], zorder=50)
        logo_ax.set_facecolor("#FFF7E8")
        logo_ax.imshow(logo_img)
        logo_ax.set_xticks([])
        logo_ax.set_yticks([])
        for spine in logo_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#C7A45D")
            spine.set_linewidth(2.0)

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
             fontsize=28, fontweight="bold", color="#FFF7E8")
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

    draw_home_plate(ax_lhh)

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

    draw_home_plate(ax_rhh)

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
        "Spin","Stuff+","Loc+","CSW%","Whiff%","Strike%","Zone%","Ext","RelH"
    ]].round(2).rename(columns={"Ext": "RelExt", "RelH": "RelHt"})

    tbl = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0.08, 1, 0.92]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    cell_width = 1.0 / len(table_df.columns)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(0.042)
        cell.set_width(cell_width)

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
    # STUFF+ GRID (2x3)
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
    # LOC+ GRID (2x3)
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
    fig, ax = plt.subplots(figsize=(7, 6.4))
    fig.patch.set_facecolor("#FFFDF8")

    if df.empty:
        ax.text(0.5, 0.5, "No movement data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x_min, x_max = df["HB"].min() - 2, df["HB"].max() + 2
    y_min, y_max = df["IVB"].min() - 2, df["IVB"].max() + 2

    style_fordham_axes(ax, "Movement Clusters")
    ax.axvspan(0, x_max, color="#E8F1FF", alpha=0.74, zorder=0)
    ax.axvspan(x_min, 0, color="#F9E6E2", alpha=0.74, zorder=0)

    for pitch, sub in df.groupby("pitch_abbr"):
        ax.scatter(sub["HB"], sub["IVB"], label=pitch, s=52, alpha=0.86, edgecolor="white", linewidth=0.7)

    ax.axhline(0, color=FORDHAM_MAROON, linewidth=1.7, alpha=0.65)
    ax.axvline(0, color=FORDHAM_MAROON, linewidth=1.7, alpha=0.65)

    ax.set_xlabel("Horizontal Break (HB)")
    ax.set_ylabel("Induced Vertical Break (IVB)")
    ax.legend(loc="best", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()

    return fig

# -----------------------------
# Release Drift Figure
# -----------------------------
def build_release_figure(pitcher_df):
    df = pitcher_df.dropna(subset=["RelS", "RelH"])
    fig, ax = plt.subplots(figsize=(7, 6.4))
    fig.patch.set_facecolor("#FFFDF8")

    if df.empty:
        ax.text(0.5, 0.5, "No release data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x_min, x_max = df["RelS"].min() - 0.5, df["RelS"].max() + 0.5
    y_min, y_max = df["RelH"].min() - 0.5, df["RelH"].max() + 0.5

    style_fordham_axes(ax, "Release Drift")
    for pitch, sub in df.groupby("pitch_abbr"):
        ax.scatter(sub["RelS"], sub["RelH"], label=pitch, s=52, alpha=0.86, edgecolor="white", linewidth=0.7)

    ax.axhline(df["RelH"].mean(), color=FORDHAM_MAROON, linestyle=":", linewidth=1.7)
    ax.axvline(df["RelS"].mean(), color=FORDHAM_MAROON, linestyle=":", linewidth=1.7)

    ax.set_xlabel("Release Side (RelS)")
    ax.set_ylabel("Release Height (RelH)")
    ax.legend(loc="best", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()

    return fig

# -----------------------------
# Release Extension Figure
# -----------------------------
def build_release_extension_figure(pitcher_df):
    if "Ext" not in pitcher_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No release extension data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    df = pitcher_df.copy()
    df["Ext"] = pd.to_numeric(df["Ext"], errors="coerce")
    df = df.dropna(subset=["Ext"]).copy()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    fig.patch.set_facecolor("#FFFDF8")

    if df.empty:
        ax.text(0.5, 0.5, "No release extension data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    sort_cols = [c for c in ["GameDate", "Date", "Inning", "PAofInning", "PitchofPA", "PitchNo", "PitchNumber"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    df["PitchIndex"] = np.arange(1, len(df) + 1)

    style_fordham_axes(ax, "Release Extension by Pitch")
    for pitch, sub in df.groupby("pitch_abbr"):
        ax.scatter(sub["PitchIndex"], sub["Ext"], label=pitch, s=46, alpha=0.88, edgecolor="white", linewidth=0.7)
        if len(sub) >= 3:
            ax.plot(sub["PitchIndex"], sub["Ext"].rolling(3, min_periods=1).mean(), alpha=0.45)

    mean_ext = df["Ext"].mean()
    ax.axhline(mean_ext, color=FORDHAM_MAROON, linestyle=":", linewidth=1.8, label=f"Avg {mean_ext:.2f} ft")

    ax.set_xlabel("Pitch #")
    ax.set_ylabel("Release Extension (ft)")
    ax.legend(loc="best", fontsize=8)

    y_pad = 0.35
    ax.set_ylim(df["Ext"].min() - y_pad, df["Ext"].max() + y_pad)
    fig.tight_layout()

    return fig

# -----------------------------
# MAIN PAGE 6 FUNCTION
# -----------------------------
def pitcher_profile_page():
    st.header("Pitcher Profile")

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

    # Normalize both sides
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
    st.subheader("Game Log")

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
    st.subheader("Generate Game Report")

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
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"{pitcher}_{g['GameDate'].date()}_{g['Opponent']}.pdf",
            mime="application/pdf"
        )

    st.markdown("---")

    # -----------------------------
    # TRENDS
    # -----------------------------
    st.subheader("Season Trends")

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
    st.subheader("Release Drift")
    st.pyplot(build_release_figure(pitcher_df))

    st.markdown("---")

    # -----------------------------
    # MOVEMENT CLUSTERS
    # -----------------------------
    st.subheader("Movement Clusters")
    st.pyplot(build_movement_figure(pitcher_df))

    st.markdown("---")

    # -----------------------------
    # RELEASE EXTENSION
    # -----------------------------
    st.subheader("Release Extension")
    st.pyplot(build_release_extension_figure(pitcher_df))


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
        fig.figimage(logo_img, xo=40, yo=fig.bbox.ymax + 1200, zorder=50)

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

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PA-LEVEL ENGINE (USED BY ALL TABS)
# ============================================================

def get_pa_endings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the final pitch of each PA.
    Uses Date, Inning, PAofInning, PitchofPA when available.
    Falls back to last pitch per (Inning, Batter) if needed.
    """
    df = df.copy()

    pa_keys = [c for c in ["Date", "Inning", "PAofInning"] if c in df.columns]

    if "PitchofPA" in df.columns and len(pa_keys) >= 2:
        df = df.sort_values(pa_keys + ["PitchofPA"])
        return df.groupby(pa_keys).tail(1)

    fallback_keys = [c for c in ["Date", "Inning", "Batter"] if c in df.columns]
    if "PitchNo" in df.columns and len(fallback_keys) >= 2:
        df = df.sort_values(fallback_keys + ["PitchNo"])
        return df.groupby(fallback_keys).tail(1)

    return df


# ============================================================
# UNIVERSAL wOBA / wRC+ ENGINE
# ============================================================

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
    Fixed league wOBA so all tabs scale identically.
    """
    return 0.315


def compute_wrc_plus(player_woba: float, league_woba: float = 0.315) -> int:
    """
    Simple ratio-based wRC+ so spread is meaningful.
    ~100 = league average, >120 = clearly above average.
    """
    if league_woba <= 0:
        return 100
    return int(round((player_woba / league_woba) * 100))


def combine_slider_sweeper(series: pd.Series) -> pd.Series:
    return series.replace({
        "SL": "SL/SW",
        "SW": "SL/SW",
        "Slider": "SL/SW",
        "Sweeper": "SL/SW"
    })


def add_ba_slg_by_group(base_df: pd.DataFrame, group_cols) -> pd.DataFrame:
    if base_df.empty or not set(group_cols).issubset(base_df.columns):
        return pd.DataFrame(columns=list(group_cols) + ["AB", "H", "BB", "HBP", "SF", "BA", "OBP", "SLG", "OPS"])

    pa = get_pa_endings(base_df).copy()
    if pa.empty or not set(group_cols).issubset(pa.columns):
        return pd.DataFrame(columns=list(group_cols) + ["AB", "H", "BB", "HBP", "SF", "BA", "OBP", "SLG", "OPS"])

    if "KorBB" not in pa.columns:
        pa["KorBB"] = ""
    if "PitchCall" not in pa.columns:
        pa["PitchCall"] = ""
    if "PlayResult" not in pa.columns:
        pa["PlayResult"] = ""

    pa["is_ab"] = ~(
        pa["KorBB"].eq("Walk") |
        pa["PitchCall"].eq("HitByPitch") |
        pa["PlayResult"].eq("Sacrifice")
    )
    pa["is_bb"] = pa["KorBB"].eq("Walk").astype(int)
    pa["is_hbp"] = pa["PitchCall"].eq("HitByPitch").astype(int)
    pa["is_sf"] = pa["PlayResult"].eq("Sacrifice").astype(int)
    pa["hit_value"] = pa["PlayResult"].map({
        "Single": 1,
        "Double": 2,
        "Triple": 3,
        "HomeRun": 4
    }).fillna(0)
    pa["is_hit"] = (pa["hit_value"] > 0).astype(int)

    grouped = pa.groupby(group_cols, observed=False).agg(
        AB=("is_ab", "sum"),
        H=("is_hit", "sum"),
        BB=("is_bb", "sum"),
        HBP=("is_hbp", "sum"),
        SF=("is_sf", "sum"),
        TB=("hit_value", "sum")
    ).reset_index()

    grouped["BA"] = np.where(grouped["AB"] > 0, grouped["H"] / grouped["AB"], np.nan)
    obp_den = grouped["AB"] + grouped["BB"] + grouped["HBP"] + grouped["SF"]
    grouped["OBP"] = np.where(obp_den > 0, (grouped["H"] + grouped["BB"] + grouped["HBP"]) / obp_den, np.nan)
    grouped["SLG"] = np.where(grouped["AB"] > 0, grouped["TB"] / grouped["AB"], np.nan)
    grouped["BA"] = grouped["BA"].round(3)
    grouped["OBP"] = grouped["OBP"].round(3)
    grouped["SLG"] = grouped["SLG"].round(3)
    grouped["OPS"] = (grouped["OBP"] + grouped["SLG"]).round(3)
    return grouped.drop(columns=["TB"])


# ============================================================
# SHARED NORMALIZATION + CONTACT QUALITY (HITTER TAB)
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
    if "EV" not in df.columns:
        df["EV"] = np.nan
    if "LA" not in df.columns:
        df["LA"] = np.nan

    for col in ["EV", "LA", "PlateLocSide", "PlateLocHeight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Easier barrel definition
    df["hard_hit"] = np.where(df["EV"].fillna(0) >= 95, 1, 0)
    df["barrel"] = np.where(
        (df["EV"].fillna(0) >= 95) &
        (df["LA"].fillna(0).between(20, 35)),
        1, 0
    )
    df["sweet_spot"] = np.where(
        df["LA"].fillna(0).between(8, 32),
        1, 0
    )

    pitch_call = df.get("PitchCall", pd.Series("", index=df.index)).astype(str)
    swing_calls = [
        "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
        "FoulBallFieldable", "FoulTip", "InPlay", "InPlayNoOut",
        "InPlayOut"
    ]
    df["is_swing"] = pitch_call.isin(swing_calls).astype(int)
    df["is_whiff"] = pitch_call.eq("StrikeSwinging").astype(int)

    if {"PlateLocSide", "PlateLocHeight"}.issubset(df.columns):
        in_zone = (
            df["PlateLocSide"].between(-0.83, 0.83) &
            df["PlateLocHeight"].between(1.5, 3.5)
        )
        df["in_zone"] = in_zone.astype(int)
        df["is_chase"] = ((df["is_swing"] == 1) & ~in_zone).astype(int)
    else:
        df["is_chase"] = 0

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
# CONTACT QUALITY (LEADERBOARD TAB) – EV/LA FLAGS
# ============================================================

def add_contact_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]
    if "LA" not in df.columns and "Angle" in df.columns:
        df["LA"] = df["Angle"]

    # Clean EV outliers and remove fouls / bunts from EV/LA
    df.loc[df["EV"] > 118, "EV"] = np.nan

    foul = ["Foul", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"]
    if "PlayResult" in df.columns:
        df.loc[df["PlayResult"].isin(foul), ["EV", "LA"]] = np.nan

    bunts = [
        "Bunt", "BuntGroundout", "BuntPopOut", "BuntLineOut",
        "SacrificeBunt", "BuntFoul", "BuntFoulTip"
    ]
    if "TaggedHitType" in df.columns:
        df.loc[df["TaggedHitType"].isin(bunts), ["EV", "LA"]] = np.nan

    # Easier barrel definition here too
    df["hard_hit"] = (df["EV"] >= 95).astype(int)
    df["barrel"] = ((df["EV"] >= 95) & df["LA"].between(20, 35)).astype(int)
    df["sweet_spot"] = df["LA"].between(7, 32).astype(int)

    if "is_swing" not in df.columns:
        df["is_swing"] = 0
    if "is_whiff" not in df.columns:
        df["is_whiff"] = 0
    if "in_zone" not in df.columns:
        df["in_zone"] = 0

    df["is_swing"] = df["is_swing"].fillna(0).astype(int)
    df["is_whiff"] = df["is_whiff"].fillna(0).astype(int)
    df["is_chase"] = (
        (df["is_swing"] == 1) &
        (df["is_whiff"] == 1) &
        (~df["in_zone"].fillna(0).astype(bool))
    ).astype(int)

    return df


def summarize_contact_quality(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty or group_col not in df.columns:
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
        BB = (pa_end["KorBB"] == "Walk").sum() if "KorBB" in pa_end.columns else 0
        K  = (pa_end["KorBB"] == "Strikeout").sum() if "KorBB" in pa_end.columns else 0
        HBP = (pa_end["PitchCall"] == "HitByPitch").sum() if "PitchCall" in pa_end.columns else 0
        SF = (pa_end["PlayResult"] == "Sacrifice").sum() if "PlayResult" in pa_end.columns else 0
        singles = (pa_end["PlayResult"] == "Single").sum() if "PlayResult" in pa_end.columns else 0
        doubles = (pa_end["PlayResult"] == "Double").sum() if "PlayResult" in pa_end.columns else 0
        triples = (pa_end["PlayResult"] == "Triple").sum() if "PlayResult" in pa_end.columns else 0
        homers = (pa_end["PlayResult"] == "HomeRun").sum() if "PlayResult" in pa_end.columns else 0
        H = singles + doubles + triples + homers
        TB = singles + 2 * doubles + 3 * triples + 4 * homers
        AB = PA - BB - HBP - SF
        obp_den = AB + BB + HBP + SF

        player_woba = compute_woba(g)
        player_wrc_plus = compute_wrc_plus(player_woba, lgwOBA)

        swings = g["is_swing"].sum() if "is_swing" in g.columns else 0
        whiffs = g["is_whiff"].sum() if "is_whiff" in g.columns else 0
        chases = g["is_chase"].sum() if "is_chase" in g.columns else 0

        bip = get_true_bip_with_ev(g) if {"EV", "PitchCall"}.issubset(g.columns) else pd.DataFrame()

        hard = bip["hard_hit"].mean() if not bip.empty else np.nan
        barrel = bip["barrel"].mean() if not bip.empty else np.nan
        sweet = bip["sweet_spot"].mean() if not bip.empty else np.nan
        avg_ev = bip["EV"].mean() if not bip.empty else np.nan
        max_ev = bip["EV"].max() if not bip.empty else np.nan
        avg_la = bip["LA"].mean() if not bip.empty else np.nan

        rows.append({
            group_col: name,
            "PA": PA,
            "AB": AB,
            "H": H,
            "BB": BB,
            "K": K,
            "BA": round(H / AB, 3) if AB > 0 else np.nan,
            "OBP": round((H + BB + HBP) / obp_den, 3) if obp_den > 0 else np.nan,
            "SLG": round(TB / AB, 3) if AB > 0 else np.nan,
            "OPS": round((H + BB + HBP) / obp_den + TB / AB, 3) if obp_den > 0 and AB > 0 else np.nan,
            "wOBA": round(player_woba, 3),
            "wRC+": player_wrc_plus,
            "Swings": swings,
            "Whiffs": whiffs,
            "Chases": chases,
            "HardHit%": round(hard * 100, 1) if hard == hard else np.nan,
            "Barrel%": round(barrel * 100, 1) if barrel == barrel else np.nan,
            "SweetSpot%": round(sweet * 100, 1) if sweet == sweet else np.nan,
            "AvgEV": round(avg_ev, 1) if avg_ev == avg_ev else np.nan,
            "MaxEV": round(max_ev, 1) if max_ev == max_ev else np.nan,
            "AvgLA": round(avg_la, 1) if avg_la == avg_la else np.nan,
            "BB%": round(BB / PA * 100, 1) if PA > 0 else 0,
            "K%": round(K / PA * 100, 1) if PA > 0 else 0,
            "Whiff%": round(whiffs / swings * 100, 1) if swings > 0 else 0,
            "Chase%": round(chases / swings * 100, 1) if swings > 0 else 0,
        })

    return pd.DataFrame(rows)




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

    bip = get_true_bip_with_ev(hdf) if {"EV", "PitchCall"}.issubset(hdf.columns) else pd.DataFrame()

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
    card["Whiff%"] = round(hdf["is_whiff"].sum() / swings * 100, 1) if swings else 0.0
    card["Chase%"] = round(hdf["is_chase"].sum() / swings * 100, 1) if swings else 0.0

    # Dominant batter side (L/R) for this hitter
    if "BatterSide" in hdf.columns and not hdf["BatterSide"].dropna().empty:
        side_raw = str(hdf["BatterSide"].mode().iloc[0]).upper()
        card["Side"] = "LHH" if side_raw.startswith("L") else "RHH"
    else:
        card["Side"] = "Unknown"

    return card


def count_effectiveness(hdf: pd.DataFrame) -> pd.DataFrame:
    if "Count" not in hdf.columns:
        return pd.DataFrame()

    hdf = hdf.copy()
    agg = hdf.groupby("Count").agg(
        N=("Count", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Chases=("is_chase", "sum")
    ).reset_index()

    bip = get_true_bip_with_ev(hdf) if {"EV", "PitchCall"}.issubset(hdf.columns) else pd.DataFrame()

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

    ba_slg = add_ba_slg_by_group(hdf, ["Count"])
    if not ba_slg.empty:
        agg = agg.merge(ba_slg, on="Count", how="left")
    else:
        agg["AB"] = np.nan
        agg["H"] = np.nan
        agg["BA"] = np.nan
        agg["SLG"] = np.nan

    agg["Swing%"] = (agg["Swings"] / agg["N"] * 100).round(1)
    agg["Whiff%"] = np.where(agg["Swings"] > 0, agg["Whiffs"] / agg["Swings"] * 100, 0).round(1)
    agg["Chase%"] = np.where(agg["Swings"] > 0, agg["Chases"] / agg["Swings"] * 100, 0).round(1)
    agg["HardHit%"] = (agg["HardHit"] * 100).round(1)
    agg["AvgEV"] = agg["AvgEV"].round(1)
    agg["AvgLA"] = agg["AvgLA"].round(1)

    return agg.sort_values("Count")


def count_pitchtype_effectiveness(hdf: pd.DataFrame) -> pd.DataFrame:
    if "pitch_abbr" not in hdf.columns or "Count" not in hdf.columns:
        return pd.DataFrame()

    hdf = hdf.copy()
    hdf["PitchGroup"] = combine_slider_sweeper(hdf["pitch_abbr"])

    agg = hdf.groupby(["Count", "PitchGroup"]).agg(
        N=("PitchGroup", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Chases=("is_chase", "sum")
    ).reset_index()

    bip = get_true_bip_with_ev(hdf) if {"EV", "PitchCall"}.issubset(hdf.columns) else pd.DataFrame()

    if not bip.empty:
        bip = bip.copy()
        bip["PitchGroup"] = combine_slider_sweeper(bip["pitch_abbr"])
        bip_agg = bip.groupby(["Count", "PitchGroup"]).agg(
            HardHit=("hard_hit", "mean"),
            AvgEV=("EV", "mean"),
            AvgLA=("LA", "mean")
        ).reset_index()
        agg = agg.merge(bip_agg, on=["Count", "PitchGroup"], how="left")
    else:
        agg["HardHit"] = np.nan
        agg["AvgEV"] = np.nan
        agg["AvgLA"] = np.nan

    ba_slg = add_ba_slg_by_group(hdf, ["Count", "PitchGroup"])
    if not ba_slg.empty:
        agg = agg.merge(ba_slg, on=["Count", "PitchGroup"], how="left")
    else:
        agg["AB"] = np.nan
        agg["H"] = np.nan
        agg["BA"] = np.nan
        agg["SLG"] = np.nan

    agg["Swing%"] = (agg["Swings"] / agg["N"] * 100).round(1)
    agg["Whiff%"] = np.where(agg["Swings"] > 0, agg["Whiffs"] / agg["Swings"] * 100, 0).round(1)
    agg["Chase%"] = np.where(agg["Swings"] > 0, agg["Chases"] / agg["Swings"] * 100, 0).round(1)
    agg["HardHit%"] = (agg["HardHit"] * 100).round(1)
    agg["AvgEV"] = agg["AvgEV"].round(1)
    agg["AvgLA"] = agg["AvgLA"].round(1)

    return agg.rename(columns={"PitchGroup": "Pitch"}).sort_values(["Count", "Pitch"])


def hitter_pitchtype_effectiveness(hdf: pd.DataFrame) -> pd.DataFrame:
    if "pitch_abbr" not in hdf.columns:
        return pd.DataFrame()

    hdf = hdf.copy()
    hdf["Pitch"] = combine_slider_sweeper(hdf["pitch_abbr"])

    agg = hdf.groupby("Pitch").agg(
        N=("Pitch", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Chases=("is_chase", "sum"),
        wOBA=("woba_value", "mean")
    ).reset_index()

    bip = get_true_bip_with_ev(hdf) if {"EV", "PitchCall"}.issubset(hdf.columns) else pd.DataFrame()
    if not bip.empty:
        bip = bip.copy()
        bip["Pitch"] = combine_slider_sweeper(bip["pitch_abbr"])
        bip_agg = bip.groupby("Pitch").agg(
            BIP=("Pitch", "count"),
            HardHit=("hard_hit", "mean"),
            AvgEV=("EV", "mean"),
            AvgLA=("LA", "mean")
        ).reset_index()
        agg = agg.merge(bip_agg, on="Pitch", how="left")
    else:
        agg["BIP"] = 0
        agg["HardHit"] = np.nan
        agg["AvgEV"] = np.nan
        agg["AvgLA"] = np.nan

    ba_slg = add_ba_slg_by_group(hdf, ["Pitch"])
    if not ba_slg.empty:
        agg = agg.merge(ba_slg, on="Pitch", how="left")
    else:
        agg["AB"] = np.nan
        agg["H"] = np.nan
        agg["BA"] = np.nan
        agg["SLG"] = np.nan

    agg["Swing%"] = (agg["Swings"] / agg["N"] * 100).round(1)
    agg["Whiff%"] = np.where(agg["Swings"] > 0, agg["Whiffs"] / agg["Swings"] * 100, 0).round(1)
    agg["Chase%"] = np.where(agg["Swings"] > 0, agg["Chases"] / agg["Swings"] * 100, 0).round(1)
    agg["HardHit%"] = (agg["HardHit"] * 100).round(1)
    agg["AvgEV"] = agg["AvgEV"].round(1)
    agg["AvgLA"] = agg["AvgLA"].round(1)
    agg["wOBA"] = agg["wOBA"].round(3)
    agg["BIP"] = agg["BIP"].fillna(0).astype(int)

    return agg[
        ["Pitch", "N", "AB", "H", "BA", "SLG", "Swing%", "Whiff%", "Chase%", "wOBA", "BIP", "HardHit%", "AvgEV", "AvgLA"]
    ].sort_values(["N", "Pitch"], ascending=[False, True])


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

        pa_end = get_pa_endings(sub)
        PA = len(pa_end)

        swings = sub["is_swing"].sum()
        whiffs = sub["is_whiff"].sum()
        chases = sub["is_chase"].sum()

        player_woba = compute_woba(sub)

        bip = get_true_bip_with_ev(sub) if {"EV", "PitchCall"}.issubset(sub.columns) else pd.DataFrame()

        hard = bip["hard_hit"].mean() if not bip.empty else np.nan
        avg_ev = bip["EV"].mean() if not bip.empty else np.nan

        rows.append({
            "Split": side,
            "PA": PA,
            "wOBA": round(player_woba, 3) if PA > 0 else np.nan,
            "Swing%": round(swings / len(sub) * 100, 1) if len(sub) else 0,
            "Whiff%": round(whiffs / swings * 100, 1) if swings > 0 else 0,
            "Chase%": round(chases / swings * 100, 1) if swings > 0 else 0,
            "HardHit%": round(hard * 100, 1) if hard == hard else np.nan,
            "AvgEV": round(avg_ev, 1) if avg_ev == avg_ev else np.nan,
        })

    return pd.DataFrame(rows)


def make_zone_heatmap(df, metric, title):
    df = df.copy()

    required = {"PlateLocSide", "PlateLocHeight"}
    if df.empty or not required.issubset(df.columns):
        return None

    for col in ["PlateLocSide", "PlateLocHeight", "EV", "ExitSpeed", "LA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]

    if "woba_value" not in df.columns:
        df["woba_value"] = 0.0

    if "hard_hit" not in df.columns:
        df["hard_hit"] = (df.get("EV", pd.Series(index=df.index, dtype=float)) >= 95).astype(int)

    if "is_swing" not in df.columns:
        df["is_swing"] = 0
    if "is_whiff" not in df.columns:
        df["is_whiff"] = 0

    if "BatterSide" in df.columns and not df["BatterSide"].dropna().empty:
        side_raw = str(df["BatterSide"].mode().iloc[0]).upper()
        hitter_side = "LHH" if side_raw.startswith("L") else "RHH"
    else:
        hitter_side = "Unknown"

    # These edges make the center cell exactly the strike zone:
    # horizontal plate width -0.83 to 0.83, vertical zone 1.5 to 3.5.
    x_edges = np.array([-2.5, -0.83, 0.83, 2.5])
    y_edges = np.array([0.0, 1.5, 3.5, 5.0])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    df["x_bin"] = pd.cut(
        df["PlateLocSide"], bins=x_edges, labels=[0, 1, 2],
        include_lowest=True
    )
    df["y_bin"] = pd.cut(
        df["PlateLocHeight"], bins=y_edges, labels=[0, 1, 2],
        include_lowest=True
    )
    df = df.dropna(subset=["x_bin", "y_bin"]).copy()

    if df.empty:
        return None

    df["x_bin"] = df["x_bin"].astype(int)
    df["y_bin"] = df["y_bin"].astype(int)
    full_index = pd.MultiIndex.from_product(
        [[0, 1, 2], [0, 1, 2]],
        names=["y_bin", "x_bin"]
    )

    if metric in ["AvgEV", "HardHit%"]:
        bip = get_true_bip_with_ev(df)
    else:
        bip = df

    if metric == "Swing%":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        values = (grouped["is_swing"].sum() / grouped["is_swing"].count() * 100)
        samples = grouped["is_swing"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "Swing%"
        cmap_name = "RdYlBu_r"
        vmin, vmax = 0, 100

    elif metric == "Whiff%":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        swings = grouped["is_swing"].sum()
        whiffs = grouped["is_whiff"].sum()
        values = whiffs / swings.replace(0, np.nan) * 100
        samples = swings.reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "Whiff%"
        cmap_name = "RdYlBu_r"
        vmin, vmax = 0, 100

    elif metric == "HardHit%":
        grouped = bip.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["hard_hit"].mean() * 100
        samples = grouped["hard_hit"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "HardHit%"
        cmap_name = "YlOrRd"
        vmin, vmax = 0, 100

    elif metric == "AvgEV":
        grouped = bip.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["EV"].mean()
        samples = grouped["EV"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = ""
        colorbar_label = "Avg EV"
        cmap_name = "YlOrRd"
        vmin, vmax = 60, 105

    elif metric == "wOBA":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["woba_value"].mean()
        samples = grouped["woba_value"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = ""
        colorbar_label = "wOBA"
        cmap_name = "YlOrRd"
        vmin, vmax = 0, 1.2

    else:
        return None

    fig, ax = plt.subplots(figsize=(5.1, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f4f4")

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#eeeeee")
    masked_grid = np.ma.masked_invalid(grid)
    im = ax.pcolormesh(
        x_edges, y_edges, masked_grid,
        cmap=cmap, shading="flat",
        edgecolors="white", linewidth=1.8,
        vmin=vmin, vmax=vmax
    )

    for y_i, y in enumerate(y_centers):
        for x_i, x in enumerate(x_centers):
            val = grid[y_i, x_i]
            n = int(samples[y_i, x_i]) if not np.isnan(samples[y_i, x_i]) else 0
            if np.isnan(val):
                txt = "—\nn=0"
            elif metric == "wOBA":
                txt = f"{val:.3f}\nn={n}"
            else:
                txt = f"{val:.0f}{label_suffix}\nn={n}"
            ax.text(
                x, y, txt,
                ha="center", va="center",
                color="black", fontsize=10, fontweight="bold",
                linespacing=1.15
            )

    strike_zone = plt.Rectangle(
        (-0.83, 1.5), 1.66, 2.0,
        fill=False, edgecolor="black", linewidth=2.5
    )
    ax.add_patch(strike_zone)

    plate_x = [-0.83, 0.83, 0.83, 0, -0.83, -0.83]
    plate_y = [0, 0, 0.17, 0.34, 0.17, 0]
    ax.plot(plate_x, plate_y, color="black", linewidth=1.8)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.0, 5.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-0.83, 0, 0.83])
    ax.set_yticks([1.5, 2.5, 3.5])
    ax.tick_params(labelsize=8, length=0)
    ax.set_xlabel("PlateLocSide (catcher view)", fontsize=9)
    ax.set_ylabel("PlateLocHeight", fontsize=9)
    ax.set_title(title, fontsize=15, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        2.35, 4.75, hitter_side,
        ha="right", va="top",
        fontsize=12, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25")
    )

    if masked_grid.count() > 0:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


def get_true_bip_with_ev(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]
    if "EV" not in df.columns:
        df["EV"] = np.nan

    df["EV"] = pd.to_numeric(df["EV"], errors="coerce")

    pitch_call = df.get("PitchCall", pd.Series("", index=df.index)).astype(str).str.strip()
    tagged_hit_type = df.get("TaggedHitType", pd.Series("", index=df.index)).astype(str).str.strip()
    play_result = df.get("PlayResult", pd.Series("", index=df.index)).astype(str).str.strip()
    bunt_labels = {
        "Bunt", "BuntGroundout", "BuntPopOut", "BuntLineOut",
        "SacrificeBunt", "BuntFoul", "BuntFoulTip"
    }

    true_bip = pitch_call.eq("InPlay")
    usable_ev = df["EV"].notna() & (df["EV"] >= 45)
    excluded_contact = tagged_hit_type.isin(bunt_labels) | play_result.isin(bunt_labels)

    out = df[true_bip & usable_ev & ~excluded_contact].copy()
    if "LA" not in out.columns and "Angle" in out.columns:
        out["LA"] = out["Angle"]
    if "LA" not in out.columns:
        out["LA"] = np.nan
    out["LA"] = pd.to_numeric(out["LA"], errors="coerce")
    out["hard_hit"] = (out["EV"] >= 95).astype(int)
    out["barrel"] = ((out["EV"] >= 98) & out["LA"].between(26, 30)).astype(int)
    out["sweet_spot"] = out["LA"].between(8, 32).astype(int)

    return out


def make_savant_zone_heatmap(df, metric, title, subtitle=None):
    df = df.copy()
    required = {"PlateLocSide", "PlateLocHeight"}
    if df.empty or not required.issubset(df.columns):
        return None

    for col in ["PlateLocSide", "PlateLocHeight", "EV", "ExitSpeed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]
    if "EV" not in df.columns:
        df["EV"] = np.nan

    if "is_swing" not in df.columns:
        df["is_swing"] = 0
    if "is_whiff" not in df.columns:
        df["is_whiff"] = 0
    if "is_csw" not in df.columns:
        df["is_csw"] = 0
    if "hard_hit" not in df.columns:
        df["hard_hit"] = (df["EV"] >= 95).astype(int)
    if "woba_value" not in df.columns:
        df["woba_value"] = 0.0

    x_edges = np.linspace(-0.83, 0.83, 4)
    y_edges = np.linspace(1.5, 3.5, 4)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    in_zone = (
        df["PlateLocSide"].between(-0.83, 0.83) &
        df["PlateLocHeight"].between(1.5, 3.5)
    )
    total_pitches = len(df)
    df = df[in_zone].copy()
    if df.empty:
        return None

    df["x_bin"] = pd.cut(df["PlateLocSide"], bins=x_edges, labels=[0, 1, 2], include_lowest=True)
    df["y_bin"] = pd.cut(df["PlateLocHeight"], bins=y_edges, labels=[0, 1, 2], include_lowest=True)
    df = df.dropna(subset=["x_bin", "y_bin"]).copy()
    if df.empty:
        return None

    df["x_bin"] = df["x_bin"].astype(int)
    df["y_bin"] = df["y_bin"].astype(int)
    full_index = pd.MultiIndex.from_product([[0, 1, 2], [0, 1, 2]], names=["y_bin", "x_bin"])

    if metric == "Usage%":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        counts = grouped["PitchCall"].count() if "PitchCall" in df.columns else grouped["PlateLocSide"].count()
        values = counts / max(total_pitches, 1) * 100
        samples = counts.reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "Pitch%"
        cmap_name = "Blues"
        vmin, vmax = 0, max(20, np.nanmax(grid) if np.isfinite(grid).any() else 20)

    elif metric == "Swing%":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["is_swing"].sum() / grouped["is_swing"].count() * 100
        samples = grouped["is_swing"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "Swing%"
        cmap_name = "RdYlBu_r"
        vmin, vmax = 0, 100

    elif metric == "Whiff%":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        swings = grouped["is_swing"].sum()
        whiffs = grouped["is_whiff"].sum()
        values = whiffs / swings.replace(0, np.nan) * 100
        samples = swings.reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "Whiff%"
        cmap_name = "RdYlBu_r"
        vmin, vmax = 0, 100

    elif metric == "CSW%":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["is_csw"].sum() / grouped["is_csw"].count() * 100
        samples = grouped["is_csw"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "CSW%"
        cmap_name = "RdYlBu_r"
        vmin, vmax = 0, 100

    elif metric == "HardHit%":
        bip = get_true_bip_with_ev(df)
        grouped = bip.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["hard_hit"].mean() * 100
        samples = grouped["hard_hit"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "HardHit%"
        cmap_name = "YlOrRd"
        vmin, vmax = 0, 100

    elif metric == "AvgEV":
        bip = get_true_bip_with_ev(df)
        grouped = bip.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["EV"].mean()
        samples = grouped["EV"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = ""
        colorbar_label = "Avg EV"
        cmap_name = "YlOrRd"
        vmin, vmax = 60, 105

    elif metric == "wOBA":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["woba_value"].mean()
        samples = grouped["woba_value"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = ""
        colorbar_label = "wOBA"
        cmap_name = "YlOrRd"
        vmin, vmax = 0, 1.2

    else:
        return None

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f4f4")

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#eeeeee")
    im = ax.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(grid),
        cmap=cmap, shading="flat",
        edgecolors="white", linewidth=2.2,
        vmin=vmin, vmax=vmax
    )

    for y_i, y in enumerate(y_centers):
        for x_i, x in enumerate(x_centers):
            val = grid[y_i, x_i]
            n = int(samples[y_i, x_i]) if not np.isnan(samples[y_i, x_i]) else 0
            if np.isnan(val):
                text = "—\nn=0"
            elif metric == "wOBA":
                text = f"{val:.3f}\nn={n}"
            else:
                text = f"{val:.0f}{label_suffix}\nn={n}"
            ax.text(
                x, y, text,
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="black", linespacing=1.12
            )

    ax.add_patch(plt.Rectangle((-0.83, 1.5), 1.66, 2.0, fill=False, edgecolor="black", linewidth=2.6))
    ax.set_xlim(-0.83, 0.83)
    ax.set_ylim(1.5, 3.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#555555"
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig




  

# ============================================================
# HITTER DEVELOPMENT & APPROACH PAGE
# ============================================================

def hitter_development_page(all_pitches_df: pd.DataFrame):

    st.title("Hitter Development & Approach")

    df = normalize_hitter_columns(all_pitches_df)
    df = add_contact_quality_local(df)

    # Only FOR_RAM hitters (no pitchers)
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

    # Dominant side for this hitter
    if "BatterSide" in hdf.columns and not hdf["BatterSide"].dropna().empty:
        side_raw = str(hdf["BatterSide"].mode().iloc[0]).upper()
        hitter_side = "LHH" if side_raw.startswith("L") else "RHH"
    else:
        hitter_side = "Unknown"

    lgwOBA = compute_league_woba(df)

    # HITTER CARD
    st.subheader(f"Hitter Card - {hitter_side}")

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
        st.metric("Side", card["Side"])
        st.metric("HardHit%", f"{card['HardHit%']}%")
        st.metric("Barrel%", f"{card['Barrel%']}%")
        st.metric("Avg EV", f"{card['AvgEV']}")
        st.metric("Max EV", f"{card['MaxEV']}")

    # COUNT-BASED EFFECTIVENESS
    st.subheader("Count-Based Effectiveness")
    count_df = count_effectiveness(hdf)
    st.dataframe(count_df, use_container_width=True)

    # COUNT × PITCH TYPE EFFECTIVENESS
    st.subheader("Count x Pitch Type Effectiveness")
    cpt_df = count_pitchtype_effectiveness(hdf)
    st.dataframe(cpt_df, use_container_width=True)

    # SPLITS VS LHP / RHP
    st.subheader("Splits vs LHP / RHP")
    splits_df = hitter_splits(hdf)
    if splits_df.empty:
        st.info("No pitcher handedness data available.")
    else:
        st.dataframe(splits_df, use_container_width=True)

    # ZONE HEATMAPS (catcher‑view, consistent IN/OUT)
    st.subheader("Zone Heatmaps (Catcher View)")

    colA, colB = st.columns(2)

    with colA:
        st.pyplot(make_zone_heatmap(hdf, "Swing%", "Swing% Heatmap"))
        st.pyplot(make_zone_heatmap(hdf, "AvgEV", "Avg EV Heatmap"))

    with colB:
        st.pyplot(make_zone_heatmap(hdf, "HardHit%", "HardHit% Heatmap"))
        st.pyplot(make_zone_heatmap(hdf, "Whiff%", "Whiff% Heatmap"))

    # SPRAY PROFILE (GB/FB/LD + PULL/MID/OPPO)
    st.subheader("Spray Profile (GB/LD/FB + Pull/Mid/Oppo)")
    spray_df = hitter_spray_profile(hdf)
    if spray_df.empty:
        st.info("Not enough batted-ball data for spray profile.")
    else:
        st.dataframe(spray_df, use_container_width=True)

    # SEQUENCING
    st.subheader("Pitch-to-Pitch Sequencing (Hitter Reaction)")

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
        f"**Best damage sequence:** {best_row['prev_pitch']} -> {best_row['pitch_abbr']} "
        f"(wOBA {best_row['wOBA']:.3f}, HardHit% {best_row['HardHit%']}%, N={int(best_row['N'])})"
    )

    st.markdown(
        f"**Toughest sequence:** {worst_row['prev_pitch']} -> {worst_row['pitch_abbr']} "
        f"(Whiff% {worst_row['Whiff%']}%, N={int(worst_row['N'])})"
    )



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
# SPRAY PROFILE (GB/FB/LD + PULL/MID/OPPO)
# ============================================================

def hitter_spray_profile(hdf: pd.DataFrame) -> pd.DataFrame:
    """
    Pull/Middle/Oppo buckets using Direction and BatterSide,
    plus GB/LD/FB percentages.
    Direction: negative = LF side, positive = RF side.
    For RHH: negative = pull, positive = oppo.
    For LHH: positive = pull, negative = oppo.
    """
    required = {"Direction", "BatterSide", "EV", "LA"}
    if not required.issubset(hdf.columns):
        return pd.DataFrame()

    df = get_true_bip_with_ev(hdf)
    df["Direction"] = pd.to_numeric(df["Direction"], errors="coerce")
    df = df.dropna(subset=["Direction", "EV", "LA"]).copy()
    if df.empty:
        return pd.DataFrame()

    def classify_row(row):
        side = str(row.get("BatterSide", "")).upper()
        direction = row["Direction"]

        # Field bucket
        if direction <= -15:
            field = "LF"
        elif direction >= 15:
            field = "RF"
        else:
            field = "CF"

        # Pull / Oppo relative to handedness
        if field == "CF":
            rel = "Middle"
        else:
            if side.startswith("R"):
                rel = "Pull" if field == "LF" else "Oppo"
            elif side.startswith("L"):
                rel = "Pull" if field == "RF" else "Oppo"
            else:
                rel = "Middle"

        # Batted-ball type by LA
        la = row["LA"]
        if la < 8:
            batted_type = "GB"
        elif la <= 27:
            batted_type = "LD"
        else:
            batted_type = "FB"

        return pd.Series({"SprayBucket": rel, "BattedType": batted_type})

    spray_info = df.apply(classify_row, axis=1)
    df = pd.concat([df, spray_info], axis=1)

    if "SprayBucket" not in df.columns:
        return pd.DataFrame()

    rows = []
    for bucket, g in df.groupby("SprayBucket"):
        BIP = len(g)
        if BIP == 0:
            continue
        gb = (g["BattedType"] == "GB").mean()
        ld = (g["BattedType"] == "LD").mean()
        fb = (g["BattedType"] == "FB").mean()
        hard = (g["EV"] >= 95).mean()
        avg_ev = g["EV"].mean()
        avg_la = g["LA"].mean()

        rows.append({
            "SprayBucket": bucket,
            "BIP": BIP,
            "GB%": round(gb * 100, 1),
            "LD%": round(ld * 100, 1),
            "FB%": round(fb * 100, 1),
            "HardHit%": round(hard * 100, 1),
            "AvgEV": round(avg_ev, 1),
            "AvgLA": round(avg_la, 1),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("SprayBucket")


def make_defensive_positioning_chart(hdf: pd.DataFrame, hitter: str):
    required = {"Direction", "BatterSide", "EV", "LA"}
    if not required.issubset(hdf.columns):
        return None, pd.DataFrame()

    df = get_true_bip_with_ev(hdf)
    if df.empty or "Direction" not in df.columns:
        return None, pd.DataFrame()

    df["Direction"] = pd.to_numeric(df["Direction"], errors="coerce")
    df["LA"] = pd.to_numeric(df["LA"], errors="coerce")
    df = df.dropna(subset=["Direction", "EV", "LA"]).copy()
    if df.empty:
        return None, pd.DataFrame()

    side_raw = str(df.get("BatterSide", pd.Series(["Unknown"])).dropna().mode().iloc[0]).upper()
    hitter_side = "RHH" if side_raw.startswith("R") else "LHH" if side_raw.startswith("L") else "Unknown"

    def classify(row):
        direction = row["Direction"]
        if direction <= -15:
            field = "LF"
        elif direction >= 15:
            field = "RF"
        else:
            field = "CF"

        if field == "CF":
            spray = "Middle"
        elif hitter_side == "RHH":
            spray = "Pull" if field == "LF" else "Oppo"
        elif hitter_side == "LHH":
            spray = "Pull" if field == "RF" else "Oppo"
        else:
            spray = field

        if row["LA"] < 8:
            contact = "Ground"
        elif row["LA"] <= 27:
            contact = "Line"
        else:
            contact = "Air"

        return pd.Series({"Field": field, "Spray": spray, "ContactType": contact})

    classified = df.apply(classify, axis=1)
    df = pd.concat([df, classified], axis=1)

    rows = []
    total_bip = len(df)
    for bucket in ["Pull", "Middle", "Oppo"]:
        g = df[df["Spray"] == bucket]
        rows.append({
            "Spray": bucket,
            "BIP": len(g),
            "BIP%": round(len(g) / total_bip * 100, 1) if total_bip else 0,
            "GB%": round((g["ContactType"] == "Ground").mean() * 100, 1) if len(g) else 0,
            "Air%": round((g["ContactType"] == "Air").mean() * 100, 1) if len(g) else 0,
            "HardHit%": round((g["EV"] >= 95).mean() * 100, 1) if len(g) else 0,
            "AvgEV": round(g["EV"].mean(), 1) if len(g) else np.nan,
        })

    summary = pd.DataFrame(rows)
    spray_summary = summary.sort_values(["BIP", "HardHit%"], ascending=False)
    best = spray_summary.iloc[0]
    second = spray_summary.iloc[1] if len(spray_summary) > 1 else best

    ground = df[df["ContactType"] == "Ground"].copy()
    gb_rate = (df["ContactType"] == "Ground").mean() * 100
    air_rate = (df["ContactType"] == "Air").mean() * 100
    hard_rate = (df["EV"] >= 95).mean() * 100
    pull_rate = float(summary.loc[summary["Spray"] == "Pull", "BIP%"].iloc[0])
    middle_rate = float(summary.loc[summary["Spray"] == "Middle", "BIP%"].iloc[0])
    oppo_rate = float(summary.loc[summary["Spray"] == "Oppo", "BIP%"].iloc[0])
    pull_ground_rate = (
        (ground["Spray"].eq("Pull")).mean() * 100 if len(ground) else 0
    )
    middle_ground_rate = (
        (ground["Spray"].eq("Middle")).mean() * 100 if len(ground) else 0
    )

    raw = hdf.copy()
    tagged_hit = raw.get("TaggedHitType", pd.Series("", index=raw.index)).astype(str)
    play_result = raw.get("PlayResult", pd.Series("", index=raw.index)).astype(str)
    bunt_mask = (
        tagged_hit.str.contains("Bunt", case=False, na=False) |
        play_result.str.contains("Bunt", case=False, na=False) |
        play_result.eq("Sacrifice")
    )
    bunt_rate = bunt_mask.sum() / max(len(get_pa_endings(raw)), 1) * 100

    base_positions = {
        "LF": (-2.1, 1.95),
        "CF": (0, 2.65),
        "RF": (2.1, 1.95),
        "3B": (-0.95, 0.78),
        "SS": (-0.45, 1.06),
        "2B": (0.45, 1.06),
        "1B": (0.95, 0.78),
    }
    positions = base_positions.copy()

    if best["Spray"] == "Pull" and hitter_side == "RHH":
        primary_of = "Shade OF toward LF / left-center"
        pull_side_note = "3B and SS protect pull-side grounders"
        of_shift = -0.28
    elif best["Spray"] == "Pull" and hitter_side == "LHH":
        primary_of = "Shade OF toward RF / right-center"
        pull_side_note = "2B and 1B protect pull-side grounders"
        of_shift = 0.28
    elif best["Spray"] == "Oppo" and hitter_side == "RHH":
        primary_of = "Shade OF toward RF / right-center"
        pull_side_note = "2B and 1B stay ready opposite way"
        of_shift = 0.20
    elif best["Spray"] == "Oppo" and hitter_side == "LHH":
        primary_of = "Shade OF toward LF / left-center"
        pull_side_note = "SS and 3B stay ready opposite way"
        of_shift = -0.20
    else:
        primary_of = "Keep OF straight up / center-heavy"
        pull_side_note = "Keep infield balanced through the middle"
        of_shift = 0.0

    positions["LF"] = (base_positions["LF"][0] + of_shift, base_positions["LF"][1])
    positions["CF"] = (base_positions["CF"][0] + of_shift * 0.65, base_positions["CF"][1])
    positions["RF"] = (base_positions["RF"][0] + of_shift, base_positions["RF"][1])

    alignment = "Standard infield"
    if bunt_rate >= 8:
        alignment = "Corners in / 3B bunt alert"
        positions["3B"] = (-0.72, 0.46)
        positions["1B"] = (0.78, 0.48)
        positions["SS"] = (-0.35, 1.04)
        positions["2B"] = (0.42, 1.04)
    elif gb_rate >= 45 and pull_ground_rate >= 45:
        alignment = "Pull-side infield shift"
        if hitter_side == "RHH":
            positions["3B"] = (-1.08, 0.72)
            positions["SS"] = (-0.58, 1.02)
            positions["2B"] = (0.08, 1.05)
            positions["1B"] = (0.82, 0.72)
        elif hitter_side == "LHH":
            positions["3B"] = (-0.82, 0.72)
            positions["SS"] = (-0.08, 1.05)
            positions["2B"] = (0.58, 1.02)
            positions["1B"] = (1.08, 0.72)
    elif gb_rate >= 45 and middle_ground_rate >= 35:
        alignment = "Middle infield pinch"
        positions["SS"] = (-0.25, 1.07)
        positions["2B"] = (0.25, 1.07)
    elif hard_rate >= 35 and max(pull_rate, oppo_rate) >= 35:
        alignment = "Guard lines"
        positions["3B"] = (-1.12, 0.74)
        positions["1B"] = (1.12, 0.74)

    depth_note = "Outfield no-doubles depth" if air_rate >= 45 and hard_rate >= 35 else "Normal OF depth"
    if gb_rate >= 50:
        depth_note = f"{alignment}; prioritize ground-ball lanes"

    summary["Alignment Read"] = summary["Spray"].map({
        "Pull": "Primary shift side" if best["Spray"] == "Pull" else "Secondary",
        "Middle": "Pinch middle" if middle_ground_rate >= 35 else "Standard",
        "Oppo": "Respect opposite field" if oppo_rate >= 30 else "Standard"
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eef6ee")
    ax.set_aspect("equal")
    ax.axis("off")

    diamond = np.array([[0, 0], [1, 0.7], [0, 1.4], [-1, 0.7], [0, 0]])
    ax.plot(diamond[:, 0], diamond[:, 1], color="#7a4a20", linewidth=2.5)
    ax.fill(diamond[:, 0], diamond[:, 1], color="#d9a15f", alpha=0.45)

    theta = np.linspace(25, 155, 120)
    x_arc = 2.9 * np.cos(np.deg2rad(theta))
    y_arc = 2.9 * np.sin(np.deg2rad(theta)) - 0.2
    ax.plot(x_arc, y_arc, color="#2f7d32", linewidth=3)

    for label, (x, y) in positions.items():
        ax.scatter(x, y, s=420, color="#A00000", edgecolor="white", linewidth=1.5, zorder=5)
        ax.text(x, y, label, ha="center", va="center", color="white", fontweight="bold", fontsize=10, zorder=6)

    ax.text(0, -0.28, "HOME", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0, 3.25, f"Best Defensive Positioning vs {hitter}", ha="center", fontsize=16, fontweight="bold")
    ax.text(0, 3.02, f"{hitter_side} | BIP={total_bip}", ha="center", fontsize=10, color="#555555")

    rec = (
        f"{primary_of}\n"
        f"{alignment}: {pull_side_note}\n"
        f"{depth_note}\n"
        f"Primary spray: {best['Spray']} ({best['BIP%']}% BIP); next: {second['Spray']} ({second['BIP%']}%)"
    )
    ax.text(
        0, -0.62, rec,
        ha="center", va="top", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#A00000", boxstyle="round,pad=0.45")
    )

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-0.9, 3.45)
    fig.tight_layout()
    return fig, summary


def _fmt_pdf_value(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, (float, np.floating)):
        if abs(value) < 1:
            return f"{value:.3f}".replace("0.", ".")
        return f"{value:.1f}"
    return str(value)


def _safe_pdf_name(name):
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")


def _add_report_table(ax, df, title, max_rows=10, font_size=8):
    ax.axis("off")
    ax.set_title(title, color="#FFF7E8", fontsize=14, fontweight="bold", loc="left", pad=10)

    if df is None or df.empty:
        ax.text(0.02, 0.55, "No data available", color="#CDBFAF", fontsize=10, ha="left", va="center")
        return

    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(_fmt_pdf_value)

    tbl = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0, 0, 1, 0.86]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#4E4036")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor(FORDHAM_MAROON)
            cell.set_text_props(color="#FFF7E8", weight="bold")
        else:
            cell.set_facecolor("#211C1A" if r % 2 else "#171514")
            cell.set_text_props(color="#F8EFE2")


def _scouting_cover_fig(title, subtitle, metric_pairs):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color=FORDHAM_MAROON, transform=ax.transAxes))
    ax.text(0.05, 0.93, "FORDHAM BASEBALL SCOUTING ZONE", color="#FFF7E8", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.80, title, color="#FFF7E8", fontsize=28, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.75, subtitle, color="#CDBFAF", fontsize=12, transform=ax.transAxes)

    cols = 4
    start_x, start_y = 0.05, 0.62
    box_w, box_h = 0.21, 0.12
    for i, (label, value) in enumerate(metric_pairs):
        x = start_x + (i % cols) * 0.235
        y = start_y - (i // cols) * 0.155
        ax.add_patch(plt.Rectangle((x, y), box_w, box_h, facecolor="#211C1A", edgecolor=FORDHAM_GOLD, linewidth=1.2, transform=ax.transAxes))
        ax.text(x + 0.018, y + 0.076, str(label), color="#CDBFAF", fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.018, y + 0.030, _fmt_pdf_value(value), color="#FFF7E8", fontsize=18, fontweight="bold", transform=ax.transAxes)

    ax.text(
        0.05, 0.08,
        "Generated from TrackMan pitch-by-pitch data. Contact metrics use true in-play batted balls with usable EV.",
        color="#CDBFAF", fontsize=9, transform=ax.transAxes
    )
    return fig


def build_hitter_scouting_pdf(hdf: pd.DataFrame, hitter: str, team: str) -> bytes:
    hdf = hdf.copy()
    lgwoba = compute_league_woba(hdf)
    card = compute_hitter_card(hdf, lgwoba)

    slash = add_ba_slg_by_group(hdf.assign(Player=hitter), ["Player"])
    ba = slash["BA"].iloc[0] if not slash.empty else np.nan
    obp = slash["OBP"].iloc[0] if not slash.empty else np.nan
    slg = slash["SLG"].iloc[0] if not slash.empty else np.nan
    ops = slash["OPS"].iloc[0] if not slash.empty else np.nan

    metric_pairs = [
        ("Team", team), ("Side", card.get("Side", "Unknown")), ("PA", card.get("PA")), ("AB", card.get("AB")),
        ("BA", ba), ("OBP", obp), ("SLG", slg), ("OPS", ops),
        ("wOBA", card.get("wOBA")), ("wRC+", card.get("wRC+")), ("Avg EV", card.get("AvgEV")), ("HardHit%", card.get("HardHit%")),
    ]

    pitch_table = hitter_pitchtype_effectiveness(hdf)
    if not pitch_table.empty:
        pitch_table = pitch_table[["Pitch", "N", "BA", "SLG", "Swing%", "Whiff%", "Chase%", "AvgEV", "HardHit%"]]

    count_table = count_effectiveness(hdf)
    if not count_table.empty:
        count_table = count_table[["Count", "N", "BA", "SLG", "Swing%", "Whiff%", "AvgEV", "HardHit%"]]

    spray_table = hitter_spray_profile(hdf)
    splits_table = hitter_splits(hdf)

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig = _scouting_cover_fig(hitter, "Hitter scouting report", metric_pairs)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#100D0C")
        gs = fig.add_gridspec(2, 2, left=0.05, right=0.95, top=0.92, bottom=0.07, hspace=0.32, wspace=0.18)
        _add_report_table(fig.add_subplot(gs[0, 0]), pitch_table, "Effectiveness vs Pitch Type", max_rows=12)
        _add_report_table(fig.add_subplot(gs[0, 1]), count_table, "Count-Based Effectiveness", max_rows=12)
        _add_report_table(fig.add_subplot(gs[1, 0]), spray_table, "Spray Profile", max_rows=8)
        _add_report_table(fig.add_subplot(gs[1, 1]), splits_table, "Splits vs Pitcher Handedness", max_rows=8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = make_savant_zone_heatmap(hdf, "AvgEV", "In-Zone Avg EV", "True BIP only")
        if fig:
            fig.patch.set_facecolor("#100D0C")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


def build_pitcher_scouting_pdf(pdf_df: pd.DataFrame, pitcher: str, team: str) -> bytes:
    pdf_df = pdf_df.copy()
    for col in ["pitch_abbr", "Velo", "IVB", "HB", "Ext", "RelH", "in_zone", "is_swing", "is_whiff", "BatterSide"]:
        if col not in pdf_df.columns:
            pdf_df[col] = np.nan
    pdf_df["pitch_abbr"] = pdf_df["pitch_abbr"].fillna("UNK")

    total = len(pdf_df)
    swings = pdf_df["is_swing"].sum() if "is_swing" in pdf_df.columns else 0
    whiffs = pdf_df["is_whiff"].sum() if "is_whiff" in pdf_df.columns else 0
    csw = pdf_df["is_csw"].mean() * 100 if "is_csw" in pdf_df.columns and total else np.nan
    zone = pdf_df["in_zone"].mean() * 100 if "in_zone" in pdf_df.columns and total else np.nan
    strike = pdf_df["is_strike"].mean() * 100 if "is_strike" in pdf_df.columns and total else np.nan
    whiff_pct = whiffs / swings * 100 if swings else np.nan
    bip = get_true_bip_with_ev(pdf_df) if {"EV", "PitchCall"}.issubset(pdf_df.columns) else pd.DataFrame()

    metric_pairs = [
        ("Team", team), ("Pitches", total), ("Strike%", strike), ("Zone%", zone),
        ("CSW%", csw), ("Whiff%", whiff_pct), ("Avg EV Allowed", bip["EV"].mean() if not bip.empty else np.nan),
        ("HardHit% Allowed", (bip["EV"] >= 95).mean() * 100 if not bip.empty else np.nan),
    ]

    arsenal = pdf_df.groupby("pitch_abbr").agg(
        N=("pitch_abbr", "count"),
        Velo=("Velo", "mean"),
        IVB=("IVB", "mean"),
        HB=("HB", "mean"),
        Ext=("Ext", "mean"),
        RelHt=("RelH", "mean"),
        Zone=("in_zone", "mean"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
    ).reset_index().rename(columns={"pitch_abbr": "Pitch"})
    arsenal["Usage%"] = arsenal["N"] / max(arsenal["N"].sum(), 1) * 100
    arsenal["Zone%"] = arsenal["Zone"] * 100
    arsenal["Whiff%"] = np.where(arsenal["Swings"] > 0, arsenal["Whiffs"] / arsenal["Swings"] * 100, np.nan)

    if not bip.empty:
        bb = bip.groupby("pitch_abbr").agg(AvgEV=("EV", "mean"), HardHit=("EV", lambda x: (x >= 95).mean() * 100)).reset_index()
        arsenal = arsenal.merge(bb.rename(columns={"pitch_abbr": "Pitch", "HardHit": "HardHit%"}), on="Pitch", how="left")
    else:
        arsenal["AvgEV"] = np.nan
        arsenal["HardHit%"] = np.nan

    arsenal = arsenal[["Pitch", "N", "Usage%", "Velo", "IVB", "HB", "Ext", "RelHt", "Zone%", "Whiff%", "AvgEV", "HardHit%"]].round(1)

    splits = pdf_df.groupby(["BatterSide", "pitch_abbr"]).agg(
        N=("pitch_abbr", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Zone=("in_zone", "mean"),
    ).reset_index().rename(columns={"pitch_abbr": "Pitch"})
    splits["Whiff%"] = np.where(splits["Swings"] > 0, splits["Whiffs"] / splits["Swings"] * 100, np.nan)
    splits["Zone%"] = splits["Zone"] * 100
    splits = splits[["BatterSide", "Pitch", "N", "Whiff%", "Zone%"]].round(1)

    buf = BytesIO()
    with PdfPages(buf) as out_pdf:
        fig = _scouting_cover_fig(pitcher, "Pitcher scouting report", metric_pairs)
        out_pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#100D0C")
        gs = fig.add_gridspec(2, 2, left=0.05, right=0.95, top=0.92, bottom=0.07, hspace=0.32, wspace=0.18)
        _add_report_table(fig.add_subplot(gs[0, :]), arsenal.sort_values("N", ascending=False), "Pitch Arsenal", max_rows=12, font_size=7)
        _add_report_table(fig.add_subplot(gs[1, 0]), splits.sort_values(["BatterSide", "N"], ascending=[True, False]), "Batter-Side Splits", max_rows=12)
        ax = fig.add_subplot(gs[1, 1])
        ax.axis("off")
        ax.set_title("Report Notes", color="#FFF7E8", fontsize=14, fontweight="bold", loc="left", pad=10)
        ax.text(
            0.02, 0.78,
            "Use this page as the quick scout card before building a game plan.\n\n"
            "Zone%, Whiff%, EV allowed, movement, release height, and extension are grouped by pitch type.\n\n"
            "Pair this with the Pitcher Advanced Info page for location heatmaps and sequencing.",
            color="#F8EFE2", fontsize=11, va="top", linespacing=1.45
        )
        out_pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for fig_builder in [build_movement_figure, build_release_figure, build_release_extension_figure]:
            fig = fig_builder(pdf_df)
            out_pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


def scouting_zone_page(all_pitches_df: pd.DataFrame):
    st.title("Scouting Zone")
    st.caption("Create team-filtered hitter and pitcher scouting PDFs from the TrackMan database.")

    if all_pitches_df.empty:
        st.error("No pitch-by-pitch data loaded.")
        return

    df = normalize_hitter_columns(all_pitches_df)
    df = add_contact_quality_local(df)

    teams = sorted(set(
        df.get("BatterTeam", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() +
        df.get("PitcherTeam", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    ))
    if not teams:
        st.warning("No team values found in BatterTeam or PitcherTeam.")
        return

    c1, c2, c3 = st.columns([1.1, 1.0, 1.4])
    with c1:
        default_idx = teams.index("FOR_RAM") if "FOR_RAM" in teams else 0
        team = st.selectbox("Team", teams, index=default_idx)
    with c2:
        report_type = st.radio("Report Type", ["Hitters", "Pitchers"], horizontal=True)

    if report_type == "Hitters":
        team_df = df[df["BatterTeam"].astype(str) == team].copy()
        players = sorted(team_df["Batter"].dropna().astype(str).unique())
        if not players:
            st.info("No hitters found for this team.")
            return
        with c3:
            player = st.selectbox("Player", players)
        player_df = team_df[team_df["Batter"].astype(str) == player].copy()
        card = compute_hitter_card(player_df, compute_league_woba(player_df))
        preview = pd.DataFrame([{
            "Player": player,
            "Team": team,
            "Side": card.get("Side"),
            "PA": card.get("PA"),
            "wOBA": card.get("wOBA"),
            "wRC+": card.get("wRC+"),
            "AvgEV": card.get("AvgEV"),
            "HardHit%": card.get("HardHit%"),
        }])
        st.dataframe(preview, hide_index=True, use_container_width=True)
        pdf_bytes = build_hitter_scouting_pdf(player_df, player, team)
        file_name = f"{_safe_pdf_name(player)}_{team}_hitter_scout.pdf"
    else:
        team_df = df[df["PitcherTeam"].astype(str) == team].copy()
        players = sorted(team_df["Pitcher"].dropna().astype(str).unique())
        if not players:
            st.info("No pitchers found for this team.")
            return
        with c3:
            player = st.selectbox("Pitcher", players)
        player_df = team_df[team_df["Pitcher"].astype(str) == player].copy()
        swings = player_df["is_swing"].sum() if "is_swing" in player_df.columns else 0
        preview = pd.DataFrame([{
            "Pitcher": player,
            "Team": team,
            "Pitches": len(player_df),
            "Strike%": round(player_df["is_strike"].mean() * 100, 1) if "is_strike" in player_df.columns and len(player_df) else np.nan,
            "Zone%": round(player_df["in_zone"].mean() * 100, 1) if "in_zone" in player_df.columns and len(player_df) else np.nan,
            "Whiff%": round(player_df["is_whiff"].sum() / swings * 100, 1) if swings else np.nan,
        }])
        st.dataframe(preview, hide_index=True, use_container_width=True)
        pdf_bytes = build_pitcher_scouting_pdf(player_df, player, team)
        file_name = f"{_safe_pdf_name(player)}_{team}_pitcher_scout.pdf"

    st.download_button(
        "Download Scouting PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
        use_container_width=True
    )


# ============================================================
# CONTACT QUALITY LEADERBOARD PAGE
# ============================================================

def contact_quality_leaderboard_page(all_pitches_df: pd.DataFrame):
    st.markdown("## Contact Quality Leaderboard")

    df = all_pitches_df.copy()

    df = df.rename(columns={
        "ExitSpeed": "EV",
        "Angle": "LA",
        "Direction": "Spray"
    })

    df = add_contact_quality(df)

    teams = sorted(set(
        df.get("BatterTeam", pd.Series(dtype=str)).dropna().unique().tolist() +
        df.get("PitcherTeam", pd.Series(dtype=str)).dropna().unique().tolist()
    ))

    if not teams:
        st.warning("No team info found.")
        return

    default = "FOR_RAM" if "FOR_RAM" in teams else teams[0]
    team = st.selectbox("Select Team", teams, index=teams.index(default))

    mode = st.radio("View:", ["Hitters", "Pitchers"], horizontal=True)

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


def hitter_development_page(all_pitches_df: pd.DataFrame):

    st.title("Hitter Development & Approach")

    df = normalize_hitter_columns(all_pitches_df)
    df = add_contact_quality_local(df)

    # Only FOR_RAM hitters (no pitchers)
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

    # Show hitter handedness
    handed = hdf.get("BatterSide", pd.Series(["Unknown"])).iloc[0]
    st.markdown(f"**Handedness:** {handed}")

    lgwOBA = compute_league_woba(df)

    # HITTER CARD
    st.subheader("Hitter Card")

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
    st.subheader("Count-Based Effectiveness")
    count_df = count_effectiveness(hdf)
    st.dataframe(count_df, use_container_width=True)

    # COUNT × PITCH TYPE EFFECTIVENESS
    st.subheader("Count x Pitch Type Effectiveness")
    cpt_df = count_pitchtype_effectiveness(hdf)
    st.dataframe(cpt_df, use_container_width=True)

    # SPLITS VS LHP / RHP
    st.subheader("Splits vs LHP / RHP")
    splits_df = hitter_splits(hdf)
    if splits_df.empty:
        st.info("No pitcher handedness data available.")
    else:
        st.dataframe(splits_df, use_container_width=True)

    # ZONE HEATMAPS
    st.subheader("Zone Heatmaps")

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

        fig4 = make_zone_heatmap(hdf, "AvgEV", "Avg EV Heatmap")
        if fig4:
            st.pyplot(fig4)

    # SPRAY PROFILE
    st.subheader("Spray Profile (GB/LD/FB + Pull/Mid/Oppo)")
    spray_df = hitter_spray_profile(hdf)
    if spray_df.empty:
        st.info("Not enough batted-ball data for spray profile.")
    else:
        st.dataframe(spray_df, use_container_width=True)

    # SEQUENCING
    st.subheader("Pitch-to-Pitch Sequencing (Hitter Reaction)")

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
        f"**Best damage sequence:** {best_row['prev_pitch']} -> {best_row['pitch_abbr']} "
        f"(wOBA {best_row['wOBA']:.3f}, HardHit% {best_row['HardHit%']}%, N={int(best_row['N'])})"
    )

    st.markdown(
        f"**Toughest sequence:** {worst_row['prev_pitch']} -> {worst_row['pitch_abbr']} "
        f"(Whiff% {worst_row['Whiff%']}%, N={int(worst_row['N'])})"
    )


# ============================================================
# PITCHER DEVELOPMENT & SEQUENCING TAB
# (Assumes filter_fordham_only and get_pitcher_list exist elsewhere)
# ============================================================

def sequencing_page(all_pitches_df: pd.DataFrame):
    st.markdown("## Pitcher Development & Sequencing")

    df = all_pitches_df.copy()
    df = filter_fordham_only(df)  # helper defined elsewhere

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

    pitchers = get_pitcher_list(df)  # helper defined elsewhere
    if not pitchers:
        st.warning("No FOR_RAM pitcher data available.")
        return

    pitcher = st.selectbox("Select Pitcher", pitchers, key="seq_pitcher_select")

    pdf = df[df["Pitcher"] == pitcher].copy()
    if pdf.empty:
        st.warning("No data for this pitcher.")
        return

    bip = get_true_bip_with_ev(pdf) if {"EV", "PitchCall"}.issubset(pdf.columns) else pd.DataFrame()

    # SECTION 1 — ARSENAL OVERVIEW
    st.markdown("### Arsenal Overview")

    arsenal = pdf.groupby("pitch_abbr").agg(
        N=("pitch_abbr", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Chases=("is_chase", "sum"),
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

    arsenal_slash = add_ba_slg_by_group(pdf, ["pitch_abbr"])
    if not arsenal_slash.empty:
        arsenal = arsenal.join(arsenal_slash.set_index("pitch_abbr")[["BA", "SLG"]], how="left")
    else:
        arsenal["BA"] = np.nan
        arsenal["SLG"] = np.nan

    arsenal["Usage%"] = (arsenal["N"] / arsenal["N"].sum() * 100).round(1)
    arsenal["Whiff%"] = np.where(
        arsenal["Swings"] > 0,
        (arsenal["Whiffs"] / arsenal["Swings"] * 100).round(1),
        0.0
    )
    arsenal["Chase%"] = np.where(
        arsenal["Swings"] > 0,
        (arsenal["Chases"] / arsenal["Swings"] * 100).round(1),
        0.0
    )
    arsenal["InZone%"] = (arsenal["InZone"] * 100).round(1)
    arsenal["HardHit%"] = (arsenal["HardHit"].fillna(0) * 100).round(1)
    arsenal["AvgEV"] = arsenal["AvgEV"].fillna(0).round(1)

    st.dataframe(
        arsenal[["Usage%", "BA", "SLG", "Whiff%", "Chase%", "InZone%", "HardHit%", "AvgEV"]],
        use_container_width=True
    )

    # SECTION 2 — COUNT-BASED EFFECTIVENESS
    st.markdown("### Count-Based Effectiveness")

    count_grid = pdf.groupby(["Count", "pitch_abbr"]).agg(
        N=("pitch_abbr", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Chases=("is_chase", "sum"),
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

    count_grid["Whiff%"] = np.where(
        count_grid["Swings"] > 0,
        (count_grid["Whiffs"] / count_grid["Swings"] * 100).round(1),
        0.0
    )
    count_grid["Chase%"] = np.where(
        count_grid["Swings"] > 0,
        (count_grid["Chases"] / count_grid["Swings"] * 100).round(1),
        0.0
    )
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

    # SECTION 3 — STRIKE ZONE 9-BOX
    st.markdown("### Strike Zone 9-Box Breakdown")
    st.caption("Baseball Savant-style in-zone map from the catcher view.")

    pitch_options = ["All"]
    if "pitch_abbr" in pdf.columns:
        pitch_options += sorted(pdf["pitch_abbr"].dropna().astype(str).unique())
    selected_zone_pitch = st.selectbox(
        "Pitch Type",
        pitch_options,
        key=f"pitcher_zone_pitch_{pitcher}"
    )
    zone_pdf = pdf.copy()
    if selected_zone_pitch != "All":
        zone_pdf = zone_pdf[zone_pdf["pitch_abbr"].astype(str) == selected_zone_pitch].copy()

    zoneA, zoneB = st.columns(2)
    with zoneA:
        fig_zone_usage = make_savant_zone_heatmap(
            zone_pdf, "Usage%", "Pitch Location%", "Share of selected pitches"
        )
        if fig_zone_usage:
            st.pyplot(fig_zone_usage)

        fig_zone_csw = make_savant_zone_heatmap(
            zone_pdf, "CSW%", "CSW% By Zone", "Called strikes + whiffs"
        )
        if fig_zone_csw:
            st.pyplot(fig_zone_csw)

    with zoneB:
        fig_zone_whiff = make_savant_zone_heatmap(
            zone_pdf, "Whiff%", "Whiff% By Zone", "Whiffs per swing"
        )
        if fig_zone_whiff:
            st.pyplot(fig_zone_whiff)

        fig_zone_ev = make_savant_zone_heatmap(
            zone_pdf, "AvgEV", "Avg EV Allowed", "True BIP only"
        )
        if fig_zone_ev:
            st.pyplot(fig_zone_ev)

    # SECTION 3 — RELEASE CONSISTENCY
    st.markdown("### Release Consistency")

    rel = pdf.groupby("pitch_abbr").agg(
        RelH_std=("RelH", "std"),
        RelS_std=("RelS", "std")
    ).round(3)

    st.dataframe(rel, use_container_width=True)

    # SECTION 4 — PITCH-TO-PITCH SEQUENCING
    st.markdown("### Pitch-to-Pitch Sequencing")

    sort_cols = [c for c in ["Date", "Inning", "PitchNumber"] if c in pdf.columns]
    if sort_cols:
        pdf = pdf.sort_values(sort_cols)

    pdf["PrevPitch"] = pdf["pitch_abbr"].shift(1)
    pdf["PrevPitcher"] = pdf["Pitcher"].shift(1)

    seq = pdf[pdf["PrevPitcher"] == pitcher].copy()

    seq_stats = seq.groupby(["PrevPitch", "pitch_abbr"]).agg(
        N=("pitch_abbr", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum")
    ).reset_index()

    seq_bip = get_true_bip_with_ev(seq) if {"EV", "PitchCall"}.issubset(seq.columns) else pd.DataFrame()
    if not seq_bip.empty:
        seq_bb = seq_bip.groupby(["PrevPitch", "pitch_abbr"]).agg(
            HardHit=("EV", lambda x: (x >= 90).mean())
        ).reset_index()
        seq_stats = seq_stats.merge(seq_bb, on=["PrevPitch", "pitch_abbr"], how="left")
    else:
        seq_stats["HardHit"] = 0.0

    seq_stats["Whiff%"] = np.where(
        seq_stats["Swings"] > 0,
        (seq_stats["Whiffs"] / seq_stats["Swings"] * 100).round(1),
        0.0
    )
    seq_stats["HardHit%"] = (seq_stats["HardHit"].fillna(0) * 100).round(1)

    st.dataframe(
        seq_stats[["PrevPitch", "pitch_abbr", "N", "Whiff%", "HardHit%"]],
        use_container_width=True
    )

    # SECTION 5 — LHH vs RHH SPLITS
    st.markdown("### LHH vs RHH Splits")

    splits = pdf.groupby(["BatterSide", "pitch_abbr"]).agg(
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum")
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

    split_slash = add_ba_slg_by_group(pdf, ["BatterSide", "pitch_abbr"])
    if not split_slash.empty:
        splits = splits.merge(
            split_slash[["BatterSide", "pitch_abbr", "BA", "SLG"]],
            on=["BatterSide", "pitch_abbr"],
            how="left"
        )
    else:
        splits["BA"] = np.nan
        splits["SLG"] = np.nan

    splits["Whiff%"] = np.where(
        splits["Swings"] > 0,
        (splits["Whiffs"] / splits["Swings"] * 100).round(1),
        0.0
    )
    splits["HardHit%"] = (splits["HardHit"].fillna(0) * 100).round(1)
    splits["AvgEV"] = splits["AvgEV"].fillna(0).round(1)

    st.dataframe(
        splits[["BatterSide", "pitch_abbr", "BA", "SLG", "Whiff%", "HardHit%", "AvgEV"]],
        use_container_width=True
    )

    # SECTION 6 — SMART DEVELOPMENT RECOMMENDATIONS
    st.markdown("### Development Recommendations")

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

    st.title("Hitter Development & Approach")

    df = normalize_hitter_columns(all_pitches_df)
    df = add_contact_quality_local(df)

    # FOR_RAM hitters only (no pitchers leaking in)
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
    st.subheader("Hitter Card")

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

    # COUNT-BASED EFFECTIVENESS (NO wOBA COLUMN)
    st.subheader("Count-Based Effectiveness")
    count_df = count_effectiveness(hdf)
    st.dataframe(count_df, use_container_width=True)

    # PITCH-TYPE EFFECTIVENESS
    st.subheader("Hitter Effectiveness vs Pitch Type")
    pitchtype_df = hitter_pitchtype_effectiveness(hdf)
    if pitchtype_df.empty:
        st.info("No pitch-type data available for this hitter.")
    else:
        st.dataframe(pitchtype_df, use_container_width=True)

    # COUNT × PITCH TYPE EFFECTIVENESS (NO wOBA COLUMN)
    st.subheader("Count x Pitch Type Effectiveness")
    cpt_df = count_pitchtype_effectiveness(hdf)
    st.dataframe(cpt_df, use_container_width=True)

    # SPLITS VS LHP / RHP (PA-BASED wOBA)
    st.subheader("Splits vs LHP / RHP")
    splits_df = hitter_splits(hdf)
    if splits_df.empty:
        st.info("No pitcher handedness data available.")
    else:
        st.dataframe(splits_df, use_container_width=True)

    # ZONE HEATMAPS
    st.subheader("Zone Heatmaps")

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

        fig4 = make_zone_heatmap(hdf, "AvgEV", "Avg EV Heatmap")
        if fig4:
            st.pyplot(fig4)

    st.subheader("Strike Zone 9-Box Breakdown")
    st.caption("Baseball Savant-style 3x3 map inside the strike zone only.")

    zone_hdf = hdf.copy()
    zone_pitch_options = ["All"]
    if "pitch_abbr" in zone_hdf.columns:
        zone_hdf["PitchGroup"] = combine_slider_sweeper(zone_hdf["pitch_abbr"])
        zone_pitch_options += sorted(zone_hdf["PitchGroup"].dropna().astype(str).unique())
    selected_zone_pitch = st.selectbox(
        "Pitch Type",
        zone_pitch_options,
        key=f"hitter_zone_pitch_{hitter}"
    )
    if selected_zone_pitch != "All":
        zone_hdf = zone_hdf[zone_hdf["PitchGroup"] == selected_zone_pitch].copy()

    zoneA, zoneB = st.columns(2)
    with zoneA:
        fig5 = make_savant_zone_heatmap(zone_hdf, "Swing%", "In-Zone Swing%", "Hitter decision map")
        if fig5:
            st.pyplot(fig5)

        fig6 = make_savant_zone_heatmap(zone_hdf, "Whiff%", "In-Zone Whiff%", "Whiffs per swing")
        if fig6:
            st.pyplot(fig6)

    with zoneB:
        fig7 = make_savant_zone_heatmap(zone_hdf, "HardHit%", "In-Zone HardHit%", "True BIP only")
        if fig7:
            st.pyplot(fig7)

        fig8 = make_savant_zone_heatmap(zone_hdf, "AvgEV", "In-Zone Avg EV", "True BIP only")
        if fig8:
            st.pyplot(fig8)

    # SPRAY PROFILE (GB/LD/FB + PULL/MID/OPPO)
    st.subheader("Spray Profile (GB/LD/FB + Pull/Mid/Oppo)")

    spray_df = hitter_spray_profile(hdf)
    if spray_df.empty:
        st.info("Not enough directional / EV / LA data for spray profile.")
    else:
        st.dataframe(spray_df, use_container_width=True)

    st.subheader("Best Defensive Positioning")
    pos_fig, pos_df = make_defensive_positioning_chart(hdf, hitter)
    if pos_fig is None:
        st.info("Not enough true batted-ball direction data for defensive positioning.")
    else:
        plt.close(pos_fig)
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

    # SEQUENCING
    st.subheader("Pitch-to-Pitch Sequencing (Hitter Reaction)")

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
        f"**Best damage sequence:** {best_row['prev_pitch']} -> {best_row['pitch_abbr']} "
        f"(wOBA {best_row['wOBA']:.3f}, HardHit% {best_row['HardHit%']}%, N={int(best_row['N'])})"
    )

    st.markdown(
        f"**Toughest sequence:** {worst_row['prev_pitch']} -> {worst_row['pitch_abbr']} "
        f"(Whiff% {worst_row['Whiff%']}%, N={int(worst_row['N'])})"
    )


def glossary_page():
    st.title("Advanced Stats Glossary")

    st.markdown("### Pitching Metrics")
    pitching_terms = pd.DataFrame([
        {"Stat": "Usage%", "What it means": "Share of pitches thrown as that pitch type.", "App logic": "Pitch type count divided by total pitches in the selected sample."},
        {"Stat": "Zone%", "What it means": "How often pitches land in the strike zone.", "App logic": "PlateLocSide from -0.83 to 0.83 and PlateLocHeight from 1.5 to 3.5."},
        {"Stat": "Strike%", "What it means": "Percent of pitches that count as strikes.", "App logic": "Called strikes, swinging strikes, fouls, balls in play, and strikeout pitch calls flagged as strikes."},
        {"Stat": "CSW%", "What it means": "Called strikes plus whiffs.", "App logic": "Called strike or swinging strike divided by total pitches."},
        {"Stat": "Whiff%", "What it means": "Misses per swing.", "App logic": "Swinging strikes divided by swings."},
        {"Stat": "Chase%", "What it means": "Swings outside the strike zone.", "App logic": "Swings on pitches outside the zone divided by swings."},
        {"Stat": "Stuff+", "What it means": "Pitch quality estimate based on raw pitch traits.", "App logic": "Uses velocity, movement, spin, extension, and release inputs in the app's Stuff+ model."},
        {"Stat": "Loc+", "What it means": "Command/location quality estimate.", "App logic": "Rewards competitive locations using count, zone, chase, called-strike, and damage signals."},
        {"Stat": "RelExt", "What it means": "Release extension in feet.", "App logic": "Average TrackMan Extension by pitch type."},
        {"Stat": "RelHt", "What it means": "Release height in feet.", "App logic": "Average TrackMan RelHeight by pitch type."},
    ])
    st.dataframe(pitching_terms, hide_index=True, use_container_width=True)

    st.markdown("### Hitting / Contact Metrics")
    hitting_terms = pd.DataFrame([
        {"Stat": "BA", "What it means": "Batting average.", "App logic": "Hits divided by at-bats. Walks, HBP, and sacrifice plays are removed from AB."},
        {"Stat": "OBP", "What it means": "On-base percentage.", "App logic": "(Hits + walks + HBP) divided by AB + walks + HBP + sacrifice plays."},
        {"Stat": "SLG", "What it means": "Slugging percentage.", "App logic": "Total bases divided by at-bats."},
        {"Stat": "OPS", "What it means": "OBP plus SLG.", "App logic": "OBP + SLG from PA-ending pitch rows."},
        {"Stat": "Avg EV", "What it means": "Average exit velocity.", "App logic": "Only PitchCall == InPlay with usable EV, excluding bunts; low tracking noise below 45 mph is filtered out."},
        {"Stat": "Max EV", "What it means": "Hardest tracked batted ball.", "App logic": "Maximum EV from the same true in-play contact pool."},
        {"Stat": "HardHit%", "What it means": "Share of hard contact.", "App logic": "Batted balls at 95 mph or harder divided by true in-play batted balls."},
        {"Stat": "Barrel%", "What it means": "High-value contact window.", "App logic": "EV at least 98 mph with launch angle from 26 to 30 degrees."},
        {"Stat": "SweetSpot%", "What it means": "Launch angles most likely to produce line drives and productive fly balls.", "App logic": "Launch angle from 8 to 32 degrees."},
        {"Stat": "wOBA", "What it means": "Weighted on-base average.", "App logic": "PA-ending events weighted as BB .69, HBP .72, 1B .88, 2B 1.247, 3B 1.578, HR 2.031."},
        {"Stat": "wRC+", "What it means": "Run creation relative to average.", "App logic": "Player wOBA divided by fixed league wOBA .315, scaled to 100."},
    ])
    st.dataframe(hitting_terms, hide_index=True, use_container_width=True)

    st.markdown("### Zone And Positioning Logic")
    zone_terms = pd.DataFrame([
        {"Area": "Strike Zone Heatmaps", "App logic": "Uses plate width from -0.83 to 0.83 feet and zone height from 1.5 to 3.5 feet."},
        {"Area": "9-Box Breakdown", "App logic": "Splits only the strike zone into equal 3-by-3 boxes, Baseball Savant style."},
        {"Area": "Pitch Type Dropdown", "App logic": "Filters the 9-box sample to all pitches or one selected pitch type before calculating the zone values."},
        {"Area": "Defensive Positioning", "App logic": "Uses true BIP direction, launch angle, EV, handedness, pull/middle/oppo rates, ground-ball rate, air rate, hard-hit rate, and bunt frequency."},
        {"Area": "Infield Alignment", "App logic": "Can recommend standard, pull-side shift, middle pinch, guard lines, or corners-in / 3B bunt alert."},
        {"Area": "Outfield Alignment", "App logic": "Shades toward the primary spray bucket and moves deeper when air contact plus hard-hit rate are elevated."},
    ])
    st.dataframe(zone_terms, hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    inject_fordham_theme(show_logo=True)
    st.markdown(
        """
        <div class="fordham-hero">
            <h1>Fordham Baseball Advanced Analytics</h1>
            <p>Pitching plans, hitter development, TrackMan contact quality, and game-report visuals in one staff dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load all processed pitch-by-pitch data ONCE
    all_pitches_df = prepare_data()

    page_options = {
        "Reports": ["Postgame Summary", "Season Summary", "Pitcher Profile"],
        "Leaderboards": ["Stuff+", "Location+", "Pitch-Type Grids", "Contact Quality"],
        "Development": ["Pitcher Advanced Info", "Hitter Advanced Info", "Umpire Scorecard"],
        "Scouting Zone": ["Player Reports"],
        "Glossary": ["Advanced Stats Glossary"],
    }

    st.markdown('<div class="nav-panel">', unsafe_allow_html=True)
    section = st.radio(
        "Section",
        list(page_options.keys()),
        horizontal=True,
        label_visibility="collapsed"
    )

    pages = page_options[section]
    page = pages[0] if len(pages) == 1 else st.radio(
        "Page",
        pages,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f'<div class="app-section-label">{section}</div>', unsafe_allow_html=True)

    if page == "Postgame Summary":
        postgame_page()
    elif page == "Season Summary":
        season_page()
    elif page == "Pitcher Profile":
        pitcher_profile_page()
    elif page == "Stuff+":
        stuff_leaderboard_page()
    elif page == "Location+":
        location_leaderboard_page()
    elif page == "Pitch-Type Grids":
        pitchtype_grids_page()
    elif page == "Contact Quality":
        contact_quality_leaderboard_page(all_pitches_df)
    elif page == "Pitcher Advanced Info":
        sequencing_page(all_pitches_df)
    elif page == "Hitter Advanced Info":
        hitter_development_page(all_pitches_df)
    elif page == "Umpire Scorecard":
        umpire_scorecard_page()
    elif page == "Player Reports":
        scouting_zone_page(all_pitches_df)
    else:
        glossary_page()


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if check_password():
    main()
else:
    st.stop()
