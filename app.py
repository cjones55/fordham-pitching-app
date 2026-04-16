#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import sys
from pathlib import Path

# ============================================================
# GLOBAL PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fordham Pitching Analyzer",
    page_icon="⚾",
    layout="wide"
)

# ============================================================
# FIX PYTHONPATH FOR UTILS
# ============================================================
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

# ============================================================
# PASSWORD PROTECTION
# ============================================================
PASSWORD = "Baseball_1"

def check_password():
    st.sidebar.title("Login")
    pw = st.sidebar.text_input("Enter password", type="password")
    if pw == PASSWORD:
        return True
    elif pw:
        st.sidebar.error("Incorrect password")
    return False

# ============================================================
# PAGE 1 — POSTGAME SUMMARY
# ============================================================
def postgame_page():
    import streamlit as st
    import pandas as pd
    from io import BytesIO
    from pathlib import Path
    from utils.shared import (
        load_models, basic_clean, filter_fordham,
        add_flags, compute_stuffplus, compute_locationplus
    )
    from utils.plotting import postgame_or_season_card

    st.title("Postgame Summary – Stuff+ & Location+")

    uploaded = st.file_uploader("Upload single-game TrackMan CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded, encoding="latin1", sep=None, engine="python")
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

        title = f"{pitcher} – Postgame Summary"
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
            file_name=f"{pitcher.replace(',','')}_Postgame_Summary.png",
            mime="image/png"
        )
    else:
        st.info("Upload a single-game CSV to generate a postgame summary.")

# ============================================================
# PAGE 2 — SEASON SUMMARY
# ============================================================
def season_page():
    import streamlit as st
    import pandas as pd
    from io import BytesIO
    from pathlib import Path
    from utils.shared import (
        load_models, basic_clean, filter_fordham,
        add_flags, compute_stuffplus, compute_locationplus
    )
    from utils.plotting import postgame_or_season_card

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
            return

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

# ============================================================
# PAGE 3 — STUFF+ LEADERBOARD
# ============================================================
def stuff_leaderboard_page():
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO
    from utils.shared import (
        load_models, basic_clean, filter_fordham,
        compute_stuffplus
    )

    BACKGROUND = "#1e1e1e"
    MAROON = "#A00000"
    CREAM = "#F5F0E1"

    st.title("Stuff+ Leaderboard")

    uploads = st.file_uploader(
        "Upload season CSVs (multi-select)",
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
            return

        df = pd.concat(dfs, ignore_index=True)
        stuff_model, stuff_league, _, _ = load_models()

        df = basic_clean(df)
        df = filter_fordham(df)
        df = compute_stuffplus(df, stuff_model, stuff_league)

        agg = df.groupby("Pitcher").agg(
            Stuff_plus=("Stuff+","mean"),
            N=("Stuff+","count")
        ).reset_index()

        min_pitches = st.slider("Minimum pitches", 10, 200, 25, 5)
        agg = agg[agg["N"] >= min_pitches]
        agg = agg.sort_values("Stuff_plus", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 13))
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(BACKGROUND)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            0.5, 1.1,
            "Fordham Baseball — Total Stuff+",
            color=MAROON,
            fontsize=30,
            fontweight="bold",
            ha="center",
            va="top"
        )

        y_start = .95
        y_step = 0.052

        for i, row in enumerate(agg.itertuples()):
            y = y_start - i * y_step
            ax.text(0.12, y, row.Pitcher, color=CREAM, fontsize=19, va="center")
            ax.text(0.88, y, f"{round(row.Stuff_plus, 1)}",
                    color=MAROON, fontsize=19, va="center", ha="right")

        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
        buf.seek(0)

        st.download_button(
            label="Download PNG",
            data=buf,
            file_name="fordham_total_stuff_leaderboard.png",
            mime="image/png"
        )
    else:
        st.info("Upload season CSVs to build the Stuff+ leaderboard.")

# ============================================================
# PAGE 4 — LOCATION+ LEADERBOARD
# ============================================================
def location_leaderboard_page():
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO
    from utils.shared import (
        load_models, basic_clean, filter_fordham,
        compute_locationplus
    )

    BACKGROUND = "#1e1e1e"
    MAROON = "#A00000"
    CREAM = "#F5F0E1"

    st.title("Location+ Leaderboard")

    uploads = st.file_uploader(
        "Upload season CSVs (multi-select)",
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
            return

        df = pd.concat(dfs, ignore_index=True)
        _, _, loc_model, loc_league = load_models()

        df = basic_clean(df)
        df = filter_fordham(df)
        df = compute_locationplus(df, loc_model, loc_league)

        agg = df.groupby("Pitcher").agg(
            Loc_plus=("Loc+","mean"),
            N=("Loc+","count")
        ).reset_index()

        min_pitches = st.slider("Minimum pitches", 10, 200, 25, 5)
        agg = agg[agg["N"] >= min_pitches]
        agg = agg.sort_values("Loc_plus", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 13))
        fig.patch.set_facecolor(BACKGROUND)
        ax.set_facecolor(BACKGROUND)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            0.5, 1.1,
            "Fordham Baseball — Total Location+",
            color=MAROON,
            fontsize=30,
            fontweight="bold",
            ha="center",
            va="top"
        )

        y_start = .95
        y_step = 0.052

        for i, row in enumerate(agg.itertuples()):
            y = y_start - i * y_step
            ax.text(0.12, y, row.Pitcher, color=CREAM, fontsize=19, va="center")
            ax.text(0.88, y, f"{round(row.Loc_plus, 1)}",
                    color=MAROON, fontsize=19, va="center", ha="right")

        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
        buf.seek(0)

        st.download_button(
            label="Download PNG",
            data=buf,
            file_name="fordham_total_location_leaderboard.png",
            mime="image/png"
        )
    else:
        st.info("Upload season CSVs to build the Location+ leaderboard.")

# ============================================================
# PAGE 5 — PITCH-TYPE GRIDS
# ============================================================
def pitchtype_grids_page():
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO
    from pathlib import Path
    import numpy as np
    from utils.shared import (
        load_models, basic_clean, filter_fordham,
        compute_locationplus
    )

    BACKGROUND = "#1e1e1e"
    GOLD = "#A00000"
    WHITE = "white"

    pitch_colors = {
        "FB": "#1f77b4",
        "SI": "#17becf",
        "FC": "#ff7f0e",
        "SL": "#d62728",
        "CU": "#9467bd",
        "CH": "#2ca02c",
        "SW": "#8c564b"
    }

    st.title("Pitch-type Grids – Location+")

    uploads = st.file_uploader(
        "Upload season CSVs (multi-select)",
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
            return

        df = pd.concat(dfs, ignore_index=True)
        _, _, loc_model, loc_league = load_models()

        df = basic_clean(df)
        df = filter_fordham(df)
        df = compute_locationplus(df, loc_model, loc_league)

        agg = df.groupby(["Pitcher","pitch_abbr"]).agg(
            Loc_plus=("Loc+","mean"),
            N=("Loc+","count")
        ).reset_index()

        min_pitches = st.slider("Minimum pitches per pitcher/pitch type", 5, 50, 10, 5)
        agg = agg[agg["N"] >= min_pitches]

        pitch_types = sorted(agg["pitch_abbr"].unique())

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.patch.set_facecolor(BACKGROUND)
        axes = axes.flatten()

        pitch_types_extended = pitch_types + ["__LOGO__", "__EMPTY__"]
        pitch_types_extended = pitch_types_extended[:9]

        for ax, pitch in zip(axes, pitch_types_extended):
            ax.set_facecolor(BACKGROUND)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(0, 1)
            for spine in ax.spines.values():
                spine.set_visible(False)

            if pitch == "__LOGO__":
                logo_path = Path("assets") / "rams.png"
                if logo_path.exists():
                    ax.imshow(mpimg.imread(logo_path))
                ax.axis("off")
                continue

            if pitch == "__EMPTY__":
                ax.axis("off")
                continue

            sub = agg[agg["pitch_abbr"] == pitch].sort_values("Loc_plus", ascending=False).head(10)

            ax.text(
                0.05, 0.96,
                f"{pitch} – Top 10 Loc+",
                color=GOLD,
                fontsize=14,
                fontweight="bold",
                va="top"
            )

            y_start = 0.76
            y_step = 0.10

            for i, row in enumerate(sub.itertuples()):
                y = y_start - i * y_step
                ax.text(0.05, y, row.Pitcher, color=WHITE, fontsize=14, va="center")
                ax.text(
                    0.95, y,
                    f"{round(row.Loc_plus, 1)}",
                    color=pitch_colors.get(pitch, WHITE),
                    fontsize=14,
                    va="center",
                    ha="right"
                )

        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
        buf.seek(0)

        st.download_button(
            label="Download PNG",
            data=buf,
            file_name="fordham_pitchtype_location_grid.png",
            mime="image/png"
        )
    else:
        st.info("Upload season CSVs to build pitch-type grids.")

# ============================================================
# MAIN APP
# ============================================================
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

# ============================================================
# ENTRY POINT
# ============================================================
if check_password():
    main()
else:
    st.stop()
