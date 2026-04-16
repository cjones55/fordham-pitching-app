#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def main():
    st.set_page_config(page_title="Stuff+ Leaderboard", layout="wide")

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


if __name__ == "__main__":
    main()
