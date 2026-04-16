#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def main():
    st.set_page_config(page_title="Pitch-type Grids", layout="wide")

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


if __name__ == "__main__":
    main()

