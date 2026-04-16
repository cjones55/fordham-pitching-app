#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:36:44 2026

@author: chrisjones
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

HEADER_MAROON = "#A00000"
BACKGROUND_DARK = "#1e1e1e"

pitch_colors = {
    "FB": "#1f77b4",
    "SI": "#17becf",
    "FC": "#ff7f0e",
    "SL": "#d62728",
    "CU": "#9467bd",
    "CH": "#2ca02c",
    "SW": "#8c564b"
}

def style_axes(ax):
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("white")

def postgame_or_season_card(pdf, title, summary, logo_path: Path | None = None):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BACKGROUND_DARK)

    if logo_path and logo_path.exists():
        logo_img = mpimg.imread(logo_path)
        fig.figimage(logo_img, xo=40, yo=fig.bbox.ymax + 300, zorder=50, alpha=1.0)

    fig.suptitle(title, fontsize=26, fontweight="bold", color=HEADER_MAROON, y=0.97)
    plt.text(0.5, 0.93, summary, ha="center", va="center", color="white", fontsize=14)

    # Movement
    ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=2)
    style_axes(ax1)
    ax1.set_facecolor(BACKGROUND_DARK)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)

    throws = pdf["PitcherThrows"].iloc[0] if "PitcherThrows" in pdf.columns else "Right"
    if throws.upper().startswith("R"):
        arm_xmin, arm_xmax = 0, 25
        glove_xmin, glove_xmax = -25, 0
    else:
        arm_xmin, arm_xmax = -25, 0
        glove_xmin, glove_xmax = 0, 25

    ax1.axvspan(arm_xmin, arm_xmax, facecolor=(0.10, 0.30, 0.60, 0.10), zorder=0)
    ax1.axvspan(glove_xmin, glove_xmax, facecolor=(0.60, 0.10, 0.10, 0.10), zorder=0)
    ax1.axhline(0, color="white", linestyle=":", linewidth=1.2)
    ax1.axvline(0, color="white", linestyle=":", linewidth=1.2)

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax1.scatter(row["HB"], row["IVB"], s=40, color=c, edgecolor="white", linewidth=0.5)

    centroids = pdf.groupby("pitch_abbr")[["HB", "IVB"]].mean().reset_index()
    for _, row in centroids.iterrows():
        pitch = row["pitch_abbr"]
        c = pitch_colors.get(pitch, "white")
        ax1.scatter(row["HB"], row["IVB"], s=250, color=c, edgecolor="white", linewidth=1.5)
        ax1.text(row["HB"], row["IVB"], pitch, color="white", fontsize=9, weight="bold", ha="center")

    ax1.set_title("Movement", color="white")

    # Location plots
    def draw_mlb_zone(ax):
        ax.set_facecolor(BACKGROUND_DARK)
        style_axes(ax)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        zone_x = [-0.83, 0.83, 0.83, -0.83, -0.83]
        zone_y = [1.5, 1.5, 3.5, 3.5, 1.5]
        ax.plot(zone_x, zone_y, color="white", linewidth=2.5)
        ax.fill_between([-0.83, 0.83], 1.5, 3.5, color="white", alpha=0.06)

    def draw_home_plate(ax):
        plate_x = [-0.83, 0.83, 0.83, 0, -0.83, -0.83]
        plate_y = [0, 0, 0.17, 0.34, 0.17, 0]
        ax.plot(plate_x, plate_y, color="white", linewidth=2)
        ax.fill(plate_x, plate_y, color="white", alpha=0.10)

    axL = plt.subplot2grid((5, 4), (0, 1), rowspan=2)
    draw_mlb_zone(axL)
    draw_home_plate(axL)
    LHH = pdf[pdf["BatterSide"] == "Left"]
    for _, row in LHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axL.scatter(row["PlateLocSide"], row["PlateLocHeight"], s=85, color=c, edgecolor="white")
    axL.set_title("LHH", color="white")

    axR = plt.subplot2grid((5, 4), (0, 2), rowspan=2)
    draw_mlb_zone(axR)
    draw_home_plate(axR)
    RHH = pdf[pdf["BatterSide"] == "Right"]
    for _, row in RHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axR.scatter(row["PlateLocSide"], row["PlateLocHeight"], s=85, color=c, edgecolor="white")
    axR.set_title("RHH", color="white")

    # Release
    axRel = plt.subplot2grid((5, 4), (0, 3), rowspan=2)
    style_axes(axRel)
    axRel.set_facecolor(BACKGROUND_DARK)
    axRel.set_xlim(-4, 4)
    axRel.set_ylim(3, 7)
    axRel.set_aspect("equal", adjustable="box")
    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        axRel.scatter(row["RelS"], row["RelH"], s=25, color=c, edgecolor="white")
    axRel.set_title("Release", color="white")

    # Table
    axT = plt.subplot2grid((5, 4), (2, 0), colspan=4, rowspan=2)
    axT.axis("off")

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
    agg["Whiff%"] = np.where(
        agg["Swings"] > 0,
        (agg["Whiffs"] / agg["Swings"] * 100).round(1),
        0.0
    )
    agg["Strike%"] = (agg["Strikes"] / agg["N"] * 100).round(1)
    agg["Zone%"] = (agg["InZone"] / agg["N"] * 100).round(1)

    table_df = agg[[
        "Pitch","N","Usage%","Velo","IVB","HB",
        "Spin","Stuff+","Loc+","CSW%","Whiff%","Strike%","Zone%"
    ]].round(1)

    tbl = axT.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_MAROON)
            cell.set_text_props(color="white", weight="bold")
        else:
            pitch = table_df.iloc[r - 1]["Pitch"]
            bg = pitch_colors.get(pitch, BACKGROUND_DARK)
            cell.set_facecolor(bg)
            cell.set_text_props(color="white")

    # Footer
    axFooter = plt.subplot2grid((5, 4), (4, 0), colspan=4)
    axFooter.axis("off")
    axFooter.text(
        0.5, 0.55, summary,
        ha="center", va="center",
        fontsize=14, color="white", weight="bold"
    )

    return fig
