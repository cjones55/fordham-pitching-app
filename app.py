#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:35:30 2026

@author: chrisjones
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Fordham Pitching Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='text-align:center; color:#A00000;'>Fordham Baseball – Pitching Analytics</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:white;'>Stuff+, Location+, leaderboards, and pitch-type grids in one place.</p>",
    unsafe_allow_html=True
)

st.sidebar.title("Navigation")
st.sidebar.write("Use the sidebar to switch pages.")

st.sidebar.markdown("---")
st.sidebar.write("Models expected in `models/`:")
st.sidebar.code(
    "stuff_lgbm_model.pkl\n"
    "stuff_lgbm_league.pkl\n"
    "location_lgbm_model.pkl\n"
    "location_lgbm_league.pkl"
)

st.markdown(
    """
    ### Pages
    - **Postgame Summary** – upload a single game CSV, pick a pitcher, get a full Stuff+ / Loc+ card.
    - **Season Summary** – upload multiple CSVs, aggregate by pitcher, generate season card.
    - **Stuff+ Leaderboard** – season Stuff+ leaderboard.
    - **Location+ Leaderboard** – season Location+ leaderboard.
    - **Pitch-type Grids** – Top-10 Stuff+ and Loc+ by pitch type.
    """,
    unsafe_allow_html=True
)
