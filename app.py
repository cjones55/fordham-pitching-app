#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st

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
# NORMAL IMPORTS (NOW THAT FILENAMES ARE VALID MODULES)
# ============================================================

import postgame_summary as postgame
import season_summary as season
import stuff_leaderboard as stuff_lb
import location_leaderboard as loc_lb
import pitchtype_grids as grids


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Fordham Pitching Analyzer",
        page_icon="⚾",
        layout="wide"
    )

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
        postgame.main()

    with tab2:
        season.main()

    with tab3:
        stuff_lb.main()

    with tab4:
        loc_lb.main()

    with tab5:
        grids.main()


# ============================================================
# ENTRY POINT
# ============================================================

if check_password():
    main()
else:
    st.stop()
