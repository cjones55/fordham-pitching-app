#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import importlib.util
import sys
from pathlib import Path

# ============================================================
# FIX: MAKE REPO + UTILS IMPORTABLE IN STREAMLIT CLOUD
# ============================================================

ROOT = Path(__file__).resolve().parent
UTILS = ROOT / "utils"

# Force Python to treat repo as a package
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UTILS))


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
# DYNAMIC PAGE LOADER (WORKS WITH FILENAMES STARTING WITH NUMBERS)
# ============================================================

def load_page(path, name):
    file_path = ROOT / path
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load your pages exactly as they exist
postgame = load_page("1_Postgame_Summary.py", "postgame")
season = load_page("2_Season_Summary.py", "season")
stuff_lb = load_page("3_Stuff_Leaderboard.py", "stuff_lb")
loc_lb = load_page("4_Location_Leaderboard.py", "loc_lb")
grids = load_page("5_Pitchtype_Grids.py", "grids")


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
