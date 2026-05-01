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
import ftplib
import tempfile
import re
import textwrap
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
DATA_DIR = ROOT / "data"
SCOUTING_DATA_DIR = ROOT / "scouting_2026_trackman"
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
            df = add_perceived_velocity(df)

            stuff_model, stuff_league, loc_model, loc_league = load_models()
            df = compute_stuffplus(df, stuff_model, stuff_league)
            df = compute_locationplus(df, loc_model, loc_league)

            processed.append(df)
        except:
            continue

    if not processed:
        return pd.DataFrame()

    return pd.concat(processed, ignore_index=True)


def get_scouting_csv_files():
    return sorted(SCOUTING_DATA_DIR.glob("*.csv"))


@st.cache_data(show_spinner=False)
def get_scouting_csv_count():
    return len(get_scouting_csv_files())


@st.cache_data(show_spinner=False)
def build_scouting_team_index():
    csvs = get_scouting_csv_files()
    rows = []
    team_cols = ["BatterTeam", "PitcherTeam"]

    for path in csvs:
        try:
            team_df = pd.read_csv(
                path,
                encoding="latin1",
                usecols=lambda col: col in team_cols,
                dtype=str,
                low_memory=False,
            )
        except Exception:
            continue

        entry = {"file": str(path), "BatterTeams": set(), "PitcherTeams": set()}
        if "BatterTeam" in team_df.columns:
            entry["BatterTeams"] = set(team_df["BatterTeam"].dropna().astype(str).str.strip())
        if "PitcherTeam" in team_df.columns:
            entry["PitcherTeams"] = set(team_df["PitcherTeam"].dropna().astype(str).str.strip())
        if entry["BatterTeams"] or entry["PitcherTeams"]:
            rows.append(entry)

    teams = sorted(set().union(*(r["BatterTeams"] | r["PitcherTeams"] for r in rows)) if rows else set())
    return rows, teams


def _scouting_files_for_team(team: str):
    index_rows, _ = build_scouting_team_index()
    team = str(team)
    files = [
        Path(row["file"])
        for row in index_rows
        if team in row["BatterTeams"] or team in row["PitcherTeams"]
    ]
    return sorted(files)


@st.cache_data(show_spinner=False)
def prepare_scouting_data(team=None):
    csvs = sorted(SCOUTING_DATA_DIR.glob("*.csv"))
    if not csvs:
        return prepare_data()

    if team:
        csvs = _scouting_files_for_team(team)
        if not csvs:
            return pd.DataFrame()

    processed = []
    stuff_model, stuff_league, loc_model, loc_league = load_models()
    for path in csvs:
        try:
            raw = pd.read_csv(path, encoding="latin1", sep=None, engine="python")
            if "Pitcher" not in raw.columns:
                continue
            if team and {"BatterTeam", "PitcherTeam"}.intersection(raw.columns):
                team_mask = pd.Series(False, index=raw.index)
                if "BatterTeam" in raw.columns:
                    team_mask |= raw["BatterTeam"].astype(str).str.strip().eq(str(team))
                if "PitcherTeam" in raw.columns:
                    team_mask |= raw["PitcherTeam"].astype(str).str.strip().eq(str(team))
                raw = raw[team_mask].copy()
                if raw.empty:
                    continue
            df = basic_clean(raw)
            df = add_flags(df)
            df = add_perceived_velocity(df)
            df = compute_stuffplus(df, stuff_model, stuff_league)
            df = compute_locationplus(df, loc_model, loc_league)
            processed.append(df)
        except Exception:
            continue

    if not processed:
        return pd.DataFrame() if team else prepare_data()
    return pd.concat(processed, ignore_index=True)


def should_import_trackman_game_csv(remote_path: str) -> bool:
    name = Path(str(remote_path)).name.lower()
    full = str(remote_path).lower()
    if not re.match(r"^2026\d{4}-.+-\d+\.csv$", name):
        return False
    if "_unverified" in name:
        return False
    excluded_terms = [
        "unverified", "practice", "playerpositioning", "positional", "position", "bullpen",
        "scrimmage", "intrasquad", "test"
    ]
    return not any(term in full for term in excluded_terms)


def validate_trackman_game_csv(local_path: Path) -> bool:
    try:
        sample = pd.read_csv(local_path, nrows=25, encoding="latin1", sep=None, engine="python")
    except Exception:
        return False

    required = {"Pitcher", "Batter", "PitchCall", "TaggedPitchType"}
    if not required.issubset(sample.columns):
        return False

    if "GameID" in sample.columns and sample["GameID"].dropna().astype(str).str.contains("practice", case=False).any():
        return False

    if "GameID" in sample.columns and sample["GameID"].dropna().astype(str).str.contains("2026").any():
        return True

    if "Date" in sample.columns:
        dates = pd.to_datetime(sample["Date"], errors="coerce")
        return bool(dates.dt.year.eq(2026).any())

    return True


def is_fordham_trackman_csv(local_path: Path, team_code="FOR_RAM") -> bool:
    try:
        sample = pd.read_csv(
            local_path,
            usecols=lambda col: col in {"PitcherTeam", "BatterTeam"},
            encoding="latin1",
            dtype=str,
            low_memory=False,
        )
    except Exception:
        return False
    if sample.empty:
        return False
    target = str(team_code).upper()
    teams = pd.concat([
        sample.get("PitcherTeam", pd.Series(dtype=str)),
        sample.get("BatterTeam", pd.Series(dtype=str)),
    ], ignore_index=True).dropna().astype(str).str.strip().str.upper()
    return teams.eq(target).any()


def copy_fordham_csv_to_data(local_path: Path, target_name: str):
    if not is_fordham_trackman_csv(local_path):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / Path(target_name).name
    target.write_bytes(local_path.read_bytes())
    return True


def _ftp_join(parent: str, child: str) -> str:
    parent = parent.rstrip("/")
    if not parent:
        return f"/{child.strip('/')}"
    return f"{parent}/{child.strip('/')}"


def _normalize_ftp_dir(path: str) -> str:
    path = str(path or "/").strip()
    if not path:
        return "/"
    return path if path.startswith("/") else f"/{path}"


def _walk_ftp_files(ftp, root_dir: str, recursive=True):
    stack = [root_dir or "/"]
    while stack:
        current = stack.pop()
        try:
            entries = list(ftp.mlsd(current))
            for name, facts in entries:
                if name in {".", ".."}:
                    continue
                path = _ftp_join(current, name)
                if facts.get("type") == "dir":
                    if recursive:
                        stack.append(path)
                elif facts.get("type") == "file":
                    yield path
        except Exception:
            old_dir = None
            try:
                old_dir = ftp.pwd()
                ftp.cwd(current)
                names = ftp.nlst()
            except Exception:
                names = []
            finally:
                if old_dir:
                    try:
                        ftp.cwd(old_dir)
                    except Exception:
                        pass

            for item in names:
                name = Path(item).name
                if name in {".", ".."}:
                    continue
                path = item if str(item).startswith("/") else _ftp_join(current, item)
                try:
                    old_dir = ftp.pwd()
                    ftp.cwd(path)
                    ftp.cwd(old_dir)
                    if recursive:
                        stack.append(path)
                except Exception:
                    yield path


def _candidate_ftp_roots(base_dir: str, months=None, day_filter="", csv_folder="CSV"):
    base = _normalize_ftp_dir(base_dir).rstrip("/")
    months = [str(m).zfill(2) for m in (months or []) if str(m).strip()]
    day_filter = str(day_filter or "").strip()
    csv_folder = str(csv_folder or "").strip().strip("/")

    def with_csv(path):
        if not csv_folder or path.rstrip("/").lower().endswith(f"/{csv_folder.lower()}"):
            return path
        return _ftp_join(path, csv_folder)

    roots = []
    if months:
        for month in months:
            month_root = _ftp_join(base, month)
            if day_filter:
                roots.append(with_csv(_ftp_join(month_root, day_filter.zfill(2))))
            else:
                for day in range(1, 32):
                    roots.append(with_csv(_ftp_join(month_root, f"{day:02d}")))
    else:
        roots.append(with_csv(base))
    return roots


def import_trackman_2026_from_ftp(
    host,
    username,
    password,
    remote_dir="/",
    port=21,
    use_tls=False,
    timeout=120,
    passive=True,
    recursive=True,
    max_downloads=None,
    months=None,
    day_filter="",
    csv_folder="CSV",
    skip_existing=True,
):
    SCOUTING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ftp_cls = ftplib.FTP_TLS if use_tls else ftplib.FTP
    imported = []
    skipped = []
    scanned = 0

    with ftp_cls() as ftp:
        ftp.connect(host=host, port=int(port), timeout=int(timeout))
        ftp.login(user=username, passwd=password)
        ftp.set_pasv(passive)
        if use_tls:
            ftp.prot_p()

        stop_import = False
        for root in _candidate_ftp_roots(remote_dir or "/", months=months, day_filter=day_filter, csv_folder=csv_folder):
            if stop_import:
                break
            for remote_path in _walk_ftp_files(ftp, root, recursive=recursive):
                scanned += 1
                if not should_import_trackman_game_csv(remote_path):
                    skipped.append((remote_path, "filtered"))
                    continue

                relative_key = remote_path.strip("/").replace("/", "__").replace(" ", "_")
                target = SCOUTING_DATA_DIR / relative_key
                if skip_existing and target.exists():
                    skipped.append((remote_path, "already imported"))
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp_path = Path(tmp.name)
                    try:
                        ftp.retrbinary(f"RETR {remote_path}", tmp.write)
                    except Exception as exc:
                        skipped.append((remote_path, f"download failed: {exc}"))
                        tmp_path.unlink(missing_ok=True)
                        continue

                if not validate_trackman_game_csv(tmp_path):
                    tmp_path.unlink(missing_ok=True)
                    skipped.append((remote_path, "not validated as 2026 game TrackMan CSV"))
                    continue

                target.write_bytes(tmp_path.read_bytes())
                copy_fordham_csv_to_data(tmp_path, target.name)
                tmp_path.unlink(missing_ok=True)
                imported.append(target.name)
                if max_downloads and len(imported) >= int(max_downloads):
                    stop_import = True
                    break

    get_scouting_csv_count.clear()
    build_scouting_team_index.clear()
    prepare_scouting_data.clear()
    return imported, skipped, scanned


def import_trackman_2026_from_sftp(
    host,
    username,
    password,
    remote_dir="/",
    port=22,
    timeout=120,
    max_downloads=None,
    months=None,
    day_filter="",
    csv_folder="CSV",
    skip_existing=True,
):
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("SFTP support requires paramiko. Install it with: pip install paramiko") from exc

    SCOUTING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    imported = []
    skipped = []
    scanned = 0

    transport = paramiko.Transport((host, int(port)))
    transport.banner_timeout = int(timeout)
    transport.auth_timeout = int(timeout)
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        stop_import = False
        for root in _candidate_ftp_roots(remote_dir or "/", months=months, day_filter=day_filter, csv_folder=csv_folder):
            if stop_import:
                break
            try:
                entries = sftp.listdir_attr(root)
            except Exception as exc:
                skipped.append((root, f"list failed: {exc}"))
                continue

            for entry in entries:
                remote_path = _ftp_join(root, entry.filename)
                scanned += 1
                if not should_import_trackman_game_csv(remote_path):
                    skipped.append((remote_path, "filtered"))
                    continue

                relative_key = remote_path.strip("/").replace("/", "__").replace(" ", "_")
                target = SCOUTING_DATA_DIR / relative_key
                if skip_existing and target.exists():
                    skipped.append((remote_path, "already imported"))
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    sftp.get(remote_path, str(tmp_path))
                except Exception as exc:
                    skipped.append((remote_path, f"download failed: {exc}"))
                    tmp_path.unlink(missing_ok=True)
                    continue

                if not validate_trackman_game_csv(tmp_path):
                    tmp_path.unlink(missing_ok=True)
                    skipped.append((remote_path, "not validated as 2026 game TrackMan CSV"))
                    continue

                target.write_bytes(tmp_path.read_bytes())
                copy_fordham_csv_to_data(tmp_path, target.name)
                tmp_path.unlink(missing_ok=True)
                imported.append(target.name)
                if max_downloads and len(imported) >= int(max_downloads):
                    stop_import = True
                    break
    finally:
        sftp.close()
        transport.close()

    get_scouting_csv_count.clear()
    build_scouting_team_index.clear()
    prepare_scouting_data.clear()
    return imported, skipped, scanned


def import_trackman_2026_from_server(protocol, **kwargs):
    if protocol == "SFTP":
        kwargs.pop("passive", None)
        kwargs.pop("recursive", None)
        return import_trackman_2026_from_sftp(**kwargs)
    kwargs["use_tls"] = protocol == "FTPS"
    return import_trackman_2026_from_ftp(**kwargs)

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

    BACKGROUND = "#100D0C"
    PANEL = "#181412"
    PANEL_ALT = "#211C1A"
    GRID = "#C7A45D"
    TEXT = "#FFF7E8"
    MUTED = "#CDBFAF"
    HEADER_MAROON = FORDHAM_MAROON

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
    swings = pdf["is_swing"].sum() if "is_swing" in pdf.columns else 0
    walks = pdf["KorBB"].eq("Walk").sum()
    strikeouts = pdf["KorBB"].eq("Strikeout").sum()
    hbp = pdf["PitchCall"].eq("HitByPitch").sum()
    hits = pdf["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()
    hr = pdf["PlayResult"].eq("HomeRun").sum()

    outs_on_play = pdf["OutsOnPlay"].sum() if "OutsOnPlay" in pdf.columns else 0
    total_outs = outs_on_play + strikeouts
    ip = total_outs // 3 + (total_outs % 3) / 10 if total_outs else 0.0

    strike_pct = round(pdf["is_strike"].mean() * 100, 1)
    zone_pct = round(pdf["in_zone"].mean() * 100, 1) if "in_zone" in pdf.columns else np.nan
    whiff_pct = round(whiffs / swings * 100, 1) if swings else 0.0
    stuff_avg = round(pdf["Stuff+"].mean(), 1) if "Stuff+" in pdf.columns and len(pdf) else np.nan
    loc_avg = round(pdf["Loc+"].mean(), 1) if "Loc+" in pdf.columns and len(pdf) else np.nan

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
        PerceivedVelo=("PerceivedVelo", "mean"),
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
    fig = plt.figure(figsize=(24, 12))
    fig.patch.set_facecolor(BACKGROUND)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.78, bottom=0.06, wspace=0.24, hspace=0.34)

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        height_ratios=[2.15, 1.0, 1.0],
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
        logo_ax = fig.add_axes([0.035, 0.855, 0.095, 0.105], zorder=50)
        logo_ax.set_facecolor(BACKGROUND)
        logo_ax.imshow(logo_img)
        logo_ax.set_xticks([])
        logo_ax.set_yticks([])
        for spine in logo_ax.spines.values():
            spine.set_visible(False)

    # -----------------------------
    # TITLE + SUMMARY
    # -----------------------------
    title = f"{pitcher}"
    subtitle = "Season Summary" if str(opponent).lower() == "season" else f"Fordham vs {opponent}"

    fig.text(0.5, 0.965, title, ha="center", va="center",
             fontsize=30, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.927, f"{subtitle} | {game_date}", ha="center", va="center",
             fontsize=15, color=MUTED, fontweight="bold")

    card_items = [
        ("Pitches", total_pitches),
        ("IP", f"{ip:.1f}"),
        ("H", hits),
        ("BB", walks),
        ("K", strikeouts),
        ("Whiff%", whiff_pct),
        ("Strike%", strike_pct),
        ("Zone%", zone_pct),
        ("Stuff+", stuff_avg),
        ("Loc+", loc_avg),
    ]
    start_x, card_w, gap = 0.155, 0.068, 0.009
    y0, h = 0.842, 0.055
    for i, (label, value) in enumerate(card_items):
        x = start_x + i * (card_w + gap)
        rect = plt.Rectangle((x, y0), card_w, h, transform=fig.transFigure,
                             facecolor=PANEL_ALT, edgecolor=GRID, linewidth=1.15, zorder=10)
        fig.add_artist(rect)
        fig.text(x + card_w / 2, y0 + 0.034, _fmt_pdf_value(value, label), ha="center", va="center",
                 fontsize=16, color=TEXT, fontweight="bold", zorder=20)
        fig.text(x + card_w / 2, y0 + 0.013, label, ha="center", va="center",
                 fontsize=8.5, color=MUTED, fontweight="bold", zorder=20)

    # -----------------------------
    # MOVEMENT
    # -----------------------------
    ax_move = fig.add_subplot(gs[0, 0])
    ax_move.set_facecolor(PANEL)
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

    ax_move.axhline(0, color=TEXT, linestyle=":", linewidth=1.2, alpha=0.72)
    ax_move.axvline(0, color=TEXT, linestyle=":", linewidth=1.2, alpha=0.72)

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_move.scatter(row["HB"], row["IVB"], s=55, color=c, edgecolor="white", linewidth=0.5)

    centroids = pdf.groupby("pitch_abbr")[["HB", "IVB"]].mean().reset_index()
    for _, row in centroids.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_move.scatter(row["HB"], row["IVB"], s=330, color=c, edgecolor="white", linewidth=1.5)
        ax_move.text(row["HB"], row["IVB"], row["pitch_abbr"],
                     color="white", fontsize=15, weight="bold", ha="center")

    ax_move.set_title("Movement Profile", color=TEXT, fontsize=18, weight="bold", pad=12)
    ax_move.set_xlabel("Horizontal Break", color=MUTED, fontsize=10, fontweight="bold")
    ax_move.set_ylabel("Induced Vertical Break", color=MUTED, fontsize=10, fontweight="bold")
    ax_move.tick_params(colors=MUTED)
    ax_move.grid(True, color=GRID, alpha=0.14, linewidth=0.8)
    for spine in ax_move.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(1.1)

    # -----------------------------
    # LHH (with HOME PLATE)
    # -----------------------------
    ax_lhh = fig.add_subplot(gs[0, 1])
    ax_lhh.set_facecolor(PANEL)
    ax_lhh.set_title(f"LHH Locations ({len(LHH_pdf)})", color=TEXT, fontsize=16, weight="bold", pad=12)
    ax_lhh.set_aspect(1.6)

    ax_lhh.set_xlim(-2.5, 2.5)
    ax_lhh.set_ylim(0, 5)

    zone_x = [-0.83, 0.83, 0.83, -0.83, -0.83]
    zone_y = [1.5, 1.5, 3.5, 3.5, 1.5]
    ax_lhh.plot(zone_x, zone_y, color=TEXT, linewidth=2.5)

    draw_home_plate(ax_lhh)

    LHH = pdf[pdf["BatterSide"] == "Left"]
    for _, row in LHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_lhh.scatter(row["PlateLocSide"], row["PlateLocHeight"],
                       s=140, color=c, edgecolor="white", linewidth=0.6)

    ax_lhh.tick_params(colors=MUTED, labelsize=11)
    ax_lhh.grid(True, color=GRID, alpha=0.10, linewidth=0.7)
    for spine in ax_lhh.spines.values():
        spine.set_color(GRID)

    # -----------------------------
    # RHH (with HOME PLATE)
    # -----------------------------
    ax_rhh = fig.add_subplot(gs[0, 2])
    ax_rhh.set_facecolor(PANEL)
    ax_rhh.set_title(f"RHH Locations ({len(RHH_pdf)})", color=TEXT, fontsize=16, weight="bold", pad=12)
    ax_rhh.set_aspect(1.6)

    ax_rhh.set_xlim(-2.5, 2.5)
    ax_rhh.set_ylim(0, 5)
    ax_rhh.plot(zone_x, zone_y, color=TEXT, linewidth=2.5)

    draw_home_plate(ax_rhh)

    RHH = pdf[pdf["BatterSide"] == "Right"]
    for _, row in RHH.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_rhh.scatter(row["PlateLocSide"], row["PlateLocHeight"],
                       s=140, color=c, edgecolor="white", linewidth=0.6)

    ax_rhh.tick_params(colors=MUTED, labelsize=11)
    ax_rhh.grid(True, color=GRID, alpha=0.10, linewidth=0.7)
    for spine in ax_rhh.spines.values():
        spine.set_color(GRID)

    # -----------------------------
    # RELEASE
    # -----------------------------
    ax_rel = fig.add_subplot(gs[0, 3])
    ax_rel.set_facecolor(PANEL)
    ax_rel.set_title("Release Window", color=TEXT, fontsize=16, weight="bold", pad=12)

    ax_rel.set_aspect(1.4)

    ax_rel.set_xlim(-3.2, 3.2)
    ax_rel.set_ylim(3.2, 6.8)

    ax_rel.axhline(np.mean(pdf["RelH"]), color=TEXT, linestyle=":", linewidth=1.2, alpha=0.72)
    ax_rel.axvline(np.mean(pdf["RelS"]), color=TEXT, linestyle=":", linewidth=1.2, alpha=0.72)

    for _, row in pdf.iterrows():
        c = pitch_colors.get(row["pitch_abbr"], "white")
        ax_rel.scatter(row["RelS"], row["RelH"], s=40, color=c, edgecolor="white", linewidth=0.6)

    ax_rel.set_xlabel("Release Side", color=MUTED, fontsize=9, fontweight="bold")
    ax_rel.set_ylabel("Release Height", color=MUTED, fontsize=9, fontweight="bold")
    ax_rel.tick_params(colors=MUTED, labelsize=10)
    ax_rel.grid(True, color=GRID, alpha=0.10, linewidth=0.7)
    for spine in ax_rel.spines.values():
        spine.set_color(GRID)

    # -----------------------------
    # TABLE
    # -----------------------------
    ax_table = fig.add_subplot(gs[1:, :])
    ax_table.axis("off")

    table_df = agg[[
        "Pitch","N","Usage%","Velo","PerceivedVelo","IVB","HB",
        "Spin","Stuff+","Loc+","CSW%","Whiff%","Strike%","Zone%","Ext","RelH"
    ]].rename(columns={"Ext": "RelExt", "RelH": "RelHt", "PerceivedVelo": "PerVelo"})
    table_display = table_df.copy()
    for col in table_display.columns:
        table_display[col] = table_display[col].map(lambda value, c=col: _fmt_pdf_value(value, c))

    tbl = ax_table.table(
        cellText=table_display.values,
        colLabels=table_display.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0.08, 1, 0.92]
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    cell_width = 1.0 / len(table_df.columns)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(0.042)
        cell.set_width(cell_width)

        if r == 0:
            cell.set_facecolor(HEADER_MAROON)
            cell.set_text_props(color=TEXT, weight="bold")
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.9)
        else:
            pitch = table_df.iloc[r - 1]["Pitch"]
            if c == 0:
                bg = pitch_colors.get(pitch, PANEL_ALT)
                cell.set_facecolor(bg)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(PANEL_ALT if r % 2 else PANEL)
                cell.set_text_props(color=TEXT, weight="bold")
            cell.set_edgecolor("#4E4036")
            cell.set_linewidth(0.7)

    # -----------------------------
    # FOOTER
    # -----------------------------
    fig.text(
        0.98, 0.03,
        f"Fordham Baseball Analytics | {game_date}",
        ha="right", va="center",
        fontsize=12, color=MUTED
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


def build_fastball_perceived_velocity_figure(pitcher_df):
    df = add_perceived_velocity(pitcher_df)
    if "pitch_abbr" not in df.columns:
        df["pitch_abbr"] = ""
    df = df[is_fastball_pitch(df["pitch_abbr"])].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    fig.patch.set_facecolor("#FFFDF8")
    if df.empty or "PerceivedVelo" not in df.columns:
        ax.text(0.5, 0.5, "No fastball perceived velocity data", ha="center", va="center")
        ax.set_axis_off()
        return fig
    df["PerceivedVelo"] = pd.to_numeric(df["PerceivedVelo"], errors="coerce")
    df["Velo"] = pd.to_numeric(df["Velo"], errors="coerce")
    df = df.dropna(subset=["PerceivedVelo", "Velo"]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "No fastball perceived velocity data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    sort_cols = [c for c in ["GameDate", "Date", "Inning", "PAofInning", "PitchofPA", "PitchNo", "PitchNumber"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    df["PitchIndex"] = np.arange(1, len(df) + 1)

    style_fordham_axes(ax, "Fastball Perceived Velocity")
    ax.scatter(df["PitchIndex"], df["Velo"], s=36, color="#244B7A", alpha=0.70, edgecolor="white", linewidth=0.5, label="Actual Velo")
    ax.scatter(df["PitchIndex"], df["PerceivedVelo"], s=46, color=FORDHAM_MAROON, alpha=0.86, edgecolor="white", linewidth=0.7, label="Perceived Velo")
    if len(df) >= 3:
        ax.plot(df["PitchIndex"], df["PerceivedVelo"].rolling(5, min_periods=1).mean(), color=FORDHAM_GOLD, linewidth=2, label="PV rolling avg")
    ax.axhline(df["PerceivedVelo"].mean(), color=FORDHAM_MAROON, linestyle=":", linewidth=1.8)
    ax.set_xlabel("Fastball #")
    ax.set_ylabel("MPH")
    ax.legend(loc="best", fontsize=8)
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

    st.markdown("---")

    # -----------------------------
    # FASTBALL PERCEIVED VELOCITY
    # -----------------------------
    st.subheader("Fastball Perceived Velocity")
    st.caption(
        f"Estimated by flight distance: fastball velo x "
        f"((60.5 - {PERCEIVED_VELO_EXT_BASELINE:.1f}) / (60.5 - extension)), "
        f"then adjusted slightly for fastball IVB and spin."
    )
    st.pyplot(build_fastball_perceived_velocity_figure(pitcher_df))


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

COLLEGE_AVG_WOBA = 0.320
BARREL_EV_MIN = 92
BARREL_LA_MIN = 16
BARREL_LA_MAX = 36
PERCEIVED_VELO_EXT_BASELINE = 6.0
PERCEIVED_VELO_PLATE_DISTANCE = 60.5
PERCEIVED_VELO_IVB_BASELINE = 16.0
PERCEIVED_VELO_SPIN_BASELINE = 2300.0
PERCEIVED_VELO_IVB_WEIGHT = 0.08
PERCEIVED_VELO_SPIN_WEIGHT = 0.04
PERCEIVED_VELO_SHAPE_CAP = 1.8


def barrel_mask(ev, la):
    ev = pd.to_numeric(ev, errors="coerce")
    la = pd.to_numeric(la, errors="coerce")
    return ev.ge(BARREL_EV_MIN) & la.between(BARREL_LA_MIN, BARREL_LA_MAX)


def is_fastball_pitch(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(["FB", "FF", "FA", "Fastball", "FourSeamFastBall", "FourSeamFastball", "Four-Seam"])


def add_perceived_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Velo" not in df.columns and "RelSpeed" in df.columns:
        df["Velo"] = df["RelSpeed"]
    if "Ext" not in df.columns and "Extension" in df.columns:
        df["Ext"] = df["Extension"]
    if "IVB" not in df.columns and "InducedVertBreak" in df.columns:
        df["IVB"] = df["InducedVertBreak"]
    if "Spin" not in df.columns and "SpinRate" in df.columns:
        df["Spin"] = df["SpinRate"]
    if "Velo" not in df.columns or "Ext" not in df.columns:
        df["PerceivedVelo"] = np.nan
        return df
    velo = pd.to_numeric(df["Velo"], errors="coerce")
    ext = pd.to_numeric(df["Ext"], errors="coerce")
    baseline_flight_distance = PERCEIVED_VELO_PLATE_DISTANCE - PERCEIVED_VELO_EXT_BASELINE
    actual_flight_distance = (PERCEIVED_VELO_PLATE_DISTANCE - ext).clip(lower=50.0, upper=58.5)
    extension_adjusted = velo * (baseline_flight_distance / actual_flight_distance)

    ivb = pd.to_numeric(df.get("IVB", np.nan), errors="coerce")
    spin = pd.to_numeric(df.get("Spin", np.nan), errors="coerce")
    shape_adjustment = (
        (ivb - PERCEIVED_VELO_IVB_BASELINE).fillna(0) * PERCEIVED_VELO_IVB_WEIGHT
        + ((spin - PERCEIVED_VELO_SPIN_BASELINE).fillna(0) / 100) * PERCEIVED_VELO_SPIN_WEIGHT
    ).clip(lower=-PERCEIVED_VELO_SHAPE_CAP, upper=PERCEIVED_VELO_SHAPE_CAP)
    perceived = extension_adjusted + shape_adjustment
    if "pitch_abbr" in df.columns:
        perceived = perceived.where(is_fastball_pitch(df["pitch_abbr"]))
    df["PerceivedVelo"] = perceived
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
    Fixed league wOBA so all tabs scale identically.
    """
    return COLLEGE_AVG_WOBA


def compute_wrc_plus(player_woba: float, league_woba: float = COLLEGE_AVG_WOBA) -> int:
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

    df["hard_hit"] = np.where(df["EV"].fillna(0) >= 95, 1, 0)
    df["barrel"] = barrel_mask(df["EV"], df["LA"]).astype(int)
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

    df["hard_hit"] = (df["EV"] >= 95).astype(int)
    df["barrel"] = barrel_mask(df["EV"], df["LA"]).astype(int)
    df["sweet_spot"] = df["LA"].between(7, 32).astype(int)

    if "is_swing" not in df.columns:
        df["is_swing"] = 0
    if "is_whiff" not in df.columns:
        df["is_whiff"] = 0
    if "is_chase" not in df.columns:
        in_zone_bool = (
            df["PlateLocSide"].between(-0.83, 0.83) &
            df["PlateLocHeight"].between(1.5, 3.5)
        )
        df["is_chase"] = ((df["is_swing"] == 1) & (~in_zone_bool)).astype(int)
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

    elif metric == "Chase%":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        pitches = grouped["is_chase"].count()
        chases = grouped["is_chase"].sum()
        values = chases / pitches.replace(0, np.nan) * 100
        samples = pitches.reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "Chase%"
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
    if "Distance" in out.columns:
        out["Distance"] = pd.to_numeric(out["Distance"], errors="coerce")
    if "LastTrackedDistance" in out.columns:
        out["LastTrackedDistance"] = pd.to_numeric(out["LastTrackedDistance"], errors="coerce")
    if "Bearing" in out.columns:
        out["Bearing"] = pd.to_numeric(out["Bearing"], errors="coerce")
    out["hard_hit"] = (out["EV"] >= 95).astype(int)
    out["barrel"] = barrel_mask(out["EV"], out["LA"]).astype(int)
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
    st.dataframe(style_scouting_dataframe(count_df, context="hitting"), use_container_width=True)

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


def hitter_shift_recommendations(hdf: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    if not {"Direction", "BatterSide", "EV", "LA"}.issubset(hdf.columns):
        return pd.DataFrame(), ["Not enough batted-ball direction data for a shift recommendation."]

    df = get_true_bip_with_ev(hdf)
    if df.empty:
        return pd.DataFrame(), ["Not enough true batted-ball data for a shift recommendation."]

    df = df.copy()
    df["Direction"] = pd.to_numeric(df["Direction"], errors="coerce")
    df["LA"] = pd.to_numeric(df["LA"], errors="coerce")
    df = df.dropna(subset=["Direction", "EV", "LA"])
    if df.empty:
        return pd.DataFrame(), ["Not enough tracked direction, EV, and LA data for a shift recommendation."]

    side_raw = str(df.get("BatterSide", pd.Series(["Unknown"])).dropna().mode().iloc[0]).upper()
    hitter_side = "RHH" if side_raw.startswith("R") else "LHH" if side_raw.startswith("L") else "Unknown"

    def classify(row):
        if row["Direction"] <= -15:
            field = "LF"
        elif row["Direction"] >= 15:
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
            spray = "Middle"

        if row["LA"] < 8:
            contact = "GB"
        elif row["LA"] <= 27:
            contact = "LD"
        else:
            contact = "Air"
        return pd.Series({"Spray": spray, "Contact": contact})

    df = pd.concat([df, df.apply(classify, axis=1)], axis=1)
    total = len(df)
    rows = []
    for bucket in ["Pull", "Middle", "Oppo"]:
        g = df[df["Spray"] == bucket]
        rows.append({
            "Spray": bucket,
            "BIP": len(g),
            "BIP%": round(len(g) / total * 100, 1) if total else 0,
            "GB%": round(g["Contact"].eq("GB").mean() * 100, 1) if len(g) else 0,
            "Air%": round(g["Contact"].eq("Air").mean() * 100, 1) if len(g) else 0,
            "HH%": round(g["EV"].ge(95).mean() * 100, 1) if len(g) else 0,
            "AvgEV": round(g["EV"].mean(), 1) if len(g) else np.nan,
        })
    summary = pd.DataFrame(rows)

    ground = df[df["Contact"].eq("GB")]
    gb_rate = df["Contact"].eq("GB").mean() * 100
    air_rate = df["Contact"].eq("Air").mean() * 100
    hard_rate = df["EV"].ge(95).mean() * 100
    pull_rate = float(summary.loc[summary["Spray"].eq("Pull"), "BIP%"].iloc[0])
    middle_rate = float(summary.loc[summary["Spray"].eq("Middle"), "BIP%"].iloc[0])
    oppo_rate = float(summary.loc[summary["Spray"].eq("Oppo"), "BIP%"].iloc[0])
    pull_gb = ground["Spray"].eq("Pull").mean() * 100 if len(ground) else 0
    middle_gb = ground["Spray"].eq("Middle").mean() * 100 if len(ground) else 0
    oppo_air = df[df["Contact"].isin(["LD", "Air"])]["Spray"].eq("Oppo").mean() * 100 if len(df[df["Contact"].isin(["LD", "Air"])]) else 0

    raw = hdf.copy()
    tagged_hit = raw.get("TaggedHitType", pd.Series("", index=raw.index)).astype(str)
    play_result = raw.get("PlayResult", pd.Series("", index=raw.index)).astype(str)
    bunt_mask = (
        tagged_hit.str.contains("Bunt", case=False, na=False) |
        play_result.str.contains("Bunt", case=False, na=False) |
        play_result.eq("Sacrifice")
    )
    bunt_rate = bunt_mask.sum() / max(len(get_pa_endings(raw)), 1) * 100

    if pull_gb >= 45 and gb_rate >= 45:
        infield = "Pull-side infield shift"
        if hitter_side == "RHH":
            infield_detail = "3B protects line, SS deeper pull-side, 2B shades middle."
        elif hitter_side == "LHH":
            infield_detail = "1B protects line, 2B deeper pull-side, SS shades middle."
        else:
            infield_detail = "Overload pull-side ground-ball lanes."
    elif middle_gb >= 35 and gb_rate >= 45:
        infield = "Middle pinch"
        infield_detail = "SS and 2B tighten toward the middle; protect back-side single lanes."
    elif bunt_rate >= 8:
        infield = "Corners in / 3B bunt alert"
        infield_detail = "3B can play bunt depth; 1B holds ready for push bunt or slash."
    elif hard_rate >= 35 and max(pull_rate, oppo_rate) >= 35:
        infield = "Guard lines"
        infield_detail = "Corners protect extra-base contact; middle stays balanced."
    else:
        infield = "Standard infield"
        infield_detail = "No extreme ground-ball shift signal; play straight with normal depth."

    primary_spray = summary.sort_values(["BIP", "HH%"], ascending=False).iloc[0]
    if primary_spray["Spray"] == "Pull" and hitter_side == "RHH":
        outfield = "Shade LF / left-center"
    elif primary_spray["Spray"] == "Pull" and hitter_side == "LHH":
        outfield = "Shade RF / right-center"
    elif primary_spray["Spray"] == "Oppo" and hitter_side == "RHH":
        outfield = "Respect RF / right-center"
    elif primary_spray["Spray"] == "Oppo" and hitter_side == "LHH":
        outfield = "Respect LF / left-center"
    else:
        outfield = "Straight up / center-heavy"

    depth = "No-doubles depth" if air_rate >= 45 and hard_rate >= 35 else "Normal depth"
    if gb_rate >= 50:
        depth = "Normal OF depth; prioritize infield ground-ball lanes"

    notes = [
        f"Infield: {infield}. {infield_detail}",
        f"Outfield: {outfield}; {depth}.",
        f"Primary spray is {primary_spray['Spray']} ({primary_spray['BIP%']}% BIP, {primary_spray['HH%']}% HH).",
        f"Ground-ball read: {gb_rate:.1f}% GB, {pull_gb:.1f}% of grounders pull-side, {middle_gb:.1f}% through middle.",
        f"Oppo air read: {oppo_air:.1f}% of air/line contact goes opposite field.",
    ]
    if bunt_rate >= 5:
        notes.append(f"Bunt/slash alert: bunt indicators show {bunt_rate:.1f}% of PA.")

    summary["Shift Read"] = summary["Spray"].map({
        "Pull": "Shift side" if pull_rate >= max(middle_rate, oppo_rate) else "Secondary",
        "Middle": "Pinch middle" if middle_gb >= 35 else "Standard",
        "Oppo": "Respect oppo air" if oppo_air >= 20 or oppo_rate >= 30 else "Standard",
    })
    return summary, notes


def build_hitter_spray_chart(hdf: pd.DataFrame, hitter: str = "Hitter"):
    required = {"Direction", "BatterSide", "EV", "LA"}
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    fig.patch.set_facecolor("#100D0C")
    ax.set_facecolor("#16391F")
    ax.set_aspect("equal")
    ax.axis("off")

    if not required.issubset(hdf.columns):
        ax.text(0, 1.5, "No spray data", ha="center", va="center", color="#FFF7E8", fontsize=14)
        return fig

    df = get_true_bip_with_ev(hdf)
    if df.empty:
        ax.text(0, 1.5, "No spray data", ha="center", va="center", color="#FFF7E8", fontsize=14)
        return fig

    df = df.copy()
    df["Direction"] = pd.to_numeric(df["Direction"], errors="coerce")
    df["LA"] = pd.to_numeric(df["LA"], errors="coerce")
    df = df.dropna(subset=["Direction", "EV", "LA"])
    if df.empty:
        ax.text(0, 1.5, "No spray data", ha="center", va="center", color="#FFF7E8", fontsize=14)
        return fig

    side_raw = str(df.get("BatterSide", pd.Series(["Unknown"])).dropna().mode().iloc[0]).upper()
    hitter_side = "RHH" if side_raw.startswith("R") else "LHH" if side_raw.startswith("L") else "Unknown"

    feet_scale = 2.78 / 395
    base_path = 90 * feet_scale
    second_base = np.sqrt(2 * 90**2) * feet_scale
    base_xy = base_path / np.sqrt(2)
    infield = np.array([[0, 0], [base_xy, base_xy], [0, second_base], [-base_xy, base_xy], [0, 0]])
    ax.fill(infield[:, 0], infield[:, 1], color="#A66B35", alpha=0.78, zorder=1)
    ax.plot(infield[:, 0], infield[:, 1], color="#E5C28A", linewidth=2.2, zorder=2)

    spray_dirs = np.linspace(-45, 45, 160)
    fence_dist = np.interp(spray_dirs, [-45, 0, 45], [338, 395, 320])
    fence_r = fence_dist * feet_scale
    fence_x = fence_r * np.sin(np.deg2rad(spray_dirs))
    fence_y = fence_r * np.cos(np.deg2rad(spray_dirs)) - 0.18
    ax.plot(fence_x, fence_y, color="#C7A45D", linewidth=3, zorder=2)
    for direction, label, dist in [(-45, "LF 338", 338), (0, "CF 395", 395), (45, "RF 320", 320)]:
        r = dist * feet_scale
        x = r * np.sin(np.deg2rad(direction))
        y = r * np.cos(np.deg2rad(direction)) - 0.18
        ax.plot([0, x], [0, y], color="#E7D3A4", alpha=0.34, linewidth=1.2)
        label_r = r + 0.16
        ax.text(label_r * np.sin(np.deg2rad(direction)), label_r * np.cos(np.deg2rad(direction)) - 0.18, label, color="#FFF7E8", fontsize=10, fontweight="bold", ha="center")

    ax.text(-1.55, 1.08, "PULL" if hitter_side == "RHH" else "OPPO", color="#FFF7E8", alpha=0.72, fontsize=10, fontweight="bold", ha="center")
    ax.text(0, 2.30, "MIDDLE", color="#FFF7E8", alpha=0.72, fontsize=10, fontweight="bold", ha="center")
    ax.text(1.55, 1.08, "OPPO" if hitter_side == "RHH" else "PULL", color="#FFF7E8", alpha=0.72, fontsize=10, fontweight="bold", ha="center")
    for lane_direction, lane_label in [(-28, "5-6"), (0, "MIF"), (28, "3-4")]:
        lane_r = 145 * feet_scale
        lane_x = lane_r * np.sin(np.deg2rad(lane_direction))
        lane_y = lane_r * np.cos(np.deg2rad(lane_direction)) - 0.05
        ax.plot([0, lane_x], [0, lane_y], color="#FFF7E8", alpha=0.13, linewidth=0.9, linestyle="--", zorder=2)
        ax.text(lane_x, lane_y + 0.04, lane_label, color="#FFF7E8", alpha=0.62, fontsize=8.5, fontweight="bold", ha="center")

    def fallback_distance(ev, la):
        ev = float(ev)
        la = float(la)
        if la < 5:
            return np.clip(8 + (ev - 45) * 2.3, 5, 165)
        launch_quality = np.exp(-((la - 24) / 22) ** 2)
        return np.clip(25 + (ev - 50) * 5.2 * launch_quality, 20, 390)

    def plot_point(row):
        direction = float(row["Bearing"]) if pd.notna(row.get("Bearing", np.nan)) else float(row["Direction"])
        plot_direction = float(np.clip(direction, -45, 45))
        distance = row.get("Distance", np.nan)
        if pd.isna(distance) or float(distance) <= 0:
            distance = row.get("LastTrackedDistance", np.nan)
        if pd.isna(distance) or float(distance) <= 0:
            distance = fallback_distance(row["EV"], row["LA"])
        radius = np.clip(float(distance), 3, 410) * feet_scale
        x = radius * np.sin(np.deg2rad(plot_direction))
        y = radius * np.cos(np.deg2rad(plot_direction)) - 0.08
        la = float(row["LA"])
        if la < 8:
            marker, color = "o", "#E7C66A"
            line_width = 0.9 + min(max(float(row["EV"]) - 75, 0), 25) / 25 * 1.0
            guide_radius = max(radius, 135 * feet_scale)
            guide_x = guide_radius * np.sin(np.deg2rad(plot_direction))
            guide_y = guide_radius * np.cos(np.deg2rad(plot_direction)) - 0.08
            ax.plot(
                [0, guide_x], [0, guide_y],
                color="#F2D37A",
                linewidth=line_width,
                alpha=0.38,
                solid_capstyle="round",
                zorder=3,
            )
            if radius < guide_radius:
                ax.scatter(guide_x, guide_y, s=20, marker="x", color="#F2D37A", linewidth=0.8, alpha=0.48, zorder=4)
        elif la <= 27:
            marker, color = "D", "#F04E45"
        else:
            marker, color = "^", "#8EC5FF"
        size = 38 + max(float(row["EV"]) - 80, 0) * 3.2
        edge = "#FFFFFF" if row["EV"] >= 95 else "#1A1412"
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor=edge, linewidth=0.8, alpha=0.88, zorder=5)

    df.apply(plot_point, axis=1)

    ax.scatter([], [], marker="o", color="#E7C66A", label="Ground ball")
    ax.scatter([], [], marker="D", color="#F04E45", label="Line drive")
    ax.scatter([], [], marker="^", color="#8EC5FF", label="Air ball")
    ax.scatter([], [], marker="o", color="#222222", edgecolor="#FFFFFF", label="95+ EV")
    ax.legend(loc="lower right", fontsize=8, facecolor="#211C1A", edgecolor="#C7A45D", labelcolor="#FFF7E8")

    ax.text(0, -0.22, "HOME", ha="center", va="center", color="#FFF7E8", fontsize=9, fontweight="bold")
    ax.set_title(f"Spray Chart - {hitter} ({hitter_side})", color="#FFF7E8", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-0.38, 3.02)
    fig.tight_layout()
    return fig


def _compact_spray_pdf_tables(spray_table, shift_table):
    spray_view = pd.DataFrame() if spray_table is None else spray_table.copy()
    shift_view = pd.DataFrame() if shift_table is None else shift_table.copy()

    if not spray_view.empty:
        spray_cols = [c for c in ["Spray", "BIP", "GB%", "LD%", "FB%", "HH%"] if c in spray_view.columns]
        spray_view = spray_view[spray_cols].rename(columns={
            "Spray": "Zone",
            "GB%": "GB",
            "LD%": "LD",
            "FB%": "FB",
            "HH%": "HH",
        })

    if not shift_view.empty:
        shift_cols = [c for c in ["Spray", "BIP%", "GB%", "Air%", "Shift Read"] if c in shift_view.columns]
        shift_view = shift_view[shift_cols].rename(columns={
            "Spray": "Zone",
            "BIP%": "BIP",
            "GB%": "GB",
            "Air%": "Air",
            "Shift Read": "Read",
        })
        if "Read" in shift_view.columns:
            shift_view["Read"] = shift_view["Read"].replace({
                "Shift side": "Shift",
                "Pinch middle": "Pinch",
                "Respect oppo air": "Oppo Air",
            })

    return spray_view, shift_view


def _append_hitter_spray_shift_page(pdf, hdf, spray_table):
    shift_table, _ = hitter_shift_recommendations(hdf)
    spray_pdf_table, shift_pdf_table = _compact_spray_pdf_tables(spray_table, shift_table)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    gs = fig.add_gridspec(
        2, 3,
        left=0.035, right=0.97, top=0.93, bottom=0.06,
        hspace=0.24, wspace=0.14,
        width_ratios=[1.38, 1.38, 1.12]
    )
    spray_img = _fig_to_image(build_hitter_spray_chart(hdf, ""))
    ax_chart = fig.add_subplot(gs[:, 0:2])
    ax_chart.imshow(spray_img)
    ax_chart.axis("off")
    _add_report_table(fig.add_subplot(gs[0, 2]), spray_pdf_table, "Spray Contact", max_rows=8, font_size=7.4, context="hitting")
    _add_report_table(fig.add_subplot(gs[1, 2]), shift_pdf_table, "Shift Reads", max_rows=8, font_size=7.4, context="hitting")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


RATE_3_DECIMAL_COLS = {
    "BA", "OBP", "SLG", "OPS", "wOBA",
    "BA Allowed", "OBP Allowed", "SLG Allowed", "OPS Allowed"
}
INTEGER_COLS = {
    "N", "PA", "AB", "H", "BB", "K", "BIP", "Swings", "Whiffs", "Chases",
    "Strikes", "InZone", "Zone", "CSW", "Pitches"
}


def _fmt_pdf_value(value, col=None):
    if pd.isna(value):
        return "-"
    col_name = str(col or "")
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if col_name in INTEGER_COLS:
            return f"{int(round(val))}"
        if col_name in RATE_3_DECIMAL_COLS:
            return f"{val:.3f}".replace("0.", ".")
        if col_name.endswith("%") or col_name in {
            "Velo", "PerVelo", "PerceivedVelo", "IVB", "HB", "Spin", "Ext", "RelExt",
            "RelH", "RelHt", "AvgEV", "MaxEV", "AvgLA", "Stuff+", "Loc+", "wRC+"
        }:
            return f"{val:.1f}"
        if abs(val) < 1:
            return f"{val:.3f}".replace("0.", ".")
        if abs(val - round(val)) < 0.000001:
            return f"{int(round(val))}"
        return f"{val:.1f}"
    return str(value)


GOOD_HIGH_COLS = {
    "BA", "OBP", "SLG", "OPS", "wOBA", "wRC+", "AvgEV", "MaxEV", "AvgLA",
    "HardHit%", "HH%", "Barrel%", "SweetSpot%", "Stuff+", "Loc+", "Strike%",
    "Zone%", "CSW%", "Whiff%", "K%", "BB%", "Swing%", "Usage%", "Velo", "PerVelo", "PerceivedVelo", "IVB", "Ext"
}
GOOD_LOW_COLS = {
    "Chase%", "Avg EV Allowed", "HH% Allowed", "HardHit% Allowed",
    "BA Allowed", "OBP Allowed", "SLG Allowed", "OPS Allowed"
}


def _metric_direction(col, context=None):
    name = str(col)
    if name in {"K%", "Whiff%"} and context == "hitting":
        return -1
    if name == "BB%" and context == "pitching":
        return -1
    if name in GOOD_LOW_COLS or "Allowed" in name:
        return -1
    if name in GOOD_HIGH_COLS:
        return 1
    return 0


def _value_to_color(value, col, series=None, context=None):
    direction = _metric_direction(col, context=context)
    if direction == 0:
        return None
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None

    if series is not None:
        nums = pd.to_numeric(series, errors="coerce").dropna()
    else:
        nums = pd.Series(dtype=float)
    if len(nums) >= 3 and nums.max() != nums.min():
        lo, hi = nums.quantile(0.10), nums.quantile(0.90)
        if hi == lo:
            lo, hi = nums.min(), nums.max()
    else:
        lo, hi = val - 1, val + 1
    score = (val - lo) / (hi - lo) if hi != lo else 0.5
    score = float(np.clip(score, 0, 1))
    if direction < 0:
        score = 1 - score

    bad = np.array([20, 38, 75])
    mid = np.array([36, 28, 26])
    good = np.array([210, 40, 40])
    if score < 0.5:
        t = score / 0.5
        rgb = bad * (1 - t) + mid * t
    else:
        t = (score - 0.5) / 0.5
        rgb = mid * (1 - t) + good * t
    return tuple(int(x) for x in rgb)


def style_scouting_dataframe(df: pd.DataFrame, context=None):
    if df is None or df.empty:
        return df

    def style_col(col):
        styles = []
        for value in col:
            rgb = _value_to_color(value, col.name, col, context=context)
            if rgb is None:
                styles.append("")
            else:
                styles.append(f"background-color: rgb{rgb}; color: #fff8e9; font-weight: 650;")
        return styles

    formatters = {col: (lambda value, c=col: _fmt_pdf_value(value, c)) for col in df.columns}
    return df.style.apply(style_col, axis=0).format(formatters)


def _safe_pdf_name(name):
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")


def _add_report_table(ax, df, title, max_rows=10, font_size=8, context=None):
    ax.axis("off")
    ax.set_title(title, color="#FFF7E8", fontsize=14, fontweight="bold", loc="left", pad=10)

    if df is None or df.empty:
        ax.text(0.02, 0.55, "No data available", color="#CDBFAF", fontsize=10, ha="left", va="center")
        return

    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(lambda value, c=col: _fmt_pdf_value(value, c))

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
            face = "#211C1A" if r % 2 else "#171514"
            col_name = view.columns[c] if c < len(view.columns) else ""
            if col_name in df.columns and r - 1 < len(df):
                rgb = _value_to_color(df.iloc[r - 1][col_name], col_name, df[col_name], context=context)
                if rgb is not None:
                    face = "#{:02x}{:02x}{:02x}".format(*rgb)
            cell.set_facecolor(face)
            cell.set_text_props(color="#F8EFE2")


def _rename_compact_report_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
    "SprayBucket": "Spray",
        "HardHit%": "HH%",
        "BatterSide": "Side",
        "pitch_abbr": "Pitch",
        "PitchGroup": "Pitch",
        "PerceivedVelo": "PerVelo",
    })


def _add_notes_panel(
    ax, title, notes, footer=None, max_notes=5, wrap_width=58,
    title_size=16, note_size=9.5, number_size=12, footer_size=9
):
    ax.axis("off")
    ax.set_title(title, color="#FFF7E8", fontsize=title_size, fontweight="bold", loc="left", pad=10)
    ax.add_patch(plt.Rectangle((0, 0.03), 1, 0.84, facecolor="#171514", edgecolor=FORDHAM_GOLD, linewidth=1.1, transform=ax.transAxes))
    y = 0.79
    for i, note in enumerate([n for n in notes if n][:max_notes], start=1):
        wrapped = "\n".join(textwrap.wrap(str(note), width=wrap_width, break_long_words=False))
        ax.text(0.055, y, f"{i}.", color=FORDHAM_GOLD, fontsize=number_size, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.115, y, wrapped, color="#F8EFE2", fontsize=note_size, transform=ax.transAxes, va="top", linespacing=1.25)
        y -= 0.105 + 0.045 * wrapped.count("\n")
        if y < 0.16:
            break
    if footer:
        ax.text(0.055, 0.07, footer, color="#CDBFAF", fontsize=footer_size, transform=ax.transAxes, va="bottom")


def _best_row_note(df, sort_col, min_n=8, ascending=False):
    if df is None or df.empty or sort_col not in df.columns:
        return None
    view = df.copy()
    if "N" in view.columns:
        view = view[pd.to_numeric(view["N"], errors="coerce").fillna(0) >= min_n]
    view[sort_col] = pd.to_numeric(view[sort_col], errors="coerce")
    view = view.dropna(subset=[sort_col])
    if view.empty:
        return None
    return view.sort_values(sort_col, ascending=ascending).iloc[0]


def _count_series(df):
    if {"Balls", "Strikes"}.issubset(df.columns):
        balls = pd.to_numeric(df["Balls"], errors="coerce").fillna(-1).astype(int)
        strikes = pd.to_numeric(df["Strikes"], errors="coerce").fillna(-1).astype(int)
        return balls.astype(str) + "-" + strikes.astype(str)
    if "Count" in df.columns:
        return df["Count"].astype(str)
    return pd.Series("", index=df.index)


def _first_pitch_hitter_damage(hdf, min_bip=2):
    if "pitch_abbr" not in hdf.columns:
        return None
    df = hdf.copy()
    df["Count"] = _count_series(df)
    first = df[df["Count"].eq("0-0")].copy()
    if first.empty:
        return None
    first["Pitch"] = combine_slider_sweeper(first["pitch_abbr"])
    damage = hitter_pitchtype_effectiveness(first)
    if damage.empty or "SLG" not in damage.columns:
        return None
    if "BIP" in damage.columns:
        damage = damage[pd.to_numeric(damage["BIP"], errors="coerce").fillna(0) >= min_bip]
    damage["SLG"] = pd.to_numeric(damage["SLG"], errors="coerce")
    damage = damage.dropna(subset=["SLG"])
    return damage.sort_values(["SLG", "AvgEV"], ascending=False).iloc[0] if not damage.empty else None


def _two_strike_hitter_chase(hdf, min_swings=3):
    if "pitch_abbr" not in hdf.columns:
        return None
    df = hdf.copy()
    strikes = pd.to_numeric(df.get("Strikes", pd.Series(index=df.index, dtype=float)), errors="coerce")
    two_k = df[strikes.ge(2)].copy()
    if two_k.empty:
        return None
    two_k["Pitch"] = combine_slider_sweeper(two_k["pitch_abbr"])
    grouped = two_k.groupby("Pitch").agg(
        N=("Pitch", "count"),
        Swings=("is_swing", "sum"),
        Chases=("is_chase", "sum"),
        Whiffs=("is_whiff", "sum"),
    ).reset_index()
    grouped = grouped[pd.to_numeric(grouped["Swings"], errors="coerce").fillna(0) >= min_swings]
    if grouped.empty:
        return None
    grouped["Chase%"] = grouped["Chases"] / grouped["Swings"] * 100
    grouped["Whiff%"] = np.where(grouped["Swings"] > 0, grouped["Whiffs"] / grouped["Swings"] * 100, np.nan)
    return grouped.sort_values(["Chase%", "Whiff%", "N"], ascending=False).iloc[0]


def _first_pitch_pitcher_usage(pdf_df):
    if "pitch_abbr" not in pdf_df.columns:
        return None
    df = pdf_df.copy()
    df["Count"] = _count_series(df)
    first = df[df["Count"].eq("0-0")].copy()
    if first.empty:
        return None
    first["Pitch"] = combine_slider_sweeper(first["pitch_abbr"])
    grouped = first.groupby("Pitch").agg(N=("Pitch", "count")).reset_index()
    grouped["Usage%"] = grouped["N"] / max(grouped["N"].sum(), 1) * 100
    return grouped.sort_values(["Usage%", "N"], ascending=False).iloc[0]


def _two_strike_pitcher_chase(pdf_df, min_swings=3):
    if "pitch_abbr" not in pdf_df.columns:
        return None
    strikes = pd.to_numeric(pdf_df.get("Strikes", pd.Series(index=pdf_df.index, dtype=float)), errors="coerce")
    two_k = pdf_df[strikes.ge(2)].copy()
    if two_k.empty:
        return None
    two_k["Pitch"] = combine_slider_sweeper(two_k["pitch_abbr"])
    grouped = two_k.groupby("Pitch").agg(
        N=("Pitch", "count"),
        Swings=("is_swing", "sum"),
        Chases=("is_chase", "sum"),
        Whiffs=("is_whiff", "sum"),
    ).reset_index()
    grouped = grouped[pd.to_numeric(grouped["Swings"], errors="coerce").fillna(0) >= min_swings]
    if grouped.empty:
        return None
    grouped["Chase%"] = grouped["Chases"] / grouped["Swings"] * 100
    grouped["Whiff%"] = np.where(grouped["Swings"] > 0, grouped["Whiffs"] / grouped["Swings"] * 100, np.nan)
    return grouped.sort_values(["Chase%", "Whiff%", "N"], ascending=False).iloc[0]


def hitter_quick_read_notes(hdf, card, pitch_table, count_table, spray_table, splits_table):
    notes = []
    notes.append(
        f"{card.get('Side', 'Unknown')} hitter, {card.get('PA', '-')} PA, "
        f"{_fmt_pdf_value(card.get('wOBA'))} wOBA, {card.get('wRC+', '-')} wRC+, "
        f"{_fmt_pdf_value(card.get('AvgEV'))} Avg EV."
    )

    first_damage = _first_pitch_hitter_damage(hdf)
    if first_damage is not None:
        notes.append(
            f"First-pitch damage: {first_damage.get('Pitch', '-')} is the pitch he has hurt most "
            f"on 0-0 counts ({_fmt_pdf_value(first_damage.get('SLG'))} SLG, {_fmt_pdf_value(first_damage.get('AvgEV'))} Avg EV)."
        )

    two_strike_chase = _two_strike_hitter_chase(hdf)
    if two_strike_chase is not None:
        notes.append(
            f"Two-strike chase pitch: {two_strike_chase.get('Pitch', '-')} has drawn "
            f"{_fmt_pdf_value(two_strike_chase.get('Chase%'))}% chase and {_fmt_pdf_value(two_strike_chase.get('Whiff%'))}% whiff."
        )

    damage = _best_row_note(pitch_table, "SLG")
    if damage is not None:
        notes.append(
            f"Overall damage bucket: {damage.get('Pitch', '-')} "
            f"({_fmt_pdf_value(damage.get('SLG'))} SLG, {_fmt_pdf_value(damage.get('AvgEV'))} Avg EV)."
        )

    whiff = _best_row_note(pitch_table, "Whiff%")
    if whiff is not None:
        notes.append(
            f"Overall miss bucket: {whiff.get('Pitch', '-')} has a "
            f"{_fmt_pdf_value(whiff.get('Whiff%'))}% whiff rate."
        )

    count = _best_row_note(count_table, "SLG", min_n=5)
    if count is not None:
        notes.append(
            f"Count to respect: {count.get('Count', '-')} has produced "
            f"{_fmt_pdf_value(count.get('SLG'))} SLG."
        )

    if spray_table is not None and not spray_table.empty and "BIP" in spray_table.columns:
        spray_sort_cols = ["BIP"] + (["HH%"] if "HH%" in spray_table.columns else [])
        spray = spray_table.sort_values(spray_sort_cols, ascending=False).iloc[0]
        spray_detail = (
            f"{_fmt_pdf_value(spray.get('BIP%'))}% of BIP"
            if "BIP%" in spray_table.columns else
            f"{_fmt_pdf_value(spray.get('BIP'))} tracked BIP"
        )
        notes.append(
            f"Spray lean: {spray.get('Spray', '-')} with {spray_detail} "
            f"and {_fmt_pdf_value(spray.get('HH%'))}% HH."
        )

    if splits_table is not None and not splits_table.empty and "wOBA" in splits_table.columns:
        split = _best_row_note(splits_table.rename(columns={"Side": "Pitcher Hand", "Split": "Pitcher Hand"}), "wOBA", min_n=5)
        if split is not None:
            notes.append(
                f"Best handedness split: {split.get('Pitcher Hand', split.get('BatterSide', '-'))} "
                f"at {_fmt_pdf_value(split.get('wOBA'))} wOBA."
            )

    return notes


def pitcher_quick_read_notes(pdf_df, arsenal, splits, allowed, pa_rates):
    notes = []
    total = len(pdf_df)
    notes.append(
        f"{total} tracked pitches, {_fmt_pdf_value(pdf_df['Stuff+'].mean() if 'Stuff+' in pdf_df.columns else np.nan)} Stuff+, "
        f"{_fmt_pdf_value(pdf_df['Loc+'].mean() if 'Loc+' in pdf_df.columns else np.nan)} Loc+, "
        f"{_fmt_pdf_value(allowed.get('BA'))}/{_fmt_pdf_value(allowed.get('OBP'))}/{_fmt_pdf_value(allowed.get('SLG'))} slash allowed."
    )

    first_usage = _first_pitch_pitcher_usage(pdf_df)
    if first_usage is not None:
        notes.append(
            f"First-pitch usage: {first_usage.get('Pitch', '-')} is his most-used 0-0 pitch "
            f"({_fmt_pdf_value(first_usage.get('Usage%'))}% usage)."
        )

    two_strike_chase = _two_strike_pitcher_chase(pdf_df)
    if two_strike_chase is not None:
        notes.append(
            f"Two-strike chase pitch: {two_strike_chase.get('Pitch', '-')} has generated "
            f"{_fmt_pdf_value(two_strike_chase.get('Chase%'))}% chase and {_fmt_pdf_value(two_strike_chase.get('Whiff%'))}% whiff."
        )

    stuff = _best_row_note(arsenal, "Stuff+", min_n=8)
    if stuff is not None:
        notes.append(
            f"Best raw pitch quality: {stuff.get('Pitch', '-')} with {_fmt_pdf_value(stuff.get('Stuff+'))} Stuff+ "
            f"and {_fmt_pdf_value(stuff.get('Whiff%'))}% Whiff."
        )

    command = _best_row_note(arsenal, "Loc+", min_n=8)
    if command is not None:
        notes.append(
            f"Best command shape: {command.get('Pitch', '-')} with {_fmt_pdf_value(command.get('Loc+'))} Loc+ "
            f"and {_fmt_pdf_value(command.get('Zone%'))}% Zone."
        )

    risk = _best_row_note(arsenal, "HardHit%", min_n=5)
    if risk is not None:
        notes.append(
            f"Contact risk: {risk.get('Pitch', '-')} has allowed {_fmt_pdf_value(risk.get('HardHit%'))}% HH "
            f"and {_fmt_pdf_value(risk.get('AvgEV'))} Avg EV."
        )

    if splits is not None and not splits.empty and "SLG" in splits.columns:
        split = _best_row_note(splits, "SLG", min_n=5)
        if split is not None:
            notes.append(
                f"Split to monitor: {split.get('Side', '-')} vs {split.get('Pitch', '-')} has allowed "
                f"{_fmt_pdf_value(split.get('BA'))}/{_fmt_pdf_value(split.get('OBP'))}/{_fmt_pdf_value(split.get('SLG'))}."
            )

    notes.append(f"PA rates: {_fmt_pdf_value(pa_rates.get('BB%'))}% BB and {_fmt_pdf_value(pa_rates.get('K%'))}% K.")
    return notes


def pitcher_allowed_slash(pdf_df: pd.DataFrame) -> dict:
    slash = add_ba_slg_by_group(pdf_df.assign(Player="Allowed"), ["Player"])
    if slash.empty:
        return {"BA": np.nan, "OBP": np.nan, "SLG": np.nan, "OPS": np.nan}
    row = slash.iloc[0]
    return {k: row.get(k, np.nan) for k in ["BA", "OBP", "SLG", "OPS"]}


def pitcher_pa_rates(pdf_df: pd.DataFrame) -> dict:
    pa = get_pa_endings(pdf_df)
    if pa.empty:
        return {"BB%": np.nan, "K%": np.nan}
    bb = pa.get("KorBB", pd.Series("", index=pa.index)).eq("Walk").sum()
    k = pa.get("KorBB", pd.Series("", index=pa.index)).eq("Strikeout").sum()
    return {"BB%": bb / len(pa) * 100, "K%": k / len(pa) * 100}


def team_hitting_metrics(team_df: pd.DataFrame) -> dict:
    if team_df.empty:
        return {}
    card = compute_hitter_card(team_df, compute_league_woba(team_df))
    slash = add_ba_slg_by_group(team_df.assign(Team="Team"), ["Team"])
    row = slash.iloc[0].to_dict() if not slash.empty else {}
    return {
        "PA": card.get("PA"),
        "BA": row.get("BA", np.nan),
        "OBP": row.get("OBP", np.nan),
        "SLG": row.get("SLG", np.nan),
        "OPS": row.get("OPS", np.nan),
        "wOBA": card.get("wOBA"),
        "wRC+": card.get("wRC+"),
        "BB%": card.get("BB%"),
        "K%": card.get("K%"),
        "AvgEV": card.get("AvgEV"),
        "HH%": card.get("HardHit%"),
        "Whiff%": card.get("Whiff%"),
        "Chase%": card.get("Chase%"),
    }


def summarize_pitching_staff(team_df: pd.DataFrame) -> pd.DataFrame:
    if team_df.empty or "Pitcher" not in team_df.columns:
        return pd.DataFrame()

    rows = []
    for pitcher, g in team_df.groupby("Pitcher"):
        if g.empty:
            continue
        pa = get_pa_endings(g)
        allowed = pitcher_allowed_slash(g)
        rates = pitcher_pa_rates(g)
        swings = g["is_swing"].sum() if "is_swing" in g.columns else 0
        whiffs = g["is_whiff"].sum() if "is_whiff" in g.columns else 0
        bip = get_true_bip_with_ev(g) if {"EV", "PitchCall"}.issubset(g.columns) else pd.DataFrame()
        rows.append({
            "Pitcher": pitcher,
            "Pitches": len(g),
            "BF": len(pa),
            "BA": allowed.get("BA"),
            "OBP": allowed.get("OBP"),
            "SLG": allowed.get("SLG"),
            "OPS": allowed.get("OPS"),
            "Stuff+": g["Stuff+"].mean() if "Stuff+" in g.columns else np.nan,
            "Loc+": g["Loc+"].mean() if "Loc+" in g.columns else np.nan,
            "Strike%": g["is_strike"].mean() * 100 if "is_strike" in g.columns and len(g) else np.nan,
            "Zone%": g["in_zone"].mean() * 100 if "in_zone" in g.columns and len(g) else np.nan,
            "CSW%": g["is_csw"].mean() * 100 if "is_csw" in g.columns and len(g) else np.nan,
            "Whiff%": whiffs / swings * 100 if swings else np.nan,
            "BB%": rates.get("BB%"),
            "K%": rates.get("K%"),
            "AvgEV": bip["EV"].mean() if not bip.empty else np.nan,
            "HH%": (bip["EV"] >= 95).mean() * 100 if not bip.empty else np.nan,
        })

    return pd.DataFrame(rows).round(3)


def team_pitching_metrics(team_df: pd.DataFrame) -> dict:
    if team_df.empty:
        return {}
    allowed = pitcher_allowed_slash(team_df)
    rates = pitcher_pa_rates(team_df)
    swings = team_df["is_swing"].sum() if "is_swing" in team_df.columns else 0
    whiffs = team_df["is_whiff"].sum() if "is_whiff" in team_df.columns else 0
    bip = get_true_bip_with_ev(team_df) if {"EV", "PitchCall"}.issubset(team_df.columns) else pd.DataFrame()
    return {
        "Pitches": len(team_df),
        "BF": len(get_pa_endings(team_df)),
        "BA": allowed.get("BA"),
        "OBP": allowed.get("OBP"),
        "SLG": allowed.get("SLG"),
        "OPS": allowed.get("OPS"),
        "Stuff+": team_df["Stuff+"].mean() if "Stuff+" in team_df.columns else np.nan,
        "Loc+": team_df["Loc+"].mean() if "Loc+" in team_df.columns else np.nan,
        "Strike%": team_df["is_strike"].mean() * 100 if "is_strike" in team_df.columns and len(team_df) else np.nan,
        "Zone%": team_df["in_zone"].mean() * 100 if "in_zone" in team_df.columns and len(team_df) else np.nan,
        "CSW%": team_df["is_csw"].mean() * 100 if "is_csw" in team_df.columns and len(team_df) else np.nan,
        "Whiff%": whiffs / swings * 100 if swings else np.nan,
        "BB%": rates.get("BB%"),
        "K%": rates.get("K%"),
        "AvgEV": bip["EV"].mean() if not bip.empty else np.nan,
        "HH%": (bip["EV"] >= 95).mean() * 100 if not bip.empty else np.nan,
    }


def team_hitter_tendencies(hitters_df: pd.DataFrame) -> pd.DataFrame:
    required = {"Batter", "Direction", "BatterSide", "EV", "LA", "PitchCall"}
    if hitters_df.empty or not required.issubset(hitters_df.columns):
        return pd.DataFrame()

    rows = []
    for hitter, g in hitters_df.groupby("Batter"):
        bip = get_true_bip_with_ev(g)
        if bip.empty:
            continue
        bip = bip.copy()
        bip["Direction"] = pd.to_numeric(bip["Direction"], errors="coerce")
        bip["LA"] = pd.to_numeric(bip["LA"], errors="coerce")
        bip = bip.dropna(subset=["Direction", "LA"])
        if bip.empty:
            continue

        side_raw = str(bip.get("BatterSide", pd.Series([""])).dropna().mode().iloc[0]).upper()
        hitter_side = "LHH" if side_raw.startswith("L") else "RHH" if side_raw.startswith("R") else "UNK"

        def spray_bucket(row):
            if row["Direction"] <= -15:
                field = "LF"
            elif row["Direction"] >= 15:
                field = "RF"
            else:
                field = "Middle"
            if field == "Middle":
                return "Middle"
            if hitter_side == "RHH":
                return "Pull" if field == "LF" else "Oppo"
            if hitter_side == "LHH":
                return "Pull" if field == "RF" else "Oppo"
            return "Middle"

        bip["Spray"] = bip.apply(spray_bucket, axis=1)
        bip["BattedType"] = np.select(
            [bip["LA"].lt(8), bip["LA"].between(8, 27), bip["LA"].gt(27)],
            ["GB", "LD", "FB"],
            default="UNK"
        )

        total = len(bip)
        pull = bip["Spray"].eq("Pull").mean() * 100
        middle = bip["Spray"].eq("Middle").mean() * 100
        oppo = bip["Spray"].eq("Oppo").mean() * 100
        gb = bip["BattedType"].eq("GB").mean() * 100
        pull_gb = ((bip["Spray"].eq("Pull")) & (bip["BattedType"].eq("GB"))).mean() * 100
        middle_gb = ((bip["Spray"].eq("Middle")) & (bip["BattedType"].eq("GB"))).mean() * 100
        oppo_air = ((bip["Spray"].eq("Oppo")) & (bip["BattedType"].isin(["LD", "FB"]))).mean() * 100
        hh = bip["EV"].ge(95).mean() * 100

        tags = []
        if pull >= 45:
            tags.append("Pull-heavy")
        elif oppo >= 35:
            tags.append("Oppo-capable")
        elif middle >= 40:
            tags.append("Middle-field")
        if pull_gb >= 25:
            tags.append("Pull GB alert")
        if middle_gb >= 25:
            tags.append("Middle GB")
        if oppo_air >= 20:
            tags.append("Oppo air")
        if hh >= 35:
            tags.append("Hard contact")

        rows.append({
            "Hitter": hitter,
            "Side": hitter_side,
            "BIP": total,
            "Pull%": round(pull, 1),
            "Middle%": round(middle, 1),
            "Oppo%": round(oppo, 1),
            "GB%": round(gb, 1),
            "Pull GB%": round(pull_gb, 1),
            "Middle GB%": round(middle_gb, 1),
            "Oppo Air%": round(oppo_air, 1),
            "HH%": round(hh, 1),
            "AvgEV": round(bip["EV"].mean(), 1),
            "Tendency": "; ".join(tags) if tags else "Balanced",
        })

    return pd.DataFrame(rows).sort_values(["BIP", "HH%"], ascending=False)


def _table_columns(df: pd.DataFrame, cols):
    if df is None or df.empty:
        return pd.DataFrame()
    return df[[c for c in cols if c in df.columns]].copy()


def _save_paginated_report_table(pdf, df, title, rows_per_page=24, font_size=6.2, context=None):
    if df is None or df.empty:
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#100D0C")
        _add_report_table(fig.add_subplot(111), df, title, max_rows=1, font_size=font_size, context=context)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    for start in range(0, len(df), rows_per_page):
        chunk = df.iloc[start:start + rows_per_page]
        end = min(start + rows_per_page, len(df))
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#100D0C")
        ax = fig.add_axes([0.04, 0.06, 0.92, 0.86])
        page_title = f"{title} ({start + 1}-{end} of {len(df)})" if len(df) > rows_per_page else title
        _add_report_table(ax, chunk, page_title, max_rows=len(chunk), font_size=font_size, context=context)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    return img


def _append_hitter_zone_summary_page(pdf, hdf):
    source_figs = [
        make_savant_zone_heatmap(hdf, "AvgEV", "In-Zone Avg EV", "True BIP only"),
        make_savant_zone_heatmap(hdf, "Whiff%", "In-Zone Whiff%", "Whiffs per swing"),
        make_full_zone_heatmap(hdf, "Chase%", "Chase% Full Zone"),
    ]
    images = []
    for fig in source_figs:
        if fig:
            fig.patch.set_facecolor("#100D0C")
            for ax in fig.axes:
                ax.title.set_color("#FFF7E8")
                ax.xaxis.label.set_color("#FFF7E8")
                ax.yaxis.label.set_color("#FFF7E8")
                ax.tick_params(colors="#FFF7E8")
            images.append(_fig_to_image(fig))

    if not images:
        return

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    fig.suptitle("Hitter Zone Summary", color="#FFF7E8", fontsize=18, fontweight="bold", y=0.97)
    cols = len(images)
    gs = fig.add_gridspec(1, cols, left=0.03, right=0.97, top=0.91, bottom=0.06, wspace=0.05)
    for i, img in enumerate(images):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pitcher_side_pitch_splits(pdf_df: pd.DataFrame) -> pd.DataFrame:
    if "BatterSide" not in pdf_df.columns or "pitch_abbr" not in pdf_df.columns:
        return pd.DataFrame()

    base = pdf_df.groupby(["BatterSide", "pitch_abbr"]).agg(
        N=("pitch_abbr", "count"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
        Zone=("in_zone", "mean"),
    ).reset_index()
    total_by_side = base.groupby("BatterSide")["N"].transform("sum")
    base["Usage%"] = np.where(total_by_side > 0, base["N"] / total_by_side * 100, np.nan)
    base["Whiff%"] = np.where(base["Swings"] > 0, base["Whiffs"] / base["Swings"] * 100, np.nan)
    base["Zone%"] = base["Zone"] * 100

    slash = add_ba_slg_by_group(pdf_df, ["BatterSide", "pitch_abbr"])
    if not slash.empty:
        base = base.merge(slash[["BatterSide", "pitch_abbr", "BA", "OBP", "SLG"]], on=["BatterSide", "pitch_abbr"], how="left")
    else:
        base["BA"] = np.nan
        base["OBP"] = np.nan
        base["SLG"] = np.nan

    return (
        base.rename(columns={"BatterSide": "Side", "pitch_abbr": "Pitch"})
        [["Side", "Pitch", "N", "Usage%", "BA", "OBP", "SLG", "Whiff%", "Zone%"]]
        .round(3)
    )


def make_full_zone_heatmap(df, metric, title):
    return make_zone_heatmap(df, metric, title)


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
    box_w, box_h = 0.21, 0.105
    for i, (label, value) in enumerate(metric_pairs):
        x = start_x + (i % cols) * 0.235
        y = start_y - (i // cols) * 0.13
        ax.add_patch(plt.Rectangle((x, y), box_w, box_h, facecolor="#211C1A", edgecolor=FORDHAM_GOLD, linewidth=1.2, transform=ax.transAxes))
        ax.text(x + 0.018, y + 0.067, str(label), color="#CDBFAF", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.018, y + 0.024, _fmt_pdf_value(value), color="#FFF7E8", fontsize=16, fontweight="bold", transform=ax.transAxes)

    ax.text(
        0.05, 0.08,
        "Generated from TrackMan pitch-by-pitch data. Contact metrics use true in-play batted balls with usable EV.",
        color="#CDBFAF", fontsize=9, transform=ax.transAxes
    )
    return fig


def _append_hitter_scouting_pages(pdf, hdf: pd.DataFrame, hitter: str, team: str):
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
        ("BB%", card.get("BB%")), ("K%", card.get("K%")), ("wOBA", card.get("wOBA")), ("wRC+", card.get("wRC+")),
        ("Avg EV", card.get("AvgEV")), ("HH%", card.get("HardHit%")),
        ("Stuff+ Faced", hdf["Stuff+"].mean() if "Stuff+" in hdf.columns else np.nan),
        ("Loc+ Faced", hdf["Loc+"].mean() if "Loc+" in hdf.columns else np.nan),
    ]

    pitch_table = hitter_pitchtype_effectiveness(hdf)
    if not pitch_table.empty:
        pitch_table = pitch_table[["Pitch", "N", "BA", "SLG", "Swing%", "Whiff%", "Chase%", "AvgEV", "HardHit%"]]
        pitch_table = _rename_compact_report_cols(pitch_table)

    count_table = count_effectiveness(hdf)
    if not count_table.empty:
        count_table = count_table[["Count", "N", "BA", "SLG", "Swing%", "Whiff%", "AvgEV", "HardHit%"]]
        count_table = _rename_compact_report_cols(count_table)

    spray_table = _rename_compact_report_cols(hitter_spray_profile(hdf))
    splits_table = hitter_splits(hdf)
    quick_notes = hitter_quick_read_notes(hdf, card, pitch_table, count_table, spray_table, splits_table)

    fig = _scouting_cover_fig(hitter, "Hitter scouting report", metric_pairs)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.95, top=0.92, bottom=0.07, hspace=0.32, wspace=0.18)
    _add_report_table(fig.add_subplot(gs[0, 0]), pitch_table, "Effectiveness vs Pitch Type", max_rows=12, context="hitting")
    _add_report_table(fig.add_subplot(gs[0, 1]), count_table, "Count-Based Effectiveness", max_rows=12, context="hitting")
    _add_report_table(fig.add_subplot(gs[1, :]), splits_table, "Splits vs Pitcher Handedness", max_rows=8, context="hitting")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    _append_hitter_spray_shift_page(pdf, hdf, spray_table)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.95, top=0.92, bottom=0.07, hspace=0.34, wspace=0.30, width_ratios=[1.3, 1.0])
    _add_notes_panel(
        fig.add_subplot(gs[:, 0]),
        "Quick Read",
        quick_notes,
        footer="Use this page as the short hitter plan before reviewing the zone heatmaps.",
        max_notes=5,
        wrap_width=48
    )
    damage_view = pitch_table.sort_values("SLG", ascending=False) if pitch_table is not None and not pitch_table.empty and "SLG" in pitch_table.columns else pitch_table
    _add_report_table(fig.add_subplot(gs[0, 1]), damage_view, "Damage Buckets", max_rows=6, font_size=7, context="hitting")
    discipline_view = pitch_table.sort_values("Whiff%", ascending=False) if pitch_table is not None and not pitch_table.empty and "Whiff%" in pitch_table.columns else pitch_table
    _add_report_table(fig.add_subplot(gs[1, 1]), discipline_view, "Miss / Chase Buckets", max_rows=6, font_size=7, context="hitting")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    _append_hitter_zone_summary_page(pdf, hdf)


def build_hitter_scouting_pdf(hdf: pd.DataFrame, hitter: str, team: str) -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        _append_hitter_scouting_pages(pdf, hdf, hitter, team)

    buf.seek(0)
    return buf.getvalue()


def _append_pitcher_scouting_pages(out_pdf, pdf_df: pd.DataFrame, pitcher: str, team: str):
    pdf_df = pdf_df.copy()
    pdf_df = add_perceived_velocity(pdf_df)
    for col in ["pitch_abbr", "Velo", "PerceivedVelo", "IVB", "HB", "Ext", "RelH", "Stuff+", "Loc+", "in_zone", "is_swing", "is_whiff", "BatterSide"]:
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
    allowed = pitcher_allowed_slash(pdf_df)
    pa_rates = pitcher_pa_rates(pdf_df)

    metric_pairs = [
        ("Team", team), ("Pitches", total), ("Strike%", strike), ("Zone%", zone),
        ("CSW%", csw), ("Whiff%", whiff_pct), ("Stuff+", pdf_df["Stuff+"].mean() if "Stuff+" in pdf_df.columns else np.nan),
        ("Loc+", pdf_df["Loc+"].mean() if "Loc+" in pdf_df.columns else np.nan),
        ("BA", allowed["BA"]), ("OBP", allowed["OBP"]), ("SLG", allowed["SLG"]), ("BB%", pa_rates["BB%"]),
        ("K%", pa_rates["K%"]), ("Avg EV Allowed", bip["EV"].mean() if not bip.empty else np.nan),
        ("HH% Allowed", (bip["EV"] >= 95).mean() * 100 if not bip.empty else np.nan),
    ]

    arsenal = pdf_df.groupby("pitch_abbr").agg(
        N=("pitch_abbr", "count"),
        Velo=("Velo", "mean"),
        PerceivedVelo=("PerceivedVelo", "mean"),
        IVB=("IVB", "mean"),
        HB=("HB", "mean"),
        Ext=("Ext", "mean"),
        RelHt=("RelH", "mean"),
        Stuff_plus=("Stuff+", "mean"),
        Loc_plus=("Loc+", "mean"),
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

    arsenal = arsenal.rename(columns={"Stuff_plus": "Stuff+", "Loc_plus": "Loc+"})
    arsenal = arsenal[["Pitch", "N", "Usage%", "Velo", "PerceivedVelo", "IVB", "HB", "Ext", "RelHt", "Stuff+", "Loc+", "Zone%", "Whiff%", "AvgEV", "HardHit%"]].round(1)

    splits = pitcher_side_pitch_splits(pdf_df)
    quick_notes = pitcher_quick_read_notes(pdf_df, arsenal, splits, allowed, pa_rates)

    fig = _scouting_cover_fig(pitcher, "Pitcher scouting report", metric_pairs)
    out_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.95, top=0.92, bottom=0.07, hspace=0.32, wspace=0.18)
    _add_report_table(fig.add_subplot(gs[0, :]), _rename_compact_report_cols(arsenal.sort_values("N", ascending=False)), "Pitch Arsenal", max_rows=12, font_size=6, context="pitching")
    _add_report_table(fig.add_subplot(gs[1, 0]), splits.sort_values(["Side", "N"], ascending=[True, False]), "Batter-Side Splits", max_rows=12, context="pitching")
    ax = fig.add_subplot(gs[1, 1])
    _add_notes_panel(
        ax,
        "Quick Read",
        quick_notes,
        footer="Pair this page with movement and location views before building the game plan.",
        max_notes=4,
        wrap_width=36
    )
    out_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = build_movement_figure(pdf_df)
    out_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pitcher_scouting_pdf(pdf_df: pd.DataFrame, pitcher: str, team: str) -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as out_pdf:
        _append_pitcher_scouting_pages(out_pdf, pdf_df, pitcher, team)

    buf.seek(0)
    return buf.getvalue()


def _append_section_divider(pdf, title, subtitle):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color=FORDHAM_MAROON, transform=ax.transAxes))
    ax.text(0.05, 0.93, "FORDHAM BASEBALL SCOUTING ZONE", color="#FFF7E8", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.55, title, color="#FFF7E8", fontsize=34, fontweight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.47, subtitle, color="#CDBFAF", fontsize=14, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_team_scouting_pdf(df: pd.DataFrame, team: str, include_individual_reports=False, packet_scope="All Players", max_players=None) -> bytes:
    df = df.copy()
    hitters_df = df[df.get("BatterTeam", pd.Series("", index=df.index)).astype(str).eq(str(team))].copy()
    pitchers_df = df[df.get("PitcherTeam", pd.Series("", index=df.index)).astype(str).eq(str(team))].copy()

    hitting = team_hitting_metrics(hitters_df)
    pitching = team_pitching_metrics(pitchers_df)

    hitter_summary = summarize_contact_quality(hitters_df, "Batter").sort_values("PA", ascending=False) if not hitters_df.empty else pd.DataFrame()
    pitcher_summary = summarize_pitching_staff(pitchers_df).sort_values("Pitches", ascending=False) if not pitchers_df.empty else pd.DataFrame()
    tendency_summary = team_hitter_tendencies(hitters_df)

    hitter_cols = ["Batter", "PA", "BA", "OBP", "SLG", "OPS", "wOBA", "wRC+", "BB%", "K%", "AvgEV", "HardHit%", "Whiff%", "Chase%"]
    pitcher_cols = ["Pitcher", "Pitches", "BF", "BA", "OBP", "SLG", "OPS", "Stuff+", "Loc+", "Strike%", "Zone%", "CSW%", "Whiff%", "BB%", "K%", "AvgEV", "HH%"]
    tendency_cols = ["Hitter", "Side", "BIP", "Pull%", "Middle%", "Oppo%", "GB%", "Pull GB%", "Middle GB%", "Oppo Air%", "HH%", "AvgEV", "Tendency"]
    hitter_summary = _table_columns(hitter_summary, hitter_cols)
    pitcher_summary = _table_columns(pitcher_summary, pitcher_cols)
    tendency_summary = _table_columns(tendency_summary, tendency_cols)

    pitch_mix = pd.DataFrame()
    if not pitchers_df.empty and "pitch_abbr" in pitchers_df.columns:
        pitch_mix = pitchers_df.groupby("pitch_abbr").agg(
            N=("pitch_abbr", "count"),
            Velo=("Velo", "mean"),
            IVB=("IVB", "mean"),
            HB=("HB", "mean"),
            Stuff_plus=("Stuff+", "mean"),
            Loc_plus=("Loc+", "mean"),
            Zone=("in_zone", "mean"),
            Swings=("is_swing", "sum"),
            Whiffs=("is_whiff", "sum"),
        ).reset_index().rename(columns={"pitch_abbr": "Pitch", "Stuff_plus": "Stuff+", "Loc_plus": "Loc+"})
        pitch_mix["Usage%"] = pitch_mix["N"] / max(pitch_mix["N"].sum(), 1) * 100
        pitch_mix["Zone%"] = pitch_mix["Zone"] * 100
        pitch_mix["Whiff%"] = np.where(pitch_mix["Swings"] > 0, pitch_mix["Whiffs"] / pitch_mix["Swings"] * 100, np.nan)
        pitch_mix = pitch_mix[["Pitch", "N", "Usage%", "Velo", "IVB", "HB", "Stuff+", "Loc+", "Zone%", "Whiff%"]].round(1).sort_values("N", ascending=False)

    metric_pairs = [
        ("Team", team), ("Hitter PA", hitting.get("PA")), ("Team BA", hitting.get("BA")), ("Team OBP", hitting.get("OBP")),
        ("Team SLG", hitting.get("SLG")), ("Team OPS", hitting.get("OPS")), ("Team wOBA", hitting.get("wOBA")), ("Team wRC+", hitting.get("wRC+")),
        ("Avg EV", hitting.get("AvgEV")), ("HH%", hitting.get("HH%")), ("Staff Pitches", pitching.get("Pitches")), ("BF", pitching.get("BF")),
        ("BA Allowed", pitching.get("BA")), ("SLG Allowed", pitching.get("SLG")), ("Staff Stuff+", pitching.get("Stuff+")), ("Staff Loc+", pitching.get("Loc+")),
    ]

    notes = [
        f"Offense: {_fmt_pdf_value(hitting.get('BA'))}/{_fmt_pdf_value(hitting.get('OBP'))}/{_fmt_pdf_value(hitting.get('SLG'))}, {_fmt_pdf_value(hitting.get('wOBA'))} wOBA, {hitting.get('wRC+', '-')} wRC+.",
        f"Contact: {_fmt_pdf_value(hitting.get('AvgEV'))} Avg EV, {_fmt_pdf_value(hitting.get('HH%'))}% HH, {_fmt_pdf_value(hitting.get('Whiff%'))}% whiff, {_fmt_pdf_value(hitting.get('Chase%'))}% chase.",
        f"Pitching: {_fmt_pdf_value(pitching.get('BA'))}/{_fmt_pdf_value(pitching.get('OBP'))}/{_fmt_pdf_value(pitching.get('SLG'))} allowed, {_fmt_pdf_value(pitching.get('K%'))}% K, {_fmt_pdf_value(pitching.get('BB%'))}% BB.",
        f"Run prevention indicators: {_fmt_pdf_value(pitching.get('Stuff+'))} Stuff+, {_fmt_pdf_value(pitching.get('Loc+'))} Loc+, {_fmt_pdf_value(pitching.get('Zone%'))}% Zone, {_fmt_pdf_value(pitching.get('Whiff%'))}% Whiff.",
    ]

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig = _scouting_cover_fig(team, "Team scouting report", metric_pairs)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#100D0C")
        gs = fig.add_gridspec(2, 2, left=0.05, right=0.95, top=0.92, bottom=0.07, hspace=0.32, wspace=0.22)
        _add_notes_panel(fig.add_subplot(gs[:, 0]), "Team Snapshot", notes, footer="Team report uses selected-team batting rows and pitching rows from the scouting database.", max_notes=4, wrap_width=48)
        _add_report_table(fig.add_subplot(gs[0, 1]), hitter_summary.sort_values("PA", ascending=False).head(8) if not hitter_summary.empty else hitter_summary, "Top Hitters", max_rows=8, font_size=6.5, context="hitting")
        _add_report_table(fig.add_subplot(gs[1, 1]), pitcher_summary.sort_values("Pitches", ascending=False).head(8) if not pitcher_summary.empty else pitcher_summary, "Top Pitchers", max_rows=8, font_size=6.2, context="pitching")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _save_paginated_report_table(
            pdf,
            tendency_summary.sort_values(["Pull GB%", "Pull%"], ascending=False) if not tendency_summary.empty else tendency_summary,
            "Hitter Tendencies",
            rows_per_page=20,
            font_size=5.8,
            context="hitting"
        )

        if not pitch_mix.empty:
            fig = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor("#100D0C")
            _add_report_table(fig.add_subplot(111), pitch_mix, "Team Pitch Mix", max_rows=14, font_size=7, context="pitching")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        _save_paginated_report_table(pdf, hitter_summary.sort_values("PA", ascending=False) if not hitter_summary.empty else hitter_summary, "All Hitter Info", rows_per_page=22, font_size=5.8, context="hitting")
        _save_paginated_report_table(pdf, pitcher_summary.sort_values("Pitches", ascending=False) if not pitcher_summary.empty else pitcher_summary, "All Pitcher Info", rows_per_page=22, font_size=5.6, context="pitching")

        if include_individual_reports:
            hitters = hitter_summary.sort_values("PA", ascending=False)["Batter"].dropna().astype(str).tolist() if "Batter" in hitter_summary.columns else []
            pitchers = pitcher_summary.sort_values("Pitches", ascending=False)["Pitcher"].dropna().astype(str).tolist() if "Pitcher" in pitcher_summary.columns else []
            if packet_scope == "Hitters Only":
                pitchers = []
            elif packet_scope == "Pitchers Only":
                hitters = []
            if max_players:
                hitters = hitters[:int(max_players)]
                pitchers = pitchers[:int(max_players)]

            if hitters:
                _append_section_divider(pdf, "Individual Hitter Reports", f"{len(hitters)} hitters from {team}")
                for hitter in hitters:
                    player_df = hitters_df[hitters_df["Batter"].astype(str).eq(hitter)].copy()
                    if not player_df.empty:
                        _append_hitter_scouting_pages(pdf, player_df, hitter, team)

            if pitchers:
                _append_section_divider(pdf, "Individual Pitcher Reports", f"{len(pitchers)} pitchers from {team}")
                for pitcher in pitchers:
                    player_df = pitchers_df[pitchers_df["Pitcher"].astype(str).eq(pitcher)].copy()
                    if not player_df.empty:
                        _append_pitcher_scouting_pages(pdf, player_df, pitcher, team)

    buf.seek(0)
    return buf.getvalue()


def scouting_zone_page(all_pitches_df: pd.DataFrame):
    st.title("Scouting Zone")
    st.caption("Create team-filtered hitter and pitcher scouting PDFs from the scouting TrackMan database.")

    with st.expander("Import 2026 TrackMan game CSVs from FileZilla", expanded=False):
        st.caption("Credentials are used only for this import and are not saved in the code.")
        i1, i2, i3 = st.columns([1.3, 0.7, 0.8])
        with i1:
            ftp_host = st.text_input("Host", placeholder="ftp.example.com")
        with i2:
            ftp_port = st.number_input("Port", min_value=1, max_value=65535, value=21, step=1)
        with i3:
            ftp_protocol = st.selectbox("Protocol", ["FTP", "FTPS", "SFTP"])
            st.caption("TrackMan FileZilla access usually works as FTP on port 21. Use SFTP on port 22 only if your account requires it.")

        i4, i5, i6 = st.columns([1.0, 1.0, 1.1])
        with i4:
            ftp_user = st.text_input("Username")
        with i5:
            ftp_password = st.text_input("Password", type="password")
        with i6:
            ftp_remote_dir = st.text_input("Remote folder", value="v3/2026")

        m1, m2, m3 = st.columns([1.2, 1.0, 0.8])
        with m1:
            ftp_months = st.multiselect(
                "Months",
                [f"{m:02d}" for m in range(1, 13)],
                default=["01", "02", "03", "04"]
            )
        with m2:
            ftp_day = st.text_input("Optional day folder", placeholder="Example: 15")
        with m3:
            ftp_csv_folder = st.text_input("CSV folder", value="CSV")

        i7, i8, i9, i10, i11 = st.columns([0.8, 0.8, 0.8, 1.0, 0.8])
        with i7:
            ftp_timeout = st.number_input("Timeout seconds", min_value=30, max_value=600, value=180, step=30)
        with i8:
            ftp_passive = st.checkbox("Passive mode", value=True)
        with i9:
            ftp_recursive = st.checkbox("Search subfolders", value=True)
        with i10:
            max_downloads = st.number_input("Max downloads (0 = all)", min_value=0, max_value=50000, value=25, step=25)
        with i11:
            skip_existing = st.checkbox("Skip existing", value=True)

        st.caption("Import rules: only game CSV names like 20260426-FordhamUniversity-1.csv are imported. _unverified, practice, playerpositioning, positional, bullpen, scrimmage, intrasquad, and test files are skipped.")
        st.caption("Folder pattern supported: /v3/2026/month/day/CSV. Select months to scan each day folder. Use Optional day folder for a one-day test like /v3/2026/04/04/CSV. Start with 25 downloads, then set Max downloads to 0 for the full pull.")
        if st.button("Import 2026 Game CSVs", use_container_width=True):
            if not ftp_host or not ftp_user or not ftp_password:
                st.error("Host, username, and password are required.")
            else:
                try:
                    with st.spinner("Connecting and importing game CSVs..."):
                        imported, skipped, scanned = import_trackman_2026_from_server(
                            protocol=ftp_protocol,
                            host=ftp_host.strip(),
                            username=ftp_user.strip(),
                            password=ftp_password,
                            remote_dir=ftp_remote_dir.strip() or "/",
                            port=int(ftp_port),
                            timeout=int(ftp_timeout),
                            max_downloads=(None if int(max_downloads) == 0 else int(max_downloads)),
                            months=ftp_months,
                            day_filter=ftp_day,
                            csv_folder=ftp_csv_folder,
                            skip_existing=skip_existing,
                            passive=ftp_passive,
                            recursive=ftp_recursive,
                        )
                    st.success(f"Scanned {scanned} files. Imported {len(imported)} into {SCOUTING_DATA_DIR.name}. Skipped {len(skipped)}.")
                    if imported:
                        st.dataframe(pd.DataFrame({"Imported Files": imported}), use_container_width=True, hide_index=True)
                    if skipped:
                        skipped_df = pd.DataFrame(skipped, columns=["Remote Path", "Reason"]).head(50)
                        st.dataframe(skipped_df, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

    if st.button("Refresh Scouting File Index", use_container_width=True):
        get_scouting_csv_count.clear()
        build_scouting_team_index.clear()
        prepare_scouting_data.clear()
        st.rerun()

    csv_count = get_scouting_csv_count()
    if csv_count:
        with st.spinner("Building team index from scouting CSVs..."):
            index_rows, teams = build_scouting_team_index()
        st.caption(f"Data source: {SCOUTING_DATA_DIR.name} ({csv_count:,} files, {len(teams):,} teams indexed)")
    else:
        fallback_df = prepare_data()
        teams = sorted(set(
            fallback_df.get("BatterTeam", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() +
            fallback_df.get("PitcherTeam", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        ))
        st.caption("Data source: current app data fallback")

    if not teams:
        st.warning("No team values found in BatterTeam or PitcherTeam.")
        return

    mode = st.radio("Scouting View", ["PDF Reports", "2026 Leaderboards"], horizontal=True)

    c1, c2, c3 = st.columns([1.1, 1.0, 1.4])
    with c1:
        default_idx = teams.index("FOR_RAM") if "FOR_RAM" in teams else 0
        team = st.selectbox("Team", teams, index=default_idx)

    if csv_count:
        selected_files = _scouting_files_for_team(team)
        st.caption(f"Selected team file set: {len(selected_files):,} CSVs involving {team}.")
        if not selected_files:
            st.info("No scouting CSVs found for that team.")
            return
        with st.spinner(f"Loading {team} scouting data..."):
            scouting_df = prepare_scouting_data(team)
    else:
        scouting_df = fallback_df

    if scouting_df.empty:
        st.error("No pitch-by-pitch data loaded for that team.")
        return

    df = normalize_hitter_columns(scouting_df)
    df = add_contact_quality_local(df)
    st.caption("Color scale: dark blue = weaker / worse, bright red = stronger / better. Scales are based on the selected table using the app's 2026 TrackMan definitions.")

    if mode == "2026 Leaderboards":
        with c2:
            leaderboard_type = st.radio("Leaderboard", ["Hitters", "Pitchers"], horizontal=True)
        if leaderboard_type == "Hitters":
            sub = df[df["BatterTeam"].astype(str) == team].copy()
            summary = summarize_contact_quality(sub, "Batter").sort_values("wRC+", ascending=False)
            table_context = "hitting"
        else:
            sub = df[df["PitcherTeam"].astype(str) == team].copy()
            summary = summarize_contact_quality(sub, "Pitcher")
            if "Stuff+" in sub.columns and not summary.empty:
                stuff_summary = sub.groupby("Pitcher").agg(Stuff_plus=("Stuff+", "mean")).reset_index()
                loc_summary = sub.groupby("Pitcher").agg(Loc_plus=("Loc+", "mean")).reset_index()
                summary = summary.merge(stuff_summary, on="Pitcher", how="left").merge(loc_summary, on="Pitcher", how="left")
                summary = summary.rename(columns={"Stuff_plus": "Stuff+", "Loc_plus": "Loc+"})
            summary = summary.sort_values("HardHit%", ascending=True)
            table_context = "pitching"
        st.dataframe(style_scouting_dataframe(summary, context=table_context), use_container_width=True, hide_index=True)
        return

    with c2:
        report_type = st.radio("Report Type", ["Hitters", "Pitchers", "Team Report"], horizontal=True)

    if report_type == "Team Report":
        team_hitters = df[df["BatterTeam"].astype(str) == team].copy()
        team_pitchers = df[df["PitcherTeam"].astype(str) == team].copy()
        with c3:
            include_individual_reports = st.checkbox(
                "Include every individual player report",
                value=False,
                help="Adds full hitter and pitcher report pages after the team overview. This can create a long PDF and may take a bit to generate."
            )
            packet_scope = st.radio(
                "Individual Packet",
                ["All Players", "Hitters Only", "Pitchers Only"],
                horizontal=True,
                disabled=not include_individual_reports
            )
        hitting = team_hitting_metrics(team_hitters)
        pitching = team_pitching_metrics(team_pitchers)
        preview = pd.DataFrame([{
            "Team": team,
            "Hitter PA": hitting.get("PA"),
            "BA": hitting.get("BA"),
            "OBP": hitting.get("OBP"),
            "SLG": hitting.get("SLG"),
            "wOBA": hitting.get("wOBA"),
            "wRC+": hitting.get("wRC+"),
            "Pitchers": team_pitchers["Pitcher"].nunique() if "Pitcher" in team_pitchers.columns else 0,
            "Staff Stuff+": round(pitching.get("Stuff+", np.nan), 1) if pd.notna(pitching.get("Stuff+", np.nan)) else np.nan,
            "Staff Loc+": round(pitching.get("Loc+", np.nan), 1) if pd.notna(pitching.get("Loc+", np.nan)) else np.nan,
        }])
        st.dataframe(style_scouting_dataframe(preview, context="hitting"), hide_index=True, use_container_width=True)
        with st.spinner("Building team scouting PDF..."):
            pdf_bytes = build_team_scouting_pdf(
                df,
                team,
                include_individual_reports=include_individual_reports,
                packet_scope=packet_scope
            )
        scope_slug = "" if not include_individual_reports else f"{_safe_pdf_name(packet_scope).lower()}_"
        file_name = f"{_safe_pdf_name(team)}_{scope_slug}team_scouting_report.pdf"
    elif report_type == "Hitters":
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
        st.dataframe(style_scouting_dataframe(preview, context="hitting"), hide_index=True, use_container_width=True)
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
        st.dataframe(style_scouting_dataframe(preview, context="pitching"), hide_index=True, use_container_width=True)
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
        st.dataframe(style_scouting_dataframe(summary, context="hitting"), use_container_width=True)

    else:
        sub = df[df["PitcherTeam"] == team]
        summary = summarize_contact_quality(sub, "Pitcher")
        if "Stuff+" in sub.columns and not summary.empty:
            stuff_summary = sub.groupby("Pitcher").agg(Stuff_plus=("Stuff+", "mean")).reset_index()
            summary = summary.merge(stuff_summary, on="Pitcher", how="left")
            summary = summary.rename(columns={"Stuff_plus": "Stuff+"})
        summary = summary.sort_values("HardHit%", ascending=True)
        st.dataframe(style_scouting_dataframe(summary, context="pitching"), use_container_width=True)


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
    st.dataframe(style_scouting_dataframe(count_df, context="hitting"), use_container_width=True)

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
        "BatterSide", "Date", "Inning", "PitchNumber", "Ext", "Velo", "PerceivedVelo"
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df = add_perceived_velocity(df)
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
        Velo=("Velo", "mean"),
        PerceivedVelo=("PerceivedVelo", "mean"),
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
        style_scouting_dataframe(
            arsenal[["Usage%", "Velo", "PerceivedVelo", "BA", "SLG", "Whiff%", "Chase%", "InZone%", "HardHit%", "AvgEV"]].rename(columns={"PerceivedVelo": "PerVelo"}),
            context="pitching"
        ),
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
        style_scouting_dataframe(count_grid[
            [
                "Count", "pitch_abbr", "N",
                "Whiff%", "Chase%", "Zone%", "CSW%", "K%",
                "HardHit%", "AvgEV"
            ]
        ], context="pitching"),
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
        RelS_std=("RelS", "std"),
        Ext_std=("Ext", "std")
    ).round(3)

    st.dataframe(style_scouting_dataframe(rel, context="pitching"), use_container_width=True)

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
        style_scouting_dataframe(seq_stats[["PrevPitch", "pitch_abbr", "N", "Whiff%", "HardHit%"]], context="pitching"),
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
        style_scouting_dataframe(splits[["BatterSide", "pitch_abbr", "BA", "SLG", "Whiff%", "HardHit%", "AvgEV"]], context="pitching"),
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
                (not np.isnan(rel_mean["RelS_std"]) and rel.loc[pitch, "RelS_std"] > rel_mean["RelS_std"] * 1.5) or
                (not np.isnan(rel_mean["Ext_std"]) and rel.loc[pitch, "Ext_std"] > rel_mean["Ext_std"] * 1.5)
            ):
                recs.append(
                    f"Improve release consistency on **{pitch}** — large variance in release height, side, or extension."
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
    st.dataframe(style_scouting_dataframe(count_df, context="hitting"), use_container_width=True)

    # PITCH-TYPE EFFECTIVENESS
    st.subheader("Hitter Effectiveness vs Pitch Type")
    pitchtype_df = hitter_pitchtype_effectiveness(hdf)
    if pitchtype_df.empty:
        st.info("No pitch-type data available for this hitter.")
    else:
        st.dataframe(style_scouting_dataframe(pitchtype_df, context="hitting"), use_container_width=True)

    # COUNT × PITCH TYPE EFFECTIVENESS (NO wOBA COLUMN)
    st.subheader("Count x Pitch Type Effectiveness")
    cpt_df = count_pitchtype_effectiveness(hdf)
    st.dataframe(style_scouting_dataframe(cpt_df, context="hitting"), use_container_width=True)

    # SPLITS VS LHP / RHP (PA-BASED wOBA)
    st.subheader("Splits vs LHP / RHP")
    splits_df = hitter_splits(hdf)
    if splits_df.empty:
        st.info("No pitcher handedness data available.")
    else:
        st.dataframe(style_scouting_dataframe(splits_df, context="hitting"), use_container_width=True)

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
    st.pyplot(build_hitter_spray_chart(hdf, hitter))

    spray_df = hitter_spray_profile(hdf)
    if spray_df.empty:
        st.info("Not enough directional / EV / LA data for spray profile.")
    else:
        st.dataframe(style_scouting_dataframe(spray_df, context="hitting"), use_container_width=True)

    st.subheader("Best Defensive Positioning")
    pos_fig, pos_df = make_defensive_positioning_chart(hdf, hitter)
    if pos_fig is None:
        st.info("Not enough true batted-ball direction data for defensive positioning.")
    else:
        plt.close(pos_fig)
        st.dataframe(style_scouting_dataframe(pos_df, context="hitting"), use_container_width=True, hide_index=True)

    # SEQUENCING
    st.subheader("Pitch-to-Pitch Sequencing (Hitter Reaction)")

    seq_df = hitter_sequencing(hdf)

    if seq_df.empty:
        st.info("Not enough sequencing data for this hitter.")
        return

    st.dataframe(style_scouting_dataframe(seq_df, context="hitting"), use_container_width=True)

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
        {"Stat": "PerVelo", "What it means": "Fastball perceived velocity adjusted for reaction distance and carry traits.", "App logic": f"Fastballs only: Velo x ((60.5 - {PERCEIVED_VELO_EXT_BASELINE:.1f}) / (60.5 - Extension)), plus a capped IVB/spin adjustment using {PERCEIVED_VELO_IVB_BASELINE:.0f} in IVB and {PERCEIVED_VELO_SPIN_BASELINE:.0f} rpm as baselines."},
        {"Stat": "IVB", "What it means": "Induced vertical break, often read as fastball ride or vertical movement.", "App logic": "TrackMan InducedVertBreak renamed to IVB and averaged by pitch type or sample."},
        {"Stat": "HB", "What it means": "Horizontal break.", "App logic": "TrackMan HorzBreak renamed to HB and averaged by pitch type or sample."},
        {"Stat": "Spin", "What it means": "Pitch spin rate in rpm.", "App logic": "TrackMan SpinRate renamed to Spin and averaged by pitch type or sample."},
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
        {"Stat": "Barrel%", "What it means": "High-value college contact window.", "App logic": f"EV at least {BARREL_EV_MIN} mph with launch angle from {BARREL_LA_MIN} to {BARREL_LA_MAX} degrees."},
        {"Stat": "SweetSpot%", "What it means": "Launch angles most likely to produce line drives and productive fly balls.", "App logic": "Launch angle from 8 to 32 degrees."},
        {"Stat": "wOBA", "What it means": "Weighted on-base average.", "App logic": "PA-ending events weighted as BB .69, HBP .72, 1B .88, 2B 1.247, 3B 1.578, HR 2.031."},
        {"Stat": "wRC+", "What it means": "Run creation relative to average.", "App logic": f"Player wOBA divided by fixed college average wOBA {COLLEGE_AVG_WOBA:.3f}, scaled to 100."},
        {"Stat": "Spray", "What it means": "Pull, middle, and opposite-field batted-ball direction.", "App logic": "Uses TrackMan Direction/Bearing and hitter handedness to convert field direction into hitter-relative spray buckets."},
        {"Stat": "Shift Read", "What it means": "Defensive alignment cue against a hitter.", "App logic": "Combines spray bucket, ground-ball rate, hard-hit rate, opposite-field air contact, and bunt indicators."},
    ])
    st.dataframe(hitting_terms, hide_index=True, use_container_width=True)

    st.markdown("### Color Scale Logic")
    color_terms = pd.DataFrame([
        {"Area": "Table Colors", "App logic": "Dark blue means weaker or worse within the selected table. Bright red means stronger or better."},
        {"Area": "Hitter Context", "App logic": "Higher BA, OBP, SLG, OPS, wOBA, wRC+, BB%, Avg EV, HH%, and Barrel% grade positively. Higher K%, Whiff%, and Chase% grade negatively."},
        {"Area": "Pitcher Context", "App logic": "Higher Stuff+, Loc+, Zone%, CSW%, Whiff%, K%, and Strike% grade positively. Higher BB%, BA allowed, SLG allowed, Avg EV allowed, and HH% allowed grade negatively."},
        {"Area": "Benchmarking", "App logic": "Color scales use the selected table's 2026 TrackMan distribution so grades match the app's exact definitions and data source."},
    ])
    st.dataframe(color_terms, hide_index=True, use_container_width=True)

    st.markdown("### Zone And Positioning Logic")
    zone_terms = pd.DataFrame([
        {"Area": "Strike Zone Heatmaps", "App logic": "Uses plate width from -0.83 to 0.83 feet and zone height from 1.5 to 3.5 feet."},
        {"Area": "9-Box Breakdown", "App logic": "Splits only the strike zone into equal 3-by-3 boxes, Baseball Savant style."},
        {"Area": "Pitch Type Dropdown", "App logic": "Filters the 9-box sample to all pitches or one selected pitch type before calculating the zone values."},
        {"Area": "Spray Chart", "App logic": "Plots batted-ball depth from TrackMan Distance when available, Bearing/Direction for angle, and scales the field to LF 338, CF 395, RF 320."},
        {"Area": "Ground-Ball Lines", "App logic": "Short ground balls keep the true landing point but add an extended infield direction guide toward the 5-6, middle, or 3-4 lane."},
        {"Area": "Defensive Positioning", "App logic": "Uses true BIP direction, launch angle, EV, handedness, pull/middle/oppo rates, ground-ball rate, air rate, hard-hit rate, and bunt frequency."},
        {"Area": "Infield Alignment", "App logic": "Can recommend standard, pull-side shift, middle pinch, guard lines, or corners-in / 3B bunt alert."},
        {"Area": "Outfield Alignment", "App logic": "Shades toward the primary spray bucket and moves deeper when air contact plus hard-hit rate are elevated."},
    ])
    st.dataframe(zone_terms, hide_index=True, use_container_width=True)

    st.markdown("### Report Formatting")
    report_terms = pd.DataFrame([
        {"Area": "Postgame / Season Graphics", "App logic": "Both reports use the same pitcher summary engine: metric cards, movement profile, LHH/RHH location maps, release window, and arsenal table."},
        {"Area": "Arsenal Table", "App logic": "Pitch color is used as a pitch-type anchor while numeric cells use clean dark rows for readability."},
        {"Area": "Decimal Display", "App logic": "Slash-line stats and wOBA show three decimals. Percentages, velocity, movement, extension, Stuff+, and Loc+ show one decimal. Counts show whole numbers."},
    ])
    st.dataframe(report_terms, hide_index=True, use_container_width=True)


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
