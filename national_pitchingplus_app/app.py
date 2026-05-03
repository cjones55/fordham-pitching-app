from __future__ import annotations

import os
import re
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DEFAULT_DATA_DIR = (APP_ROOT.parent / "scouting_2026_trackman").resolve()
LOGO_DIR = APP_ROOT / "team_logos"
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

try:
    from shared import (
        load_models,
        basic_clean,
        add_flags,
        compute_stuffplus,
        compute_locationplus,
    )
except Exception:
    load_models = basic_clean = add_flags = compute_stuffplus = compute_locationplus = None

PITCH_COLORS = {
    "FB": "#2D7FF9",
    "SI": "#13B5C8",
    "FC": "#F59E0B",
    "SL": "#EF4444",
    "SW": "#BE185D",
    "CB": "#8B5CF6",
    "CU": "#8B5CF6",
    "CH": "#10B981",
    "SP": "#10B981",
    "KN": "#CDBFAF",
    "UNK": "#94A3B8",
}

TEAM_NAMES = {
    "FOR_RAM": "Fordham Rams",
    "BIN_BEA": "Binghamton Bearcats",
    "TEN_VOL": "Tennessee Volunteers",
    "FLA_GAT": "Florida Gators",
    "FLA__GAT": "Florida Gators",
    "GEO_BUL": "Georgia Bulldogs",
    "LSU_TIG": "LSU Tigers",
    "ARK_RAZ": "Arkansas Razorbacks",
    "OLE_REB": "Ole Miss Rebels",
    "VIR_CAV": "Virginia Cavaliers",
    "UNC_SEA": "UNCW Seahawks",
    "GEO_PAT": "George Mason Patriots",
}

TEAM_COLORS = {
    "FOR_RAM": ("#8C1515", "#C7A45D"),
    "BIN_BEA": ("#005A43", "#FFFFFF"),
    "TEN_VOL": ("#FF8200", "#58595B"),
    "FLA_GAT": ("#0021A5", "#FA4616"),
    "FLA__GAT": ("#0021A5", "#FA4616"),
    "GEO_BUL": ("#BA0C2F", "#000000"),
    "LSU_TIG": ("#461D7C", "#FDD023"),
    "ARK_RAZ": ("#9D2235", "#FFFFFF"),
    "OLE_REB": ("#CE1126", "#14213D"),
    "VIR_CAV": ("#232D4B", "#F84C1E"),
    "UNC_SEA": ("#006666", "#CBA052"),
    "GEO_PAT": ("#006633", "#FFCC33"),
}


st.set_page_config(
    page_title="CBBReports",
    page_icon="CB",
    layout="wide",
)


def inject_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: #090b10;
            color: #f8fafc;
        }
        div[data-testid="stHeader"] {
            background: transparent;
        }
        .pp-hero {
            border: 1px solid rgba(148, 163, 184, .22);
            background: linear-gradient(135deg, #111827 0%, #111827 55%, #1f2937 100%);
            padding: 24px 28px;
            border-radius: 10px;
            margin-bottom: 16px;
        }
        .pp-hero h1 {
            margin: 0;
            font-size: 34px;
            letter-spacing: 0;
            color: #fff7ed;
        }
        .pp-hero p {
            margin: 6px 0 0 0;
            color: #cbd5e1;
            font-size: 15px;
        }
        .paywall {
            max-width: 760px;
            margin: 40px auto;
            padding: 28px;
            border-radius: 10px;
            border: 1px solid rgba(245, 158, 11, .28);
            background: #111827;
        }
        .metric-card {
            border: 1px solid rgba(148, 163, 184, .22);
            background: #111827;
            border-radius: 8px;
            padding: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_secret_list(name: str) -> list[str]:
    try:
        values = st.secrets.get("auth", {}).get(name, [])
    except Exception:
        values = []
    if isinstance(values, str):
        values = [values]
    env_values = os.environ.get("PITCHINGPLUS_ACCESS_CODES", "")
    if env_values:
        values = list(values) + [v.strip() for v in env_values.split(",") if v.strip()]
    return [str(v).strip() for v in values if str(v).strip()]


def get_secret_value(section: str, name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(section, {}).get(name, default))
    except Exception:
        return default


def check_paywall() -> bool:
    if st.session_state.get("pp_authenticated"):
        return True

    app_name = get_secret_value("auth", "app_name", "CBBReports")
    checkout_url = get_secret_value("auth", "checkout_url", "")
    support_email = get_secret_value("auth", "support_email", "")
    valid_codes = set(get_secret_list("access_codes"))
    if not valid_codes:
        valid_codes = {"DEMO-2026"}

    st.markdown(
        f"""
        <div class="paywall">
            <h1>{app_name}</h1>
            <p>College Baseball Pitching Plus: national pitcher reports, postgame graphics, and player stat cards from the 2026 TrackMan database.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("paywall_form"):
        access_code = st.text_input("Access code", type="password", placeholder="Enter your customer access code")
        submitted = st.form_submit_button("Unlock Reports", use_container_width=True)

    if submitted:
        if access_code.strip() in valid_codes:
            st.session_state["pp_authenticated"] = True
            st.rerun()
        st.error("Invalid access code.")

    cols = st.columns(2)
    if checkout_url:
        cols[0].link_button("Buy Access", checkout_url, use_container_width=True)
    if support_email:
        cols[1].markdown(f"Need access? `{support_email}`")
    if valid_codes == {"DEMO-2026"}:
        st.caption("Local demo mode: use access code DEMO-2026. In production, set customer codes in Streamlit secrets.")
    st.caption("Payment should be handled by Stripe, Gumroad, or another checkout provider. This app enforces customer access after purchase.")
    return False


def data_dir() -> Path:
    configured = get_secret_value("data", "scouting_data_dir", "")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_DIR


def safe_team_name(code: str) -> str:
    code = str(code or "").strip().upper()
    if not code:
        return "Unknown Team"
    if code in TEAM_NAMES:
        return TEAM_NAMES[code]
    parts = [p for p in re.split(r"_+", code) if p]
    return " ".join(p.title() for p in parts) if parts else code


def team_colors(code: str) -> tuple[str, str]:
    code = str(code or "").strip().upper()
    if code in TEAM_COLORS:
        return TEAM_COLORS[code]
    palette = [
        ("#991B1B", "#FBBF24"), ("#0F766E", "#F8FAFC"), ("#1D4ED8", "#F97316"),
        ("#4C1D95", "#FACC15"), ("#111827", "#E5E7EB"), ("#166534", "#FDE68A"),
    ]
    key = sum((i + 1) * ord(ch) for i, ch in enumerate(code))
    return palette[key % len(palette)]


def logo_path_for_team(code: str) -> Path | None:
    code = str(code or "").strip().upper()
    for suffix in [".png", ".jpg", ".jpeg"]:
        path = LOGO_DIR / f"{code}{suffix}"
        if path.exists():
            return path
    return None


def pitch_abbr(value: str) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "fastball": "FB", "four-seam": "FB", "four seam": "FB", "sinker": "SI",
        "two-seam": "SI", "cutter": "FC", "slider": "SL", "sweeper": "SW",
        "curveball": "CB", "changeup": "CH", "splitter": "SP", "knuckleball": "KN",
    }
    return mapping.get(raw, str(value or "UNK").strip().upper()[:2] or "UNK")


def clean_pitch_data(df: pd.DataFrame) -> pd.DataFrame:
    if all(func is not None for func in [load_models, basic_clean, add_flags, compute_stuffplus, compute_locationplus]):
        try:
            out = basic_clean(df.copy())
            out = add_flags(out)
            stuff_model, stuff_league, loc_model, loc_league = load_models(MODELS_DIR)
            out = compute_stuffplus(out, stuff_model, stuff_league)
            out = compute_locationplus(out, loc_model, loc_league)
        except Exception:
            out = df.copy()
        else:
            if "TaggedPitchType" in out.columns:
                out["Pitch"] = out["pitch_abbr"].fillna(out["TaggedPitchType"].map(pitch_abbr))
            else:
                out["Pitch"] = out.get("pitch_abbr", pd.Series("UNK", index=out.index)).fillna("UNK")
            rename_after = {
                "RelSpeed": "Velo",
                "InducedVertBreak": "IVB",
                "HorzBreak": "HB",
                "SpinRate": "Spin",
                "RelHeight": "RelH",
                "RelSide": "RelS",
                "Extension": "Ext",
                "ExitSpeed": "EV",
                "Angle": "LA",
            }
            out = out.rename(columns={k: v for k, v in rename_after.items() if k in out.columns and v not in out.columns})
            for col in ["Velo", "IVB", "HB", "Spin", "RelH", "RelS", "Ext", "PlateLocHeight", "PlateLocSide", "EV", "LA", "OutsOnPlay", "Stuff+", "Loc+"]:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
            if "Date" in out.columns:
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
            return out

    out = df.copy()
    rename = {
        "RelSpeed": "Velo",
        "InducedVertBreak": "IVB",
        "HorzBreak": "HB",
        "SpinRate": "Spin",
        "RelHeight": "RelH",
        "RelSide": "RelS",
        "Extension": "Ext",
        "ExitSpeed": "EV",
        "Angle": "LA",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    for col in ["Velo", "IVB", "HB", "Spin", "RelH", "RelS", "Ext", "PlateLocHeight", "PlateLocSide", "EV", "LA", "OutsOnPlay"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Pitch"] = out.get("TaggedPitchType", pd.Series("UNK", index=out.index)).map(pitch_abbr)
    call = out.get("PitchCall", pd.Series("", index=out.index)).astype(str)
    out["is_whiff"] = call.str.contains("Swinging|StrikeSwinging", case=False, na=False)
    out["is_swing"] = call.str.contains("Swinging|Foul|InPlay", case=False, na=False)
    out["is_csw"] = call.str.contains("StrikeCalled|StrikeSwinging|Swinging", case=False, na=False)
    out["is_strike"] = call.str.contains("Strike|Foul|InPlay", case=False, na=False)
    if {"PlateLocSide", "PlateLocHeight"}.issubset(out.columns):
        out["in_zone"] = out["PlateLocSide"].between(-0.83, 0.83) & out["PlateLocHeight"].between(1.5, 3.5)
    else:
        out["in_zone"] = False
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
    return out


@st.cache_data(show_spinner=False)
def csv_files(folder: str) -> list[str]:
    return [str(p) for p in sorted(Path(folder).glob("*.csv"))]


@st.cache_data(show_spinner=False)
def build_index(folder: str) -> pd.DataFrame:
    rows = []
    usecols = ["Date", "Pitcher", "PitcherTeam", "BatterTeam", "HomeTeam", "AwayTeam", "GameID", "GameUID"]
    for path in csv_files(folder):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype=str, low_memory=False)
        except Exception:
            continue
        if "Pitcher" not in df.columns or "PitcherTeam" not in df.columns:
            continue
        for (team, pitcher), g in df.dropna(subset=["Pitcher", "PitcherTeam"]).groupby(["PitcherTeam", "Pitcher"]):
            rows.append({
                "TeamCode": str(team).strip(),
                "Team": safe_team_name(team),
                "Pitcher": str(pitcher).strip(),
                "Files": 1,
                "Pitches": len(g),
            })
    if not rows:
        return pd.DataFrame(columns=["TeamCode", "Team", "Pitcher", "Files", "Pitches"])
    idx = pd.DataFrame(rows)
    return idx.groupby(["TeamCode", "Team", "Pitcher"], as_index=False).agg(Files=("Files", "sum"), Pitches=("Pitches", "sum"))


@st.cache_data(show_spinner=True)
def load_pitcher_data(folder: str, team_code: str, pitcher: str) -> pd.DataFrame:
    chunks = []
    for path in csv_files(folder):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not {"Pitcher", "PitcherTeam"}.issubset(df.columns):
            continue
        mask = df["PitcherTeam"].astype(str).str.strip().eq(str(team_code)) & df["Pitcher"].astype(str).str.strip().eq(str(pitcher))
        if mask.any():
            chunks.append(df[mask].copy())
    return clean_pitch_data(pd.concat(chunks, ignore_index=True)) if chunks else pd.DataFrame()


def pitcher_stat_card(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    pa = df[df.get("KorBB", pd.Series("", index=df.index)).astype(str).isin(["Walk", "Strikeout"]) |
            df.get("PlayResult", pd.Series("", index=df.index)).astype(str).isin(["Single", "Double", "Triple", "HomeRun", "Out", "Error", "FieldersChoice", "Sacrifice"])]
    swings = df["is_swing"].sum()
    whiffs = df["is_whiff"].sum()
    hits = df.get("PlayResult", pd.Series("", index=df.index)).astype(str).isin(["Single", "Double", "Triple", "HomeRun"]).sum()
    walks = df.get("KorBB", pd.Series("", index=df.index)).astype(str).eq("Walk").sum()
    strikeouts = df.get("KorBB", pd.Series("", index=df.index)).astype(str).eq("Strikeout").sum()
    outs = pd.to_numeric(df.get("OutsOnPlay", pd.Series(0, index=df.index)), errors="coerce").fillna(0).sum() + strikeouts
    ip = int(outs // 3) + (outs % 3) / 10 if outs else np.nan
    ab = max(len(pa) - walks, 0)
    tb = (
        df.get("PlayResult", pd.Series("", index=df.index)).astype(str).eq("Single").sum()
        + 2 * df.get("PlayResult", pd.Series("", index=df.index)).astype(str).eq("Double").sum()
        + 3 * df.get("PlayResult", pd.Series("", index=df.index)).astype(str).eq("Triple").sum()
        + 4 * df.get("PlayResult", pd.Series("", index=df.index)).astype(str).eq("HomeRun").sum()
    )
    return {
        "Pitches": len(df),
        "Games": df.get("GameID", pd.Series(dtype=str)).nunique() if "GameID" in df.columns else df.get("Date", pd.Series(dtype=str)).nunique(),
        "IP": ip,
        "K": strikeouts,
        "BB": walks,
        "K%": strikeouts / len(pa) * 100 if len(pa) else np.nan,
        "BB%": walks / len(pa) * 100 if len(pa) else np.nan,
        "BAA": hits / ab if ab else np.nan,
        "SLG": tb / ab if ab else np.nan,
        "Velo": df["Velo"].mean() if "Velo" in df.columns else np.nan,
        "MaxVelo": df["Velo"].max() if "Velo" in df.columns else np.nan,
        "Stuff+": df["Stuff+"].mean() if "Stuff+" in df.columns else np.nan,
        "Loc+": df["Loc+"].mean() if "Loc+" in df.columns else np.nan,
        "Whiff%": whiffs / swings * 100 if swings else np.nan,
        "Zone%": df["in_zone"].mean() * 100 if "in_zone" in df.columns else np.nan,
        "CSW%": df["is_csw"].mean() * 100 if "is_csw" in df.columns else np.nan,
    }


def fmt(value, stat="") -> str:
    if pd.isna(value):
        return "-"
    if stat in {"BAA", "SLG"}:
        return f"{float(value):.3f}".replace("0.", ".")
    if stat in {"Pitches", "Games", "K", "BB"}:
        return f"{int(round(float(value))):,}"
    return f"{float(value):.1f}"


def build_player_card_png(df: pd.DataFrame, pitcher: str, team_code: str) -> bytes:
    card = pitcher_stat_card(df)
    primary, accent = team_colors(team_code)
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#090B10")
    ax.set_facecolor("#090B10")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, .78), 1, .22, transform=ax.transAxes, color=primary))
    ax.text(.04, .91, pitcher, transform=ax.transAxes, color="white", fontsize=28, fontweight="bold", va="center")
    ax.text(.04, .835, safe_team_name(team_code), transform=ax.transAxes, color=accent, fontsize=15, fontweight="bold")

    logo = logo_path_for_team(team_code)
    if logo:
        try:
            img = Image.open(logo)
            logo_ax = fig.add_axes([.80, .80, .13, .13])
            logo_ax.imshow(img)
            logo_ax.axis("off")
        except Exception:
            pass

    items = ["Pitches", "Games", "IP", "K", "BB", "K%", "BB%", "BAA", "SLG", "Velo", "MaxVelo", "Stuff+", "Loc+", "Whiff%", "Zone%", "CSW%"]
    for i, key in enumerate(items):
        x = .045 + (i % 8) * .116
        y = .58 - (i // 8) * .25
        ax.add_patch(plt.Rectangle((x, y), .095, .15, transform=ax.transAxes, facecolor="#111827", edgecolor="#334155", linewidth=1))
        ax.text(x + .0475, y + .09, fmt(card.get(key), key), transform=ax.transAxes, color="#fff7ed", ha="center", fontsize=16, fontweight="bold")
        ax.text(x + .0475, y + .035, key, transform=ax.transAxes, color="#cbd5e1", ha="center", fontsize=8.5, fontweight="bold")

    ax.text(.04, .08, "CBBReports | College Baseball Pitching Plus | 2026 TrackMan", transform=ax.transAxes, color="#94a3b8", fontsize=10)
    out = BytesIO()
    fig.savefig(out, format="png", dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


def build_pitcher_summary_png(df: pd.DataFrame, pitcher: str, team_code: str, game_id: str | None = None, summary_type: str = "Postgame") -> bytes:
    if summary_type == "Season" or not game_id:
        game_df = df.copy()
        label = "Season Summary"
        date = "2026 Season"
    else:
        game_df = df[df["GameID"].astype(str).eq(str(game_id))].copy() if "GameID" in df.columns else df.copy()
        if game_df.empty:
            game_df = df.copy()
        label = "Postgame Summary"
        date = game_df["Date"].dropna().astype(str).iloc[0] if "Date" in game_df.columns and not game_df["Date"].dropna().empty else "2026"
    card = pitcher_stat_card(game_df)
    primary, accent = team_colors(team_code)
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#090B10")
    gs = fig.add_gridspec(2, 3, height_ratios=[.75, 1.35], width_ratios=[1.25, 1, 1], hspace=.32, wspace=.26)

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.add_patch(plt.Rectangle((0, .05), 1, .9, transform=title_ax.transAxes, color="#111827"))
    title_ax.add_patch(plt.Rectangle((0, .05), .015, .9, transform=title_ax.transAxes, color=accent))
    title_ax.text(.035, .62, pitcher, color="#fff7ed", fontsize=28, fontweight="bold", transform=title_ax.transAxes)
    title_ax.text(.035, .30, f"{safe_team_name(team_code)} | {label} | {date}" + (f" | {game_id}" if game_id and summary_type != "Season" else ""), color="#cbd5e1", fontsize=13, transform=title_ax.transAxes)
    for i, key in enumerate(["Pitches", "IP", "K", "BB", "Stuff+", "Loc+", "Whiff%", "Zone%", "CSW%"]):
        x = .46 + i * .055
        title_ax.text(x, .62, fmt(card.get(key), key), color="#fff7ed", fontsize=16, fontweight="bold", ha="center", transform=title_ax.transAxes)
        title_ax.text(x, .34, key, color="#94a3b8", fontsize=8, fontweight="bold", ha="center", transform=title_ax.transAxes)

    ax_move = fig.add_subplot(gs[1, 0])
    ax_move.set_facecolor("#111827")
    ax_move.axhline(0, color="#64748b", linestyle=":", linewidth=1)
    ax_move.axvline(0, color="#64748b", linestyle=":", linewidth=1)
    for pitch, g in game_df.groupby("Pitch"):
        ax_move.scatter(g["HB"], g["IVB"], s=42, color=PITCH_COLORS.get(pitch, "#94a3b8"), edgecolor="white", linewidth=.35, label=pitch, alpha=.88)
    ax_move.set_title("Movement", color="#fff7ed", fontweight="bold")
    ax_move.set_xlabel("HB", color="#cbd5e1")
    ax_move.set_ylabel("IVB", color="#cbd5e1")
    ax_move.tick_params(colors="#cbd5e1")
    ax_move.grid(color="#334155", alpha=.35)
    ax_move.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#fff7ed", fontsize=8)

    ax_zone = fig.add_subplot(gs[1, 1])
    ax_zone.set_facecolor("#111827")
    ax_zone.plot([-0.83, .83, .83, -.83, -.83], [1.5, 1.5, 3.5, 3.5, 1.5], color="#fff7ed", linewidth=2)
    for pitch, g in game_df.groupby("Pitch"):
        ax_zone.scatter(g["PlateLocSide"], g["PlateLocHeight"], s=45, color=PITCH_COLORS.get(pitch, "#94a3b8"), edgecolor="white", linewidth=.35, alpha=.88)
    ax_zone.set_xlim(-2.2, 2.2)
    ax_zone.set_ylim(.5, 4.5)
    ax_zone.set_title("Locations", color="#fff7ed", fontweight="bold")
    ax_zone.tick_params(colors="#cbd5e1")
    ax_zone.grid(color="#334155", alpha=.28)

    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis("off")
    arsenal = game_df.groupby("Pitch").agg(
        N=("Pitch", "count"),
        Velo=("Velo", "mean"),
        IVB=("IVB", "mean"),
        HB=("HB", "mean"),
        StuffPlus=("Stuff+", "mean") if "Stuff+" in game_df.columns else ("Pitch", "count"),
        LocPlus=("Loc+", "mean") if "Loc+" in game_df.columns else ("Pitch", "count"),
        Zone=("in_zone", "mean"),
        CSW=("is_csw", "mean"),
    ).reset_index()
    arsenal["Zone%"] = arsenal["Zone"] * 100
    arsenal["CSW%"] = arsenal["CSW"] * 100
    arsenal = arsenal.rename(columns={"StuffPlus": "Stuff+", "LocPlus": "Loc+"})
    view = arsenal[["Pitch", "N", "Velo", "IVB", "HB", "Stuff+", "Loc+", "Zone%", "CSW%"]].copy()
    for col in view.columns:
        if col != "Pitch":
            view[col] = view[col].map(lambda v, c=col: fmt(v, c))
    table = ax_table.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        if r == 0:
            cell.set_facecolor(primary)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="#fff7ed", weight="bold")
    ax_table.set_title("Arsenal", color="#fff7ed", fontweight="bold", pad=18)

    out = BytesIO()
    fig.savefig(out, format="png", dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


def main():
    inject_style()
    if not check_paywall():
        return

    st.markdown(
        """
        <div class="pp-hero">
            <h1>College Baseball Pitching Plus</h1>
            <p>CBBReports: build postgame graphics, season summaries, and player stat cards for any pitcher in the 2026 TrackMan database.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    folder = data_dir()
    if not folder.exists():
        st.error(f"Data folder not found: {folder}")
        return

    index = build_index(str(folder))
    if index.empty:
        st.error("No pitcher index could be built from the TrackMan folder.")
        return

    teams = index[["TeamCode", "Team"]].drop_duplicates().sort_values("Team")
    c1, c2, c3 = st.columns([1.2, 1.4, 1.1])
    with c1:
        team_code = st.selectbox("Team", teams["TeamCode"].tolist(), format_func=lambda c: f"{safe_team_name(c)} ({c})")
    team_pitchers = index[index["TeamCode"].eq(team_code)].sort_values(["Pitches", "Pitcher"], ascending=[False, True])
    with c2:
        pitcher = st.selectbox("Pitcher", team_pitchers["Pitcher"].tolist())
    with c3:
        view = st.radio("Graphic", ["Player Stat Card", "Postgame Summary", "Season Summary"], horizontal=False)

    with st.spinner("Loading pitcher TrackMan data..."):
        df = load_pitcher_data(str(folder), team_code, pitcher)
    if df.empty:
        st.warning("No tracked pitches found for that pitcher.")
        return

    card = pitcher_stat_card(df)
    m = st.columns(8)
    for col, key in zip(m, ["Pitches", "Games", "Velo", "MaxVelo", "Stuff+", "Loc+", "K%", "Whiff%"]):
        col.metric(key, fmt(card.get(key), key))

    if view == "Player Stat Card":
        png = build_player_card_png(df, pitcher, team_code)
        st.image(png, use_container_width=True)
        st.download_button("Download Player Stat Card PNG", png, file_name=f"{pitcher.replace(',', '').replace(' ', '_')}_stat_card.png", mime="image/png", use_container_width=True)
    elif view == "Postgame Summary":
        if "GameID" in df.columns:
            games = df.groupby("GameID").agg(Date=("Date", "first"), Pitches=("Pitch", "count")).reset_index().sort_values("Date")
            game_id = st.selectbox("Game", games["GameID"].astype(str).tolist(), format_func=lambda g: f"{games.loc[games['GameID'].astype(str).eq(str(g)), 'Date'].iloc[0]} | {g}")
        else:
            game_id = "Season"
        png = build_pitcher_summary_png(df, pitcher, team_code, game_id, summary_type="Postgame")
        st.image(png, use_container_width=True)
        st.download_button("Download Postgame Summary PNG", png, file_name=f"{pitcher.replace(',', '').replace(' ', '_')}_{game_id}_postgame.png", mime="image/png", use_container_width=True)
    else:
        png = build_pitcher_summary_png(df, pitcher, team_code, summary_type="Season")
        st.image(png, use_container_width=True)
        st.download_button("Download Season Summary PNG", png, file_name=f"{pitcher.replace(',', '').replace(' ', '_')}_season_summary.png", mime="image/png", use_container_width=True)

    with st.expander("Team logos"):
        st.write("Add licensed PNG logos to `national_pitchingplus_app/team_logos/TEAM_CODE.png`. Missing logos use a branded color fallback.")
        logo = logo_path_for_team(team_code)
        if logo:
            st.image(str(logo), width=120)
        else:
            primary, accent = team_colors(team_code)
            st.markdown(
                f"<div style='background:{primary};color:{accent};padding:16px;border-radius:8px;font-weight:800;width:260px'>{safe_team_name(team_code)}</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
