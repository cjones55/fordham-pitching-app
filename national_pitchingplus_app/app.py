from __future__ import annotations

import os
import re
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Polygon
from PIL import Image


APP_ROOT     = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DEFAULT_DATA_DIR = (PROJECT_ROOT / "scouting_2026_trackman").resolve()
LOGO_DIR   = APP_ROOT / "team_logos"
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

try:
    from shared import load_models, basic_clean, add_flags, compute_stuffplus, compute_locationplus
    _SHARED_OK = True
except Exception:
    load_models = basic_clean = add_flags = compute_stuffplus = compute_locationplus = None
    _SHARED_OK = False

# ── Pitch colours (exact match to Fordham app) ────────────────────────────────
PITCH_COLORS = {
    "FB": "#1f77b4", "FF": "#1f77b4",
    "SI": "#17becf", "FT": "#17becf",
    "FC": "#ff9f1c",
    "SL": "#e63946",
    "SW": "#b56576",
    "CU": "#7b2cbf", "CB": "#7b2cbf",
    "CH": "#2a9d8f", "SP": "#2a9d8f",
    "KN": "#CDBFAF",
}

PITCH_LABELS = {
    "FB":"Fastball","SI":"Sinker","FC":"Cutter","SL":"Slider",
    "SW":"Sweeper","CU":"Curveball","CB":"Curveball","CH":"Changeup",
    "SP":"Splitter","KN":"Knuckleball",
}

# ── Full team dicts (ported from Fordham app) ─────────────────────────────────
TEAM_NAMES = {
    "FOR_RAM":"Fordham Rams","FOR_RAM1":"Fordham Rams",
    "FLA__GAT":"Florida Gators","FLA_GAT":"Florida Gators",
    "TEN_VOL":"Tennessee Volunteers","HAW_WAR":"Hawaii Warriors",
    "VIR_CAV":"Virginia Cavaliers","STE_HAT":"Stetson Hatters",
    "UCLA":"UCLA Bruins","ARK_RAZ":"Arkansas Razorbacks",
    "OLE_REB":"Ole Miss Rebels","ORE_BEA":"Oregon State Beavers",
    "NEB":"Nebraska Cornhuskers","OKL_SOO":"Oklahoma Sooners",
    "MIZ_TIG":"Missouri Tigers","BAY_BEA":"Baylor Bears",
    "BIN_BEA":"Binghamton Bearcats","RUT_SCA":"Rutgers Scarlet Knights",
    "SPU_PEA":"Saint Peter's Peacocks","GON_BUL":"Gonzaga Bulldogs",
    "KAN_JAY":"Kansas Jayhawks","MIN_GOL":"Minnesota Golden Gophers",
    "SAN_DON":"San Francisco Dons","SAN_AZT":"San Diego State Aztecs",
    "ECU_PIR":"East Carolina Pirates","FAU_OWL":"Florida Atlantic Owls",
    "FGCU":"Florida Gulf Coast Eagles","UNC_SEA":"UNCW Seahawks",
    "UNCW":"UNCW Seahawks","VCU_RAM":"VCU Rams",
    "JMU_DUK":"James Madison Dukes","UIC_FLA":"UIC Flames",
    "UAB_BLA":"UAB Blazers","MTSU_BLU":"Middle Tennessee Blue Raiders",
    "ULM_WAR":"ULM Warhawks","WIC_SHO":"Wichita State Shockers",
    "PEP_WAV":"Pepperdine Waves","NOT_IRI":"Notre Dame Fighting Irish",
    "TOW_TIG":"Towson Tigers","TRO_T":"Troy Trojans",
    "CSU_BAK":"Cal State Bakersfield Roadrunners",
    "CAL_BEA":"California Golden Bears","USC_UPS":"USC Upstate Spartans",
    "OLD_MON":"Old Dominion Monarchs","IND_SYC":"Indiana State Sycamores",
    "NMS_AGG":"New Mexico State Aggies","MUR_RAC":"Murray State Racers",
    "SIU_SAL":"Southern Illinois Salukis","SAC_HOR":"Sacramento State Hornets",
    "TAR_TEX":"Tarleton State Texans","DAV_WIL":"Davidson Wildcats",
    "WRI_RAI":"Wright State Raiders","WIN_BUL":"Winthrop Eagles",
    "QUI_BOB":"Quinnipiac Bobcats","COR_BRE":"Cornell Big Red",
    "GEO_BUL":"Georgia Bulldogs","GEO_PAT":"George Mason Patriots",
    "GEO_EAG":"Georgia Southern Eagles","GEO_GWI":"George Washington Revolutionaries",
    "GEO_HOY":"Georgetown Hoyas","GEO_PAN":"Georgia State Panthers",
    "GEO_SOU":"Georgia Southern Eagles","SAC_PIO":"Sacred Heart Pioneers",
    "ION_GAL":"Iona Gaels","ION_GAE":"Iona Gaels","STM_GAE":"Saint Mary's Gaels",
    "MAR_RED":"Marist Red Foxes","LOY_LIO":"Loyola Marymount Lions",
    "WAG_SEA":"Wagner Seahawks","FAI_STA":"Fairfield Stags",
    "MAN_JAS":"Manhattan Jaspers","NIA_EAG":"Niagara Purple Eagles",
    "CAN_GRI":"Canisius Golden Griffins","SIE_SAI":"Siena Saints",
    "MON_HAW":"Monmouth Hawks","RIC_SPI":"Richmond Spiders",
    "RHO_RAM":"Rhode Island Rams","DAY_FLY":"Dayton Flyers",
    "LAS_EXP":"La Salle Explorers","SAI_BIL":"Saint Louis Billikens",
    "STL_BIL":"Saint Louis Billikens","SBU_BON":"St. Bonaventure Bonnies",
    "STB_BON":"St. Bonaventure Bonnies","JOE_HAW":"Saint Joseph's Hawks",
    "SAI_JOE":"Saint Joseph's Hawks","STJ_HAW":"Saint Joseph's Hawks",
    "DUQ_DUK":"Duquesne Dukes","LOY_RAM":"Loyola Chicago Ramblers",
    "UMASS":"Massachusetts Minutemen","MAS_MIN":"Massachusetts Minutemen",
    "COL_LION":"Columbia Lions","SBU_SEA":"Stony Brook Seawolves",
    "STJ_RED":"St. John's Red Storm","DUK_BLU":"Duke Blue Devils",
    "UMASS_RIV":"UMass Lowell River Hawks","VAN_COM":"Vanderbilt Commodores",
    "AKR_ZIP":"Akron Zips","ALA_CRI":"Alabama Crimson Tide",
    "BOC_EAG":"Boston College Eagles","OHIO_BOB":"Ohio Bobcats",
    "PUR_BOI":"Purdue Boilermakers","RIC_OWL":"Rice Owls",
    "SET_PIR":"Seton Hall Pirates","PIT_PAN":"Pitt Panthers",
    "BYU_COU":"BYU Cougars","CIN_BEA":"Cincinnati Bearcats",
    "CLE_TIG":"Clemson Tigers","MIC_SPA":"Michigan State Spartans",
    "MIC_WOL":"Michigan Wolverines","CEN_MIC":"Central Michigan Chippewas",
    "LOU_CAJ":"Louisiana Ragin' Cajuns","LOU_CAR":"Louisville Cardinals",
    "FDU_KNI":"Fairleigh Dickinson Knights","FRE_BUL":"Fresno State Bulldogs",
    "LSU_TIG":"LSU Tigers","MIA_HUR":"Miami Hurricanes",
    "MSU_BDG":"Mississippi State Bulldogs","NIU_HUS":"NIU Huskies",
    "ARI_SUN":"Arizona State Sun Devils","AIR_FOR":"Air Force Falcons",
    "ARM_BLA":"Army Black Knights","NAV_MID":"Navy Midshipmen",
    "XAV_MUS":"Xavier Musketeers","AUB_TIG":"Auburn Tigers",
    "NEV_WOL":"Nevada Wolf Pack","UCF_KNI":"UCF Knights",
    "HOU_COG":"Houston Cougars","YAL_BUL":"Yale Bulldogs",
    "NOR_TAR":"North Carolina Tar Heels","NOR_WOL":"NC State Wolfpack",
    "ORE_DUC":"Oregon Ducks","UCO_HUS":"UConn Huskies","CON_HUS":"UConn Huskies",
    "BOS_COL":"Boston College Eagles","DEL_BLU":"Delaware Blue Hens",
    "HOF_PRI":"Hofstra Pride","DRE_DRA":"Drexel Dragons",
    "NOR_HUS":"Northeastern Huskies","ELON_PHO":"Elon Phoenix",
    "CAM_CAM":"Campbell Camels","CHS_COU":"Charleston Cougars",
    "BRY_BUL":"Bryant Bulldogs","LIU_SHA":"LIU Sharks",
    "MER_WAR":"Merrimack Warriors","MAI_BLA":"Maine Black Bears",
    "ALB_GRE":"UAlbany Great Danes","ALB_DAN":"UAlbany Great Danes",
    "UML_RIV":"UMass Lowell River Hawks","LOW_RIV":"UMass Lowell River Hawks",
    "LEH_MOU":"Lehigh Mountain Hawks","LAF_LEO":"Lafayette Leopards",
    "BUC_BIS":"Bucknell Bison","HOL_CRO":"Holy Cross Crusaders",
    "COL_GAT":"Colgate Raiders","RIDER_BRO":"Rider Broncs","RID_BRO":"Rider Broncs",
    "OLD_DOM":"Old Dominion Monarchs","CHA_49E":"Charlotte 49ers",
    "LIB_FLA":"Liberty Flames","APP_MOU":"App State Mountaineers",
    "WVU_MOU":"West Virginia Mountaineers","WES_MOU":"West Virginia Mountaineers",
    "GEO_STA":"Georgia State Panthers","COA_CHA":"Coastal Carolina Chanticleers",
    "UMA_AMH":"UMass Amherst Minutemen","UMBC_RET":"UMBC Retrievers",
    "UNC_SPA":"UNC Greensboro Spartans","USF_BUL":"South Florida Bulls",
    "FLO_SEM":"Florida State Seminoles","HIG_PAN":"High Point Panthers",
    "OSU_BUC":"Ohio State Buckeyes","PEN_NIT":"Penn State Nittany Lions",
    "PEN_QUA":"Penn Quakers","PRI_TIG":"Princeton Tigers",
    "STA_CAR":"Stanford Cardinal","TCU_HFG":"TCU Horned Frogs",
    "TEX_BOB":"Texas State Bobcats","TEX_LON":"Texas Longhorns",
    "TEX_RAI":"Texas Tech Red Raiders","TUL_GRE":"Tulane Green Wave",
    "VIR_TEC":"Virginia Tech Hokies","WAK_DEA":"Wake Forest Demon Deacons",
    "WAS_HUS":"Washington Huskies","KEN_WIL":"Kentucky Wildcats",
    "ILL_ILL":"Illinois Fighting Illini","KAN_WIL":"Kansas State Wildcats",
    "IU":"Indiana Hoosiers","DAL_PAT":"Dallas Baptist Patriots",
    "GIT_YEL":"Georgia Tech Yellow Jackets","SAN_TOR":"San Diego Toreros",
    "SAL_JAG":"South Alabama Jaguars","SAN_GAU":"UC Santa Barbara Gauchos",
    "CAL_FUL":"Cal State Fullerton Titans","CAL_MUS":"Cal Poly Mustangs",
    "CAL_ANT":"UC Irvine Anteaters","CAL_AGO":"UC Davis Aggies",
    "POR_PIL":"Portland Pilots","GRA_CAN":"Grand Canyon Lopes",
    "OKL_COW":"Oklahoma State Cowboys","NOR_AGG":"North Carolina A&T Aggies",
    "CRE_BLU":"Creighton Bluejays","ETS_BUC":"ETSU Buccaneers",
    "SOU_MIS":"Southern Miss Golden Eagles","LIP_BIS":"Lipscomb Bisons",
    "BUT_BUL":"Butler Bulldogs",
}

TEAM_COLORS = {
    "FOR_RAM":("#8C1515","#C7A45D"),"FOR_RAM1":("#8C1515","#C7A45D"),
    "FLA__GAT":("#0021A5","#FA4616"),"FLA_GAT":("#0021A5","#FA4616"),
    "TEN_VOL":("#FF8200","#58595B"),"HAW_WAR":("#024731","#A5ACAF"),
    "VIR_CAV":("#232D4B","#F84C1E"),"STE_HAT":("#006747","#B9975B"),
    "UCLA":("#2774AE","#FFD100"),"ARK_RAZ":("#9D2235","#FFFFFF"),
    "OLE_REB":("#CE1126","#14213D"),"ORE_BEA":("#DC4405","#000000"),
    "NEB":("#E41C38","#FFFFFF"),"MIZ_TIG":("#F1B82D","#000000"),
    "BAY_BEA":("#154734","#FFB81C"),"BIN_BEA":("#005A43","#FFFFFF"),
    "RUT_SCA":("#CC0033","#5F6A72"),"SPU_PEA":("#003DA5","#FFFFFF"),
    "GON_BUL":("#041E42","#C8102E"),"KAN_JAY":("#0051BA","#E8000D"),
    "MIN_GOL":("#7A0019","#FFCC33"),"SAN_DON":("#00543C","#FDBB30"),
    "SAN_AZT":("#A6192E","#000000"),"ECU_PIR":("#592A8A","#FDC82F"),
    "FAU_OWL":("#003366","#CC0000"),"FGCU":("#002D72","#007A33"),
    "UNC_SEA":("#006666","#CBA052"),"UNCW":("#006666","#CBA052"),
    "VCU_RAM":("#FFB300","#000000"),"JMU_DUK":("#450084","#CBB677"),
    "UIC_FLA":("#001E62","#D50032"),"UAB_BLA":("#1E6B52","#FFC845"),
    "MTSU_BLU":("#0066CC","#C0C0C0"),"ULM_WAR":("#840029","#F1B82D"),
    "WIC_SHO":("#FFCD00","#000000"),"PEP_WAV":("#00205C","#F37021"),
    "NOT_IRI":("#0C2340","#C99700"),"TOW_TIG":("#FFBB00","#000000"),
    "TRO_T":("#8A2432","#B3A369"),"CSU_BAK":("#005DAA","#FFC72C"),
    "CAL_BEA":("#003262","#FDB515"),"OLD_MON":("#003057","#7C878E"),
    "IND_SYC":("#0142BC","#FFFFFF"),"NMS_AGG":("#861F41","#000000"),
    "MUR_RAC":("#002144","#ECAC00"),"SIU_SAL":("#720000","#000000"),
    "SAC_HOR":("#043927","#C4B581"),"TAR_TEX":("#4B116F","#FFFFFF"),
    "DAV_WIL":("#AC1A2F","#000000"),"WRI_RAI":("#026937","#FFCC33"),
    "WIN_BUL":("#660000","#FFD200"),"QUI_BOB":("#00205B","#C8102E"),
    "COR_BRE":("#B31B1B","#FFFFFF"),"GEO_BUL":("#BA0C2F","#000000"),
    "GEO_PAT":("#006633","#FFCC33"),"GEO_EAG":("#011E41","#A99260"),
    "GEO_GWI":("#033C5A","#AA9868"),"GEO_HOY":("#041E42","#8D817B"),
    "GEO_PAN":("#0039A6","#C60C30"),"GEO_SOU":("#011E41","#A99260"),
    "SAC_PIO":("#CE1141","#FFFFFF"),"ION_GAL":("#6F2C91","#FFB81C"),
    "ION_GAE":("#6F2C91","#FFB81C"),"STM_GAE":("#D80024","#003A70"),
    "MAR_RED":("#B31B1B","#FFFFFF"),"LOY_LIO":("#A50034","#003B5C"),
    "WAG_SEA":("#006747","#FFFFFF"),"FAI_STA":("#C8102E","#003A70"),
    "MAN_JAS":("#00703C","#FFFFFF"),"NIA_EAG":("#4B116F","#C99700"),
    "CAN_GRI":("#0C2340","#FFCC00"),"SIE_SAI":("#006747","#FFB81C"),
    "MON_HAW":("#041E42","#A7A9AC"),"RIC_SPI":("#990000","#000066"),
    "RHO_RAM":("#68ABE8","#002147"),"DAY_FLY":("#CE1141","#00539B"),
    "LAS_EXP":("#00205B","#FDB515"),"SAI_BIL":("#003DA5","#C8C9C7"),
    "STL_BIL":("#003DA5","#C8C9C7"),"SBU_BON":("#54261A","#FDB515"),
    "STB_BON":("#54261A","#FDB515"),"JOE_HAW":("#9E1B32","#A7A8AA"),
    "SAI_JOE":("#9E1B32","#A7A8AA"),"STJ_HAW":("#9E1B32","#A7A8AA"),
    "DUQ_DUK":("#041E42","#BA0C2F"),"LOY_RAM":("#8D0034","#FFC72C"),
    "UMASS":("#971B2F","#FFFFFF"),"MAS_MIN":("#971B2F","#FFFFFF"),
    "COL_LION":("#75AADB","#FFFFFF"),"SBU_SEA":("#990000","#1F1F1F"),
    "STJ_RED":("#BA0C2F","#FFFFFF"),"DUK_BLU":("#012169","#FFFFFF"),
    "UMASS_RIV":("#003DA5","#C0C0C0"),"VAN_COM":("#000000","#B3A369"),
    "AKR_ZIP":("#041E42","#A89968"),"ALA_CRI":("#9E1B32","#FFFFFF"),
    "BOC_EAG":("#003263","#BC9B6A"),"OHIO_BOB":("#00694E","#FFFFFF"),
    "PUR_BOI":("#CEB888","#000000"),"RIC_OWL":("#00205B","#FFFFFF"),
    "SET_PIR":("#003366","#FFFFFF"),"PIT_PAN":("#003594","#FFB81C"),
    "BYU_COU":("#002255","#FFFFFF"),"CIN_BEA":("#E00122","#000000"),
    "CLE_TIG":("#F66733","#522D80"),"MIC_SPA":("#18453B","#FFFFFF"),
    "MIC_WOL":("#00274C","#FFCB05"),"CEN_MIC":("#6A0032","#FFCB05"),
    "LOU_CAJ":("#CE181E","#000000"),"LOU_CAR":("#AD0000","#000000"),
    "FDU_KNI":("#0033A0","#C8102E"),"FRE_BUL":("#003A70","#C41230"),
    "LSU_TIG":("#461D7C","#FDD023"),"MIA_HUR":("#F47321","#005030"),
    "MSU_BDG":("#660000","#FFFFFF"),"NIU_HUS":("#BA0C2F","#000000"),
    "ARI_SUN":("#8C1D40","#FFC627"),"AIR_FOR":("#003087","#8A8D8F"),
    "ARM_BLA":("#000000","#D4AF37"),"NAV_MID":("#00205B","#C5B783"),
    "XAV_MUS":("#0C2340","#9EA2A2"),"AUB_TIG":("#0C2340","#E87722"),
    "NEV_WOL":("#003366","#A7A9AC"),"UCF_KNI":("#000000","#BA9B37"),
    "HOU_COG":("#C8102E","#FFFFFF"),"YAL_BUL":("#00356B","#FFFFFF"),
    "NOR_TAR":("#7BAFD4","#13294B"),"NOR_WOL":("#CC0000","#000000"),
    "ORE_DUC":("#154733","#FEE123"),"UCO_HUS":("#000E2F","#FFFFFF"),
    "CON_HUS":("#000E2F","#FFFFFF"),"BOS_COL":("#003263","#BC9B6A"),
    "DEL_BLU":("#00539B","#FFD200"),"HOF_PRI":("#003591","#FFB81C"),
    "DRE_DRA":("#07294D","#FFC600"),"NOR_HUS":("#CC0000","#000000"),
    "ELON_PHO":("#73000A","#B59A57"),"CAM_CAM":("#F47920","#000000"),
    "CHS_COU":("#73000A","#000000"),"BRY_BUL":("#000000","#C8102E"),
    "LIU_SHA":("#69BE28","#002F6C"),"MER_WAR":("#002D72","#FDB515"),
    "MAI_BLA":("#003263","#B9975B"),"ALB_GRE":("#46166B","#EEB211"),
    "ALB_DAN":("#46166B","#EEB211"),"UML_RIV":("#003DA5","#C0C0C0"),
    "LOW_RIV":("#003DA5","#C0C0C0"),"LEH_MOU":("#653600","#FFFFFF"),
    "LAF_LEO":("#800000","#FFFFFF"),"BUC_BIS":("#E87722","#002F6C"),
    "HOL_CRO":("#602D89","#FFFFFF"),"COL_GAT":("#821019","#FFFFFF"),
    "RIDER_BRO":("#981E32","#FFFFFF"),"RID_BRO":("#981E32","#FFFFFF"),
    "OLD_DOM":("#003057","#7C878E"),"CHA_49E":("#005035","#A49665"),
    "LIB_FLA":("#002D62","#C41230"),"APP_MOU":("#000000","#FFCC00"),
    "WVU_MOU":("#002855","#EAAA00"),"WES_MOU":("#002855","#EAAA00"),
    "GEO_STA":("#0039A6","#C60C30"),"COA_CHA":("#006F71","#A17A2C"),
    "UMA_AMH":("#971B2F","#FFFFFF"),"UMBC_RET":("#FFCC00","#000000"),
    "UNC_SPA":("#003366","#FFC72C"),"USF_BUL":("#006747","#CFC493"),
    "FLO_SEM":("#782F40","#CEB888"),"HIG_PAN":("#6B2D8B","#FFFFFF"),
    "OSU_BUC":("#BB0000","#666666"),"PEN_NIT":("#1E407C","#FFFFFF"),
    "PEN_QUA":("#011F5B","#990000"),"PRI_TIG":("#E87722","#000000"),
    "STA_CAR":("#8C1515","#FFFFFF"),"TCU_HFG":("#4D1979","#A3A9AC"),
    "TEX_BOB":("#501214","#AC9155"),"TEX_LON":("#BF5700","#FFFFFF"),
    "TEX_RAI":("#CC0000","#000000"),"TUL_GRE":("#006747","#418FDE"),
    "VIR_TEC":("#630031","#CF4420"),"WAK_DEA":("#9E7E38","#000000"),
    "WAS_HUS":("#4B2E83","#E8D3A2"),"KEN_WIL":("#0033A0","#FFFFFF"),
    "ILL_ILL":("#E84A27","#13294B"),"KAN_WIL":("#512888","#D1A82D"),
    "IU":("#990000","#DFBBBB"),"DAL_PAT":("#003087","#C8102E"),
    "GIT_YEL":("#003057","#B3A369"),"SAN_TOR":("#002147","#A5843B"),
    "SAL_JAG":("#00205B","#B9975B"),"SAN_GAU":("#003660","#FDD023"),
    "CAL_FUL":("#00274C","#F47920"),"CAL_MUS":("#154734","#C8B560"),
    "CAL_ANT":("#003764","#FFD200"),"CAL_AGO":("#022851","#DAAA00"),
    "POR_PIL":("#6E0E19","#C5B783"),"GRA_CAN":("#492E7F","#B59A57"),
    "OKL_COW":("#FF6600","#000000"),"NOR_AGG":("#004684","#FFD966"),
    "CRE_BLU":("#005CA9","#FFFFFF"),"ETS_BUC":("#041E42","#FFCC00"),
    "SOU_MIS":("#FFC72C","#000000"),"LIP_BIS":("#00205B","#C8A84B"),
    "BUT_BUL":("#13294B","#747F7F"),
}

BG   = "#1e1e1e"
TXT  = "#FFFFFF"
TXT2 = "#CCCCCC"

st.set_page_config(page_title="CBBReports", page_icon="⚾", layout="wide")


# ── Cached model loader (once per session) ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_models():
    if not _SHARED_OK or load_models is None:
        return None, None, None, None
    try:
        sm, sl, lm, ll = load_models(MODELS_DIR)
        return sm, sl, lm, ll
    except Exception:
        return None, None, None, None


# ── Helpers ───────────────────────────────────────────────────────────────────

def readable_text_color(bg_hex: str) -> str:
    h = bg_hex.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return "#000000" if (0.299*r + 0.587*g + 0.114*b)/255 > 0.55 else "#FFFFFF"


def safe_team_name(code: str) -> str:
    code = str(code or "").strip()
    if code in TEAM_NAMES:
        return TEAM_NAMES[code]
    parts = [p for p in re.split(r"_+", code) if p]
    return " ".join(p.title() for p in parts) if parts else code


def get_team_colors(code: str) -> tuple[str, str]:
    code = str(code or "").strip()
    if code in TEAM_COLORS:
        return TEAM_COLORS[code]
    palette = [("#991B1B","#FBBF24"),("#0F766E","#F8FAFC"),("#1D4ED8","#F97316"),
               ("#4C1D95","#FACC15"),("#166534","#FDE68A"),("#7C3AED","#FCD34D")]
    key = sum((i+1)*ord(ch) for i,ch in enumerate(code))
    return palette[key % len(palette)]


def logo_path_for_team(code: str) -> Path | None:
    for suffix in [".png",".jpg",".jpeg"]:
        p = LOGO_DIR / f"{str(code or '').strip()}{suffix}"
        if p.exists():
            return p
    return None


def pc(pt: str) -> str:
    return PITCH_COLORS.get(str(pt).upper()[:2], "#888888")


def fmt(v, stat="") -> str:
    if pd.isna(v):
        return "—"
    if stat in {"BAA","SLG"}:
        return f"{float(v):.3f}".replace("0.",".")
    if stat in {"Pitches","Games","K","BB","N"}:
        return f"{int(round(float(v))):,}"
    if stat == "Usage%":
        return f"{float(v):.1f}%"
    return f"{float(v):.1f}"


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_secret_list(name):
    try:
        values = st.secrets.get("auth",{}).get(name,[])
    except Exception:
        values = []
    if isinstance(values,str):
        values = [values]
    env = os.environ.get("PITCHINGPLUS_ACCESS_CODES","")
    if env:
        values = list(values) + [v.strip() for v in env.split(",") if v.strip()]
    return [str(v).strip() for v in values if str(v).strip()]


def get_secret_value(section,name,default=""):
    try:
        return str(st.secrets.get(section,{}).get(name,default))
    except Exception:
        return default


def inject_style():
    st.markdown("""
    <style>
    .stApp{background:#1e1e1e;color:#fff}
    div[data-testid="stHeader"]{background:transparent}
    .cbb-hero{background:#111;border:1px solid #333;border-radius:8px;padding:20px 24px;margin-bottom:16px}
    .cbb-hero h1{margin:0;font-size:28px;color:#fff}
    .cbb-hero p{margin:4px 0 0;color:#aaa;font-size:14px}
    .paywall{max-width:680px;margin:40px auto;padding:28px;border-radius:10px;border:1px solid #444;background:#111}
    </style>""", unsafe_allow_html=True)


def check_paywall() -> bool:
    if st.session_state.get("pp_authenticated"):
        return True
    app_name     = get_secret_value("auth","app_name","CBBReports")
    checkout_url = get_secret_value("auth","checkout_url","")
    support_email= get_secret_value("auth","support_email","")
    valid_codes  = set(get_secret_list("access_codes")) or {"DEMO-2026"}
    st.markdown(f"""
    <div class="paywall">
      <h1 style="color:#fff;margin:0 0 8px">{app_name}</h1>
      <p style="color:#aaa">College Baseball Pitching Plus — national pitcher reports,
      postgame graphics, and stat cards powered by the 2026 TrackMan database.</p>
    </div>""", unsafe_allow_html=True)
    with st.form("paywall_form"):
        code = st.text_input("Access code", type="password", placeholder="Enter your access code")
        if st.form_submit_button("Unlock Reports ⚾", use_container_width=True):
            if code.strip() in valid_codes:
                st.session_state["pp_authenticated"] = True
                st.rerun()
            st.error("Invalid access code.")
    c1,c2 = st.columns(2)
    if checkout_url:
        c1.link_button("Buy Access →", checkout_url, use_container_width=True)
    if support_email:
        c2.markdown(f"<p style='color:#aaa;padding-top:8px'>{support_email}</p>", unsafe_allow_html=True)
    if valid_codes == {"DEMO-2026"}:
        st.caption("Demo mode — use code DEMO-2026")
    return False


# ── Data pipeline ─────────────────────────────────────────────────────────────

def data_dir() -> Path:
    configured = get_secret_value("data","scouting_data_dir","")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_DIR


@st.cache_data(show_spinner=False)
def csv_files(folder: str) -> list[str]:
    return [str(p) for p in sorted(Path(folder).glob("*.csv"))]


@st.cache_data(show_spinner="Building pitcher index…")
def build_index(folder: str) -> pd.DataFrame:
    """Returns (TeamCode, Pitcher, Pitches, Files) where Files is a list of paths."""
    usecols = ["Pitcher","PitcherTeam"]
    rows = []
    for path in csv_files(folder):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in usecols,
                             dtype=str, low_memory=False)
        except Exception:
            continue
        if "Pitcher" not in df.columns or "PitcherTeam" not in df.columns:
            continue
        df = df.dropna(subset=["Pitcher","PitcherTeam"])
        df["Pitcher"]     = df["Pitcher"].str.strip()
        df["PitcherTeam"] = df["PitcherTeam"].str.strip()
        for (team, pitcher), g in df.groupby(["PitcherTeam","Pitcher"]):
            rows.append({"TeamCode":team,"Pitcher":pitcher,
                         "Pitches":len(g),"File":path})
    if not rows:
        return pd.DataFrame(columns=["TeamCode","Team","Pitcher","Pitches","Files"])
    raw = pd.DataFrame(rows)
    idx = (raw.groupby(["TeamCode","Pitcher"], as_index=False)
              .agg(Pitches=("Pitches","sum"), Files=("File", list)))
    idx["Team"] = idx["TeamCode"].map(safe_team_name)
    return idx


def _fallback_clean(df: pd.DataFrame) -> pd.DataFrame:
    PITCH_MAP = {
        "Fastball":"FB","FourSeamFastBall":"FB","FourSeamFastball":"FB","4-Seam":"FB",
        "Sinker":"SI","Cutter":"FC","Slider":"SL","Sweeper":"SW",
        "Curveball":"CU","CurveBall":"CU","ChangeUp":"CH","Changeup":"CH",
    }
    out = df.copy()
    rename = {"RelSpeed":"Velo","InducedVertBreak":"IVB","HorzBreak":"HB",
              "SpinRate":"Spin","RelHeight":"RelH","RelSide":"RelS",
              "Extension":"Ext","ExitSpeed":"EV","Angle":"LA"}
    out = out.rename(columns={k:v for k,v in rename.items() if k in out.columns})
    if "TaggedPitchType" in out.columns:
        out["Pitch"] = out["TaggedPitchType"].map(PITCH_MAP).fillna(
            out["TaggedPitchType"].astype(str).str[:2].str.upper())
    else:
        out["Pitch"] = "UNK"
    call = out.get("PitchCall", pd.Series("", index=out.index)).astype(str)
    out["is_whiff"]  = call.isin(["StrikeSwinging"])
    out["is_swing"]  = call.isin(["StrikeSwinging","FoulBall","FoulBallNotFieldable","InPlay","InPlayNoOut","InPlayOut"])
    out["is_csw"]    = call.isin(["StrikeCalled","StrikeSwinging"])
    out["is_strike"] = call.isin(["StrikeCalled","StrikeSwinging","FoulBall","FoulBallNotFieldable","InPlay","InPlayNoOut","InPlayOut"])
    if {"PlateLocSide","PlateLocHeight"}.issubset(out.columns):
        out["in_zone"] = out["PlateLocSide"].between(-0.83,0.83) & out["PlateLocHeight"].between(1.5,3.5)
    else:
        out["in_zone"] = False
    for col in ["Velo","IVB","HB","Spin","RelH","RelS","Ext","PlateLocHeight","PlateLocSide"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
    return out


def _add_perceived_velo(df: pd.DataFrame) -> pd.DataFrame:
    if "Velo" not in df.columns or "Ext" not in df.columns:
        return df
    velo = pd.to_numeric(df["Velo"], errors="coerce")
    ext  = pd.to_numeric(df["Ext"],  errors="coerce")
    df["PerceivedVelo"] = velo * (60.5 / (60.5 - ext.clip(upper=7)))
    return df


def clean_pitch_data(df: pd.DataFrame) -> pd.DataFrame:
    sm, sl, lm, ll = _get_models()
    if _SHARED_OK and basic_clean is not None and sm is not None:
        try:
            out = basic_clean(df.copy())
            out = add_flags(out)
            out = compute_stuffplus(out, sm, sl)
            out = compute_locationplus(out, lm, ll)
            out["Pitch"] = out.get("pitch_abbr", pd.Series("UNK", index=out.index)).fillna("UNK")
            rename_extra = {"RelSpeed":"Velo","InducedVertBreak":"IVB","HorzBreak":"HB",
                            "SpinRate":"Spin","RelHeight":"RelH","RelSide":"RelS",
                            "Extension":"Ext","ExitSpeed":"EV","Angle":"LA"}
            out = out.rename(columns={k:v for k,v in rename_extra.items()
                                       if k in out.columns and v not in out.columns})
            for col in ["Velo","IVB","HB","Spin","RelH","RelS","Ext","PlateLocHeight","PlateLocSide","Stuff+","Loc+"]:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
            out = _add_perceived_velo(out)
            if "Date" in out.columns:
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
            return out
        except Exception:
            pass
    out = _fallback_clean(df)
    return _add_perceived_velo(out)


@st.cache_data(show_spinner="Loading pitcher data…")
def load_pitcher_data(folder: str, team_code: str, pitcher: str,
                      file_list: tuple) -> pd.DataFrame:
    """Load only the specific files that contain this pitcher (fast)."""
    chunks = []
    for path in file_list:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not {"Pitcher","PitcherTeam"}.issubset(df.columns):
            continue
        mask = (df["PitcherTeam"].astype(str).str.strip() == str(team_code)) & \
               (df["Pitcher"].astype(str).str.strip() == str(pitcher))
        if mask.any():
            chunks.append(df[mask].copy())
    return clean_pitch_data(pd.concat(chunks, ignore_index=True)) if chunks else pd.DataFrame()


# ── Stats ─────────────────────────────────────────────────────────────────────

def pitcher_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    pr  = df.get("PlayResult", pd.Series("", index=df.index)).astype(str)
    kbb = df.get("KorBB",      pd.Series("", index=df.index)).astype(str)
    pa_mask = kbb.isin(["Walk","Strikeout"]) | pr.isin(
        ["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice","Sacrifice"])
    pa = df[pa_mask]
    hits  = pr.isin(["Single","Double","Triple","HomeRun"]).sum()
    walks = kbb.eq("Walk").sum()
    ks    = kbb.eq("Strikeout").sum()
    outs  = pd.to_numeric(df.get("OutsOnPlay",0), errors="coerce").fillna(0).sum() + ks
    ab    = max(len(pa) - walks, 0)
    tb    = pr.eq("Single").sum()+2*pr.eq("Double").sum()+3*pr.eq("Triple").sum()+4*pr.eq("HomeRun").sum()
    swings = df.get("is_swing", pd.Series(False, index=df.index)).sum()
    whiffs = df.get("is_whiff", pd.Series(False, index=df.index)).sum()
    return {
        "Pitches": len(df),
        "Games":   df.get("GameID", df.get("Date", pd.Series(dtype=str))).nunique(),
        "IP":      int(outs//3)+(outs%3)/10 if outs else np.nan,
        "K": ks, "BB": walks,
        "K%":     ks/len(pa)*100    if len(pa) else np.nan,
        "BB%":    walks/len(pa)*100 if len(pa) else np.nan,
        "BAA":    hits/ab  if ab else np.nan,
        "SLG":    tb/ab    if ab else np.nan,
        "Velo":   df["Velo"].mean()   if "Velo"   in df.columns else np.nan,
        "MaxVelo":df["Velo"].max()    if "Velo"   in df.columns else np.nan,
        "FB Velo": df.loc[df.get("Pitch","").eq("FB"),"Velo"].mean() if "Pitch" in df.columns and "Velo" in df.columns else np.nan,
        "FB PercVelo": df.loc[df.get("Pitch","").eq("FB"),"PerceivedVelo"].mean() if "Pitch" in df.columns and "PerceivedVelo" in df.columns else np.nan,
        "Stuff+": df["Stuff+"].mean() if "Stuff+" in df.columns else np.nan,
        "Loc+":   df["Loc+"].mean()   if "Loc+"   in df.columns else np.nan,
        "Whiff%": whiffs/swings*100   if swings   else np.nan,
        "Zone%":  df.get("in_zone",pd.Series(False,index=df.index)).mean()*100,
        "CSW%":   df.get("is_csw", pd.Series(False,index=df.index)).mean()*100,
    }


def arsenal_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Pitch" not in df.columns:
        return pd.DataFrame()
    total = len(df)
    agg_dict = {"N":("Pitch","count"),"Velo":("Velo","mean"),
                "IVB":("IVB","mean"),"HB":("HB","mean"),"Spin":("Spin","mean")}
    agg = df.groupby("Pitch").agg(**agg_dict).reset_index()
    agg["Usage%"] = agg["N"] / total * 100
    if "Stuff+" in df.columns:
        agg["Stuff+"] = df.groupby("Pitch")["Stuff+"].mean().reindex(agg["Pitch"]).values
    if "Loc+" in df.columns:
        agg["Loc+"]   = df.groupby("Pitch")["Loc+"].mean().reindex(agg["Pitch"]).values
    if "is_whiff" in df.columns and "is_swing" in df.columns:
        w = df.groupby("Pitch").apply(
            lambda g: g["is_whiff"].sum()/g["is_swing"].sum()*100
            if g["is_swing"].sum() else np.nan)
        agg["Whiff%"] = agg["Pitch"].map(w)
    if "in_zone" in df.columns:
        agg["Zone%"] = df.groupby("Pitch")["in_zone"].mean().reindex(agg["Pitch"]).values * 100
    if "is_csw" in df.columns:
        agg["CSW%"]  = df.groupby("Pitch")["is_csw"].mean().reindex(agg["Pitch"]).values * 100
    return agg.sort_values("N", ascending=False).reset_index(drop=True)


# ── Graphics (mirroring Fordham postgame_or_season_card layout) ───────────────

def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TXT2, which="both", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#444444")

def _draw_zone(ax):
    ax.set_facecolor(BG)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.plot([-0.83,0.83,0.83,-0.83,-0.83],[1.5,1.5,3.5,3.5,1.5], color="white", linewidth=2.5)
    ax.fill_between([-0.83,0.83],1.5,3.5, color="white", alpha=0.06)
    # home plate
    ax.plot([-0.83,0.83,0.83,0,-0.83,-0.83],[0,0,0.17,0.34,0.17,0], color="white", linewidth=2)

def build_summary_png(df: pd.DataFrame, pitcher: str, team_code: str,
                      game_id: str | None = None, label: str = "Season Summary") -> bytes:
    game_df = df.copy()
    date_str = "2026 Season"
    if game_id and "GameID" in df.columns:
        filtered = df[df["GameID"].astype(str).eq(str(game_id))]
        if not filtered.empty:
            game_df = filtered
            date_str = str(game_df["Date"].dropna().iloc[0]) if "Date" in game_df.columns else "2026"

    primary, accent = get_team_colors(team_code)
    txt_on_primary  = readable_text_color(primary)
    card  = pitcher_stats(game_df)
    arsen = arsenal_table(game_df)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)

    # ── title + stats header ──────────────────────────────────────────────────
    fig.text(0.5, 0.975, pitcher,
             ha="center", color=primary, fontsize=26, fontweight="bold")
    stat_keys = ["Pitches","IP","K","BB","FB Velo","FB PercVelo","MaxVelo","Stuff+","Loc+","K%","Whiff%","Zone%","CSW%"]
    summary_str = "   ".join(f"{k}: {fmt(card.get(k),k)}" for k in stat_keys)
    fig.text(0.5, 0.945, f"{safe_team_name(team_code)}  ·  {label}  ·  {date_str}",
             ha="center", color=TXT2, fontsize=13)
    fig.text(0.5, 0.922, summary_str, ha="center", color=TXT2, fontsize=11)

    logo = logo_path_for_team(team_code)
    if logo:
        try:
            img = Image.open(logo)
            logo_ax = fig.add_axes([0.88, 0.91, 0.09, 0.08])
            logo_ax.imshow(img); logo_ax.axis("off")
        except Exception:
            pass

    # ── subplot grid: mirrors Fordham postgame_or_season_card ─────────────────
    # rows 0-1: 4 scatter panels
    ax_move = plt.subplot2grid((5,4), (0,0), rowspan=2, fig=fig)
    ax_lhh  = plt.subplot2grid((5,4), (0,1), rowspan=2, fig=fig)
    ax_rhh  = plt.subplot2grid((5,4), (0,2), rowspan=2, fig=fig)
    ax_rel  = plt.subplot2grid((5,4), (0,3), rowspan=2, fig=fig)
    # row 2-3: full-width arsenal table
    ax_tbl  = plt.subplot2grid((5,4), (2,0), colspan=4, rowspan=2, fig=fig)
    # row 4: pitch mix + footer
    ax_foot = plt.subplot2grid((5,4), (4,0), colspan=4, fig=fig)

    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.05, right=0.97,
                        hspace=0.35, wspace=0.30)

    # Movement chart
    _style_ax(ax_move)
    throws = game_df["PitcherThrows"].iloc[0] if "PitcherThrows" in game_df.columns else "Right"
    arm_x  = (0,25)  if throws.upper().startswith("R") else (-25,0)
    glv_x  = (-25,0) if throws.upper().startswith("R") else (0,25)
    ax_move.axvspan(*arm_x, facecolor=(0.10,0.30,0.60,0.10))
    ax_move.axvspan(*glv_x, facecolor=(0.60,0.10,0.10,0.10))
    ax_move.axhline(0, color="white", linestyle=":", linewidth=1.2)
    ax_move.axvline(0, color="white", linestyle=":", linewidth=1.2)
    ax_move.set_xlim(-25,25); ax_move.set_ylim(-25,25)
    for _, row in game_df.iterrows():
        ax_move.scatter(row.get("HB"), row.get("IVB"), s=40,
                        color=pc(row["Pitch"]), edgecolor="white", linewidth=0.5)
    for pt, g in game_df.groupby("Pitch"):
        cx, cy = g["HB"].mean(), g["IVB"].mean()
        ax_move.scatter(cx, cy, s=250, color=pc(pt), edgecolor="white", linewidth=1.5)
        ax_move.text(cx, cy, pt, color="white", fontsize=9, weight="bold", ha="center", va="center")
    ax_move.set_title("Movement", color="white", fontsize=11, fontweight="bold")
    ax_move.set_xlabel("Horizontal Break", color=TXT2, fontsize=8)
    ax_move.set_ylabel("Induced Vert Break", color=TXT2, fontsize=8)

    # LHH location
    _draw_zone(ax_lhh)
    lhh = game_df[game_df.get("BatterSide","").eq("Left") if "BatterSide" in game_df.columns else pd.Series(False,index=game_df.index)]
    for _, row in lhh.iterrows():
        ax_lhh.scatter(row.get("PlateLocSide"), row.get("PlateLocHeight"),
                       s=70, color=pc(row["Pitch"]), edgecolor="white", linewidth=0.5)
    ax_lhh.set_title("vs LHH", color="white", fontsize=11, fontweight="bold")

    # RHH location
    _draw_zone(ax_rhh)
    rhh = game_df[game_df["BatterSide"].eq("Right")] if "BatterSide" in game_df.columns else game_df
    for _, row in rhh.iterrows():
        ax_rhh.scatter(row.get("PlateLocSide"), row.get("PlateLocHeight"),
                       s=70, color=pc(row["Pitch"]), edgecolor="white", linewidth=0.5)
    ax_rhh.set_title("vs RHH", color="white", fontsize=11, fontweight="bold")

    # Release point
    _style_ax(ax_rel)
    ax_rel.set_xlim(-4,4); ax_rel.set_ylim(3,7)
    ax_rel.set_aspect("equal", adjustable="box")
    if "RelS" in game_df.columns and "RelH" in game_df.columns:
        for _, row in game_df.iterrows():
            ax_rel.scatter(row.get("RelS"), row.get("RelH"),
                           s=25, color=pc(row["Pitch"]), edgecolor="white", linewidth=0.3)
    ax_rel.set_title("Release Point", color="white", fontsize=11, fontweight="bold")
    ax_rel.set_xlabel("Horizontal", color=TXT2, fontsize=8)
    ax_rel.set_ylabel("Height", color=TXT2, fontsize=8)
    ax_rel.invert_xaxis()

    # Arsenal table
    ax_tbl.axis("off")
    if not arsen.empty:
        cols_show = ["Pitch","N","Usage%","Velo","IVB","HB","Spin"]
        for x in ["Stuff+","Loc+","Whiff%","Zone%","CSW%"]:
            if x in arsen.columns:
                cols_show.append(x)
        view = arsen[cols_show].copy()
        for col in cols_show:
            if col != "Pitch":
                view[col] = view[col].apply(lambda v: fmt(v, col))
        tbl = ax_tbl.table(cellText=view.values, colLabels=view.columns,
                           loc="center", cellLoc="center", bbox=[0,0,1,1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        for (r,c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(primary)
                cell.set_text_props(color=txt_on_primary, weight="bold")
            else:
                pt = view.iloc[r-1]["Pitch"] if r-1 < len(view) else ""
                cell.set_facecolor(pc(pt))
                cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("#2a2a2a")

    # Footer: pitch mix bar + legend
    ax_foot.axis("off")
    if not arsen.empty and "Usage%" in arsen.columns:
        x_cur = 0.0
        for _, row in arsen.iterrows():
            w = row["Usage%"] / 100.0
            ax_foot.add_patch(plt.Rectangle((x_cur,0.5), w, 0.45,
                facecolor=pc(row["Pitch"]), edgecolor=BG, linewidth=0.8,
                transform=ax_foot.transAxes))
            if w > 0.05:
                ax_foot.text(x_cur + w/2, 0.72, f"{row['Pitch']} {row['Usage%']:.0f}%",
                    transform=ax_foot.transAxes, color="white",
                    ha="center", va="center", fontsize=9, fontweight="bold")
            x_cur += w
    ax_foot.text(0.5, 0.15, "CBBReports  ·  College Baseball Pitching Plus  ·  2026 TrackMan",
                 transform=ax_foot.transAxes, ha="center", color=TXT2, fontsize=9)

    out = BytesIO()
    fig.savefig(out, format="png", dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


def build_stat_card_png(df: pd.DataFrame, pitcher: str, team_code: str) -> bytes:
    primary, accent = get_team_colors(team_code)
    txt_on = readable_text_color(primary)
    card  = pitcher_stats(df)
    arsen = arsenal_table(df)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.axis("off")

    ax.add_patch(plt.Rectangle((0,0.78),1,0.22, transform=ax.transAxes,
                                color=primary, zorder=2))
    ax.text(0.03,0.90, pitcher, transform=ax.transAxes,
            color=txt_on, fontsize=26, fontweight="bold", va="center", zorder=3)
    ax.text(0.03,0.83, f"{safe_team_name(team_code)}  ·  2026 Season",
            transform=ax.transAxes, color=accent, fontsize=12, fontweight="bold",
            va="center", zorder=3)

    logo = logo_path_for_team(team_code)
    if logo:
        try:
            img = Image.open(logo)
            logo_ax = fig.add_axes([0.83,0.81,0.13,0.15])
            logo_ax.imshow(img); logo_ax.axis("off")
        except Exception:
            pass

    stat_keys = ["Pitches","Games","IP","K","BB","K%","BB%","BAA",
                 "SLG","Velo","MaxVelo","Stuff+","Loc+","Whiff%","Zone%","CSW%"]
    tw, th = 0.104, 0.145
    for i, key in enumerate(stat_keys):
        ci, ri = i % 8, i // 8
        x = 0.025 + ci*(tw+0.008)
        y = 0.585 - ri*0.21
        ax.add_patch(plt.Rectangle((x,y),tw,th, transform=ax.transAxes,
                                   facecolor="#111111", edgecolor="#333333",
                                   linewidth=0.8, zorder=2))
        ax.text(x+tw/2, y+th*0.62, fmt(card.get(key),key), transform=ax.transAxes,
                color=TXT, ha="center", fontsize=15, fontweight="bold", zorder=3)
        ax.text(x+tw/2, y+th*0.18, key, transform=ax.transAxes,
                color=TXT2, ha="center", fontsize=8, fontweight="bold", zorder=3)

    if not arsen.empty and "Usage%" in arsen.columns:
        bx, by, bh = 0.025, 0.06, 0.065
        tw_total = 0.97 - bx
        x_cur = bx
        for _, row in arsen.iterrows():
            sw = (row["Usage%"]/100) * tw_total
            if sw < 0.005: continue
            ax.add_patch(plt.Rectangle((x_cur,by),sw,bh, transform=ax.transAxes,
                                       facecolor=pc(row["Pitch"]), edgecolor=BG,
                                       linewidth=0.5, zorder=2))
            if sw > 0.045:
                ax.text(x_cur+sw/2, by+bh/2, f"{row['Pitch']}\n{row['Usage%']:.0f}%",
                        transform=ax.transAxes, color="white",
                        ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=3)
            x_cur += sw
        ax.text(bx, by-0.03, "Pitch Mix", transform=ax.transAxes, color=TXT2, fontsize=8)

    ax.text(0.025, 0.013, "CBBReports  ·  College Baseball Pitching Plus  ·  2026 TrackMan",
            transform=ax.transAxes, color=TXT2, fontsize=8.5)

    out = BytesIO()
    fig.savefig(out, format="png", dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    inject_style()
    if not check_paywall():
        return

    st.markdown("""
    <div class="cbb-hero">
        <h1>⚾ College Baseball Pitching Plus</h1>
        <p>CBBReports — postgame graphics, season summaries, and stat cards for any pitcher in the 2026 TrackMan database.</p>
    </div>""", unsafe_allow_html=True)

    folder = data_dir()
    if not folder.exists():
        st.error(f"Data folder not found: {folder}")
        return

    # Warm models in background so first pitcher load is fast
    _get_models()

    index = build_index(str(folder))
    if index.empty:
        st.error("No pitchers found in the TrackMan folder.")
        return

    # Filter to known D1 teams
    known = index[index["TeamCode"].isin(TEAM_NAMES)].copy()
    if known.empty:
        known = index.copy()

    teams = known[["TeamCode","Team"]].drop_duplicates().sort_values("Team")

    c1, c2, c3 = st.columns([1.3, 1.5, 1.0])
    with c1:
        team_code = st.selectbox("Team", teams["TeamCode"].tolist(),
                                 format_func=safe_team_name)
    team_rows = known[known["TeamCode"].eq(team_code)].sort_values(
        ["Pitches","Pitcher"], ascending=[False,True])
    with c2:
        pitcher = st.selectbox(
            "Pitcher", team_rows["Pitcher"].tolist(),
            format_func=lambda p: f"{p}  ({int(team_rows.loc[team_rows.Pitcher==p,'Pitches'].iloc[0]):,} pitches)")
    with c3:
        view = st.radio("Report", ["Stat Card","Postgame Summary","Season Summary"])

    # Get file list from index (fast path — only read relevant files)
    row = known[(known["TeamCode"]==team_code) & (known["Pitcher"]==pitcher)]
    file_list = tuple(row["Files"].iloc[0]) if not row.empty else ()

    df = load_pitcher_data(str(folder), team_code, pitcher, file_list)
    if df.empty:
        st.warning("No tracked pitches found for that pitcher.")
        return

    has_models = "Stuff+" in df.columns and df["Stuff+"].notna().any()
    if not has_models:
        st.info("Stuff+ / Loc+ models unavailable — showing raw TrackMan metrics only.")

    card = pitcher_stats(df)
    stat_keys = ["Pitches","Games","FB Velo","FB PercVelo","MaxVelo","Stuff+","Loc+","K%","Whiff%","Zone%","CSW%"]
    cols = st.columns(len(stat_keys))
    for col, key in zip(cols, stat_keys):
        col.metric(key, fmt(card.get(key), key))

    primary, accent = get_team_colors(team_code)

    if view == "Stat Card":
        png = build_stat_card_png(df, pitcher, team_code)
        st.image(png, use_container_width=True)
        st.download_button("⬇ Download Stat Card", png,
            file_name=f"{pitcher.replace(', ','_')}_stat_card.png",
            mime="image/png", use_container_width=True)

    elif view == "Postgame Summary":
        if "GameID" in df.columns:
            games = (df.groupby("GameID")
                       .agg(Date=("Date","first"), Pitches=("Pitch","count"))
                       .reset_index().sort_values("Date"))
            gid = st.selectbox(
                "Game", games["GameID"].astype(str).tolist(),
                format_func=lambda g: (
                    f"{games.loc[games['GameID'].astype(str).eq(g),'Date'].iloc[0]}  ·  "
                    f"{int(games.loc[games['GameID'].astype(str).eq(g),'Pitches'].iloc[0])} pitches"))
        else:
            gid = None
            st.info("No GameID column — showing full season.")
        png = build_summary_png(df, pitcher, team_code, gid, "Postgame Summary")
        st.image(png, use_container_width=True)
        st.download_button("⬇ Download Postgame PNG", png,
            file_name=f"{pitcher.replace(', ','_')}_postgame.png",
            mime="image/png", use_container_width=True)

    else:
        png = build_summary_png(df, pitcher, team_code, label="Season Summary")
        st.image(png, use_container_width=True)
        st.download_button("⬇ Download Season Summary PNG", png,
            file_name=f"{pitcher.replace(', ','_')}_season.png",
            mime="image/png", use_container_width=True)

    with st.expander("Arsenal breakdown"):
        arsen = arsenal_table(df)
        if not arsen.empty:
            show_cols = [c for c in ["Pitch","N","Usage%","Velo","IVB","HB","Spin",
                                      "Stuff+","Loc+","Whiff%","Zone%","CSW%"] if c in arsen.columns]
            view_df = arsen[show_cols].copy()
            for col in show_cols:
                if col != "Pitch":
                    view_df[col] = view_df[col].apply(lambda v: fmt(v, col))
            st.dataframe(view_df, hide_index=True, use_container_width=True)

    with st.expander("Team info"):
        logo = logo_path_for_team(team_code)
        if logo:
            st.image(str(logo), width=100)
        else:
            st.markdown(
                f"<div style='background:{primary};color:{accent};"
                f"padding:12px 18px;border-radius:6px;font-weight:800;"
                f"display:inline-block'>{safe_team_name(team_code)}</div>",
                unsafe_allow_html=True)
        st.caption(f"Team code: `{team_code}`")


if __name__ == "__main__":
    main()
