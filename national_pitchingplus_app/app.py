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
from matplotlib.patches import FancyArrowPatch, Polygon
from PIL import Image


APP_ROOT     = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DEFAULT_DATA_DIR = (PROJECT_ROOT / "scouting_2026_trackman").resolve()
LOGO_DIR   = APP_ROOT / "team_logos"
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

try:
    from shared import load_models, basic_clean, add_flags, compute_stuffplus, compute_locationplus
except Exception:
    load_models = basic_clean = add_flags = compute_stuffplus = compute_locationplus = None

# ── Pitch colours (matches Fordham app exactly) ──────────────────────────────
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
    "FB": "Fastball", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider",   "SW": "Sweeper", "CU": "Curveball",
    "CB": "Curveball","CH": "Changeup","SP": "Splitter",
    "KN": "Knuckleball",
}

# ── Full team name dict (ported from Fordham app) ────────────────────────────
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
    "RHI_RAM":"Rhode Island Rams","COL_LION":"Columbia Lions",
    "SBU_SEA":"Stony Brook Seawolves","STJ_RED":"St. John's Red Storm",
    "DUK_BLU":"Duke Blue Devils","UMASS_RIV":"UMass Lowell River Hawks",
    "VAN_COM":"Vanderbilt Commodores","AKR_ZIP":"Akron Zips",
    "ALA_CRI":"Alabama Crimson Tide","BOC_EAG":"Boston College Eagles",
    "OHIO_BOB":"Ohio Bobcats","PUR_BOI":"Purdue Boilermakers",
    "RIC_OWL":"Rice Owls","SET_PIR":"Seton Hall Pirates",
    "PIT_PAN":"Pitt Panthers","BYU_COU":"BYU Cougars",
    "CIN_BEA":"Cincinnati Bearcats","CLE_TIG":"Clemson Tigers",
    "MIC_SPA":"Michigan State Spartans","MIC_WOL":"Michigan Wolverines",
    "CEN_MIC":"Central Michigan Chippewas","LOU_CAJ":"Louisiana Ragin' Cajuns",
    "LOU_CAR":"Louisville Cardinals","FDU_KNI":"Fairleigh Dickinson Knights",
    "FRE_BUL":"Fresno State Bulldogs","LSU_TIG":"LSU Tigers",
    "MIA_HUR":"Miami Hurricanes","MSU_BDG":"Mississippi State Bulldogs",
    "NIU_HUS":"NIU Huskies","ARI_SUN":"Arizona State Sun Devils",
    "AIR_FOR":"Air Force Falcons","ARM_BLA":"Army Black Knights",
    "NAV_MID":"Navy Midshipmen","XAV_MUS":"Xavier Musketeers",
    "AUB_TIG":"Auburn Tigers","NEV_WOL":"Nevada Wolf Pack",
    "UCF_KNI":"UCF Knights","HOU_COG":"Houston Cougars",
    "YAL_BUL":"Yale Bulldogs","NOR_TAR":"North Carolina Tar Heels",
    "NOR_WOL":"NC State Wolfpack","ORE_DUC":"Oregon Ducks",
    "UCO_HUS":"UConn Huskies","CON_HUS":"UConn Huskies",
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
    "VAN_COM":"Vanderbilt Commodores","VIR_TEC":"Virginia Tech Hokies",
    "WAK_DEA":"Wake Forest Demon Deacons","WAS_HUS":"Washington Huskies",
    "KEN_WIL":"Kentucky Wildcats","ILL_ILL":"Illinois Fighting Illini",
    "KAN_WIL":"Kansas State Wildcats","IU":"Indiana Hoosiers",
    "DAL_PAT":"Dallas Baptist Patriots","GIT_YEL":"Georgia Tech Yellow Jackets",
    "SAN_TOR":"San Diego Toreros","SAL_JAG":"South Alabama Jaguars",
    "SAN_GAU":"UC Santa Barbara Gauchos","CAL_FUL":"Cal State Fullerton Titans",
    "CAL_MUS":"Cal Poly Mustangs","CAL_ANT":"UC Irvine Anteaters",
    "CAL_AGO":"UC Davis Aggies","POR_PIL":"Portland Pilots",
    "GRA_CAN":"Grand Canyon Lopes","OKL_COW":"Oklahoma State Cowboys",
    "NOR_AGG":"North Carolina A&T Aggies","CRE_BLU":"Creighton Bluejays",
    "ETS_BUC":"ETSU Buccaneers","SOU_MIS":"Southern Miss Golden Eagles",
    "LIP_BIS":"Lipscomb Bisons","BUT_BUL":"Butler Bulldogs",
}

# ── Full team colour dict (ported from Fordham app) ──────────────────────────
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
    "POR_PIL":("#6E0E19","#C5B783"),"GRA_CAN":("#492E7F","#B59B6A"),
    "OKL_COW":("#FF6600","#000000"),"NOR_AGG":("#004684","#FFD966"),
    "CRE_BLU":("#005CA9","#FFFFFF"),"ETS_BUC":("#041E42","#FFCC00"),
    "SOU_MIS":("#FFC72C","#000000"),"LIP_BIS":("#00205B","#C8A84B"),
    "BUT_BUL":("#13294B","#747F7F"),"CRE_BLU":("#005CA9","#FFFFFF"),
}

BG      = "#0D0D0D"
PANEL   = "#111827"
BORDER  = "#1E293B"
TEXT_HI = "#FFFFFF"
TEXT_LO = "#94A3B8"
GRID_C  = "#1E293B"

st.set_page_config(page_title="CBBReports", page_icon="⚾", layout="wide")


# ── Helpers ───────────────────────────────────────────────────────────────────

def readable_text_color(bg_hex: str) -> str:
    bg_hex = bg_hex.lstrip("#")
    r, g, b = int(bg_hex[0:2],16), int(bg_hex[2:4],16), int(bg_hex[4:6],16)
    luminance = (0.299*r + 0.587*g + 0.114*b) / 255
    return "#000000" if luminance > 0.55 else "#FFFFFF"


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
    palette = [
        ("#991B1B","#FBBF24"),("#0F766E","#F8FAFC"),("#1D4ED8","#F97316"),
        ("#4C1D95","#FACC15"),("#166534","#FDE68A"),("#7C3AED","#FCD34D"),
    ]
    key = sum((i+1)*ord(ch) for i,ch in enumerate(code))
    return palette[key % len(palette)]


def logo_path_for_team(code: str) -> Path | None:
    code = str(code or "").strip()
    for suffix in [".png", ".jpg", ".jpeg"]:
        p = LOGO_DIR / f"{code}{suffix}"
        if p.exists():
            return p
    return None


def draw_home_plate(ax, x=0, y=0.25, size=0.18):
    verts = np.array([
        [x - size, y], [x + size, y],
        [x + size, y + size*0.6],
        [x,        y + size*1.1],
        [x - size, y + size*0.6],
    ])
    ax.add_patch(Polygon(verts, closed=True, facecolor="white", edgecolor="#555", linewidth=1.2, zorder=3))


def draw_strike_zone(ax):
    rect = mpatches.FancyBboxPatch((-0.83, 1.5), 1.66, 2.0,
        boxstyle="square,pad=0", linewidth=2, edgecolor=TEXT_HI, facecolor="none", zorder=4)
    ax.add_patch(rect)
    for x in [-0.277, 0.277]:
        ax.axvline(x, 1.5/5, 3.5/5, color=TEXT_LO, linewidth=0.6, linestyle="--", alpha=0.5)
    for y in [2.17, 2.83]:
        ax.axhline(y, color=TEXT_LO, linewidth=0.6, linestyle="--", alpha=0.5,
                   xmin=(-0.83+2.2)/4.4, xmax=(0.83+2.2)/4.4)


def pitch_color(pt: str) -> str:
    return PITCH_COLORS.get(str(pt).upper()[:2], "#94A3B8")


def pitch_label(pt: str) -> str:
    return PITCH_LABELS.get(str(pt).upper()[:2], str(pt))


def fmt(value, stat="") -> str:
    if pd.isna(value):
        return "—"
    if stat in {"BAA","SLG"}:
        return f"{float(value):.3f}".replace("0.",".")
    if stat in {"Pitches","Games","K","BB","N"}:
        return f"{int(round(float(value))):,}"
    if stat in {"Usage%"}:
        return f"{float(value):.1f}%"
    return f"{float(value):.1f}"


# ── Auth helpers ─────────────────────────────────────────────────────────────

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


def inject_style():
    st.markdown("""
    <style>
    .stApp { background:#0D0D0D; color:#F8FAFC; }
    div[data-testid="stHeader"] { background:transparent; }
    .cbb-hero {
        background:linear-gradient(135deg,#111827 0%,#1a2234 100%);
        border:1px solid #1E293B; border-radius:10px;
        padding:22px 28px; margin-bottom:18px;
    }
    .cbb-hero h1 { margin:0; font-size:30px; color:#FFFFFF; }
    .cbb-hero p  { margin:6px 0 0; color:#94A3B8; font-size:14px; }
    .paywall {
        max-width:700px; margin:40px auto; padding:32px;
        border-radius:12px; border:1px solid #334155;
        background:#111827;
    }
    </style>""", unsafe_allow_html=True)


def check_paywall() -> bool:
    if st.session_state.get("pp_authenticated"):
        return True
    app_name     = get_secret_value("auth","app_name","CBBReports")
    checkout_url = get_secret_value("auth","checkout_url","")
    support_email= get_secret_value("auth","support_email","")
    valid_codes  = set(get_secret_list("access_codes"))
    if not valid_codes:
        valid_codes = {"DEMO-2026"}
    st.markdown(f"""
    <div class="paywall">
        <h1 style="color:#fff;margin:0 0 8px">{app_name}</h1>
        <p style="color:#94A3B8">College Baseball Pitching Plus — national pitcher reports,
        postgame graphics, and stat cards from the 2026 TrackMan database.</p>
    </div>""", unsafe_allow_html=True)
    with st.form("paywall_form"):
        code = st.text_input("Access code", type="password", placeholder="Enter your access code")
        if st.form_submit_button("Unlock Reports ⚾", use_container_width=True):
            if code.strip() in valid_codes:
                st.session_state["pp_authenticated"] = True
                st.rerun()
            st.error("Invalid access code.")
    c1, c2 = st.columns(2)
    if checkout_url:
        c1.link_button("Buy Access →", checkout_url, use_container_width=True)
    if support_email:
        c2.markdown(f"<p style='color:#94A3B8;padding-top:8px'>Need help? {support_email}</p>", unsafe_allow_html=True)
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


@st.cache_data(show_spinner=False)
def build_index(folder: str) -> pd.DataFrame:
    rows = []
    usecols = ["Date","Pitcher","PitcherTeam","BatterTeam","HomeTeam","AwayTeam","GameID","GameUID"]
    for path in csv_files(folder):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype=str, low_memory=False)
        except Exception:
            continue
        if "Pitcher" not in df.columns or "PitcherTeam" not in df.columns:
            continue
        for (team, pitcher), g in df.dropna(subset=["Pitcher","PitcherTeam"]).groupby(["PitcherTeam","Pitcher"]):
            rows.append({"TeamCode":str(team).strip(),"Pitcher":str(pitcher).strip(),"Pitches":len(g)})
    if not rows:
        return pd.DataFrame(columns=["TeamCode","Team","Pitcher","Pitches"])
    idx = pd.DataFrame(rows)
    idx = idx.groupby(["TeamCode","Pitcher"], as_index=False).agg(Pitches=("Pitches","sum"))
    idx["Team"] = idx["TeamCode"].map(lambda c: safe_team_name(c))
    return idx


def clean_pitch_data(df: pd.DataFrame) -> pd.DataFrame:
    if all(fn is not None for fn in [load_models, basic_clean, add_flags, compute_stuffplus, compute_locationplus]):
        try:
            out = basic_clean(df.copy())
            out = add_flags(out)
            sm, sl, lm, ll = load_models(MODELS_DIR)
            out = compute_stuffplus(out, sm, sl)
            out = compute_locationplus(out, lm, ll)
            out["Pitch"] = out.get("pitch_abbr", pd.Series("UNK", index=out.index)).fillna("UNK")
        except Exception:
            out = _fallback_clean(df)
    else:
        out = _fallback_clean(df)

    rename = {"RelSpeed":"Velo","InducedVertBreak":"IVB","HorzBreak":"HB",
               "SpinRate":"Spin","RelHeight":"RelH","RelSide":"RelS",
               "Extension":"Ext","ExitSpeed":"EV","Angle":"LA"}
    out = out.rename(columns={k:v for k,v in rename.items() if k in out.columns and v not in out.columns})
    for col in ["Velo","IVB","HB","Spin","RelH","RelS","Ext","PlateLocHeight","PlateLocSide","EV","LA","Stuff+","Loc+"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
    return out


def _fallback_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    PITCH_MAP = {
        "Fastball":"FB","FourSeamFastBall":"FB","FourSeamFastball":"FB","4-Seam":"FB",
        "Sinker":"SI","Cutter":"FC","Slider":"SL","Sweeper":"SW",
        "Curveball":"CU","CurveBall":"CU","ChangeUp":"CH","Changeup":"CH",
    }
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
    return out


@st.cache_data(show_spinner=True)
def load_pitcher_data(folder: str, team_code: str, pitcher: str) -> pd.DataFrame:
    chunks = []
    for path in csv_files(folder):
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
    pa_mask = (df.get("KorBB", pd.Series("", index=df.index)).astype(str).isin(["Walk","Strikeout"]) |
               df.get("PlayResult", pd.Series("", index=df.index)).astype(str).isin(
                   ["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice","Sacrifice"]))
    pa = df[pa_mask]
    pr  = df.get("PlayResult", pd.Series("", index=df.index)).astype(str)
    kbb = df.get("KorBB",      pd.Series("", index=df.index)).astype(str)
    hits  = pr.isin(["Single","Double","Triple","HomeRun"]).sum()
    walks = kbb.eq("Walk").sum()
    ks    = kbb.eq("Strikeout").sum()
    outs  = pd.to_numeric(df.get("OutsOnPlay", 0), errors="coerce").fillna(0).sum() + ks
    ab    = max(len(pa) - walks, 0)
    tb    = pr.eq("Single").sum() + 2*pr.eq("Double").sum() + 3*pr.eq("Triple").sum() + 4*pr.eq("HomeRun").sum()
    swings = df.get("is_swing", pd.Series(False, index=df.index)).sum()
    whiffs = df.get("is_whiff", pd.Series(False, index=df.index)).sum()
    return {
        "Pitches":  len(df),
        "Games":    df.get("GameID", df.get("Date", pd.Series(dtype=str))).nunique(),
        "IP":       int(outs//3) + (outs%3)/10 if outs else np.nan,
        "K":        ks, "BB": walks,
        "K%":       ks/len(pa)*100  if len(pa) else np.nan,
        "BB%":      walks/len(pa)*100 if len(pa) else np.nan,
        "BAA":      hits/ab if ab else np.nan,
        "SLG":      tb/ab   if ab else np.nan,
        "Velo":     df["Velo"].mean()   if "Velo"   in df.columns else np.nan,
        "MaxVelo":  df["Velo"].max()    if "Velo"   in df.columns else np.nan,
        "Stuff+":   df["Stuff+"].mean() if "Stuff+" in df.columns else np.nan,
        "Loc+":     df["Loc+"].mean()   if "Loc+"   in df.columns else np.nan,
        "Whiff%":   whiffs/swings*100   if swings else np.nan,
        "Zone%":    df.get("in_zone", pd.Series(False,index=df.index)).mean()*100,
        "CSW%":     df.get("is_csw",  pd.Series(False,index=df.index)).mean()*100,
    }


def arsenal_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Pitch" not in df.columns:
        return pd.DataFrame()
    total = len(df)
    agg = df.groupby("Pitch").agg(
        N=("Pitch","count"),
        Velo=("Velo","mean"),
        IVB=("IVB","mean"),
        HB=("HB","mean"),
        Spin=("Spin","mean"),
    ).reset_index()
    agg["Usage%"] = agg["N"] / total * 100
    if "Stuff+" in df.columns:
        agg["Stuff+"] = df.groupby("Pitch")["Stuff+"].mean().values
    if "Loc+" in df.columns:
        agg["Loc+"] = df.groupby("Pitch")["Loc+"].mean().values
    if "is_whiff" in df.columns and "is_swing" in df.columns:
        w = df.groupby("Pitch").apply(lambda g: g["is_whiff"].sum() / g["is_swing"].sum() * 100 if g["is_swing"].sum() else np.nan)
        agg["Whiff%"] = agg["Pitch"].map(w)
    if "in_zone" in df.columns:
        agg["Zone%"] = df.groupby("Pitch")["in_zone"].mean().values * 100
    if "is_csw" in df.columns:
        agg["CSW%"] = df.groupby("Pitch")["is_csw"].mean().values * 100
    return agg.sort_values("N", ascending=False).reset_index(drop=True)


# ── Graphics ──────────────────────────────────────────────────────────────────

def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_LO, labelsize=8)
    ax.spines[:].set_color(BORDER)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.grid(color=GRID_C, linewidth=0.6, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT_HI, fontsize=10, fontweight="bold", pad=6)


def _plot_movement(ax, df, title="Pitch Movement"):
    _style_ax(ax, title)
    ax.axhline(0, color=TEXT_LO, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color=TEXT_LO, linewidth=0.8, linestyle="--", alpha=0.5)
    for pt, g in df.groupby("Pitch"):
        c = pitch_color(pt)
        ax.scatter(g["HB"], g["IVB"], s=28, color=c, edgecolor="white",
                   linewidth=0.3, alpha=0.75, zorder=3)
        mx, my = g["HB"].mean(), g["IVB"].mean()
        ax.scatter(mx, my, s=120, color=c, edgecolor="white",
                   linewidth=1.2, marker="D", zorder=5)
        ax.annotate(pt, (mx, my), textcoords="offset points", xytext=(5, 4),
                    color="white", fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=c, alpha=0.85, ec="none"))
    ax.set_xlabel("Horizontal Break →", color=TEXT_LO, fontsize=8)
    ax.set_ylabel("Induced Vert Break ↑", color=TEXT_LO, fontsize=8)


def _plot_locations(ax, df, title="Pitch Locations"):
    _style_ax(ax, title)
    draw_strike_zone(ax)
    draw_home_plate(ax)
    for pt, g in df.groupby("Pitch"):
        ax.scatter(g["PlateLocSide"], g["PlateLocHeight"], s=28,
                   color=pitch_color(pt), edgecolor="white", linewidth=0.3,
                   alpha=0.75, zorder=3, label=pt)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.1, 5.0)
    ax.set_xlabel("Plate Side", color=TEXT_LO, fontsize=8)
    ax.set_ylabel("Plate Height", color=TEXT_LO, fontsize=8)


def _plot_release(ax, df, title="Release Point"):
    _style_ax(ax, title)
    for pt, g in df.groupby("Pitch"):
        c = pitch_color(pt)
        ax.scatter(g["RelS"], g["RelH"], s=28, color=c, edgecolor="white",
                   linewidth=0.3, alpha=0.75, zorder=3)
        ax.scatter(g["RelS"].mean(), g["RelH"].mean(), s=120, color=c,
                   edgecolor="white", linewidth=1.2, marker="D", zorder=5)
    ax.set_xlabel("Horizontal Release", color=TEXT_LO, fontsize=8)
    ax.set_ylabel("Vertical Release", color=TEXT_LO, fontsize=8)
    ax.invert_xaxis()


def _plot_stuffloc_quadrant(ax, df, title="Stuff+ vs Loc+"):
    _style_ax(ax, title)
    ax.axhline(100, color=TEXT_LO, linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(100, color=TEXT_LO, linewidth=1.0, linestyle="--", alpha=0.6)

    # shade quadrants subtly
    xlim, ylim = (60, 150), (60, 150)
    ax.fill_between([100, xlim[1]], 100, ylim[1], color="#14532d", alpha=0.12)  # good/good
    ax.fill_between([xlim[0], 100], ylim[0], 100, color="#7f1d1d", alpha=0.12)  # bad/bad

    ax.text(102, ylim[1]-3, "Elite", color="#4ade80", fontsize=7.5, va="top", alpha=0.8)
    ax.text(xlim[0]+1, ylim[0]+1, "Below Avg", color="#f87171", fontsize=7.5, alpha=0.8)

    has_stuff = "Stuff+" in df.columns and df["Stuff+"].notna().any()
    has_loc   = "Loc+"   in df.columns and df["Loc+"].notna().any()

    if not (has_stuff and has_loc):
        ax.text(0.5, 0.5, "Models not\navailable", transform=ax.transAxes,
                color=TEXT_LO, ha="center", va="center", fontsize=9)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        return

    for pt, g in df.groupby("Pitch"):
        ms = g["Stuff+"].mean()
        ml = g["Loc+"].mean()
        if pd.isna(ms) or pd.isna(ml):
            continue
        c = pitch_color(pt)
        ax.scatter(ms, ml, s=220, color=c, edgecolor="white",
                   linewidth=1.5, zorder=5)
        ax.annotate(pt, (ms, ml), textcoords="offset points", xytext=(6, 4),
                    color="white", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=c, alpha=0.9, ec="none"))

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Stuff+", color=TEXT_LO, fontsize=8)
    ax.set_ylabel("Loc+",   color=TEXT_LO, fontsize=8)


def _draw_arsenal_table(ax, arsen: pd.DataFrame, primary: str, accent: str):
    ax.axis("off")
    cols_show = ["Pitch","N","Usage%","Velo","IVB","HB","Spin"]
    for extra in ["Stuff+","Loc+","Whiff%","Zone%","CSW%"]:
        if extra in arsen.columns:
            cols_show.append(extra)
    view = arsen[cols_show].copy()
    for col in view.columns:
        if col == "Pitch":
            continue
        view[col] = view[col].apply(lambda v: fmt(v, col))
    tbl = ax.table(cellText=view.values, colLabels=view.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.7)
    txt_on_primary = readable_text_color(primary)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(primary)
            cell.set_text_props(color=txt_on_primary, weight="bold", size=8)
        else:
            pt = view.iloc[r-1]["Pitch"] if r-1 < len(view) else ""
            row_color = pitch_color(pt)
            cell.set_facecolor(PANEL)
            if c == 0:
                cell.set_facecolor(row_color)
                cell.set_text_props(color="white", weight="bold", size=8.5)
            else:
                cell.set_text_props(color=TEXT_HI, size=8.5)
    ax.set_title("Arsenal", color=TEXT_HI, fontsize=10, fontweight="bold", pad=10)


def build_summary_png(df: pd.DataFrame, pitcher: str, team_code: str,
                      game_id: str | None = None, label: str = "Season Summary") -> bytes:
    game_df = df.copy()
    date_str = "2026 Season"
    if game_id and "GameID" in df.columns:
        filtered = df[df["GameID"].astype(str).eq(str(game_id))]
        if not filtered.empty:
            game_df = filtered
            date_str = game_df["Date"].dropna().astype(str).iloc[0] if "Date" in game_df.columns else "2026"

    primary, accent = get_team_colors(team_code)
    txt_primary = readable_text_color(primary)
    card  = pitcher_stats(game_df)
    arsen = arsenal_table(game_df)

    fig = plt.figure(figsize=(20, 11), facecolor=BG)
    # layout: header row + 5 panels (movement, locations, Stuff+/Loc+, release, arsenal)
    gs = fig.add_gridspec(2, 5,
        height_ratios=[0.20, 0.80],
        width_ratios=[1, 1, 0.95, 0.85, 1.5],
        hspace=0.08, wspace=0.28,
        left=0.03, right=0.98, top=0.97, bottom=0.07)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = fig.add_subplot(gs[0, :])
    hdr.set_facecolor(primary)
    hdr.axis("off")

    logo = logo_path_for_team(team_code)
    if logo:
        try:
            img = Image.open(logo)
            logo_ax = fig.add_axes([0.895, 0.81, 0.08, 0.14])
            logo_ax.imshow(img)
            logo_ax.axis("off")
        except Exception:
            pass

    hdr.text(0.015, 0.72, pitcher,
             transform=hdr.transAxes, color=txt_primary,
             fontsize=26, fontweight="bold", va="center")
    hdr.text(0.015, 0.28,
             f"{safe_team_name(team_code)}  ·  {label}  ·  {date_str}",
             transform=hdr.transAxes, color=accent,
             fontsize=11, fontweight="bold", va="center")

    stat_keys = ["Pitches","IP","K","BB","Velo","MaxVelo","Stuff+","Loc+","K%","Whiff%","Zone%","CSW%"]
    n_stats = len(stat_keys)
    x_start = 0.33
    x_step  = (0.95 - x_start) / n_stats
    for i, key in enumerate(stat_keys):
        x = x_start + i * x_step + x_step / 2
        hdr.text(x, 0.75, fmt(card.get(key), key),
                 transform=hdr.transAxes, color=txt_primary,
                 fontsize=13, fontweight="bold", ha="center", va="center")
        hdr.text(x, 0.22, key,
                 transform=hdr.transAxes, color=accent,
                 fontsize=7.5, ha="center", va="center")

    # ── Panels ────────────────────────────────────────────────────────────────
    has_rel = {"RelS","RelH"}.issubset(game_df.columns) and game_df["RelS"].notna().sum() > 5

    ax_move = fig.add_subplot(gs[1, 0])
    _plot_movement(ax_move, game_df)

    ax_zone = fig.add_subplot(gs[1, 1])
    _plot_locations(ax_zone, game_df)

    ax_quad = fig.add_subplot(gs[1, 2])
    _plot_stuffloc_quadrant(ax_quad, game_df)

    ax_rel = fig.add_subplot(gs[1, 3])
    if has_rel:
        _plot_release(ax_rel, game_df)
    else:
        ax_rel.set_facecolor(PANEL)
        ax_rel.axis("off")
        ax_rel.text(0.5, 0.5, "No release\ndata", transform=ax_rel.transAxes,
                    color=TEXT_LO, ha="center", va="center", fontsize=9)

    ax_tbl = fig.add_subplot(gs[1, 4])
    if not arsen.empty:
        _draw_arsenal_table(ax_tbl, arsen, primary, accent)
    else:
        ax_tbl.axis("off")
        ax_tbl.text(0.5, 0.5, "No arsenal data", transform=ax_tbl.transAxes,
                    color=TEXT_LO, ha="center", va="center")

    # ── Pitch legend ──────────────────────────────────────────────────────────
    if "Pitch" in game_df.columns:
        pitches_present = game_df["Pitch"].dropna().unique()
        handles = [mpatches.Patch(color=pitch_color(p), label=f"{p} – {pitch_label(p)}")
                   for p in pitches_present if p in PITCH_COLORS]
        if handles:
            fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
                       facecolor=PANEL, edgecolor=BORDER,
                       labelcolor=TEXT_HI, fontsize=8, framealpha=0.9,
                       bbox_to_anchor=(0.5, -0.01))

    out = BytesIO()
    fig.savefig(out, format="png", dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


def build_stat_card_png(df: pd.DataFrame, pitcher: str, team_code: str) -> bytes:
    primary, accent = get_team_colors(team_code)
    txt_primary = readable_text_color(primary)
    card  = pitcher_stats(df)
    arsen = arsenal_table(df)

    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # header band
    ax.add_patch(plt.Rectangle((0, 0.78), 1, 0.22, transform=ax.transAxes, color=primary, zorder=2))
    ax.text(0.03, 0.91, pitcher, transform=ax.transAxes,
            color=txt_primary, fontsize=27, fontweight="bold", va="center", zorder=3)
    ax.text(0.03, 0.825, safe_team_name(team_code) + "  ·  2026 Season",
            transform=ax.transAxes, color=accent, fontsize=12, fontweight="bold", va="center", zorder=3)

    logo = logo_path_for_team(team_code)
    if logo:
        try:
            img = Image.open(logo)
            logo_ax = fig.add_axes([0.82, 0.80, 0.12, 0.16])
            logo_ax.imshow(img)
            logo_ax.axis("off")
        except Exception:
            pass

    # stat tiles — 8 per row, 2 rows
    stat_keys = ["Pitches","Games","IP","K","BB","K%","BB%","BAA",
                 "SLG","Velo","MaxVelo","Stuff+","Loc+","Whiff%","Zone%","CSW%"]
    tile_w, tile_h = 0.104, 0.145
    for i, key in enumerate(stat_keys):
        col_i = i % 8
        row_i = i // 8
        x = 0.025 + col_i * (tile_w + 0.008)
        y = 0.585 - row_i * 0.21
        ax.add_patch(plt.Rectangle((x, y), tile_w, tile_h, transform=ax.transAxes,
                                   facecolor=PANEL, edgecolor=BORDER, linewidth=0.8, zorder=2))
        ax.text(x + tile_w/2, y + tile_h*0.62, fmt(card.get(key), key),
                transform=ax.transAxes, color=TEXT_HI, ha="center",
                fontsize=15, fontweight="bold", zorder=3)
        ax.text(x + tile_w/2, y + tile_h*0.18, key,
                transform=ax.transAxes, color=TEXT_LO, ha="center",
                fontsize=8, fontweight="bold", zorder=3)

    # pitch mix bar at bottom
    if not arsen.empty and "Usage%" in arsen.columns:
        bar_y, bar_h = 0.06, 0.065
        x_cursor = 0.025
        total_w = 0.97 - 0.025
        for _, row in arsen.iterrows():
            seg_w = (row["Usage%"] / 100) * total_w
            if seg_w < 0.005:
                continue
            ax.add_patch(plt.Rectangle((x_cursor, bar_y), seg_w, bar_h,
                                       transform=ax.transAxes,
                                       facecolor=pitch_color(row["Pitch"]),
                                       edgecolor=BG, linewidth=0.5, zorder=2))
            if seg_w > 0.045:
                ax.text(x_cursor + seg_w/2, bar_y + bar_h/2,
                        f"{row['Pitch']}\n{row['Usage%']:.0f}%",
                        transform=ax.transAxes, color="white",
                        ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=3)
            x_cursor += seg_w
        ax.text(0.025, bar_y - 0.03, "Pitch Mix",
                transform=ax.transAxes, color=TEXT_LO, fontsize=8)

    ax.text(0.025, 0.013, "CBBReports  ·  College Baseball Pitching Plus  ·  2026 TrackMan",
            transform=ax.transAxes, color=TEXT_LO, fontsize=8.5)

    out = BytesIO()
    fig.savefig(out, format="png", dpi=220, facecolor=BG, bbox_inches="tight")
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

    with st.spinner("Building pitcher index…"):
        index = build_index(str(folder))
    if index.empty:
        st.error("No pitchers found in the TrackMan folder.")
        return

    # Filter to known teams only (those with proper names in our dict)
    known = index[index["TeamCode"].isin(TEAM_NAMES)].copy()
    if known.empty:
        known = index.copy()

    teams = (known[["TeamCode","Team"]]
             .drop_duplicates()
             .sort_values("Team")
             .reset_index(drop=True))

    c1, c2, c3 = st.columns([1.3, 1.5, 1.0])
    with c1:
        team_code = st.selectbox(
            "Team", teams["TeamCode"].tolist(),
            format_func=lambda c: safe_team_name(c))
    team_pitchers = (known[known["TeamCode"].eq(team_code)]
                     .sort_values(["Pitches","Pitcher"], ascending=[False,True]))
    with c2:
        pitcher = st.selectbox(
            "Pitcher",
            team_pitchers["Pitcher"].tolist(),
            format_func=lambda p: f"{p}  ({team_pitchers.loc[team_pitchers.Pitcher==p,'Pitches'].iloc[0]:,} pitches)")
    with c3:
        view = st.radio("Report Type", ["Stat Card","Postgame Summary","Season Summary"])

    with st.spinner("Loading pitcher data…"):
        df = load_pitcher_data(str(folder), team_code, pitcher)
    if df.empty:
        st.warning("No tracked pitches found for that pitcher.")
        return

    card = pitcher_stats(df)
    stat_display = ["Pitches","Games","Velo","MaxVelo","Stuff+","Loc+","K%","Whiff%","Zone%","CSW%"]
    cols = st.columns(len(stat_display))
    for col, key in zip(cols, stat_display):
        col.metric(key, fmt(card.get(key), key))

    primary, accent = get_team_colors(team_code)

    if view == "Stat Card":
        png = build_stat_card_png(df, pitcher, team_code)
        st.image(png, use_container_width=True)
        st.download_button("⬇ Download Stat Card PNG", png,
            file_name=f"{pitcher.replace(', ','_')}_stat_card.png",
            mime="image/png", use_container_width=True)

    elif view == "Postgame Summary":
        if "GameID" in df.columns:
            games = (df.groupby("GameID")
                     .agg(Date=("Date","first"), Pitches=("Pitch","count"))
                     .reset_index()
                     .sort_values("Date"))
            game_id = st.selectbox(
                "Game",
                games["GameID"].astype(str).tolist(),
                format_func=lambda g: f"{games.loc[games['GameID'].astype(str).eq(g),'Date'].iloc[0]}  ·  {int(games.loc[games['GameID'].astype(str).eq(g),'Pitches'].iloc[0])} pitches")
        else:
            game_id = None
            st.info("No GameID column — showing full season data.")
        png = build_summary_png(df, pitcher, team_code, game_id, label="Postgame Summary")
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
            display_cols = [c for c in ["Pitch","N","Usage%","Velo","IVB","HB","Spin","Stuff+","Loc+","Whiff%","Zone%","CSW%"] if c in arsen.columns]
            view_df = arsen[display_cols].copy()
            for col in display_cols:
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
                f"padding:14px 20px;border-radius:8px;font-weight:800;"
                f"display:inline-block'>{safe_team_name(team_code)}</div>",
                unsafe_allow_html=True)
        st.caption(f"Team code: `{team_code}`  ·  Add logos to `national_pitchingplus_app/team_logos/{team_code}.png`")


if __name__ == "__main__":
    main()
