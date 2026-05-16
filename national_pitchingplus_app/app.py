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


import joblib
from auth import (
    is_logged_in, render_auth_page,
    render_sidebar_user, render_profile_page, render_pricing_page,
    has_pro_access, trial_days_left,
)

APP_ROOT          = Path(__file__).resolve().parent
PROJECT_ROOT      = APP_ROOT.parent
DEFAULT_DATA_DIR  = (PROJECT_ROOT / "scouting_2026_trackman").resolve()
SCOUTING_PARQUET_1 = PROJECT_ROOT / "scouting_data_1.parquet"
SCOUTING_PARQUET_2 = PROJECT_ROOT / "scouting_data_2.parquet"
LOGO_DIR          = APP_ROOT / "team_logos"
MODELS_DIR        = PROJECT_ROOT / "models"
PITCH_CLEAN_VERSION = "2026-05-08-unknown-pitch-inference-v2"

# Stuff+ feature set (must match training)
_STUFF_FEATURES = ["Velo","IVB","HB","Spin","RelH","RelS","Ext","VAA","HAA"]

# Perceived-velo constants (mirror Fordham app exactly)
_PV_DIST       = 60.5
_PV_EXT_BASE   = 6.0
_PV_IVB_BASE   = 16.0
_PV_SPIN_BASE  = 2300.0
_PV_IVB_W      = 0.08
_PV_SPIN_W     = 0.04
_PV_SHAPE_CAP  = 1.8

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
    "GEO_BUL":"Georgia Bulldogs","GEO_COL":"George Washington Revolutionaries","GEO_COL1":"George Washington Revolutionaries","GEO_COL2":"George Washington Revolutionaries","GEO_PAT":"George Mason Patriots",
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
    "LAS_EXP":"La Salle Explorers","LAS_EXS":"La Salle Explorers",
    "SAI_BIL":"Saint Louis Billikens","STL_BIL":"Saint Louis Billikens",
    "SLU_BILL":"Saint Louis Billikens",
    "SBU_BON":"St. Bonaventure Bonnies",
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
    # Big Ten additions
    "IOW_HAW":"Iowa Hawkeyes","MAR_TER":"Maryland Terrapins",
    # SEC additions
    "TEX_A&M":"Texas A&M Aggies","TEX_A&M1":"Texas A&M Aggies","TEX_AGG":"Texas A&M Aggies",
    "SOU_GAM":"South Carolina Gamecocks",
    # Big East additions
    "UCO_HUS":"UConn Huskies","CON_HUS":"UConn Huskies",
    # Big 12 addition
    "UTA_UTE":"Utah Utes","HOU_COU":"Houston Cougars",
    # Sun Belt additions
    "LOU_CAJ":"Louisiana Ragin' Cajuns","NOR_TEX":"North Texas Mean Green",
    "SOU_JAG":"Southern Jaguars",
    # Mountain West additions
    "UTA_STA":"Utah State Aggies",
    # WAC additions
    "ARL_MAV":"UT Arlington Mavericks","CHI_STA":"Chicago State Cougars",
    "DIX_STE":"Utah Tech Trailblazers","HBU_HUS":"Houston Christian Huskies",
    # NEC additions
    "WAG_SEA":"Wagner Seahawks",
    # ASUN additions
    "EKU_COL":"Eastern Kentucky Colonels","UTM_SKY":"UT Martin Skyhawks",
    "JAC_STA":"Jacksonville State Gamecocks","AUS_GOV":"Austin Peay Governors",
    # SoCon additions
    "WOF_TER":"Wofford Terriers",
    # MVC additions
    "EIU_PAN":"Eastern Illinois Panthers",
    # CAA alternate codes
    "ELO_PHO":"Elon Phoenix",
    # Big West additions
    "LON_BEA":"Long Beach State Dirtbags",
    # Summit League additions
    "DEN_UNI":"Denver Pioneers",
    # SWAC additions
    "JAC_TIG":"Jackson State Tigers","ARK_LIO":"UAPB Golden Lions",
    "ALA_HOR":"Alabama State Hornets","ALA_ANM":"Alabama A&M Bulldogs",
    # Alternate/duplicate codes
    "OHI_BOB":"Ohio Bobcats","GEO_PAN":"Georgia State Panthers",
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
    "GEO_COL":("#033C5A","#AA9868"),"GEO_COL1":("#033C5A","#AA9868"),"GEO_COL2":("#033C5A","#AA9868"),
    "GEO_PAN":("#0039A6","#C60C30"),"GEO_SOU":("#011E41","#A99260"),
    "SAC_PIO":("#CE1141","#FFFFFF"),"ION_GAL":("#891C2C","#C8960C"),
    "ION_GAE":("#891C2C","#C8960C"),"STM_GAE":("#D80024","#003A70"),
    "MAR_RED":("#B31B1B","#FFFFFF"),"LOY_LIO":("#A50034","#003B5C"),
    "WAG_SEA":("#006747","#FFFFFF"),"FAI_STA":("#C8102E","#003A70"),
    "MAN_JAS":("#00703C","#FFFFFF"),"NIA_EAG":("#4B116F","#C99700"),
    "CAN_GRI":("#0C2340","#FFCC00"),"SIE_SAI":("#006747","#FFB81C"),
    "MON_HAW":("#041E42","#A7A9AC"),"RIC_SPI":("#990000","#000066"),
    "RHO_RAM":("#68ABE8","#002147"),"DAY_FLY":("#CE1141","#00539B"),
    "LAS_EXP":("#00205B","#FDB515"),"LAS_EXS":("#00205B","#FDB515"),
    "SAI_BIL":("#003DA5","#C8C9C7"),"STL_BIL":("#003DA5","#C8C9C7"),
    "SLU_BILL":("#003DA5","#C8C9C7"),
    "SBU_BON":("#54261A","#FDB515"),
    "STB_BON":("#54261A","#FDB515"),"JOE_HAW":("#9E1B32","#A7A8AA"),
    "SAI_JOE":("#9E1B32","#A7A8AA"),"STJ_HAW":("#9E1B32","#A7A8AA"),
    "DUQ_DUK":("#041E42","#BA0C2F"),"LOY_RAM":("#8D0034","#FFC72C"),
    "UMASS":("#971B2F","#FFFFFF"),"MAS_MIN":("#971B2F","#FFFFFF"),
    "COL_LION":("#75AADB","#FFFFFF"),"SBU_SEA":("#990000","#1F1F1F"),
    "STJ_RED":("#BA0C2F","#FFFFFF"),"DUK_BLU":("#012169","#FFFFFF"),
    "UMASS_RIV":("#003DA5","#C0C0C0"),"VAN_COM":("#000000","#B3A369"),
    "AKR_ZIP":("#041E42","#A89968"),"ALA_CRI":("#9E1B32","#FFFFFF"),
    "BOC_EAG":("#8A0000","#C9A84C"),"OHIO_BOB":("#00694E","#FFFFFF"),
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
    "HOU_COG":("#C8102E","#FFFFFF"),"HOU_COU":("#C8102E","#FFFFFF"),
    "YAL_BUL":("#00356B","#FFFFFF"),
    "NOR_TAR":("#7BAFD4","#13294B"),"NOR_WOL":("#CC0000","#000000"),
    "ORE_DUC":("#154733","#FEE123"),"UCO_HUS":("#000E2F","#FFFFFF"),
    "CON_HUS":("#000E2F","#FFFFFF"),"BOS_COL":("#8A0000","#C9A84C"),
    "DEL_BLU":("#00539B","#FFD200"),"HOF_PRI":("#003591","#FFB81C"),
    "DRE_DRA":("#07294D","#FFC600"),"NOR_HUS":("#CC0000","#000000"),
    "ELON_PHO":("#73000A","#B59A57"),"CAM_CAM":("#F47920","#000000"),
    "CHS_COU":("#73000A","#000000"),"BRY_BUL":("#000000","#C8A415"),
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
    # Big Ten additions
    "IOW_HAW":("#FFCD00","#000000"),"MAR_TER":("#E03A3E","#FFCC00"),
    # SEC additions
    "TEX_A&M":("#500000","#FFFFFF"),"TEX_A&M1":("#500000","#FFFFFF"),
    "TEX_AGG":("#500000","#FFFFFF"),"SOU_GAM":("#73000A","#000000"),
    # Big East / alternate
    "UCO_HUS":("#000E2F","#FFFFFF"),"CON_HUS":("#000E2F","#FFFFFF"),
    "HOU_COU":("#C8102E","#FFFFFF"),
    # Sun Belt additions
    "LOU_CAJ":("#CE181E","#000000"),"NOR_TEX":("#00853E","#FFFFFF"),
    # Mountain West / Big 12
    "UTA_STA":("#003263","#9EADB5"),"UTA_UTE":("#CC0000","#000000"),
    # WAC additions
    "ARL_MAV":("#003087","#FF8200"),"CHI_STA":("#006747","#FFFFFF"),
    "DIX_STE":("#00853E","#FFFFFF"),"HBU_HUS":("#002D62","#C8102E"),
    # NEC / ASUN / SoCon / MVC additions
    "WAG_SEA":("#006747","#FFFFFF"),
    "EKU_COL":("#7D0028","#B59A57"),"UTM_SKY":("#FF8200","#002147"),
    "JAC_STA":("#002D62","#C9A240"),"AUS_GOV":("#C8102E","#FFFFFF"),
    "WOF_TER":("#CEB888","#000000"),"EIU_PAN":("#004B98","#9B9EA4"),
    "ELO_PHO":("#73000A","#B59A57"),
    # Big West / Summit
    "LON_BEA":("#000000","#FFD700"),"DEN_UNI":("#912727","#C8A032"),
    # SWAC
    "JAC_TIG":("#004B8D","#FFFFFF"),"ARK_LIO":("#007A33","#FFD700"),
    "ALA_HOR":("#7F2633","#FFAD00"),"ALA_ANM":("#63003C","#F5C518"),
    "SOU_JAG":("#003087","#FFD700"),
    # Alternates
    "OHI_BOB":("#00694E","#FFFFFF"),"GEO_PAN":("#0039A6","#C60C30"),
}

# ── Conference index (D1 only) ────────────────────────────────────────────────
TEAM_CONFERENCES = {
    # ACC
    "BOC_EAG":"ACC","BOS_COL":"ACC","CAL_BEA":"ACC","CLE_TIG":"ACC","DUK_BLU":"ACC",
    "FLO_SEM":"ACC","GIT_YEL":"ACC","LOU_CAR":"ACC","MIA_HUR":"ACC","NOR_TAR":"ACC",
    "NOR_WOL":"ACC","NOT_IRI":"ACC","PIT_PAN":"ACC","STA_CAR":"ACC","VIR_CAV":"ACC",
    "VIR_TEC":"ACC","WAK_DEA":"ACC",
    # SEC
    "ALA_CRI":"SEC","ARK_RAZ":"SEC","AUB_TIG":"SEC","FLA_GAT":"SEC","FLA__GAT":"SEC",
    "GEO_BUL":"SEC","KEN_WIL":"SEC","LSU_TIG":"SEC","MIZ_TIG":"SEC","MSU_BDG":"SEC",
    "OKL_SOO":"SEC","OLE_REB":"SEC","TEN_VOL":"SEC","TEX_A&M":"SEC","TEX_A&M1":"SEC",
    "TEX_AGG":"SEC","TEX_LON":"SEC","VAN_COM":"SEC",
    # Big 12
    "ARI_SUN":"Big 12","ARI_WIL":"Big 12","BAY_BEA":"Big 12","BYU_COU":"Big 12","CIN_BEA":"Big 12",
    "HOU_COG":"Big 12","HOU_COU":"Big 12","KAN_JAY":"Big 12","KAN_WIL":"Big 12","OKL_COW":"Big 12",
    "TCU_HFG":"Big 12","TEX_RAI":"Big 12","UCF_KNI":"Big 12","UTA_UTE":"Big 12","WES_MOU":"Big 12",
    "WVU_MOU":"Big 12",
    # Big Ten
    "ILL_ILL":"Big Ten","IOW_HAW":"Big Ten","IU":"Big Ten","MAR_TER":"Big Ten","MIC_SPA":"Big Ten",
    "MIC_WOL":"Big Ten","MIN_GOL":"Big Ten","NEB":"Big Ten","ORE_DUC":"Big Ten","OSU_BUC":"Big Ten",
    "PEN_NIT":"Big Ten","PUR_BOI":"Big Ten","RUT_SCA":"Big Ten","SOU_TRO":"Big Ten","UCLA":"Big Ten",
    "USC_UPS":"Big Ten","WAS_HUS":"Big Ten",
    # American
    "CHA_49E":"American","CHA_FOR":"American","ECU_PIR":"American","FAU_OWL":"American","MT":"American","RIC_OWL":"American",
    "TUL_GRE":"American","UAB_BLA":"American","USF_BUL":"American","UTS_ROA":"American","WIC_SHO":"American",
    # Mountain West
    "AIR_FOR":"Mountain West","FRE_BUL":"Mountain West","MEX_LOB":"Mountain West","NEV_WOL":"Mountain West","SAN_AZT":"Mountain West",
    "UNL_REB":"Mountain West","UTA_STA":"Mountain West",
    # WCC
    "GON_BUL":"WCC","LOY_LIO":"WCC","PAC_TIG":"WCC","PEP_WAV":"WCC","POR_PIL":"WCC",
    "SAC_DON":"WCC","SAN_BRO":"WCC","SAN_DIE22":"WCC","SAN_DIE23":"WCC","SAN_DON":"WCC",
    "SAN_TOR":"WCC","STM_GAE":"WCC",
    # Atlantic 10
    "DAV_WIL":"Atlantic 10","DAY_FLY":"Atlantic 10","DUQ_DUK":"Atlantic 10","FOR_RAM":"Atlantic 10","FOR_RAM1":"Atlantic 10",
    "GEO_COL":"Atlantic 10","GEO_COL1":"Atlantic 10","GEO_COL2":"Atlantic 10","GEO_GWI":"Atlantic 10","GEO_PAT":"Atlantic 10","JOE_HAW":"Atlantic 10","LAS_EXP":"Atlantic 10","LAS_EXS":"Atlantic 10",
    "LAS_EXS":"Atlantic 10","LOY_RAM":"Atlantic 10","RHI_RAM":"Atlantic 10","RHO_RAM":"Atlantic 10","RIC_SPI":"Atlantic 10",
    "SAI_BIL":"Atlantic 10","SAI_JOE":"Atlantic 10","SBU_BON":"Atlantic 10","SLU_BILL":"Atlantic 10","STB_BON":"Atlantic 10",
    "STJ_HAW":"Atlantic 10","STL_BIL":"Atlantic 10","VCU_RAM":"Atlantic 10",
    # UMass is MAC, not Atlantic 10
    "UMASS":"MAC","MAS_MIN":"MAC","UMA_AMH":"MAC",
    # MAAC
    "CAN_GOL":"MAAC","CAN_GRI":"MAAC","FAI_STA":"MAAC","ION_GAE":"MAAC","ION_GAE1":"MAAC",
    "ION_GAL":"MAAC","MAN_JAS":"MAAC","MAR_RED":"MAAC","MER_WAR":"MAAC","MSM_MTN":"MAAC",
    "NIA_EAG":"MAAC","NIA_PUR":"MAAC","QUIN_BOB":"MAAC","QUI_BOB":"MAAC","RIDER_BRO":"MAAC",
    "RID_BRO":"MAAC","SAC_HEA":"MAAC","SAC_PIO":"MAAC","SIE_SAI":"MAAC","SIE_SAI1":"MAAC",
    "SPU_PEA":"MAAC","STP_PCO":"MAAC","STP_PEA":"MAAC",
    # Patriot League
    "ARM_BLA":"Patriot League","BUC_BIS":"Patriot League","COL_GAT":"Patriot League","HOL_CRO":"Patriot League","LAF_LEO":"Patriot League",
    "LAF_LEP":"Patriot League","LEH_MOU":"Patriot League","NAV_MID":"Patriot League",
    # Ivy League
    "COL_LION":"Ivy League","COR_BRE":"Ivy League","DAR_GRE":"Ivy League","PEN_QUA":"Ivy League","PRI_TIG":"Ivy League",
    "YAL_BUL":"Ivy League",
    # CAA
    "CAM_CAM":"CAA","CHS_COU":"CAA","COA_COU":"CAA","COL_CHA":"CAA","DEL_BLU":"CAA",
    "DRE_DRA":"CAA","ELON_PHO":"CAA","ELO_PHO":"CAA","HOF_PRI":"CAA","MON_HAW":"CAA",
    "NOR_AGG":"CAA","NOR_HUS":"CAA","SBU_SEA":"CAA","TOW_TIG":"CAA","UNCW":"CAA",
    "UNC_SEA":"CAA","WIL_SEA":"CAA","WM_TRI":"CAA",
    # MAC
    "AKR_ZIP":"MAC","BGS_FAL":"MAC","CEN_MIC":"MAC","EMU_EAG":"MAC","MIA_RED":"MAC",
    "NIU_HUS":"MAC","OHIO_BOB":"MAC","OHI_BOB":"MAC","TOL_ROC":"MAC",
    # Horizon League
    "MIL_UNI":"Horizon League","OAK_GOL":"Horizon League","UIC_FLA":"Horizon League","UWM_PAN":"Horizon League","WRI_RAI":"Horizon League",
    "YOU_HAR":"Horizon League","YSU_PEN":"Horizon League",
    # Missouri Valley
    "BEL_BRU":"Missouri Valley","BRA_BRA":"Missouri Valley","EVA_ACE":"Missouri Valley","ILL_RED":"Missouri Valley","IND_SYC":"Missouri Valley",
    "MUR_RAC":"Missouri Valley","SIU_SAL":"Missouri Valley","VAL_BLA":"Missouri Valley","VAL_CRU":"Missouri Valley",
    # C-USA
    "DAL_PAT":"C-USA","FLO_PAN":"C-USA","JAC_GAM":"C-USA","KEN_OWL":"C-USA","LIB_FLA":"C-USA",
    "LOU_BUL":"C-USA","MIS_BEA":"C-USA","MTSU_BLU":"C-USA","NMS_AGG":"C-USA",
    # Sun Belt
    "APP_MOU":"Sun Belt","ASU_RED":"Sun Belt","COA_CHA":"Sun Belt","GEO_EAG":"Sun Belt","GEO_PAN":"Sun Belt",
    "GEO_SOU":"Sun Belt","GEO_STA":"Sun Belt","JMU_DUK":"Sun Belt","LOU_CAJ":"Sun Belt","MAR_THU":"Sun Belt",
    "NOR_TEX":"Sun Belt","OLD_DOM":"Sun Belt","OLD_MON":"Sun Belt","SAL_JAG":"Sun Belt","SOU_GOL":"Sun Belt",
    "SOU_MIS":"Sun Belt","TEX_BOB":"Sun Belt","TRO_T":"Sun Belt","TRO_TRJ":"Sun Belt","ULM_WAR":"Sun Belt",
    # Big East
    "BUT_BUL":"Big East","CON_HUS":"Big East","CRE_BLU":"Big East","GEO_HOY":"Big East","SET_PIR":"Big East",
    "STJ_RED":"Big East","UCO_HUS":"Big East","VIL_WIL":"Big East","XAV_MUS":"Big East",
    # WAC
    "ABI_WIL":"WAC","ARL_MAV":"WAC","CAL_LAN":"WAC","CHI_STA":"WAC","CSU_BAK":"WAC",
    "DIX_STE":"WAC","GRA_CAN":"WAC","HBU_HUS":"WAC","SAC_HOR":"WAC","SEA_RED":"WAC",
    "TAR_TEX":"WAC","UTA_WOL":"WAC","UTR_VAQ":"WAC",
    # NEC
    "CCU_BLD":"NEC","DEL_STA":"NEC","FDU_KNI":"NEC","LIU_SHA":"NEC","MAR_HAW":"NEC",
    "NEW_HAV":"NEC","WAG_SEA":"NEC",
    # America East
    "ALB_DAN":"America East","ALB_GRE":"America East","BIN_BEA":"America East","BRY_BUL":"America East","LOW_RIV":"America East",
    "MAI_BLA":"America East","NJI_HIG":"America East","UMASS_RIV":"America East","UMBC_RET":"America East","UML_RIV":"America East",
    # Big West
    "CAL_AGO":"Big West","CAL_ANT":"Big West","CAL_FUL":"Big West","CAL_MAT":"Big West","CAL_MUS":"Big West",
    "HAW_WAR":"Big West","LON_BEA":"Big West","SAN_BAR1":"Big West","SAN_GAU":"Big West",
    # SoCon
    "CIT_BUL":"SoCon","ETS_BUC":"SoCon","MER_BEA":"SoCon","SAM_BUL":"SoCon","UNC_SPA":"SoCon",
    "VIR_KEY":"SoCon","WOF_TER":"SoCon",
    # ASUN
    "ALA_LIO":"ASUN","AUS_GOV":"ASUN","EKU_COL":"ASUN","FGCU":"ASUN","HIG_PAN":"ASUN",
    "JAC_STA":"ASUN","LIP_BIS":"ASUN","NOF_OSP":"ASUN","NOR_FLO":"ASUN","QUN_RYL":"ASUN",
    "STE_HAT":"ASUN","UTM_SKY":"ASUN",
    # Pac-12
    "ORE_BEA":"Pac-12",
    # Big South
    "CHA_BUC":"Big South","LON_LAN":"Big South","PRE_BLH":"Big South","RAD_HIG":"Big South","WIN_BUL":"Big South",
    "WIN_EAG":"Big South",
    # OVC
    "EIU_PAN":"OVC","LIT_TRO":"OVC","MOR_EAG":"OVC","SOU_COU":"OVC","SOU_IND16":"OVC",
    "SOU_RED":"OVC","TEN_TEC":"OVC","UTS_EAG":"OVC","WIU_LEA":"OVC",
    # Summit League
    "DEN_UNI":"Summit League","ORA_GOL":"Summit League","STM_BOB":"Summit League","STU_BOB":"Summit League","UNO_MAV":"Summit League",
    # Southland
    "LAM_CAR":"Southland","MCN_COW":"Southland","NEW_PRI":"Southland","NIC_COL":"Southland","NOR_DEM":"Southland",
    "SOU_LIO":"Southland","TEX_ISL":"Southland",
    # SWAC
    "ALA_ANM":"SWAC","ALA_HOR":"SWAC","ALB_STA":"SWAC","ALC_BRA":"SWAC","ARK_LIO":"SWAC",
    "FLO_RAT":"SWAC","GRA_TIG":"SWAC","JAC_TIG":"SWAC","MIS_DEL":"SWAC","SOU_GAM":"SWAC",
    "SOU_JAG":"SWAC",
}

# High-volume 2026 TrackMan code overrides. These sit outside the base dicts so
# we can keep improving national coverage as new tags show up in the feed.
TEAM_NAMES.update({
    "NEW_PRI": "New Orleans Privateers",
    "TRO_TRJ": "Troy Trojans",
    "LOU_BUL": "Louisiana Tech Bulldogs",
    "UTR_VAQ": "UT Rio Grande Valley Vaqueros",
    "MER_BEA": "Mercer Bears",
    "MT": "Memphis Tigers",
    "FLO_PAN": "FIU Panthers",
    "CHA_FOR": "Charlotte 49ers",
    "SAN_BRO": "Santa Clara Broncos",
    "SOU_TRO": "USC Trojans",
    "SOU_LIO": "Southeastern Louisiana Lions",
    "CSD_TRI": "UC San Diego Tritons",
    "MIS_BEA": "Missouri State Bears",
    "CAL_LAN": "California Baptist Lancers",
    "SOU_IND16": "Southern Indiana Screaming Eagles",
    "UNL_REB": "UNLV Rebels",
    "LAM_CAR": "Lamar Cardinals",
    "PRE_BLH": "Presbyterian Blue Hose",
    "ASU_RED": "Arkansas State Red Wolves",
    "MAR_THU": "Marshall Thundering Herd",
    "SOU_GOL": "Southern Miss Golden Eagles",
    "ABI_WIL": "Abilene Christian Wildcats",
    "LON_LAN": "Longwood Lancers",
    "IWC": "Incarnate Word Cardinals",
    "SOU_RED": "Southeast Missouri Redhawks",
    "EMU_EAG": "Eastern Michigan Eagles",
    "WES_HIL": "Western Kentucky Hilltoppers",
    "WCC": "Western Carolina Catamounts",
    "PAC_TIG": "Pacific Tigers",
    "TEN_TEC": "Tennessee Tech Golden Eagles",
    "UTA_WOL": "Utah Valley Wolverines",
    "KEN_OWL": "Kennesaw State Owls",
    "EVA_ACE": "Evansville Purple Aces",
    "ALA_LIO": "North Alabama Lions",
    "UTS_ROA": "UTSA Roadrunners",
    "NIC_COL": "Nicholls Colonels",
    "CAL_MAT": "CSUN Matadors",
    "JAC_GAM": "Jacksonville State Gamecocks",
    "ARI_WIL": "Arizona Wildcats",
    "SAM_BUL": "Samford Bulldogs",
    "WIN_EAG": "Winthrop Eagles",
    "VIR_KEY": "VMI Keydets",
    "SOU_COU": "SIUE Cougars",
    "MCN_COW": "McNeese Cowboys",
    "BEL_BRU": "Belmont Bruins",
    "LIT_TRO": "Little Rock Trojans",
    "QUN_RYL": "Queens Royals",
    "COL_CHA": "Charleston Cougars",
    "RAD_HIG": "Radford Highlanders",
    "CIT_BUL": "The Citadel Bulldogs",
    "SEA_RED": "Seattle U Redhawks",
    "CHA_BUC": "Charleston Southern Buccaneers",
    "TOL_ROC": "Toledo Rockets",
    "NOR_DEM": "Northwestern State Demons",
    "STU_BOB": "St. Thomas Tommies",
    "STM_BOB": "St. Thomas Tommies",
    "THO_M": "St. Thomas Tommies",
    "BRA_BRA": "Bradley Braves",
    "NOF_OSP": "North Florida Ospreys",
    "MOR_EAG": "Morehead State Eagles",
    "LAF_LEP": "Lafayette Leopards",
    "ORA_GOL": "Oral Roberts Golden Eagles",
    "WM_TRI": "William & Mary Tribe",
    "VAL_CRU": "Valparaiso Beacons",
    "MIA_RED": "Miami (OH) RedHawks",
    "OAK_GOL": "Oakland Golden Grizzlies",
    "ILL_RED": "Illinois State Redbirds",
    "VIL_WIL": "Villanova Wildcats",
    "MSM_MTN": "Mount St. Mary's Mountaineers",
    "YSU_PEN": "Youngstown State Penguins",
    "YOU_HAR": "Youngstown State Penguins",
    "UWM_PAN": "Milwaukee Panthers",
    "UNO_MAV": "Omaha Mavericks",
    "BGS_FAL": "Bowling Green Falcons",
    "SAM_BEA": "Sam Houston Bearkats",
    "HOL_CRU": "Holy Cross Crusaders",
    "WIU_LEA": "Western Illinois Leathernecks",
    "NOR_BIS": "North Dakota State Bison",
    "MEX_LOB": "New Mexico Lobos",
    "DAR_GRE": "Dartmouth Big Green",
    "TEX_ISL": "Texas A&M-Corpus Christi Islanders",
    "MIS_DEL": "Mississippi Valley State Delta Devils",
    "GRA_TIG": "Grambling State Tigers",
    "NOR_CAT": "Northwestern Wildcats",
    "LON_DIR": "Long Beach State Dirtbags",
    "STE_LUM": "Stephen F. Austin Lumberjacks",
    "HAR_CRI": "Harvard Crimson",
    "BRO_BEA": "Brown Bears",
    "MIL_UNI2": "Milwaukee Panthers",
    "WAS_COU": "Washington State Cougars",
    # ── D1 alt codes & newly identified ──────────────────────────────────────
    "BEL_COL":"Bellarmine Knights","BEL_KNI":"Bellarmine Knights",
    "BRO_COL":"Brown Bears",
    "BUT_COL1":"Butler Bulldogs",
    "BRY_STR1":"Bryant Bulldogs",
    "CAM_UNI":"Campbell Camels","CAM_UNI1":"Campbell Camels",
    "CAL_POL1":"Cal Poly Mustangs",
    "CCU_BLD":"Central Connecticut State Blue Devils",
    "CEN_COL1":"Central Connecticut State Blue Devils",
    "COP_STA":"Coppin State Eagles",
    "DEL_HOR":"Delaware State Hornets",
    "EC":"East Carolina Pirates",
    "GAR_RUN":"Gardner-Webb Runnin' Bulldogs",
    "HOW_HAW":"Howard Bison",
    "KEN_STA1":"Kennesaw State Owls",
    "LIN_UNI":"Lindenwood Lions",
    "LON_ISL22":"Long Island University Sharks",
    "MER_UNI":"Merrimack Warriors",
    "MIS_DEL1":"Mississippi Valley State Delta Devils",
    "MIS_ST.":"Missouri State Bears",
    "MIS_BEA":"Missouri State Bears",
    "NCA_BUL":"North Carolina A&T Aggies",
    "NIC_COL1":"Nicholls Colonels",
    "NOR_IOW2":"Northern Iowa Panthers",
    "PRA_ACA":"Prairie View A&M Panthers","PRA_PAN":"Prairie View A&M Panthers","PRA_PRA1":"Prairie View A&M Panthers",
    "SAD_GAU":"UC Santa Barbara Gauchos",
    "SOU_ILL":"Southern Illinois Salukis",
    "SOU_SOU8":"Southern Jaguars",
    "STE_MUS":"Stephen F. Austin Lumberjacks",
    "STO_COL":"Stonehill Skyhawks",
    "TEX_A&M":"Texas A&M Aggies","TEX_A&M1":"Texas A&M Aggies",
    "WMI_BRO":"Western Michigan Broncos",
    # ── D2 / D3 / NAIA / JUCO ────────────────────────────────────────────────
    "AND_TRO":"Anderson University Trojans",
    "AVE_MAR":"Ave Maria University Gyrenes",
    "BAR_COL":"Barry University Buccaneers",
    "BIO_UNI":"Biola University Eagles",
    "CAR_EAG":"Carson-Newman Eagles",
    "CCU_BLD":"Central Connecticut State Blue Devils",
    "CED_UNI":"Cedarville University Yellow Jackets",
    "CEN_BEA":"Central Baptist College Mustangs",
    "CEN_COL":"Central College Dutch",
    "CHI_MRN":"Chicago Marines",
    "CLA_PAN":"Clarion University Eagles",
    "CLE_CLE2":"Select/Showcase Team",
    "CMU_MAV":"Colorado Mesa University Mavericks",
    "COK_COB":"Coker University Cobras",
    "COL_CUG":"Colby College White Mules",
    "CUM_PAT":"Cumberland University Patriots",
    "EMO_HEN":"Emory & Henry College Wasps",
    "ERS_COL":"Erskine College Flying Fleet",
    "EAS_NEW":"Eastern New Mexico Greyhounds",
    "EAS_TEX":"East Texas Baptist University Tigers",
    "FER_COL":"Ferris State University Bulldogs",
    "FLA_COL":"Flagler College Saints",
    "FRA_MAR1":"Franklin & Marshall Diplomats",
    "FRE_PAC":"Fresno Pacific University Sunbirds",
    "GAS_COL":"Gadsden State Fighting Cardinals",
    "GAD_STA":"Gadsden State Fighting Cardinals",
    "GEO_FOX":"George Fox University Bruins",
    "GRA_COL":"Grace College Lancers",
    "GUL_COM":"Gulf Coast State College Commodores",
    "HAW_HIL":"Hawkeye Community College",
    "HEN_COL":"Henderson State Reddies",
    "HIL_COL2":"Hill College Rebels",
    "HIN_COM":"Hinds Community College Eagles",
    "HOL_COM":"Holmes Community College",
    "HUT_COM":"Hutchinson CC Blue Dragons",
    "ILL_WES":"Illinois Wesleyan University Titans",
    "IND_WES":"Indiana Wesleyan University Wildcats",
    "IOW_CEN":"Iowa Central CC Tritons",
    "ITA_ITA":"Italy National Team",
    "JES_UNI":"Jesuit University",
    "JOH_LOG":"John A. Logan College Volunteers",
    "JOH_UNI":"Johns Hopkins University Blue Jays",
    "JON_COL":"Jones County Junior College Bobcats",
    "JUD":"Judson University Eagles",
    "KIN_UNI":"King University Tornado",
    "KS_GF":"Kansas Select",
    "LAN_BEA":"Lane College Dragons",
    "LEB_VAL":"Lebanon Valley College Flying Dutchmen",
    "LEE_UNI":"Lee University Flames",
    "LEN_BEA":"Lenoir-Rhyne University Bears",
    "LIN_MEM":"Lincoln Memorial University Railsplitters",
    "LIN_UNI2":"Limestone University Saints",
    "LOW_COL":"Lower Columbia College Red Devils",
    "LYC_COL":"Lycoming College Warriors",
    "MAR_LIO":"Marion University Flying Knights",
    "MAR_SAI":"Marian University Saints",
    "MID_GEO":"Middle Georgia State Warriors",
    "MIS_COL1":"Missouri S&T Miners",
    "MOL_COL":"Molloy University Lions",
    "MOU_OLV":"Mount Olive University Trojans",
    "NAV_COL":"Navarro College Bulldogs",
    "NCB":"Northwestern College Eagles",
    "NEW_HAV":"New Haven Chargers",
    "NEW_WLV":"New England College Pilgrims",
    "NOK_NOR":"Northern Oklahoma Select",
    "NOR_GEO3":"North Georgia University Nighthawks",
    "NOR_GRE":"North Greenville University Crusaders",
    "NOR_MIS":"Northern Michigan University Wildcats",
    "NOV_SOU":"Nova Southeastern University Sharks",
    "ODE_COL":"Odessa College Wranglers",
    "OGL_UNI":"Oglethorpe University Petrels",
    "ORA_COA":"Orange Coast College Pirates",
    "PAN_COL":"Panola College Ponies",
    "PAR_JUN":"Paris Junior College Dragons",
    "PAT_HEN":"Patrick Henry College Patriots",
    "PEA_RIV":"Pearl River Community College Wildcats",
    "PER_BAS6":"Georgia Perimeter College",
    "POI_LOM":"Point Loma Nazarene Sea Lions",
    "POM_COL":"Pomona-Pitzer Sagehens",
    "QUI_HAW":"Quincy University Hawks",
    "REI_UNI":"Reinhardt University Eagles",
    "RHO_COL":"Rhode Island College Anchormen",
    "RIV_CIT":"Riverside City College Tigers",
    "ROA_COL":"Roanoke College Maroons",
    "ROG_WIL":"Roger Williams University Hawks",
    "SAD_GAU":"UC Santa Barbara Gauchos",
    "SAG_VAL":"Saginaw Valley State Cardinals",
    "SAN_JAC":"San Jacinto College Ravens",
    "SAN_SPA":"San Bernardino Valley College Wolverines",
    "SET_HIL":"Seton Hill University Griffins",
    "SHE_UNI":"Shenandoah University Hornets","SHE_UNI1":"Shenandoah University Hornets",
    "SHI_UNI":"Shippensburg University Raiders",
    "SHO_UNI":"Shorter University Hawks",
    "SLC_CCB":"Salt Lake Community College Bruins",
    "SLI_ROC":"Slippery Rock University The Rock",
    "SOU_ARK":"Southern Arkansas Muleriders","SOU_ARK2":"Southern Arkansas Muleriders",
    "SOU_JAC":"Southern Union State CC",
    "SOU_ORE":"Southern Oregon University Raiders",
    "SOU_UNI":"Southern Nazarene University Crimson Storm",
    "SOU_WES":"Southwestern University Pirates","SOU_WES1":"Southwestern Adventist Eagles",
    "STM_RAT":"St. Mary's University Rattlers",
    "TEM_LEO":"Temple College Leopards",
    "TEN_WES":"Tennessee Wesleyan University Bulldogs",
    "TEX_LUT":"Texas Lutheran University Bulldogs",
    "TJC_APA":"Tyler Junior College Apaches",
    "TNU":"Trevecca Nazarene University Trojans",
    "TRI_TIG":"Trinity University Tigers",
    "TUF_UNI":"Tufts University Jumbos",
    "TUS_PIO":"Tusculum University Pioneers","TUS_TUS":"Tusculum University Pioneers",
    "UNI_FIN":"University of Findlay Oilers",
    "UNI_IND":"University of Indianapolis Greyhounds",
    "UNI_MON":"University of Montevallo Falcons",
    "UNI_SCR":"University of Scranton Royals",
    "UNI_TEX":"UT Permian Basin Falcons",
    "UNI_WES":"Sciences & Arts Oklahoma Drovers",
    "UNC_PEM":"UNC Pembroke Braves",
    "VAN_UNI":"Vanguard University Lions",
    "VIR_WIS":"Virginia Wesleyan University Marlins",
    "WAB_VAL":"Wabash Valley College Warriors",
    "Wal_Sen":"Walters State CC Senators",
    "WAL_WAL4":"Walters State CC Senators",
    "WAR_UNI1":"Wartburg College Knights",
    "WES_COL":"Westminster College Titans",
    "WES_FLO5":"West Florida Argonauts",
    "WES_TEX":"West Texas A&M Buffs","WES_TEX1":"West Texas A&M Buffs",
    "WIL_CAR":"William Carey University Crusaders",
    "WIL_JEW":"William Jewell College Cardinals",
    "WMI_BRO":"Western Michigan Broncos",
    "WOR_POL":"Worcester Polytechnic Institute Engineers",
    "WOU_WOL":"Western Oregon University Wolves",
    "Gol_W":"Gold/White Select",
    "Gra_R":"Select Team",
})

TEAM_CONFERENCES.update({
    "CSD_TRI": "Big West",
    "IWC": "Southland",
    "WES_HIL": "C-USA",
    "WCC": "SoCon",
    "SAM_BEA": "C-USA",
    "NOR_BIS": "Summit League",
    "THO_M": "Summit League",
    "HOL_CRU": "Patriot League",
    "NOR_CAT": "Big Ten",
    "LON_DIR": "Big West",
    "STE_LUM": "Southland",
    "HAR_CRI": "Ivy League",
    "BRO_BEA": "Ivy League",
    "MIL_UNI2": "Horizon League",
    "WAS_COU": "Mountain West",
    "SOU_GAM": "SEC",
    # ── Newly identified D1 teams ─────────────────────────────────────────────
    "BEL_COL":"ASUN","BEL_KNI":"ASUN",
    "BRO_COL":"Ivy League",
    "BUT_COL1":"Big East",
    "BRY_STR1":"America East",
    "CAM_UNI":"Big South","CAM_UNI1":"Big South",
    "CAL_POL1":"Big West",
    "CCU_BLD":"NEC","CEN_COL1":"NEC",
    "COP_STA":"MEAC",
    "DEL_HOR":"MEAC",
    "EC":"American",
    "EMU_EAG":"MAC",
    "GAR_RUN":"Big South",
    "HOW_HAW":"MEAC",
    "KEN_STA1":"C-USA",
    "LIN_UNI":"Ohio Valley",
    "LON_ISL22":"America East",
    "MER_UNI":"NEC",
    "MIS_BEA":"Missouri Valley",
    "MIS_DEL1":"SWAC",
    "MIS_ST.":"Missouri Valley",
    "NCA_BUL":"CAA",
    "NIC_COL1":"Southland",
    "NOR_IOW2":"Missouri Valley",
    "PRA_ACA":"SWAC","PRA_PAN":"SWAC","PRA_PRA1":"SWAC",
    "SAD_GAU":"Big West",
    "SOU_ILL":"Missouri Valley",
    "SOU_SOU8":"SWAC",
    "STE_MUS":"Southland",
    "STO_COL":"NEC",
    "TEX_A&M":"SEC","TEX_A&M1":"SEC",
    "WMI_BRO":"MAC",
})

TEAM_COLORS.update({
    "NEW_PRI": ("#005EB8", "#C99700"),
    "TRO_TRJ": ("#8A2432", "#B3A369"),
    "LOU_BUL": ("#E31B23", "#003DA5"),
    "UTR_VAQ": ("#F15A22", "#005CB9"),
    "MER_BEA": ("#F76800", "#000000"),
    "MT": ("#002147", "#8C8C8C"),
    "FLO_PAN": ("#081E3F", "#B6862C"),
    "CHA_FOR": ("#005035", "#A49665"),
    "SAN_BRO": ("#862633", "#FFFFFF"),
    "SOU_TRO": ("#990000", "#FFC72C"),
    "SOU_LIO": ("#006747", "#F5C400"),
    "CSD_TRI": ("#00629B", "#FFCD00"),
    "MIS_BEA": ("#5E0009", "#F1B82D"),
    "CAL_LAN": ("#002554", "#FDB515"),
    "SOU_IND16": ("#002D62", "#E4002B"),
    "UNL_REB": ("#BA0C2F", "#000000"),
    "LAM_CAR": ("#DC0032", "#FFFFFF"),
    "PRE_BLH": ("#005DAA", "#C4CED4"),
    "ASU_RED": ("#CC092F", "#000000"),
    "MAR_THU": ("#00B140", "#000000"),
    "SOU_GOL": ("#FFC72C", "#000000"),
    "ABI_WIL": ("#4E2683", "#FFFFFF"),
    "LON_LAN": ("#002F6C", "#A7A9AC"),
    "IWC": ("#BA0C2F", "#000000"),
    "SOU_RED": ("#C8102E", "#000000"),
    "EMU_EAG": ("#006633", "#FFFFFF"),
    "WES_HIL": ("#C60C30", "#FFFFFF"),
    "WCC": ("#592C88", "#A7A9AC"),
    "PAC_TIG": ("#F58025", "#000000"),
    "TEN_TEC": ("#4F2984", "#FFDD00"),
    "UTA_WOL": ("#275D38", "#FFFFFF"),
    "KEN_OWL": ("#FDB515", "#000000"),
    "EVA_ACE": ("#522D80", "#F2A900"),
    "ALA_LIO": ("#46166B", "#FDB515"),
    "UTS_ROA": ("#0C2340", "#F15A22"),
    "NIC_COL": ("#C8102E", "#A7A8AA"),
    "CAL_MAT": ("#CE1126", "#000000"),
    "JAC_GAM": ("#002D62", "#C9A240"),
    "ARI_WIL": ("#0C234B", "#AB0520"),
    "SAM_BUL": ("#00205B", "#C8102E"),
    "WIN_EAG": ("#660000", "#FFD200"),
    "VIR_KEY": ("#A6192E", "#F7C600"),
    "SOU_COU": ("#E35205", "#000000"),
    "MCN_COW": ("#00529B", "#FFD100"),
    "BEL_BRU": ("#00205B", "#C8102E"),
    "LIT_TRO": ("#6E2639", "#A7A9AC"),
    "QUN_RYL": ("#00205B", "#B3A369"),
    "COL_CHA": ("#73000A", "#000000"),
    "RAD_HIG": ("#CC0000", "#FFFFFF"),
    "CIT_BUL": ("#3975B7", "#FFFFFF"),
    "SEA_RED": ("#AA0000", "#000000"),
    "CHA_BUC": ("#002855", "#A7A9AC"),
    "TOL_ROC": ("#15397F", "#FFCE00"),
    "NOR_DEM": ("#4B0082", "#F2A900"),
    "STU_BOB": ("#510C76", "#C0C0C0"),
    "STM_BOB": ("#510C76", "#C0C0C0"),
    "THO_M": ("#510C76", "#C0C0C0"),
    "BRA_BRA": ("#A50000", "#FFFFFF"),
    "NOF_OSP": ("#00246B", "#B3A369"),
    "MOR_EAG": ("#005EB8", "#F2A900"),
    "LAF_LEP": ("#800000", "#FFFFFF"),
    "ORA_GOL": ("#002F6C", "#C5B783"),
    "WM_TRI": ("#115740", "#B9975B"),
    "VAL_CRU": ("#381E0E", "#F2A900"),
    "MIA_RED": ("#B61E2E", "#FFFFFF"),
    "OAK_GOL": ("#B59A57", "#000000"),
    "ILL_RED": ("#CE1126", "#FFFFFF"),
    "VIL_WIL": ("#00205B", "#13B5EA"),
    "MSM_MTN": ("#002855", "#A7A9AC"),
    "YSU_PEN": ("#C8102E", "#FFFFFF"),
    "YOU_HAR": ("#C8102E", "#FFFFFF"),
    "UWM_PAN": ("#000000", "#FFBD00"),
    "UNO_MAV": ("#000000", "#D71920"),
    "BGS_FAL": ("#4F2C1D", "#FF7300"),
    "SAM_BEA": ("#F58220", "#FFFFFF"),
    "HOL_CRU": ("#602D89", "#FFFFFF"),
    "WIU_LEA": ("#663399", "#F2A900"),
    "NOR_BIS": ("#0A5640", "#FFC82E"),
    "MEX_LOB": ("#BA0C2F", "#A7A8AA"),
    "DAR_GRE": ("#00693E", "#FFFFFF"),
    "TEX_ISL": ("#0067C5", "#00A3E0"),
    "MIS_DEL": ("#006747", "#C5B783"),
    "GRA_TIG": ("#EAAA00", "#000000"),
    "NOR_CAT": ("#4E2A84", "#FFFFFF"),
    "LON_DIR": ("#000000", "#FFD700"),
    "STE_LUM": ("#512888", "#FFFFFF"),
    "HAR_CRI": ("#A51C30", "#FFFFFF"),
    "BRO_BEA": ("#4E3629", "#ED1C24"),
    "MIL_UNI2": ("#000000", "#FFBD00"),
    "WAS_COU": ("#981E32", "#5E6A71"),
    # ── D1 colors ─────────────────────────────────────────────────────────────
    "BEL_COL":("#002D62","#C8102E"),"BEL_KNI":("#002D62","#C8102E"),
    "BRO_COL":("#4E3629","#ED1C24"),
    "BUT_COL1":("#13294B","#747F7F"),
    "BRY_STR1":("#000000","#C8A415"),
    "CAM_UNI":("#F47920","#000000"),"CAM_UNI1":("#F47920","#000000"),
    "CAL_POL1":("#154734","#C8B560"),
    "CCU_BLD":("#003DA5","#FFFFFF"),"CEN_COL1":("#003DA5","#FFFFFF"),
    "COP_STA":("#00205B","#B3A369"),
    "DEL_HOR":("#002F6C","#EAA221"),
    "EC":("#592A8A","#FDC82F"),
    "GAR_RUN":("#750000","#000000"),
    "HOW_HAW":("#003A70","#E31837"),
    "KEN_STA1":("#FDB515","#000000"),
    "LIN_UNI":("#002D6C","#FFD700"),
    "LON_ISL22":("#002D6C","#69BE28"),
    "MER_UNI":("#002D72","#FDB515"),
    "MIS_BEA":("#5E0009","#F1B82D"),
    "MIS_DEL1":("#006747","#C5B783"),
    "MIS_ST.":("#5E0009","#F1B82D"),
    "NCA_BUL":("#004684","#FFD700"),
    "NIC_COL1":("#C8102E","#A7A8AA"),
    "NOR_IOW2":("#4B116F","#FFCC00"),
    "PRA_ACA":("#4F2D7F","#FFD700"),"PRA_PAN":("#4F2D7F","#FFD700"),"PRA_PRA1":("#4F2D7F","#FFD700"),
    "SAD_GAU":("#003660","#FDD023"),
    "SOU_ILL":("#720000","#000000"),
    "SOU_SOU8":("#003DA5","#F0C528"),
    "STE_MUS":("#512888","#FFFFFF"),
    "STO_COL":("#003DA5","#C8A415"),
    "TEX_A&M":("#500000","#FFFFFF"),"TEX_A&M1":("#500000","#FFFFFF"),
    "WMI_BRO":("#6C4023","#B5A167"),
})

TEAM_LOGO_ALIASES = {
    "ALB_DAN": "ALB_GRE",
    "LOW_RIV": "UML_RIV",
    "GEO_COL": "GEO_GWI",
    "GEO_COL1": "GEO_GWI",
    "GEO_COL2": "GEO_GWI",
    "SAI_JOE": "JOE_HAW",
    "STJ_HAW": "JOE_HAW",
    "STL_BIL": "SLU_BILL",
    "STB_BON": "SBU_BON",
    "RIDER_BRO": "RID_BRO",
    "MAS_MIN": "UMASS",
    "UMA_AMH": "UMASS",
    "OHI_BOB": "OHIO_BOB",
    "ELO_PHO": "ELON_PHO",
    "JAC_GAM": "JAC_STA",
    "TEX_A&M1": "TEX_A&M",
    "TEX_AGG": "TEX_A&M",
    "STU_BOB": "STM_BOB",
    "THO_M": "STM_BOB",
    "TRO_TRJ": "TRO_T",
    "MT": "MEM_TIG",
    "CHA_FOR": "CHA_49E",
    "SOU_GOL": "SOU_MIS",
    "LAF_LEP": "LAF_LEO",
    "HOL_CRU": "HOL_CRO",
    "COL_CHA": "CHS_COU",
    "WIN_EAG": "WIN_BUL",
    "UWM_PAN": "MIL_UNI",
    "MIL_UNI2": "MIL_UNI",
    "LON_DIR": "LON_BEA",
}

BG   = "#0F1218"
BG2  = "#171C24"
PANEL = "#151A22"
PANEL2 = "#1C2330"
GRID = "#344052"
TXT  = "#F8FAFC"
TXT2 = "#B8C1CC"

# ── wOBA / wRC+ constants ─────────────────────────────────────────────────────
# D1 collegiate linear weights (derived from 2019 D1 run-expectancy research,
# adjusted for 2026 environment). These are lower than FanGraphs MLB weights
# because college run values per event are lower than MLB.
WOBA_BB  = 0.64
WOBA_HBP = 0.66
WOBA_1B  = 0.80
WOBA_2B  = 1.12
WOBA_3B  = 1.41
WOBA_HR  = 1.76
# wOBA_scale = lgwOBA / lgOBP  — converts wOBA-units to runs per PA
# Derived: 0.338 / 0.387 = 0.873  (lgOBP from same 2026 D1 dataset)
WOBA_SCALE = 0.873
LG_OBP     = 0.387   # 2026 D1 average OBP
# lgR/PA  = lgwOBA / wOBA_scale = lgOBP by construction = 0.387
LG_R_PA    = LG_OBP
# 2026 D1 league-average wOBA — calibrated to .325
# Collegiate weights (BB=.64, 1B=.80, HR=1.76), denominator = AB+BB+HBP
LG_WOBA = 0.325


def _muted_text_on(hex_color: str) -> str:
    """Muted secondary text guaranteed readable against hex_color background."""
    try:
        h = str(hex_color or "#000000").lstrip("#")
        lum = (0.299*int(h[0:2],16) + 0.587*int(h[2:4],16) + 0.114*int(h[4:6],16)) / 255
        return "#444444" if lum > 0.62 else "#CDBFAF"
    except Exception:
        return "#CDBFAF"


st.set_page_config(page_title="College Baseball Plus", page_icon="CB", layout="wide")


# ── Cached model loader — loads directly from repo models/ folder ─────────────
@st.cache_resource(show_spinner=False)
def _get_models():
    try:
        sm = joblib.load(MODELS_DIR / "stuff_lgbm_model.pkl")
        sl = joblib.load(MODELS_DIR / "stuff_lgbm_league.pkl")
        lm = joblib.load(MODELS_DIR / "location_lgbm_model.pkl")
        ll = joblib.load(MODELS_DIR / "location_lgbm_league.pkl")
        return sm, sl, lm, ll
    except Exception:
        return None, None, None, None


def _compute_stuffplus(df: pd.DataFrame, model, league) -> pd.DataFrame:
    mu    = league["mean"]
    sigma = league["std"] if league["std"] > 0 else 1.0
    X = df[_STUFF_FEATURES].fillna(0)
    df["Stuff+"] = 100 + 50 * ((model.predict_proba(X)[:, 1] - mu) / sigma)
    return df


def _compute_locationplus(df: pd.DataFrame, model, league) -> pd.DataFrame:
    mu    = league["mean"]
    sigma = league["std"] if league["std"] > 0 else 1.0
    df["Balls"]   = _numeric_series(df, "Balls", 0).fillna(0).astype(int)
    df["Strikes"] = _numeric_series(df, "Strikes", 0).fillna(0).astype(int)
    if "zone" not in df.columns:
        df["zone"] = 0
    else:
        df["zone"] = pd.to_numeric(df["zone"], errors="coerce").fillna(0)
    pitch_code = df["Pitch"].astype("category").cat.codes
    X = pd.DataFrame({
        "PlateLocSide":   df["PlateLocSide"],
        "PlateLocHeight": df["PlateLocHeight"],
        "zone":           df["zone"],
        "Balls":          df["Balls"],
        "Strikes":        df["Strikes"],
        "pitch_abbr":     pitch_code,
    })
    if hasattr(model, "_n_classes") and model._n_classes is None:
        model._n_classes = 1
    df["Loc+"] = 100 + 50 * ((model.predict(X) - mu) / sigma)
    return df


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
    code = str(code or "").strip()
    if code.upper() in {"FOR_RAM", "FOR_RAM1"}:
        ram_head = PROJECT_ROOT / "static" / "rams.png"
        if ram_head.exists():
            return ram_head
    candidates = [code]
    alias = TEAM_LOGO_ALIASES.get(code)
    if alias:
        candidates.append(alias)
    for candidate in candidates:
        for suffix in [".png",".jpg",".jpeg"]:
            p = LOGO_DIR / f"{candidate}{suffix}"
            if p.exists():
                return p
    return None


def _place_logo(fig_or_ax, logo: "Path | None", primary: str, accent: str,
                bounds: tuple, use_inset: bool = False, opacity: float = 1.0) -> bool:
    """Render a trimmed, translucent team logo with no visible box."""
    if not logo:
        return False
    try:
        img = Image.open(logo).convert("RGBA")
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        pad = max(8, int(max(img.size) * 0.08))
        canvas = Image.new("RGBA", (img.size[0] + pad * 2, img.size[1] + pad * 2), (0, 0, 0, 0))
        canvas.alpha_composite(img, (pad, pad))
        img = canvas
        arr = np.array(img)
        arr[:, :, 3] = (arr[:, :, 3].astype(float) * opacity).clip(0, 255).astype(np.uint8)
        if use_inset:
            lax = fig_or_ax.inset_axes(list(bounds))
        else:
            lax = fig_or_ax.add_axes(list(bounds))
        lax.set_facecolor((0, 0, 0, 0))
        lax.patch.set_alpha(0)
        lax.imshow(arr, aspect="equal")
        lax.set_xticks([]); lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_visible(False)
        return True
    except Exception:
        return False


def _team_logo_array(logo: "Path | None", opacity: float = 0.18) -> np.ndarray | None:
    """Return a trimmed translucent logo array for in-chart watermarks."""
    if not logo:
        return None
    try:
        img = Image.open(logo).convert("RGBA")
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        arr = np.array(img)
        alpha = arr[:, :, 3].astype(float)
        arr[:, :, 3] = (alpha * opacity).clip(0, 255).astype(np.uint8)
        return arr
    except Exception:
        return None


def pc(pt: str) -> str:
    return PITCH_COLORS.get(str(pt).upper()[:2], "#888888")


def fmt(v, stat="") -> str:
    if pd.isna(v):
        return "—"
    if stat in {"BAA","SLG","BA","OBP","OPS"}:
        return f"{float(v):.3f}".replace("0.",".")
    if stat in {"Pitches","Games","K","BB","N"}:
        return f"{int(round(float(v))):,}"
    if stat == "Usage%" or str(stat).endswith("%"):
        return f"{float(v):.1f}%"
    return f"{float(v):.1f}"


def _series(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    """Return a column or a full-length default Series."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _numeric_series(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    return pd.to_numeric(_series(df, col, default), errors="coerce")


def _pitcher_outs(df: pd.DataFrame) -> float:
    """Compute pitcher outs, using TrackMan OutsOnPlay when available.

    Current deployed Parquet files may not include OutsOnPlay yet, so fall back
    to PA-ending results: strikeouts and batter/runner-out outcomes count as one
    out. Future Parquet rebuilds include OutsOnPlay for multi-out plays.
    """
    if df.empty:
        return 0.0
    pr = df.get("PlayResult", pd.Series("", index=df.index)).fillna("").astype(str)
    kbb = df.get("KorBB", pd.Series("", index=df.index)).fillna("").astype(str)
    if "OutsOnPlay" in df.columns and df["OutsOnPlay"].notna().any():
        return float(_numeric_series(df, "OutsOnPlay", 0).fillna(0).sum() + kbb.eq("Strikeout").sum())
    out_results = {"Out", "FieldersChoice", "Sacrifice"}
    return float(kbb.eq("Strikeout").sum() + pr.isin(out_results).sum())


# ── Baseball Savant–style percentile coloring ────────────────────────────────
# Breakpoints (p10, p30, p50, p70, p90) for D1 college hitters, 2025-26
_HITTER_PCTS: dict[str, tuple] = {
    # 7-point breakpoints (p2,p10,p25,p50,p75,p90,p98) from 3,766 D1 hitters
    # ≥50 PA, 2026 TrackMan parquet, collegiate wOBA weights, lgwOBA=.325
    "wRC+":   ( 66,   81,   93,  106,  119,  132,  152, True),
    "wOBA":   (.213, .263, .300, .341, .385, .427, .497, True),
    "BA":     (.159, .219, .254, .296, .341, .383, .440, True),
    "OBP":    (.248, .309, .350, .398, .440, .484, .562, True),
    "SLG":    (.220, .296, .373, .448, .551, .642, .800, True),
    "OPS":    (.487, .629, .734, .851, .983, 1.109, 1.320, True),
    "K%":     ( 6.5,  10.6, 14.3, 19.0, 24.4, 30.0, 38.2, False),
    "BB%":    ( 3.8,   6.3,  8.6, 11.3, 14.3, 17.5, 22.2, True),
    "Whiff%": ( 9.0,  13.4, 17.4, 22.5, 27.8, 33.0, 40.2, False),
    "Chase%": (21.0,  25.3, 28.6, 32.2, 35.8, 39.3, 44.1, False),
    "Avg EV": (80.1,  83.1, 85.2, 87.6, 89.9, 91.9, 94.3, True),
    "HH%":    ( 6.3,  17.0, 25.9, 35.2, 43.9, 51.1, 58.8, True),
}

# D1 pitcher percentile benchmarks (season-level, 2,088 pitchers ≥30 PA)
_D1_PITCHER_PCTS_CBB = {
    "Stuff+":  ( 75.0,  87.0, 100.0, 113.0, 125.0, True),
    "Loc+":    ( 75.0,  87.0, 100.0, 113.0, 125.0, True),
    "Velo":    ( 86.2,  88.0,  89.7,  91.5,  93.2, True),
    "CSW%":    ( 23.5,  25.7,  28.0,  30.7,  32.9, True),
    "Zone%":   ( 38.6,  41.4,  44.5,  47.2,  49.6, True),
    "Whiff%P": ( 16.1,  19.6,  23.6,  28.2,  32.7, True),  # pitcher Whiff%
    "K%P":     ( 12.2,  16.2,  20.5,  25.7,  30.5, True),
    "BB%P":    (  5.9,   8.2,  11.0,  14.7,  19.4, False),
    "GB%P":    ( 31.4,  36.8,  42.6,  49.2,  55.1, True),
    "Avg EV A":( 84.8,  86.6,  88.3,  89.7,  91.1, False),  # lower = better
}


def _pitcher_pct_rank_cbb(stat: str, val) -> float | None:
    """0–1 rank for a pitcher stat (1.0 = best). Smooth extrapolation at edges."""
    if val is None or pd.isna(val) or stat not in _D1_PITCHER_PCTS_CBB:
        return None
    p10, p25, p50, p75, p90, high = _D1_PITCHER_PCTS_CBB[stat]
    bps  = [p10, p25, p50, p75, p90]
    pcts = [0.10, 0.25, 0.50, 0.75, 0.90]
    fv   = float(val)
    if fv < bps[0]:
        pct = max(0.01, 0.10 * fv / bps[0]) if bps[0] > 0 else 0.01
    elif fv > bps[-1]:
        pct = min(0.99, 0.90 + 0.09 * (fv - bps[-1]) / max(bps[-1] * 0.25, 1))
    else:
        pct = 0.50
        for i in range(len(bps) - 1):
            if bps[i] <= fv <= bps[i + 1]:
                t = (fv - bps[i]) / (bps[i + 1] - bps[i])
                pct = pcts[i] + t * (pcts[i + 1] - pcts[i])
                break
    return (1.0 - pct) if not high else pct


def _pct_to_hex_cbb(pct: float | None, stops: list) -> str:
    """Generic stop-list → hex colour."""
    if pct is None:
        return "#2a2a3a"
    r, g, b = 165, 165, 165
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]; p1, c1 = stops[i + 1]
        if p0 <= pct <= p1:
            t = (pct - p0) / (p1 - p0) if p1 > p0 else 0
            r = int(c0[0] + t*(c1[0]-c0[0]))
            g = int(c0[1] + t*(c1[1]-c0[1]))
            b = int(c0[2] + t*(c1[2]-c0[2]))
            break
    lum = (0.299*r + 0.587*g + 0.114*b) / 255
    return f"#{r:02x}{g:02x}{b:02x}"


def _pct_label_cbb(pct: float | None) -> str:
    if pct is None: return "—"
    n = max(1, int(round(pct * 100)))
    if n >= 90: return f"{n}th ★"
    if 11 <= n <= 13: return f"{n}th"
    suffix = {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

# Savant-style gradient: blue (poor) → mid-gray (avg) → red (elite)
# Mid-gray instead of near-white keeps text readable on both sides of avg
_SAVANT_STOPS = [
    (0.00, ( 10,  46, 110)),  # #0a2e6e  deep blue    0th pct
    (0.20, ( 25,  86, 160)),  # #1956a0  blue        20th pct
    (0.40, ( 94, 163, 208)),  # #5ea3d0  light blue  40th pct
    (0.50, (120, 120, 120)),  # #787878  mid-gray    50th pct
    (0.60, (209, 100,  70)),  # #d16446  light red   60th pct
    (0.80, (209,  60,  40)),  # #d13c28  red         80th pct
    (1.00, (139,   0,   0)),  # #8b0000  dark red   100th pct
]

# Alias — same gradient used for both pitcher and hitter context;
# direction (high/low = good) is determined by the percentile rank function.
_HITTER_STOPS = _SAVANT_STOPS


def _pct_rank(stat: str, value) -> float | None:
    """0–1 percentile rank for a hitter stat (1.0 = best for hitter)."""
    if pd.isna(value) or stat not in _HITTER_PCTS:
        return None
    p2, p10, p25, p50, p75, p90, p98, high_good = _HITTER_PCTS[stat]
    bps  = [p2, p10, p25, p50, p75, p90, p98]
    pcts = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    fv   = float(value)
    if fv < bps[0]:
        pct = max(0.01, 0.02 * fv / bps[0]) if bps[0] > 0 else 0.01
    elif fv > bps[-1]:
        pct = min(0.99, 0.98 + 0.01 * (fv - bps[-1]) / max(abs(bps[-1]) * 0.15, 1))
    else:
        pct = 0.5
        for i in range(len(bps) - 1):
            if bps[i] <= fv <= bps[i+1]:
                t = (fv - bps[i]) / (bps[i+1] - bps[i])
                pct = pcts[i] + t * (pcts[i+1] - pcts[i])
                break
    return (1.0 - pct) if not high_good else pct


def _stops_color(pct: float, stops: list) -> tuple[str, str]:
    r, g, b = stops[len(stops)//2][1]
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]; p1, c1 = stops[i+1]
        if p0 <= pct <= p1:
            t = (pct - p0) / (p1 - p0) if p1 > p0 else 0
            r = int(c0[0] + t*(c1[0]-c0[0]))
            g = int(c0[1] + t*(c1[1]-c0[1]))
            b = int(c0[2] + t*(c1[2]-c0[2]))
            break
    bg  = f"#{r:02x}{g:02x}{b:02x}"
    lum = (0.299*r + 0.587*g + 0.114*b) / 255
    return bg, ("#000000" if lum > 0.52 else "#ffffff")


def _savant_color(stat: str, value) -> tuple[str, str]:
    """Savant-style: red = elite, blue = poor (for any stat, direction handled by pct_rank)."""
    pct = _pct_rank(stat, value)
    return _stops_color(pct, _SAVANT_STOPS) if pct is not None else ("#1a1a2a", "#888888")


def _hitter_color(stat: str, value) -> tuple[str, str]:
    """Hitter-perspective: RED = elite/hot, blue = poor/cold."""
    pct = _pct_rank(stat, value)
    return _stops_color(pct, _HITTER_STOPS) if pct is not None else ("#1a1a2a", "#888888")


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
    # Preconnect + async font load — avoids blocking the first paint
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap"
          rel="stylesheet" media="print" onload="this.media='all'">
    <noscript><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet"></noscript>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    :root{
        --cbb-bg:#172033;
        --cbb-panel:#202b3f;
        --cbb-panel-2:#26344a;
        --cbb-line:#3d4b63;
        --cbb-text:#f8fafc;
        --cbb-muted:#b7c3d3;
        --cbb-gold:#d6a74f;
        --cbb-red:#b91c1c;
        --cbb-blue:#38bdf8;
    }
    .stApp{
        background:
            linear-gradient(180deg,rgba(39,52,74,.94),rgba(23,32,51,.98) 420px),
            radial-gradient(circle at 18% -10%,rgba(214,167,79,.18),transparent 34%),
            radial-gradient(circle at 82% -16%,rgba(56,189,248,.18),transparent 35%),
            var(--cbb-bg);
        color:var(--cbb-text);
        font-family:'Inter',sans-serif;
    }
    div[data-testid="stHeader"]{background:transparent}
    .block-container{max-width:1420px;padding-top:2.1rem;padding-bottom:3.5rem}
    h1,h2,h3{letter-spacing:0;color:var(--cbb-text)}
    p,span,label,div{letter-spacing:0}

    div[data-testid="stMetric"]{
        background:linear-gradient(180deg,rgba(42,56,79,.96),rgba(31,43,63,.96));
        border:1px solid rgba(214,167,79,.26);
        border-radius:8px;
        padding:12px 12px 10px;
        box-shadow:0 14px 30px rgba(0,0,0,.20);
    }
    div[data-testid="stMetricValue"]{
        font-size:1.32rem!important;
        font-weight:800!important;
        color:#ffffff!important;
    }
    div[data-testid="stMetricLabel"]{
        font-size:0.68rem!important;
        color:var(--cbb-muted)!important;
        text-transform:uppercase;
        letter-spacing:.08em;
    }

    .cbb-hero{
        position:relative;
        overflow:hidden;
        background:
            linear-gradient(135deg,rgba(131,24,24,.88) 0%,rgba(37,52,75,.94) 46%,rgba(27,38,58,.97) 100%),
            repeating-linear-gradient(90deg,rgba(255,255,255,.04) 0 1px,transparent 1px 34px);
        border:1px solid rgba(214,167,79,.32);
        border-radius:10px;
        padding:28px 32px 24px;
        margin-bottom:18px;
        box-shadow:0 24px 70px rgba(0,0,0,.36);
    }
    .cbb-hero:after{
        content:"";
        position:absolute;
        inset:auto -8% -55% 48%;
        height:165%;
        background:
            linear-gradient(90deg,transparent,rgba(214,167,79,.08),transparent),
            repeating-linear-gradient(115deg,rgba(255,255,255,.055) 0 1px,transparent 1px 15px);
        transform:skewX(-16deg);
        pointer-events:none;
    }
    .hero-kicker{
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding:5px 10px;
        border:1px solid rgba(214,167,79,.34);
        border-radius:999px;
        background:rgba(15,23,42,.38);
        color:#f7d48a;
        font-size:11px;
        font-weight:800;
        text-transform:uppercase;
        letter-spacing:.10em;
        margin-bottom:10px;
    }
    .cbb-hero h1{
        position:relative;
        margin:0 0 8px;
        font-size:34px;
        line-height:1.04;
        font-weight:800;
        color:#fff;
        max-width:760px;
    }
    .cbb-hero p{
        position:relative;
        margin:0;
        color:#d7dee9;
        font-size:14px;
        line-height:1.55;
        max-width:800px;
    }
    .hero-chip-row{
        position:relative;
        display:flex;
        gap:10px;
        flex-wrap:wrap;
        margin-top:18px;
    }
    .hero-chip{
        background:rgba(255,255,255,.105);
        border:1px solid rgba(255,255,255,.20);
        border-radius:8px;
        padding:9px 12px;
        min-width:132px;
    }
    .hero-chip b{display:block;color:#ffffff;font-size:14px}
    .hero-chip span{display:block;color:#aeb9ca;font-size:11px;margin-top:1px;text-transform:uppercase;letter-spacing:.06em}

    .data-strip{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:14px;
        background:rgba(34,47,69,.92);
        border:1px solid rgba(214,167,79,.24);
        border-radius:8px;
        padding:12px 14px;
        margin:0 0 18px;
        color:var(--cbb-muted);
        box-shadow:0 12px 32px rgba(0,0,0,.20);
    }
    .data-strip b{color:#ffffff}
    .data-strip span{font-size:12px}

    .filter-row{
        background:rgba(34,47,69,.94);
        border:1px solid rgba(214,167,79,.24);
        border-radius:8px;
        padding:16px 18px;
        margin-bottom:14px;
        box-shadow:0 15px 42px rgba(0,0,0,.22);
    }
    .pitcher-card{
        background:
            linear-gradient(135deg,rgba(37,50,73,.98),rgba(31,43,63,.96)),
            repeating-linear-gradient(90deg,rgba(255,255,255,.035) 0 1px,transparent 1px 28px);
        border:1px solid rgba(214,167,79,.30);
        border-left:4px solid var(--cbb-gold);
        border-radius:8px;
        padding:18px 20px;
        margin-bottom:16px;
        box-shadow:0 16px 38px rgba(0,0,0,.25);
    }
    .pitcher-name{font-size:24px;font-weight:800;color:#fff;margin:0 0 3px}
    .pitcher-meta{font-size:13px;color:var(--cbb-muted);margin:0}
    .conf-badge{
        display:inline-block;
        padding:3px 10px;
        border-radius:999px;
        font-size:11px;
        font-weight:800;
        letter-spacing:.06em;
        margin-left:8px;vertical-align:middle;
        text-transform:uppercase;
        box-shadow:inset 0 0 0 1px rgba(255,255,255,.16);
    }

    .metric-explainer{
        background:rgba(34,47,69,.90);
        border:1px solid rgba(56,189,248,.24);
        border-radius:8px;
        padding:12px 16px;
        font-size:12px;
        color:var(--cbb-muted);
        line-height:1.6;
        margin-top:4px;
    }
    .metric-explainer b{color:#e2e8f0}

    .paywall{
        max-width:720px;
        margin:56px auto 24px;
        padding:34px;
        border-radius:10px;
        border:1px solid rgba(214,167,79,.28);
        background:
            linear-gradient(135deg,rgba(131,24,24,.68),rgba(37,52,75,.95) 52%,rgba(23,32,51,.98)),
            repeating-linear-gradient(90deg,rgba(255,255,255,.04) 0 1px,transparent 1px 34px);
        box-shadow:0 28px 80px rgba(0,0,0,.38);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-testid="stNumberInput"] input,
    textarea{
        background:#1f2a3d!important;
        color:#f8fafc!important;
        border-color:#334155!important;
        border-radius:8px!important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    input, textarea{color:#f8fafc!important}
    label, .stRadio label, .stSelectbox label, .stNumberInput label{
        color:#cbd5e1!important;
        font-weight:700!important;
        font-size:.78rem!important;
    }
    div[role="radiogroup"]{
        gap:8px;
    }
    div[role="radiogroup"] label{
        background:rgba(30,41,59,.86);
        border:1px solid rgba(203,213,225,.24);
        border-radius:8px;
        padding:8px 12px;
        min-height:38px;
    }
    div[role="radiogroup"] label:has(input:checked){
        border-color:rgba(214,167,79,.70);
        background:linear-gradient(180deg,rgba(120,53,15,.50),rgba(30,41,59,.90));
        color:#ffffff!important;
    }
    div[data-testid="stDataFrame"]{
        border:1px solid rgba(214,167,79,.18);
        border-radius:8px;
        overflow:hidden;
        box-shadow:0 16px 42px rgba(0,0,0,.22);
    }

    .stDownloadButton>button{
        background:linear-gradient(135deg,#b91c1c,#7f1d1d)!important;
        color:#fff!important;
        border:1px solid rgba(214,167,79,.24)!important;
        border-radius:8px!important;
        font-weight:800!important;
        letter-spacing:.02em!important;
    }
    .stDownloadButton>button:hover{
        background:linear-gradient(135deg,#dc2626,#991b1b)!important;
        border-color:rgba(214,167,79,.55)!important;
    }
    .stButton>button,
    .stFormSubmitButton>button,
    div[data-testid="stLinkButton"] a{
        background:#22314a!important;
        color:#f8fafc!important;
        border:1px solid rgba(214,167,79,.28)!important;
        border-radius:8px!important;
        font-weight:800!important;
    }
    .stButton>button:hover,
    .stFormSubmitButton>button:hover,
    div[data-testid="stLinkButton"] a:hover{
        background:#2b3d5a!important;
        border-color:rgba(214,167,79,.62)!important;
        color:#ffffff!important;
    }

    hr{border-color:rgba(214,167,79,.16)!important;margin:1rem 0 1.15rem}
    .stCaptionContainer, .stCaptionContainer p{color:var(--cbb-muted)!important}

    /* ── Coverage / stat strip tiles ──────────────────────────────────── */
    .cov-strip{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 20px}
    .cov-tile{
        background:rgba(34,47,69,.88);
        border:1px solid rgba(214,167,79,.22);
        border-radius:10px;
        padding:14px 20px;
        flex:1;min-width:110px;text-align:center;
        transition:border-color .2s,transform .2s;
    }
    .cov-tile:hover{border-color:rgba(214,167,79,.55);transform:translateY(-2px)}
    .cov-tile .cov-num{font-size:1.75rem;font-weight:800;color:#f8d96e;line-height:1}
    .cov-tile .cov-lbl{font-size:.66rem;color:#9BAABF;text-transform:uppercase;letter-spacing:.09em;margin-top:5px}

    /* ── Feature preview grid (free users) ────────────────────────────── */
    .feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0 20px}
    .feat-card{
        background:rgba(30,42,60,.90);
        border:1px solid rgba(56,189,248,.18);
        border-radius:10px;
        padding:16px 12px;
        text-align:center;
        transition:border-color .2s,transform .2s;
    }
    .feat-card:hover{border-color:rgba(56,189,248,.45);transform:translateY(-2px)}
    .feat-icon{font-size:1.7rem;margin-bottom:8px}
    .feat-title{font-size:.88rem;font-weight:700;color:#e2e8f0;margin-bottom:4px}
    .feat-desc{font-size:.71rem;color:#9BAABF;line-height:1.45}

    /* ── Section header bar ────────────────────────────────────────────── */
    .section-hdr{
        display:flex;align-items:center;gap:10px;
        padding:10px 0 6px;
        border-bottom:2px solid rgba(214,167,79,.24);
        margin-bottom:16px;
    }
    .section-hdr .sh-icon{font-size:1.35rem}
    .section-hdr .sh-title{font-size:1.25rem;font-weight:800;color:#fff}
    .section-hdr .sh-badge{
        margin-left:auto;
        background:rgba(214,167,79,.16);
        border:1px solid rgba(214,167,79,.34);
        border-radius:999px;
        padding:2px 10px;
        font-size:.7rem;
        font-weight:700;
        color:#f8d96e;
        letter-spacing:.06em;
    }

    /* ── Hot / fire badge ──────────────────────────────────────────────── */
    .hot-badge{
        display:inline-block;
        background:linear-gradient(135deg,#ef4444,#f97316);
        color:#fff!important;
        border-radius:999px;
        padding:1px 7px;
        font-size:.62rem;
        font-weight:800;
        letter-spacing:.05em;
        margin-left:5px;
        vertical-align:middle;
    }
    .elite-badge{
        display:inline-block;
        background:linear-gradient(135deg,#7c3aed,#4f46e5);
        color:#fff!important;
        border-radius:999px;
        padding:1px 7px;
        font-size:.62rem;
        font-weight:800;
        letter-spacing:.05em;
        margin-left:5px;
        vertical-align:middle;
    }

    /* ── Stat glow pulse ───────────────────────────────────────────────── */
    @keyframes cbb-glow{
        0%,100%{box-shadow:0 0 0 0 rgba(214,167,79,0)}
        50%{box-shadow:0 0 20px 5px rgba(214,167,79,.18)}
    }
    .stat-glow{animation:cbb-glow 3.2s ease-in-out infinite}

    /* ── Hot Right Now strip ───────────────────────────────────────────── */
    .hot-strip{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 18px}
    .hot-card{
        flex:1;min-width:140px;
        background:linear-gradient(135deg,rgba(30,42,60,.96),rgba(25,36,54,.98));
        border:1px solid rgba(214,167,79,.28);
        border-radius:12px;padding:14px 16px;text-align:center;
        transition:border-color .2s,transform .18s;cursor:default;
    }
    .hot-card:hover{border-color:rgba(214,167,79,.65);transform:translateY(-2px)}
    .hot-card .hc-crown{font-size:1.3rem;margin-bottom:4px}
    .hot-card .hc-val{font-size:1.6rem;font-weight:800;color:#f8d96e;line-height:1}
    .hot-card .hc-stat{font-size:.65rem;color:#9BAABF;text-transform:uppercase;letter-spacing:.08em;margin:3px 0}
    .hot-card .hc-name{font-size:.78rem;color:#e2e8f0;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .hot-card .hc-team{font-size:.66rem;color:#9BAABF}

    /* ── Compare tool ──────────────────────────────────────────────────── */
    .compare-wrap{
        background:rgba(30,42,60,.92);
        border:1.5px solid rgba(56,189,248,.28);
        border-radius:12px;padding:16px 18px 18px;
        margin:0 0 18px;
    }
    .compare-hdr{font-size:.72rem;font-weight:800;color:#38bdf8;
        text-transform:uppercase;letter-spacing:.10em;margin-bottom:8px}
    .vs-badge{
        display:flex;align-items:center;justify-content:center;
        font-size:1.1rem;font-weight:900;color:#9BAABF;padding:4px 0;
    }

    /* ── Recently viewed ───────────────────────────────────────────────── */
    .rv-row{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 4px}
    .rv-chip{
        background:rgba(30,42,60,.88);
        border:1px solid rgba(214,167,79,.25);
        border-radius:8px;padding:6px 12px;
        font-size:.78rem;color:#e2e8f0;font-weight:600;
        cursor:pointer;transition:border-color .18s;white-space:nowrap;
    }
    .rv-chip:hover{border-color:rgba(214,167,79,.60);color:#ffffff}

    /* ── Sub progress bar ──────────────────────────────────────────────── */
    .sub-bar-wrap{margin:8px 0 12px}
    .sub-bar-track{background:#1a2540;border-radius:999px;height:6px;overflow:hidden}
    .sub-bar-fill{height:100%;border-radius:999px;transition:width .6s ease}
    .sub-bar-label{font-size:.7rem;color:#9BAABF;margin-top:4px;display:flex;justify-content:space-between}

    /* ── Global player search ──────────────────────────────────────────── */
    .search-container{
        background:linear-gradient(135deg,rgba(30,42,60,.96),rgba(25,36,54,.98));
        border:1.5px solid rgba(214,167,79,.40);
        border-radius:12px;
        padding:14px 18px 16px;
        margin:0 0 18px;
        box-shadow:0 12px 36px rgba(0,0,0,.28);
        transition:border-color .2s;
    }
    .search-container:hover{border-color:rgba(214,167,79,.65)}
    .search-label{
        font-size:.72rem;font-weight:800;color:#f8d96e;
        text-transform:uppercase;letter-spacing:.10em;margin-bottom:4px;
    }

    /* ── Upgrade CTA banner ────────────────────────────────────────────── */
    .upgrade-banner{
        background:linear-gradient(135deg,rgba(120,53,15,.70),rgba(37,52,75,.95));
        border:1px solid rgba(214,167,79,.45);
        border-radius:12px;
        padding:22px 24px;
        margin:20px 0;
        display:flex;
        align-items:center;
        gap:18px;
        box-shadow:0 18px 50px rgba(0,0,0,.30);
    }
    .upgrade-banner .ub-icon{font-size:2.4rem}
    .upgrade-banner .ub-title{font-size:1.1rem;font-weight:800;color:#fff;margin-bottom:4px}
    .upgrade-banner .ub-desc{font-size:.82rem;color:#d7dee9;line-height:1.5}

    /* ── Multiselect & dropdown — comprehensive dark theme ─────────────── */
    /* Dropdown popup container */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] div,
    div[data-baseweb="menu"],
    ul[data-baseweb="menu"]{
        background:#1f2a3d!important;
        border:1px solid #334155!important;
        color:#f8fafc!important;
    }
    /* Every option row — all states */
    div[data-baseweb="menu"] li,
    div[data-baseweb="menu"] [role="option"],
    div[data-baseweb="menu"] [role="option"] *,
    div[data-baseweb="menu"] li *{
        background:#1f2a3d!important;
        color:#f8fafc!important;
    }
    /* Hover / focus / highlighted / selected option */
    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="menu"] li:focus,
    div[data-baseweb="menu"] [role="option"]:hover,
    div[data-baseweb="menu"] [role="option"]:focus,
    div[data-baseweb="menu"] [data-highlighted="true"],
    div[data-baseweb="menu"] [data-highlighted="true"] *,
    div[data-baseweb="menu"] [aria-selected="true"],
    div[data-baseweb="menu"] [aria-selected="true"] *{
        background:#2b3d5a!important;
        color:#ffffff!important;
    }
    /* Selected tag pills — baseweb + testid variants */
    div[data-baseweb="tag"],
    div[data-baseweb="tag"] *,
    span[data-baseweb="tag"],
    span[data-baseweb="tag"] *,
    [data-testid="stMultiSelectChip"],
    [data-testid="stMultiSelectChip"] *,
    [data-testid="stMultiSelectChip"] span,
    .stMultiSelect [data-baseweb="tag"],
    .stMultiSelect [data-baseweb="tag"] *{
        background:#22314a!important;
        color:#f8fafc!important;
    }
    div[data-baseweb="tag"],
    span[data-baseweb="tag"],
    [data-testid="stMultiSelectChip"]{
        border:1px solid rgba(214,167,79,.40)!important;
        border-radius:6px!important;
    }
    /* Tag × button */
    div[data-baseweb="tag"] [data-baseweb="icon"],
    div[data-baseweb="tag"] svg,
    [data-testid="stMultiSelectChip"] svg,
    [data-testid="stMultiSelectChip"] button{
        color:#9BAABF!important;
        fill:#9BAABF!important;
    }
    /* Search input inside multiselect */
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] input:focus{
        color:#f8fafc!important;
        background:transparent!important;
        caret-color:#f8fafc!important;
    }
    div[data-baseweb="select"] input::placeholder{
        color:#9BAABF!important;
    }
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
        if st.form_submit_button("Unlock Reports", use_container_width=True):
            if code.strip() in valid_codes:
                st.session_state["pp_authenticated"] = True
                st.rerun()
            st.error("Invalid access code.")
    c1,c2 = st.columns(2)
    if checkout_url:
        c1.link_button("Buy Access", checkout_url, use_container_width=True)
    if support_email:
        c2.markdown(f"<p style='color:#aaa;padding-top:8px'>{support_email}</p>", unsafe_allow_html=True)
    if valid_codes == {"DEMO-2026"}:
        st.caption("Demo mode — use code DEMO-2026")
    return False


# ── Data pipeline ─────────────────────────────────────────────────────────────

def data_dir() -> Path:
    configured = get_secret_value("data","scouting_data_dir","")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_DIR


def csv_files(folder: str) -> list[str]:
    # Sort descending so newest FTP-import date comes first — dedup keeps latest
    return [str(p) for p in sorted(Path(folder).glob("*.csv"), reverse=True)]


def _unique_csv_files(folder: str) -> list[str]:
    """Deduplicate scouting files by game name — keep first import only."""
    seen = set()
    unique = []
    for p in csv_files(folder):
        m = re.match(r"v3__\d{4}__\d{2}__\d{2}__CSV__(.+)", Path(p).name)
        if m:
            key = m.group(1)
            if key in seen:
                continue
            seen.add(key)
        unique.append(p)
    return unique


def _csv_folder_has_data(folder: str) -> bool:
    """True if the local CSV folder has any scouting files."""
    return bool(csv_files(folder))


@st.cache_data(ttl=30, show_spinner=False)
def data_source_signature(folder: str) -> tuple:
    """Small version stamp for the active data source.

    Streamlit caches by function arguments, not by files read inside a function.
    Including this signature in cached data loaders makes new FTP imports and
    rebuilt Parquet files visible without asking coaches to manually clear cache.
    """
    parts = _parquet_parts()
    if parts:
        part_sig = []
        for p in parts:
            try:
                stat = p.stat()
            except OSError:
                continue
            part_sig.append((p.name, stat.st_size, stat.st_mtime_ns))
        return ("parquet", tuple(part_sig))

    csvs = [Path(p) for p in csv_files(folder)]
    if csvs:
        latest_mtime = 0
        total_size = 0
        newest_names = []
        for p in csvs:
            try:
                stat = p.stat()
            except OSError:
                continue
            latest_mtime = max(latest_mtime, stat.st_mtime_ns)
            total_size += stat.st_size
            newest_names.append((stat.st_mtime_ns, p.name))
        newest_names = tuple(name for _, name in sorted(newest_names, reverse=True)[:8])
        return ("csv", len(csvs), latest_mtime, total_size, newest_names)

    return ("none", ())


def _parquet_parts() -> list[Path]:
    """Return existing Parquet part files."""
    return [p for p in (SCOUTING_PARQUET_1, SCOUTING_PARQUET_2) if p.exists()]


def _parquet_available() -> bool:
    return bool(_parquet_parts())


@st.cache_data(ttl=60, show_spinner=False)
def active_data_source(folder: str) -> str:
    """Choose the active national data source.

    Streamlit Cloud deploys with both an older tracked CSV folder and the newer
    split Parquet files. Parquet must win there or recent games disappear from
    the app even though the rebuilt Parquet has them.
    """
    if _parquet_available():
        return "parquet"
    if _csv_folder_has_data(folder):
        return "csv"
    return "none"


@st.cache_data(show_spinner="Loading team index...")
def _parquet_index_cols(source_sig: tuple) -> pd.DataFrame:
    """Read ONLY PitcherTeam+Pitcher+BatterTeam+Batter from the Parquet.
    Uses column projection so peak memory is ~40 MB instead of ~600 MB.
    Cached once per session."""
    parts = _parquet_parts()
    if not parts:
        return pd.DataFrame()
    import pyarrow.parquet as pq_io
    chunks = []
    for p in parts:
        try:
            tbl = pq_io.read_table(str(p),
                columns=["PitcherTeam","Pitcher","BatterTeam","Batter"])
            chunks.append(tbl.to_pandas())
        except Exception:
            pass
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    # Categorical dtypes slash memory 10×
    for col in out.columns:
        out[col] = out[col].astype("category")
    return out


@st.cache_data(show_spinner="Loading leaderboard data...", ttl=3600)
def _parquet_load_teams_bulk(team_codes: tuple, source_sig: tuple, role: str = "pitcher") -> pd.DataFrame:
    """Read only rows for a SET of teams in 2 passes (one per Parquet file).
    Much faster than calling _parquet_load_team() per team for leaderboards.
    role='pitcher' filters PitcherTeam; 'batter' filters BatterTeam."""
    parts = _parquet_parts()
    if not parts:
        return pd.DataFrame()
    import pyarrow.parquet as pq_io
    col = "PitcherTeam" if role == "pitcher" else "BatterTeam"
    team_set = set(team_codes)
    chunks = []
    for p in parts:
        try:
            filters = [[col, "=", tc] for tc in team_set]
            # pyarrow OR-of-equalities
            tbl = pq_io.read_table(str(p), filters=[[("PitcherTeam" if role=="pitcher" else "BatterTeam", "=", tc)] for tc in team_set])
            if len(tbl):
                chunks.append(tbl.to_pandas())
        except Exception:
            pass
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


@st.cache_data(show_spinner="Loading pitcher data...", ttl=3600)
def _parquet_load_team(team_code: str, source_sig: tuple) -> pd.DataFrame:  # noqa: F811
    """Load all rows where PitcherTeam OR BatterTeam == team_code.
    Uses pyarrow row-group filtering — only the matching rows reach Python.
    Never loads the full 2M-row dataset into memory."""
    parts = _parquet_parts()
    if not parts:
        return pd.DataFrame()
    import pyarrow.parquet as pq_io
    chunks = []
    for p in parts:
        try:
            # [[...],[...]] = OR between the two predicates
            tbl = pq_io.read_table(str(p), filters=[
                [("PitcherTeam", "=", team_code)],
                [("BatterTeam",  "=", team_code)],
            ])
            if len(tbl):
                chunks.append(tbl.to_pandas())
        except Exception:
            pass
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


_INDEX_CACHE     = DEFAULT_DATA_DIR / ".cbb_pitcher_index.pkl"
_HIT_INDEX_CACHE = DEFAULT_DATA_DIR / ".cbb_hitter_index.pkl"


def _load_disk_index(cache_path: Path, source_sig: tuple):
    """Load pitcher/hitter index from disk if CSV count matches — instant on restart."""
    import pickle
    try:
        if cache_path.exists():
            with open(cache_path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("source_sig") == source_sig:
                return saved["data"]
    except Exception:
        pass
    return None


def _save_disk_index(cache_path: Path, data, source_sig: tuple):
    import pickle
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump({"source_sig": source_sig, "data": data}, fh)
    except Exception:
        pass


@st.cache_data(show_spinner="Building pitcher index...")
def build_index(folder: str, source_sig: tuple) -> pd.DataFrame:
    """Returns (TeamCode, Pitcher, Pitches, Files) where Files is a list of paths."""
    # ── Parquet / cloud mode — column-projection only, no full load ──────────
    if active_data_source(folder) == "parquet":
        idx = _parquet_index_cols(source_sig)
        if idx.empty or "Pitcher" not in idx.columns or "PitcherTeam" not in idx.columns:
            return pd.DataFrame(columns=["TeamCode","Team","Pitcher","Pitches","Files"])
        grp = (idx[["PitcherTeam","Pitcher"]].dropna()
                 .assign(Pitcher    =lambda d: d["Pitcher"].astype(str).str.strip(),
                         PitcherTeam=lambda d: d["PitcherTeam"].astype(str).str.strip())
                 .groupby(["PitcherTeam","Pitcher"], as_index=False, observed=True)
                 .size().rename(columns={"PitcherTeam":"TeamCode","size":"Pitches"}))
        grp["Files"] = [[] for _ in range(len(grp))]
        grp["Team"]  = grp["TeamCode"].map(safe_team_name)
        return grp

    # ── CSV / local mode ─────────────────────────────────────────────────────
    all_csvs = _unique_csv_files(folder)

    cached = _load_disk_index(_INDEX_CACHE, source_sig)
    if cached is not None:
        return cached

    usecols = ["Pitcher", "PitcherTeam"]
    rows = []
    for path in all_csvs:
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
    _save_disk_index(_INDEX_CACHE, idx, source_sig)
    return idx


def _fallback_clean(df: pd.DataFrame) -> pd.DataFrame:
    PITCH_MAP = {
        # Four-seam fastball variants
        "Fastball":"FB","FourSeamFastBall":"FB","FourSeamFastball":"FB",
        "4-Seam":"FB","Four-Seam":"FB","FastBall":"FB","FF":"FB","FA":"FB",
        # Two-seam / sinker (TwoSeamFastBall was producing "TW")
        "TwoSeamFastBall":"SI","TwoSeamFastball":"SI","OneSeamFastBall":"SI",
        "Sinker":"SI","SNK":"SI","FT":"SI",
        # Breaking / offspeed
        "Cutter":"FC","Slider":"SL","Sweeper":"SW",
        "Curveball":"CU","CurveBall":"CU",
        "ChangeUp":"CH","Changeup":"CH","ChangeupFB":"CH",
        "Splitter":"SP","Split-Finger":"SP","SplitFinger":"SP",
    }
    out = df.copy()
    rename = {"RelSpeed":"Velo","InducedVertBreak":"IVB","HorzBreak":"HB",
              "SpinRate":"Spin","RelHeight":"RelH","RelSide":"RelS",
              "Extension":"Ext","ExitSpeed":"EV","Angle":"LA"}
    out = out.rename(columns={k:v for k,v in rename.items() if k in out.columns})
    if "TaggedPitchType" in out.columns:
        out["PitchRaw"] = out["TaggedPitchType"].astype(str)
        out["Pitch"] = out["TaggedPitchType"].map(PITCH_MAP).fillna(
            out["TaggedPitchType"].astype(str).str[:2].str.upper())
    else:
        out["PitchRaw"] = ""
        out["Pitch"] = "UNK"
    call = out.get("PitchCall", pd.Series("", index=out.index)).astype(str)
    out["is_whiff"]  = call.isin(["StrikeSwinging"])
    out["is_swing"]  = call.isin(["StrikeSwinging","FoulBall","FoulBallNotFieldable","InPlay","InPlayNoOut","InPlayOut"])
    out["is_csw"]    = call.isin(["StrikeCalled","StrikeSwinging"])
    out["is_strike"] = call.isin(["StrikeCalled","StrikeSwinging","FoulBall","FoulBallNotFieldable","InPlay","InPlayNoOut","InPlayOut"])
    for col in ["Velo","IVB","HB","Spin","RelH","RelS","Ext","PlateLocHeight","PlateLocSide"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if {"PlateLocSide","PlateLocHeight"}.issubset(out.columns):
        out["in_zone"] = out["PlateLocSide"].between(-0.83,0.83) & out["PlateLocHeight"].between(1.5,3.5)
    else:
        out["in_zone"] = False
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date.astype(str)
    return out


_UNKNOWN_PITCH_CODES = {"", "NA", "NAN", "NONE", "NULL", "UN", "UNK", "OT", "OTHER"}
_PITCH_INFER_FEATURES = ["Velo", "IVB", "HB", "Spin"]


def _infer_unknown_pitch_types(df: pd.DataFrame) -> pd.DataFrame:
    """Assign OT/UN/UNK pitches to the closest pitch in that pitcher's arsenal.

    TrackMan occasionally leaves a pitch as Other/Undefined even when its
    velo/movement/spin profile clearly matches one of the pitcher's established
    pitch groups. This pass only relabels close matches and keeps ambiguous
    pitches as-is.
    """
    if df.empty or "Pitch" not in df.columns:
        return df

    out = df.copy()
    out["Pitch"] = out["Pitch"].fillna("UNK").astype(str).str.upper().str.strip()
    if "PitchInferred" not in out.columns:
        out["PitchInferred"] = False

    features = [c for c in _PITCH_INFER_FEATURES if c in out.columns]
    if len(features) < 2:
        return out
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    group_cols = [c for c in ["PitcherTeam", "Pitcher"] if c in out.columns]
    if not group_cols:
        group_cols = ["Pitcher"] if "Pitcher" in out.columns else []
    groups = out.groupby(group_cols, dropna=False) if group_cols else [(None, out)]

    for _, g in groups:
        idx = g.index
        pitch = out.loc[idx, "Pitch"].astype(str).str.upper().str.strip()
        unknown_idx = idx[pitch.isin(_UNKNOWN_PITCH_CODES)]
        if len(unknown_idx) == 0:
            continue

        known = out.loc[idx[~pitch.isin(_UNKNOWN_PITCH_CODES)]].copy()
        if known.empty:
            continue
        counts = known["Pitch"].value_counts()
        keep = counts[counts >= 4].index
        known = known[known["Pitch"].isin(keep)]
        if known.empty:
            continue

        usable_features = [
            c for c in features
            if known[c].notna().sum() >= max(4, min(12, len(known) // 3))
        ]
        if len(usable_features) < 2:
            continue

        centers = known.groupby("Pitch")[usable_features].mean().dropna(how="all")
        if centers.empty:
            continue

        scale = known[usable_features].std(ddof=0).replace(0, np.nan)
        fallback_scale = pd.Series({"Velo": 2.5, "IVB": 5.0, "HB": 5.0, "Spin": 260.0})
        scale = scale.fillna(fallback_scale).fillna(1.0)

        for ridx in unknown_idx:
            row = out.loc[ridx, usable_features]
            present = row.notna()
            if present.sum() < 2:
                continue
            best_pitch, best_dist = None, np.inf
            for ptype, center in centers.iterrows():
                dims = present & center.notna()
                if dims.sum() < 2:
                    continue
                diff = ((row[dims] - center[dims]) / scale[dims]).astype(float)
                dist = float(np.sqrt(np.mean(np.square(diff))))
                if dist < best_dist:
                    best_pitch, best_dist = ptype, dist
            if best_pitch and best_dist <= 1.35:
                out.at[ridx, "Pitch"] = str(best_pitch)
                out.at[ridx, "PitchInferred"] = True

    return out


def _add_perceived_velo(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the Fordham app's add_perceived_velocity() exactly."""
    if "Velo" not in df.columns or "Ext" not in df.columns:
        df["PerceivedVelo"] = np.nan
        return df
    velo = pd.to_numeric(df["Velo"], errors="coerce")
    ext  = pd.to_numeric(df["Ext"],  errors="coerce")
    ivb  = pd.to_numeric(df.get("IVB",  np.nan), errors="coerce")
    spin = pd.to_numeric(df.get("Spin", np.nan), errors="coerce")

    baseline_dist = _PV_DIST - _PV_EXT_BASE                            # 54.5
    actual_dist   = (_PV_DIST - ext).clip(lower=50.0, upper=58.5)
    ext_adjusted  = velo * (baseline_dist / actual_dist)

    shape_adj = (
        (ivb  - _PV_IVB_BASE ).fillna(0) * _PV_IVB_W
        + ((spin - _PV_SPIN_BASE).fillna(0) / 100) * _PV_SPIN_W
    ).clip(lower=-_PV_SHAPE_CAP, upper=_PV_SHAPE_CAP)

    perceived = ext_adjusted + shape_adj
    # Only meaningful for fastball-family pitches
    if "Pitch" in df.columns:
        fb_mask = df["Pitch"].astype(str).isin(["FB","FF","FA","SI","FT"])
        perceived = perceived.where(fb_mask)
    df["PerceivedVelo"] = perceived
    return df


def clean_pitch_data(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: basic cleaning + pitch type mapping (no external deps)
    out = _fallback_clean(df.copy())
    out = _infer_unknown_pitch_types(out)

    # Step 2: run LightGBM Stuff+ and Loc+ models directly from repo models/
    sm, sl, lm, ll = _get_models()
    if sm is not None:
        try:
            rename = {"RelSpeed":"Velo","InducedVertBreak":"IVB","HorzBreak":"HB",
                      "SpinRate":"Spin","RelHeight":"RelH","RelSide":"RelS",
                      "Extension":"Ext","VertApprAngle":"VAA","HorzApprAngle":"HAA"}
            out = out.rename(columns={k:v for k,v in rename.items()
                                      if k in out.columns and v not in out.columns})
            for col in _STUFF_FEATURES + ["PlateLocSide","PlateLocHeight"]:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
            out = _compute_stuffplus(out, sm, sl)
            out = _compute_locationplus(out, lm, ll)
        except Exception:
            pass

    # Step 3: perceived velocity (Fordham formula)
    out = _add_perceived_velo(out)
    return out


@st.cache_data(show_spinner="Loading pitcher data...")
def load_pitcher_data(folder: str, team_code: str, pitcher: str,
                      file_list: tuple, source_sig: tuple,
                      clean_version: str = PITCH_CLEAN_VERSION) -> pd.DataFrame:
    """Load data for a specific pitcher — from Parquet (cloud) or CSVs (local)."""
    _ = clean_version  # included in Streamlit's cache key
    # ── Parquet / cloud mode ─────────────────────────────────────────────────
    if active_data_source(folder) == "parquet":
        team_df = _parquet_load_team(team_code, source_sig)
        if team_df.empty:
            return pd.DataFrame()
        mask = (team_df.get("Pitcher", pd.Series("", index=team_df.index))
                       .astype(str).str.strip() == str(pitcher))
        sub = team_df[mask].copy()
        return clean_pitch_data(sub) if not sub.empty else pd.DataFrame()

    # ── CSV / local mode ─────────────────────────────────────────────────────
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
    outs  = _pitcher_outs(df)
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
        "RelH":   df["RelH"].mean() if "RelH" in df.columns else np.nan,
        "Ext":    df["Ext"].mean()  if "Ext"  in df.columns else np.nan,
        "RelExt": df["Ext"].mean()  if "Ext"  in df.columns else np.nan,
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
    for col in ["RelH","RelS","Ext"]:
        if col in df.columns:
            agg[col] = df.groupby("Pitch")[col].mean().reindex(agg["Pitch"]).values
    return agg.sort_values("N", ascending=False).reset_index(drop=True)


# ── Graphics (mirroring Fordham postgame_or_season_card layout) ───────────────

def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TXT2, which="both", labelsize=10, length=3)
    for sp in ax.spines.values():
        sp.set_color("#2B3442")
        sp.set_linewidth(1.0)
    ax.grid(color=GRID, alpha=0.28, linewidth=0.7)


def _panel_title(ax, title: str, subtitle: str | None = None):
    ax.text(0.0, 1.035, title, transform=ax.transAxes, color=TXT,
            fontsize=16, fontweight="bold", ha="left", va="bottom")
    if subtitle:
        ax.text(1.0, 1.035, subtitle, transform=ax.transAxes, color=TXT2,
                fontsize=10.5, fontweight="bold", ha="right", va="bottom")


def _metric_tile(ax, x: float, y: float, w: float, h: float,
                 label: str, value: str, primary: str | None = None):
    color = primary or PANEL2
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="#2B3442",
        linewidth=0.9,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.60, value, transform=ax.transAxes,
            color=TXT, ha="center", va="center", fontsize=16,
            fontweight="bold", zorder=3)
    ax.text(x + w/2, y + h*0.20, label, transform=ax.transAxes,
            color=TXT2, ha="center", va="center", fontsize=9,
            fontweight="bold", zorder=3)

def _draw_zone(ax):
    ax.set_facecolor(PANEL)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color("#2B3442")
    ax.plot([-0.83,0.83,0.83,-0.83,-0.83],[1.5,1.5,3.5,3.5,1.5], color="#F8FAFC", linewidth=2.2)
    ax.fill_between([-0.83,0.83],1.5,3.5, color="white", alpha=0.055)
    # home plate
    ax.plot([-0.83,0.83,0.83,0,-0.83,-0.83],[0,0,0.17,0.34,0.17,0], color="#F8FAFC", linewidth=1.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def _draw_pitch_usage_panel(ax, df: pd.DataFrame):
    _style_ax(ax)
    _panel_title(ax, "Pitch Usage", "by batter side")
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.6, 3.05)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Overall", "vs LHH", "vs RHH"], color=TXT, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Usage %", color=TXT2, fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if df.empty or "Pitch" not in df.columns:
        ax.text(0.5, 0.5, "No pitch mix", transform=ax.transAxes,
                color=TXT2, ha="center", va="center", fontsize=12, fontweight="bold")
        return

    pitch_order = df["Pitch"].value_counts().index.tolist()

    def _subset(label: str) -> pd.DataFrame:
        if label == "Overall" or "BatterSide" not in df.columns:
            return df
        want = "Left" if label == "vs LHH" else "Right"
        return df[df["BatterSide"].astype(str).eq(want)]

    rows = [("Overall", 2), ("vs LHH", 1), ("vs RHH", 0)]
    for label, y in rows:
        sub = _subset(label)
        if sub.empty:
            ax.text(2, y, "No data", color=TXT2, fontsize=10, va="center", fontweight="bold")
            continue
        counts = sub["Pitch"].value_counts().reindex(pitch_order).fillna(0)
        total = float(counts.sum())
        left = 0.0
        for pitch, count in counts.items():
            if count <= 0 or total <= 0:
                continue
            width = float(count) / total * 100
            ax.barh(y, width, left=left, height=0.48, color=pc(pitch),
                    edgecolor=PANEL, linewidth=1.0)
            if width >= 16:
                ax.text(left + width / 2, y, f"{pitch}\n{width:.0f}%",
                        color="white", ha="center", va="center",
                        fontsize=8.5, fontweight="bold")
            elif width >= 10:
                ax.text(left + width / 2, y, pitch,
                        color="white", ha="center", va="center",
                        fontsize=8, fontweight="bold")
            left += width
        ax.text(106.0, y, f"{int(total)}", color=TXT2, fontsize=9,
                va="center", ha="left", fontweight="bold")

    ax.text(106.0, 2.53, "N", color=TXT2, fontsize=9,
            va="center", ha="left", fontweight="bold")

@st.cache_data(ttl=3600, show_spinner=False)
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
    txt_on  = readable_text_color(primary)
    card    = pitcher_stats(game_df)
    arsen   = arsenal_table(game_df)

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(BG)

    # ── Header bar ────────────────────────────────────────────────────────────
    hdr = fig.add_axes([0, 0.905, 1, 0.095])
    hdr.set_facecolor(primary); hdr.axis("off")
    hdr.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hdr.transAxes,
                                facecolor=primary, edgecolor="none", zorder=0))
    hdr.add_patch(plt.Rectangle((0, 0), 1, 0.08, transform=hdr.transAxes,
                                facecolor=accent, edgecolor="none", alpha=0.95, zorder=1))

    logo = logo_path_for_team(team_code)
    has_logo = _place_logo(fig, logo, primary, accent, (0.925, 0.925, 0.052, 0.055), opacity=1.0)

    # Truncate very long names so they don't run into the stat columns
    pitcher_display = pitcher if len(pitcher) <= 24 else pitcher[:23] + "…"
    hdr.text(0.018, 0.72, pitcher_display, color=txt_on, fontsize=27, fontweight="bold",
             transform=hdr.transAxes, va="center")
    conf = TEAM_CONFERENCES.get(team_code, "")
    _throws = game_df["PitcherThrows"].dropna().astype(str).mode()
    _hand   = ("LHP" if _throws.iloc[0].upper().startswith("L") else "RHP") if not _throws.empty else ""
    subtitle = f"{safe_team_name(team_code)}"
    if conf:
        subtitle += f"  ·  {conf}"
    if _hand:
        subtitle += f"  ·  {_hand}"
    subtitle += f"  ·  {label}"
    if date_str and date_str != label:
        subtitle += f"  ·  {date_str}"
    hdr.text(0.018, 0.18, subtitle, color=accent, fontsize=11.5, fontweight="bold",
             transform=hdr.transAxes, va="center")

    stat_keys = ["Pitches","IP","K","BB","FB Velo","FB PercVelo","RelH","RelExt","MaxVelo","Stuff+","Loc+","K%","Whiff%","Zone%","CSW%"]
    n_s = len(stat_keys)
    x_start = 0.300
    x_end = 0.610 if has_logo else 0.665
    for i, key in enumerate(stat_keys):
        x = x_start + i * (x_end / n_s) + (x_end / n_s) / 2
        hdr.text(x, 0.70, fmt(card.get(key), key), color=txt_on,
                 fontsize=13.5, fontweight="bold", ha="center", va="center",
                 transform=hdr.transAxes)
        hdr.text(x, 0.33, key, color=accent, fontsize=8.7, fontweight="bold",
                 ha="center", va="center", transform=hdr.transAxes)

    # ── Grid: movement, LHH/RHH locations, usage, arsenal table, footer ───────
    ax_move = plt.subplot2grid((6,4), (0,0), rowspan=3, fig=fig)
    ax_lhh  = plt.subplot2grid((6,4), (0,1), rowspan=3, fig=fig)
    ax_rhh  = plt.subplot2grid((6,4), (0,2), rowspan=3, fig=fig)
    ax_usage = plt.subplot2grid((6,4), (0,3), rowspan=3, fig=fig)
    ax_tbl  = plt.subplot2grid((6,4), (3,0), colspan=4, rowspan=2, fig=fig)
    ax_foot = plt.subplot2grid((6,4), (5,0), colspan=4, fig=fig)

    fig.subplots_adjust(top=0.875, bottom=0.03, left=0.045, right=0.975,
                        hspace=0.54, wspace=0.34)

    # Movement
    _style_ax(ax_move)
    throws = game_df["PitcherThrows"].iloc[0] if "PitcherThrows" in game_df.columns else "Right"
    arm_x  = (0,25)  if throws.upper().startswith("R") else (-25,0)
    glv_x  = (-25,0) if throws.upper().startswith("R") else (0,25)
    ax_move.axvspan(*arm_x, facecolor=(0.10,0.30,0.60,0.10))
    ax_move.axvspan(*glv_x, facecolor=(0.60,0.10,0.10,0.10))
    ax_move.axhline(0, color="#E5E7EB", linestyle=":", linewidth=1.1, alpha=0.75)
    ax_move.axvline(0, color="#E5E7EB", linestyle=":", linewidth=1.1, alpha=0.75)
    ax_move.set_xlim(-25,25); ax_move.set_ylim(-25,25)
    ax_move.set_aspect("equal", adjustable="box")
    for _, row in game_df.iterrows():
        ax_move.scatter(row.get("HB"), row.get("IVB"), s=42,
                        color=pc(row["Pitch"]), edgecolor="#F8FAFC", linewidth=0.35, alpha=0.72)
    for pt, g in game_df.groupby("Pitch"):
        cx, cy = g["HB"].mean(), g["IVB"].mean()
        ax_move.scatter(cx, cy, s=300, color=pc(pt), edgecolor="#F8FAFC", linewidth=1.4, zorder=5)
        ax_move.text(cx, cy, pt, color="white", fontsize=12, weight="bold",
                     ha="center", va="center")
    _panel_title(ax_move, "Pitch Movement", "avg dots labeled")
    ax_move.set_xlabel("Horizontal Break", color=TXT2, fontsize=11, fontweight="bold")
    ax_move.set_ylabel("Induced Vert Break", color=TXT2, fontsize=11, fontweight="bold")

    # vs LHH
    _draw_zone(ax_lhh)
    if "BatterSide" in game_df.columns:
        lhh = game_df[game_df["BatterSide"].eq("Left")]
    else:
        lhh = pd.DataFrame()
    for _, row in lhh.iterrows():
        ax_lhh.scatter(row.get("PlateLocSide"), row.get("PlateLocHeight"),
                       s=72, color=pc(row["Pitch"]), edgecolor="#F8FAFC", linewidth=0.35, alpha=0.78)
    _panel_title(ax_lhh, "vs LHH", "locations")

    # vs RHH
    _draw_zone(ax_rhh)
    rhh = game_df[game_df["BatterSide"].eq("Right")] if "BatterSide" in game_df.columns else game_df
    for _, row in rhh.iterrows():
        ax_rhh.scatter(row.get("PlateLocSide"), row.get("PlateLocHeight"),
                       s=72, color=pc(row["Pitch"]), edgecolor="#F8FAFC", linewidth=0.35, alpha=0.78)
    _panel_title(ax_rhh, "vs RHH", "locations")

    _draw_pitch_usage_panel(ax_usage, game_df)

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
        tbl.set_fontsize(13)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#2B3442")
            cell.set_linewidth(0.7)
            if r == 0:
                cell.set_facecolor(primary)
                cell.set_text_props(color=txt_on, weight="bold", size=12)
            else:
                pt = view.iloc[r-1]["Pitch"] if r-1 < len(view) else ""
                cell.set_facecolor(PANEL2 if r % 2 else PANEL)
                cell.set_text_props(color=TXT, weight="bold", size=12)
                if c == 0:
                    cell.set_facecolor(pc(pt))
                    cell.set_text_props(color="white", weight="bold", size=13)

    # Footer
    ax_foot.axis("off")
    ax_foot.text(0.5, 0.40, "CBBReports  ·  College Baseball Pitching Plus  ·  2026 TrackMan",
                 transform=ax_foot.transAxes, ha="center", color=TXT2, fontsize=10)

    out = BytesIO()
    fig.savefig(out, format="png", dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


@st.cache_data(ttl=3600, show_spinner=False)
def build_stat_card_png(df: pd.DataFrame, pitcher: str, team_code: str) -> bytes:
    primary, accent = get_team_colors(team_code)
    txt_on = readable_text_color(primary)
    card  = pitcher_stats(df)
    arsen = arsenal_table(df)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.axis("off")

    # Header band
    ax.add_patch(mpatches.FancyBboxPatch((0.015,0.765),0.97,0.22, transform=ax.transAxes,
                                         boxstyle="round,pad=0.012,rounding_size=0.030",
                                         facecolor=primary, edgecolor="none", zorder=2))
    ax.add_patch(plt.Rectangle((0.027,0.775),0.946,0.012, transform=ax.transAxes,
                               color=accent, alpha=0.95, zorder=3))

    logo = logo_path_for_team(team_code)
    _place_logo(fig, logo, primary, accent, (0.890, 0.820, 0.075, 0.115), opacity=1.0)

    # Pitcher name + team — stay left, clear of logo
    ax.text(0.045, 0.90, pitcher, transform=ax.transAxes,
            color=txt_on, fontsize=24, fontweight="bold", va="center", zorder=3)
    ax.text(0.045, 0.825, f"{safe_team_name(team_code)}  ·  Player Stat Card  ·  2026",
            transform=ax.transAxes, color=accent, fontsize=11, fontweight="bold",
            va="center", zorder=3)

    # 18 stat tiles in 3 rows of 6 — sized to stay clear of the header logo.
    stat_keys = ["Pitches","Games","IP","K","BB","K%","BB%","BAA",
                 "SLG","Velo","MaxVelo","RelH","RelExt","Stuff+","Loc+","Whiff%","Zone%","CSW%"]
    tw, th = 0.130, 0.112
    for i, key in enumerate(stat_keys):
        ci, ri = i % 6, i // 6
        x = 0.025 + ci*(tw+0.012)
        y = 0.610 - ri*0.145
        _metric_tile(ax, x, y, tw, th, key, fmt(card.get(key), key))

    if not arsen.empty and "Usage%" in arsen.columns:
        bx, by, bh = 0.025, 0.06, 0.065
        tw_total = 0.97 - bx
        x_cur = bx
        for _, row in arsen.iterrows():
            sw = (row["Usage%"]/100) * tw_total
            if sw < 0.005: continue
            ax.add_patch(mpatches.FancyBboxPatch((x_cur,by),sw,bh, transform=ax.transAxes,
                                       boxstyle="round,pad=0.002,rounding_size=0.010",
                                       facecolor=pc(row["Pitch"]), edgecolor=BG,
                                       linewidth=0.7, zorder=2))
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


# ── Leaderboard ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building leaderboard…")
def build_leaderboard(folder: str, team_codes: tuple, source_sig: tuple, min_pitches: int = 25) -> pd.DataFrame:
    """Load all files for the given teams at once, run models once, aggregate by pitcher."""
    team_set = set(team_codes)

    # ── Parquet / cloud mode — 2 reads total for entire team set ────────────
    if active_data_source(folder) == "parquet":
        bulk = _parquet_load_teams_bulk(tuple(sorted(team_set)), source_sig, role="pitcher")
        if bulk.empty:
            return pd.DataFrame()
        pt = bulk.get("PitcherTeam", pd.Series("", index=bulk.index)).astype(str).str.strip()
        all_df = clean_pitch_data(bulk[pt.isin(team_set)].copy())
    else:
        # ── CSV / local mode ─────────────────────────────────────────────────
        idx = build_index(folder, source_sig)
        pool = idx[idx["TeamCode"].isin(team_set)]
        if pool.empty:
            return pd.DataFrame()

        all_files: set = set()
        for files in pool["Files"]:
            all_files.update(files)

        chunks = []
        for path in sorted(all_files):
            try:
                df = pd.read_csv(path, low_memory=False)
                if not {"Pitcher","PitcherTeam"}.issubset(df.columns):
                    continue
                mask = df["PitcherTeam"].astype(str).str.strip().isin(team_set)
                if mask.any():
                    chunks.append(df[mask].copy())
            except Exception:
                continue

        if not chunks:
            return pd.DataFrame()
        all_df = clean_pitch_data(pd.concat(chunks, ignore_index=True))

    rows = []
    group_cols = ["PitcherTeam","Pitcher"] if "PitcherTeam" in all_df.columns else ["Pitcher"]
    for keys, pdf in all_df.groupby(group_cols):
        if len(pdf) < min_pitches:
            continue
        team_code = keys[0] if isinstance(keys, tuple) else ""
        pitcher   = keys[1] if isinstance(keys, tuple) else keys
        stats = pitcher_stats(pdf)
        stats["Pitcher"]    = pitcher
        stats["Team"]       = safe_team_name(team_code)
        stats["TeamCode"]   = team_code
        stats["Conference"] = TEAM_CONFERENCES.get(team_code, "")
        rows.append(stats)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def _color_plus(val):
    """Background color for Stuff+/Loc+ cells."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 115:   return "background:#14532d;color:#fff"
    if v >= 105:   return "background:#166534;color:#fff"
    if v >= 95:    return "background:#854d0e;color:#fff"
    if v >= 85:    return "background:#7f1d1d;color:#fff"
    return                 "background:#450a0a;color:#fff"


def leaderboard_page(folder: str, all_known: pd.DataFrame, source_sig: tuple):
    st.caption("Ranked by Stuff+ or Loc+ across any scope — full D1, conference, or single team.")

    d1 = all_known[all_known["Division"] == "D1"]

    # ── Scope filters ─────────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns([1, 1.2, 1.2, 0.8])
    with sc1:
        scope = st.radio("Scope", ["All D1", "Conference", "Team"])
    with sc2:
        if scope in ("Conference", "Team"):
            confs = sorted(d1["Conference"].replace("","—").dropna().unique())
            sel_conf = st.selectbox("Conference", confs, key="lb_conf")
        else:
            sel_conf = None
    with sc3:
        if scope == "Team":
            conf_teams = d1[d1["Conference"] == sel_conf][["TeamCode","Team"]].drop_duplicates().sort_values("Team")
            sel_team = st.selectbox("Team", conf_teams["TeamCode"].tolist(),
                                    format_func=safe_team_name, key="lb_team")
        else:
            sel_team = None
    with sc4:
        min_p  = st.number_input("Min pitches", min_value=5, max_value=200, value=25, step=5)
        sort_by = st.selectbox("Sort by", ["Stuff+","Loc+","Whiff%","CSW%","K%","FB Velo","Velo"])

    # Build team code pool
    if scope == "All D1":
        if st.button("Load Full D1 Leaderboard", use_container_width=True):
            st.session_state["lb_d1_confirmed"] = True
        if not st.session_state.get("lb_d1_confirmed"):
            st.info("The full D1 leaderboard loads all pitchers in the database. Click the button above to proceed.")
            return
        team_codes = tuple(sorted(d1["TeamCode"].unique()))
    elif scope == "Conference":
        team_codes = tuple(sorted(d1[d1["Conference"] == sel_conf]["TeamCode"].unique()))
    else:
        team_codes = (sel_team,) if sel_team else ()

    if not team_codes:
        st.warning("No teams found for this selection.")
        return

    lb = build_leaderboard(folder, team_codes, source_sig, min_pitches=int(min_p))
    if lb.empty:
        st.warning("No pitchers meet the minimum pitch threshold.")
        return

    # Sort and rank
    if sort_by in lb.columns:
        lb = lb.sort_values(sort_by, ascending=False)
    lb = lb.reset_index(drop=True)
    lb.index = lb.index + 1  # 1-based rank

    show_cols = ["#","Pitcher","Team"]
    if scope in ("All D1","Conference"):
        show_cols.append("Conference")
    for c in ["Pitches","Games","Stuff+","Loc+","FB Velo","FB PercVelo",
              "MaxVelo","K%","Whiff%","Zone%","CSW%","BAA"]:
        if c in lb.columns:
            show_cols.append(c)

    view = lb.copy()
    _medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    view["#"] = [_medals.get(i, str(i)) for i in range(1, len(view) + 1)]
    view = view[show_cols]
    for col in show_cols:
        if col not in ("#","Pitcher","Team","Conference"):
            view[col] = view[col].apply(lambda v: fmt(v, col))

    # Style Stuff+ and Loc+ columns
    def style_lb(row):
        styles = [""] * len(row)
        for col_name in ("Stuff+","Loc+"):
            if col_name in view.columns:
                idx_c = list(view.columns).index(col_name)
                styles[idx_c] = _color_plus(row[col_name])
        return styles

    st.dataframe(
        view.style.apply(style_lb, axis=1),
        use_container_width=True,
        height=min(600, 38 + len(view) * 35),
    )
    st.caption(f"{len(view)} pitcher(s) · minimum {min_p} pitches · sorted by {sort_by}")


def hr_leaderboard_section(folder: str, all_known: pd.DataFrame, source_sig: tuple):
    st.caption("Longest tracked home runs — select scope below.")
    d1 = all_known[all_known["Division"] == "D1"]

    sc1, sc2, sc3 = st.columns([1, 1.2, 1.2])
    with sc1:
        scope = st.radio("Scope", ["Conference","Team"], key="hr_scope")
    with sc2:
        confs = sorted(d1["Conference"].replace("","—").dropna().unique())
        sel_conf = st.selectbox("Conference", confs, key="hr_conf")
    with sc3:
        conf_teams = d1[d1["Conference"] == sel_conf][["TeamCode","Team"]].drop_duplicates().sort_values("Team")
        if scope == "Team":
            sel_team = st.selectbox("Team", conf_teams["TeamCode"].tolist(),
                                    format_func=safe_team_name, key="hr_team")
            team_codes = (sel_team,)
        else:
            team_codes = tuple(sorted(conf_teams["TeamCode"].unique()))

    if not team_codes:
        st.warning("No teams for this selection.")
        return

    with st.spinner("Loading home run data…"):
        hr_df = _hr_leaderboard_national(folder, team_codes, source_sig)

    if hr_df.empty:
        st.info("No home runs found for this selection.")
        return

    hr_df.index = hr_df.index + 1
    mcols = st.columns(min(4, len(hr_df)))
    for i, col in enumerate(mcols):
        r = hr_df.iloc[i]
        col.metric(f"#{i+1}  {r['Batter'].split(',')[0].strip()}",
                   f"{r['Dist (ft)']:.0f} ft",
                   f"EV {r['EV (mph)']:.0f}  ·  LA {r['LA (°)']:.0f}°"
                   if pd.notna(r.get("EV (mph)")) else "")
    st.markdown("---")
    st.dataframe(hr_df, use_container_width=True)


# ── Hitter data pipeline ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building hitter index...")
def build_hitter_index(folder: str, source_sig: tuple) -> pd.DataFrame:
    # ── Parquet / cloud mode — column-projection only ────────────────────────
    if active_data_source(folder) == "parquet":
        idx = _parquet_index_cols(source_sig)
        if idx.empty or "Batter" not in idx.columns or "BatterTeam" not in idx.columns:
            return pd.DataFrame(columns=["TeamCode","Team","Batter","PA","Files"])
        grp = (idx[["BatterTeam","Batter"]].dropna()
                 .assign(Batter    =lambda d: d["Batter"].astype(str).str.strip(),
                         BatterTeam=lambda d: d["BatterTeam"].astype(str).str.strip())
                 .groupby(["BatterTeam","Batter"], as_index=False, observed=True)
                 .size().rename(columns={"BatterTeam":"TeamCode","size":"PA"}))
        grp["Files"] = [[] for _ in range(len(grp))]
        grp["Team"]  = grp["TeamCode"].map(safe_team_name)
        return grp

    # ── CSV / local mode ─────────────────────────────────────────────────────
    all_csvs = _unique_csv_files(folder)

    cached = _load_disk_index(_HIT_INDEX_CACHE, source_sig)
    if cached is not None:
        return cached

    usecols = ["Batter", "BatterTeam"]
    rows = []
    for path in all_csvs:
        try:
            df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype=str, low_memory=False)
        except Exception:
            continue
        if "Batter" not in df.columns or "BatterTeam" not in df.columns:
            continue
        df = df.dropna(subset=["Batter", "BatterTeam"])
        df["Batter"]     = df["Batter"].str.strip()
        df["BatterTeam"] = df["BatterTeam"].str.strip()
        df = df[df["Batter"].ne("") & df["BatterTeam"].ne("")]
        for (team, batter), g in df.groupby(["BatterTeam", "Batter"]):
            rows.append({"TeamCode": team, "Batter": batter, "PA": len(g), "File": path})
    if not rows:
        return pd.DataFrame(columns=["TeamCode", "Team", "Batter", "PA", "Files"])
    raw = pd.DataFrame(rows)
    idx = raw.groupby(["TeamCode", "Batter"], as_index=False).agg(
        PA=("PA", "sum"), Files=("File", list))
    idx["Team"] = idx["TeamCode"].map(safe_team_name)
    _save_disk_index(_HIT_INDEX_CACHE, idx, source_sig)
    return idx


@st.cache_data(show_spinner="Loading hitter data...")
def load_hitter_data(folder: str, team_code: str, batter: str,
                     file_list: tuple, source_sig: tuple) -> pd.DataFrame:
    # ── Parquet / cloud mode ─────────────────────────────────────────────────
    if active_data_source(folder) == "parquet":
        team_df = _parquet_load_team(team_code, source_sig)
        if team_df.empty:
            return pd.DataFrame()
        mask = (team_df.get("Batter", pd.Series("", index=team_df.index))
                       .astype(str).str.strip() == str(batter))
        chunks = [team_df[mask].copy()]
    else:
        chunks = []
        for path in file_list:
            try:
                df = pd.read_csv(path, low_memory=False)
            except Exception:
                continue
            if not {"Batter", "BatterTeam"}.issubset(df.columns):
                continue
            mask = (df["BatterTeam"].astype(str).str.strip() == str(team_code)) & \
                   (df["Batter"].astype(str).str.strip() == str(batter))
            if mask.any():
                chunks.append(df[mask].copy())
    if not chunks or (len(chunks) == 1 and chunks[0].empty):
        return pd.DataFrame()
    all_df = pd.concat(chunks, ignore_index=True)
    rename = {"RelSpeed": "Velo", "InducedVertBreak": "IVB", "HorzBreak": "HB",
              "SpinRate": "Spin", "ExitSpeed": "EV", "Angle": "LA",
              "Distance": "Dist", "TaggedPitchType": "Pitch_raw"}
    all_df = all_df.rename(columns={k: v for k, v in rename.items() if k in all_df.columns})
    for col in ["EV", "LA", "Dist", "PlateLocSide", "PlateLocHeight", "Direction"]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")
    if "Pitch_raw" in all_df.columns and "Pitch" not in all_df.columns:
        PITCH_MAP_LOCAL = {
            "Fastball": "FB", "FourSeamFastBall": "FB", "Sinker": "SI",
            "Cutter": "FC", "Slider": "SL", "Sweeper": "SW",
            "Curveball": "CU", "CurveBall": "CU", "ChangeUp": "CH", "Changeup": "CH",
        }
        all_df["Pitch"] = all_df["Pitch_raw"].map(PITCH_MAP_LOCAL).fillna(
            all_df["Pitch_raw"].astype(str).str[:2].str.upper())
    if "Date" in all_df.columns:
        all_df["Date"] = pd.to_datetime(all_df["Date"], errors="coerce").dt.date.astype(str)
    return all_df


def hitter_stats_cbb(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    pr    = df.get("PlayResult", pd.Series("", index=df.index)).fillna("").astype(str)
    kbb   = df.get("KorBB",      pd.Series("", index=df.index)).fillna("").astype(str)
    pc_   = df.get("PitchCall",  pd.Series("", index=df.index)).fillna("").astype(str)
    pa_mask = kbb.isin(["Walk","Strikeout"]) | pr.isin(
        ["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice","Sacrifice"])
    pa = df[pa_mask]
    singles = pr.eq("Single").sum(); doubles = pr.eq("Double").sum()
    triples = pr.eq("Triple").sum(); homers  = pr.eq("HomeRun").sum()
    H  = singles + doubles + triples + homers
    TB = singles + 2*doubles + 3*triples + 4*homers
    walks = kbb.eq("Walk").sum(); ks = kbb.eq("Strikeout").sum()
    hbp   = pc_.eq("HitByPitch").sum()
    sf    = pr.eq("Sacrifice").sum()
    ab    = max(len(pa) - walks - hbp - sf, 0)
    obd   = ab + walks + hbp + sf
    swings = pc_.isin(["StrikeSwinging","FoulBall","FoulBallNotFieldable",
                        "InPlay","InPlayNoOut","InPlayOut"]).sum()
    whiffs = pc_.eq("StrikeSwinging").sum()
    in_z   = (_numeric_series(df, "PlateLocSide", 0).between(-0.83, 0.83) &
              _numeric_series(df, "PlateLocHeight", 0).between(1.5, 3.5))
    chases = (pc_.isin(["StrikeSwinging","FoulBall","FoulBallNotFieldable",
                         "InPlay","InPlayNoOut","InPlayOut"]) & ~in_z).sum()
    ev_s = pd.to_numeric(df.get("EV", pd.Series(dtype=float)), errors="coerce")
    bip_ev = ev_s[pr.isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice"]) & (ev_s > 45)]

    # wOBA — D1 collegiate linear weights, SF included in denominator
    woba_num = (WOBA_BB*walks + WOBA_HBP*hbp +
                WOBA_1B*singles + WOBA_2B*doubles + WOBA_3B*triples + WOBA_HR*homers)
    woba_den = ab + walks + hbp          # AB + BB + HBP; excludes SF (tagger inconsistency)
    woba = round(woba_num / woba_den, 3) if woba_den else np.nan

    # wRC+ = (wOBA / lgwOBA) × 100
    wrc_plus = round((woba / LG_WOBA) * 100) if not pd.isna(woba) and LG_WOBA > 0 else np.nan

    return {
        "PA": len(pa), "AB": ab, "H": H, "HR": int(homers),
        "xHB": int(doubles+triples+homers), "BB": int(walks), "K": int(ks),
        "BA":  round(H/ab, 3) if ab else np.nan,
        "OBP": round((H+walks+hbp)/obd, 3) if obd else np.nan,
        "SLG": round(TB/ab, 3) if ab else np.nan,
        "OPS": round((H+walks+hbp)/obd + TB/ab, 3) if obd and ab else np.nan,
        "wOBA":    woba,
        "wRC+":    wrc_plus,
        "K%":  ks/len(pa)*100    if len(pa) else np.nan,
        "BB%": walks/len(pa)*100 if len(pa) else np.nan,
        "Avg EV":  round(bip_ev.mean(), 1) if len(bip_ev) else np.nan,
        "HH%":     (bip_ev >= 95).mean()*100 if len(bip_ev) else np.nan,
        "Whiff%":  whiffs/swings*100 if swings else np.nan,
        "Chase%":  chases/swings*100 if swings else np.nan,
        "Games":   df.get("GameID", df.get("Date", pd.Series(dtype=str))).nunique(),
        "BIP":     int(len(bip_ev)),
    }


def _draw_spray(ax, df, color_by_ev: bool = True, team_code: str | None = None):  # noqa: C901
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Field geometry ────────────────────────────────────────────────────────
    # Wall profile (LF 330 / CF 400 / RF 310)
    t_full = np.linspace(np.radians(-48), np.radians(48), 300)
    d_wall = np.interp(t_full, [np.radians(-48), 0, np.radians(48)], [330, 400, 310])

    # Layer 1 — full fair-territory grass (outfield + infield)
    xs = np.concatenate([[0], d_wall * np.sin(t_full), [0]])
    ys = np.concatenate([[0], d_wall * np.cos(t_full), [0]])
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True,
                         facecolor="#1a3d1a", edgecolor="none", zorder=1))

    # Layer 2 — warning track (sandy strip inside wall)
    t_w = np.linspace(np.radians(-50), np.radians(50), 300)
    d_wo = np.interp(t_w, [np.radians(-50), 0, np.radians(50)], [330, 400, 310])
    d_wi = np.interp(t_w, [np.radians(-50), 0, np.radians(50)], [308, 378, 288])
    ax.add_patch(Polygon(
        list(zip(np.concatenate([d_wo*np.sin(t_w), (d_wi*np.sin(t_w))[::-1]]),
                 np.concatenate([d_wo*np.cos(t_w), (d_wi*np.cos(t_w))[::-1]]))),
        closed=True, facecolor="#5c4820", edgecolor="none", zorder=2))

    # Layer 3 — infield DIRT (fan polygon from home to ~97 ft covering base paths)
    t_id = np.linspace(np.radians(-46), np.radians(46), 200)
    ax.add_patch(Polygon(
        list(zip(np.concatenate([[0], 97*np.sin(t_id), [0]]),
                 np.concatenate([[0], 97*np.cos(t_id), [0]]))),
        closed=True, facecolor="#7a5a20", edgecolor="none", zorder=3))

    # Layer 4 — infield GRASS (the classic diamond between the bases)
    # Vertices approximate the real grass-dirt boundary at each base
    ig = [
        (0,    -2),   # behind home plate
        (-22,   7),   # 3B-line side of HP
        (-65,  62),   # third base
        (-28, 120),   # between 3B and 2B
        (0,   128),   # second base (tip toward CF)
        (28,  120),   # between 2B and 1B
        (65,   62),   # first base
        (22,    7),   # 1B-line side of HP
    ]
    ax.add_patch(Polygon(ig, closed=True, facecolor="#1e4a1a", edgecolor="none", zorder=4))

    # Team logo watermark in center field. Keep it quiet enough that spray dots
    # and direction reads stay dominant.
    logo_arr = _team_logo_array(logo_path_for_team(team_code or ""), opacity=0.22)
    if logo_arr is not None:
        h, w = logo_arr.shape[:2]
        aspect = w / max(h, 1)
        max_w, max_h = 132, 104
        logo_w = min(max_w, max_h * aspect)
        logo_h = logo_w / max(aspect, 0.01)
        cx, cy = 0, 300
        ax.imshow(
            logo_arr,
            extent=[cx - logo_w / 2, cx + logo_w / 2, cy - logo_h / 2, cy + logo_h / 2],
            zorder=4.6,
            interpolation="lanczos",
        )

    # Layer 5 — pitcher's mound dirt + rubber
    ax.add_patch(mpatches.Circle((0, 60.5), 9,
                                 facecolor="#9a7030", edgecolor="#7a5820", lw=0.8, zorder=5))
    ax.add_patch(mpatches.Rectangle((-3, 59.8), 6, 1.4,
                                    facecolor="#ddddcc", edgecolor="none", zorder=6))  # rubber

    # Layer 5 — home plate dirt circle
    ax.add_patch(mpatches.Circle((0, 0), 22, facecolor="#7a5a20", edgecolor="none", zorder=5))

    # Foul lines
    for ang in [-45, 45]:
        r = np.radians(ang)
        ax.plot([0, 415*np.sin(r)], [0, 415*np.cos(r)],
                color="#ddddc0", lw=1.8, alpha=0.55, zorder=7)

    # Outfield wall
    ax.plot(d_wall*np.sin(t_full), d_wall*np.cos(t_full),
            color="#ddddc0", lw=2.5, alpha=0.70, zorder=7)

    # Distance rings
    for ring in [200, 300, 400]:
        tr = np.linspace(np.radians(-50), np.radians(50), 120)
        ax.plot(ring*np.sin(tr), ring*np.cos(tr),
                color="#ffffff", lw=0.6, ls="--", alpha=0.15, zorder=7)
        ax.text(ring*np.sin(np.radians(46))+6, ring*np.cos(np.radians(46)),
                f"{ring}'", color=TXT2, fontsize=11, ha="left", va="center", zorder=8)

    # Bases (90-ft diamond rotated 45°)
    for bx, by in [(63.64, 63.64), (0, 127.28), (-63.64, 63.64)]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx-5, by-5), 10, 10, boxstyle="square,pad=0",
            facecolor="#eeeeee", edgecolor="#aaaaaa", lw=0.8, zorder=8))
    # Home plate
    hp = [(0,0),(8.5,5),(8.5,-1),(0,-9),(-8.5,-1),(-8.5,5)]
    ax.add_patch(Polygon(hp, closed=True,
                         facecolor="#eeeeee", edgecolor="#aaaaaa", lw=0.8, zorder=8))

    # Sector labels
    for lbl, deg in [("LF",-34),("LC",-18),("CF",0),("RC",18),("RF",34)]:
        ax.text(255*np.sin(np.radians(deg)), 255*np.cos(np.radians(deg)),
                lbl, color="#D6DDE8", fontsize=13, ha="center", va="center",
                fontweight="bold", alpha=0.80, zorder=7)

    # Pull / Oppo labels (handedness-aware)
    _bs_sp  = df.get("BatterSide", pd.Series(dtype=str)).dropna().astype(str)
    _lhh_sp = (not _bs_sp.empty) and _bs_sp.mode().iloc[0].upper().startswith("L")
    _pull_deg, _oppo_deg = (-40, 40) if not _lhh_sp else (40, -40)
    ax.text(175*np.sin(np.radians(_pull_deg)), 175*np.cos(np.radians(_pull_deg)),
            "PULL", color="#F04444", fontsize=12, ha="center", va="center",
            fontweight="bold", alpha=0.85, zorder=7)
    ax.text(175*np.sin(np.radians(_oppo_deg)), 175*np.cos(np.radians(_oppo_deg)),
            "OPPO", color="#5599dd", fontsize=12, ha="center", va="center",
            fontweight="bold", alpha=0.85, zorder=7)

    # ── BIP scatter dots — coloured by EV when available ─────────────────────
    try:
        import matplotlib.colors as mcolors
        pr  = df.get("PlayResult", pd.Series("", index=df.index)).fillna("").astype(str)
        ev_raw = pd.to_numeric(df.get("EV", pd.Series(dtype=float)), errors="coerce")
        bip_rows = df[pr.isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice"])]

        if "Dist" in bip_rows.columns and "Direction" in bip_rows.columns:
            dist   = pd.to_numeric(bip_rows["Dist"],      errors="coerce")
            ang_   = pd.to_numeric(bip_rows["Direction"], errors="coerce")
            bip_pr = pr[bip_rows.index]
            bip_ev = ev_raw[bip_rows.index]
            valid  = dist.notna() & ang_.notna() & (dist > 20)

            # EV colormap: Baseball Savant blue→white→red
            ev_cmap = mcolors.LinearSegmentedColormap.from_list("ev", [
                (0.00, "#0a2e6e"), (0.35, "#5ea3d0"), (0.50, "#787878"),
                (0.70, "#f5a17a"), (1.00, "#8b0000")
            ])
            EV_LO, EV_HI = 50.0, 105.0

            sizes = {"HomeRun": 160, "Triple": 100, "Double": 80,
                     "Single": 55, "Out": 30, "Error": 30, "FieldersChoice": 30}

            # Draws outs first (underneath), then hits, HRs on top
            for result in ["Out","Error","FieldersChoice","Single","Double","Triple","HomeRun"]:
                mask = valid & (bip_pr == result)
                if not mask.any():
                    continue
                gd = dist[mask]; ga = ang_[mask]
                gev = bip_ev[mask]
                x_c = gd * np.sin(np.radians(ga))
                y_c = gd * np.cos(np.radians(ga))
                s   = sizes.get(result, 35)
                is_hr = result == "HomeRun"
                is_out = result in ("Out","Error","FieldersChoice")

                if color_by_ev and gev.notna().any():
                    norm_ev = gev.clip(EV_LO, EV_HI)
                    colors_arr = ev_cmap((norm_ev - EV_LO) / (EV_HI - EV_LO))
                    colors_arr[gev.isna()] = [0.5, 0.5, 0.5, 0.6]
                    ec = "white" if is_hr else ("none" if is_out else "white")
                    ax.scatter(x_c, y_c, s=s, c=colors_arr,
                               edgecolors=ec, linewidth=0.8,
                               alpha=0.95 if is_hr else (0.5 if is_out else 0.88),
                               zorder=10 if is_hr else (8 if is_out else 9),
                               marker="*" if is_hr else "o")
                else:
                    flat_clr = {"HomeRun":"#ff3333","Triple":"#ffc000",
                                "Double":"#44aaff","Single":"#44dd55"}.get(result,"#888888")
                    ax.scatter(x_c, y_c, s=s, color=flat_clr,
                               edgecolors="white" if is_hr else "none",
                               linewidth=0.8, alpha=0.88,
                               zorder=10 if is_hr else 9,
                               marker="*" if is_hr else "o")

            # BIP summary text
            n_hr  = (bip_pr == "HomeRun").sum()
            n_xbh = bip_pr.isin(["Double","Triple","HomeRun"]).sum()
            n_h   = bip_pr.isin(["Single","Double","Triple","HomeRun"]).sum()
            n_out = bip_pr.isin(["Out","Error","FieldersChoice"]).sum()
            n_bip = n_h + n_out
            avg_ev_bip = bip_ev[(bip_ev > 45) & valid].mean()
            ev_label = f"  ·  Avg EV {avg_ev_bip:.1f}" if not np.isnan(avg_ev_bip) else ""
            ax.text(-375, -28,
                    f"BIP: {n_bip}  |  H: {n_h}  |  HR: {n_hr}  |  XBH: {n_xbh}{ev_label}",
                    color=TXT2, fontsize=11.5, ha="left", va="bottom", zorder=11)

    except Exception:
        pass

    ax.set_xlim(-400, 400)
    ax.set_ylim(-40, 450)
    _panel_title(ax, "Spray Chart", "dots colored by exit velocity")

    # EV colorbar beneath the spray chart
    try:
        import matplotlib.colors as mcolors2
        ev_cmap2 = mcolors2.LinearSegmentedColormap.from_list("ev2", [
            (0.00, "#0a2e6e"), (0.35, "#5ea3d0"), (0.50, "#787878"),
            (0.70, "#f5a17a"), (1.00, "#8b0000")
        ])
        cax = ax.inset_axes([0.04, 0.018, 0.68, 0.028])
        cb  = plt.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(50, 105), cmap=ev_cmap2),
            cax=cax, orientation="horizontal")
        cb.set_ticks([50, 70, 87, 95, 105])
        cb.ax.set_xticklabels(["50", "70", "87 (avg)", "95 (HH)", "105+"],
                               color=TXT2, fontsize=10.5)
        cb.outline.set_edgecolor("#333333")
        cb.ax.tick_params(colors=TXT2, size=2)
        ax.text(0.76, 0.028, "* = HR", color=TXT2, fontsize=11,
                transform=ax.transAxes, va="bottom", fontweight="bold")
    except Exception:
        pass


def _draw_hitter_zone(ax, df):
    """3×3 strike-zone heatmap: Avg EV colored via savant percentile scale."""
    ax.set_facecolor(PANEL)
    ax.axis("off")

    zx = np.linspace(-0.83, 0.83, 4)
    zy = np.linspace(1.5, 3.5, 4)
    grid   = np.full((3, 3), np.nan)
    grid_n = np.zeros((3, 3), dtype=int)

    try:
        ev = pd.to_numeric(df.get("EV", pd.Series(dtype=float)), errors="coerce")
        ps = pd.to_numeric(df.get("PlateLocSide",   pd.Series(dtype=float)), errors="coerce")
        ph = pd.to_numeric(df.get("PlateLocHeight", pd.Series(dtype=float)), errors="coerce")
        pr = df.get("PlayResult", pd.Series("", index=df.index)).fillna("").astype(str)
        bip_mask = (pr.isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice"])
                    & (ev > 45))
        for ri in range(3):
            for ci in range(3):
                mask = bip_mask & ps.between(zx[ci], zx[ci+1]) & ph.between(zy[ri], zy[ri+1])
                n = int(mask.sum())
                grid_n[ri, ci] = n
                if n >= 3:
                    grid[ri, ci] = float(ev[mask].mean())
    except Exception:
        pass

    # Draw cells — red=high EV (good for hitter), blue=low EV
    for ri in range(3):
        for ci in range(3):
            x0, x1 = zx[ci], zx[ci+1]
            y0, y1 = zy[ri], zy[ri+1]
            val = grid[ri, ci]
            n   = grid_n[ri, ci]
            cx, cy = (x0+x1)/2, (y0+y1)/2
            if not np.isnan(val):
                bg_c, txt_c = _hitter_color("Avg EV", val)
                ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                             facecolor=bg_c, edgecolor=PANEL, lw=2.5, zorder=2))
                ax.text(cx, cy + 0.08, f"{val:.0f}",
                        fontsize=18, fontweight="bold",
                        ha="center", va="center", color=txt_c, zorder=3)
                ax.text(cx, cy - 0.20, f"n={n}",
                        fontsize=10, ha="center", va="center",
                        color=txt_c, alpha=0.65, zorder=3)
            else:
                ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                             facecolor=PANEL2, edgecolor=PANEL, lw=2.5, zorder=2))
                ax.text(cx, cy, f"n={n}" if n > 0 else "—",
                        fontsize=10.5, ha="center", va="center",
                        color="#7D87A6", zorder=3)

    # Strike zone border + grid lines
    ax.plot([-0.83, 0.83, 0.83, -0.83, -0.83],
            [1.5,  1.5,  3.5,  3.5,  1.5],
            color=TXT, lw=2.4, zorder=4)
    for xg in zx[1:-1]:
        ax.plot([xg, xg], [1.5, 3.5], color="white", lw=0.7, alpha=0.35, zorder=4)
    for yg in zy[1:-1]:
        ax.plot([-0.83, 0.83], [yg, yg], color="white", lw=0.7, alpha=0.35, zorder=4)

    # Zone position labels — flip Inside/Outside for LHH
    _bs_z   = df.get("BatterSide", pd.Series(dtype=str)).dropna().astype(str)
    _lhh_z  = (not _bs_z.empty) and _bs_z.mode().iloc[0].upper().startswith("L")
    _side_lbls = ["Outside", "Middle", "Inside"] if _lhh_z else ["Inside", "Middle", "Outside"]
    for ci, lbl in enumerate(_side_lbls):
        ax.text((zx[ci]+zx[ci+1])/2, 3.75, lbl,
                color=TXT2, fontsize=11.5, ha="center", va="bottom", alpha=0.9)
    for ri, lbl in enumerate(["Low", "Mid", "High"]):
        ax.text(-1.08, (zy[ri]+zy[ri+1])/2, lbl,
                color=TXT2, fontsize=11.5, ha="right", va="center", alpha=0.9)

    # Colour scale legend (blue=low → red=high, hitter perspective)
    try:
        import matplotlib.colors as mcolors
        cmap_h = mcolors.LinearSegmentedColormap.from_list(
            "hitter", [(p, tuple(c/255 for c in rgb)) for p, rgb in _HITTER_STOPS], N=256)
        cax = ax.inset_axes([0.05, 0.02, 0.90, 0.048])
        norm = plt.Normalize(0, 1)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_h),
                          cax=cax, orientation="horizontal")
        cb.set_ticks([0, 0.5, 1])
        cb.ax.set_xticklabels(["Poor EV", "Avg", "Hard Hit"],
                               color=TXT2, fontsize=10.5)
        cb.outline.set_edgecolor("#2B3442")
        cb.ax.tick_params(colors=TXT2, size=2)
    except Exception:
        pass

    ax.set_xlim(-1.65, 1.5)
    ax.set_ylim(0.8, 4.3)
    _panel_title(ax, "Avg Exit Velocity by Zone", "3x3 strike zone")


def _draw_pitch_breakdown(ax, df, primary, txt_on):  # noqa: C901
    ax.set_facecolor(PANEL)
    ax.axis("off")

    # Use _hitter_color: red=elite for hitter, blue=poor

    try:
        pitch_col = "Pitch" if "Pitch" in df.columns else None
        if pitch_col is None or df.empty:
            ax.text(0.5, 0.5, "No pitch data", color=TXT2,
                    ha="center", va="center", transform=ax.transAxes)
            return
        rows = []
        for pitch, g in df.groupby(pitch_col):
            if str(pitch) in ("UN","TW","OT","UNK","na","nan","") or len(g) < 5:
                continue
            gpr  = g.get("PlayResult", pd.Series("", index=g.index)).fillna("").astype(str)
            gkbb = g.get("KorBB",      pd.Series("", index=g.index)).fillna("").astype(str)
            gpc  = g.get("PitchCall",  pd.Series("", index=g.index)).fillna("").astype(str)
            gev  = pd.to_numeric(g.get("EV", pd.Series(dtype=float)), errors="coerce")
            pa_m = (gkbb.isin(["Walk","Strikeout"]) |
                    gpr.isin(["Single","Double","Triple","HomeRun",
                               "Out","Error","FieldersChoice","Sacrifice"]))
            pa_    = g[pa_m]
            hits   = gpr.isin(["Single","Double","Triple","HomeRun"]).sum()
            walks_ = gkbb.eq("Walk").sum()
            ab_    = max(len(pa_) - int(walks_), 0)
            sw_    = gpc.isin(["StrikeSwinging","FoulBall","FoulBallNotFieldable",
                                "InPlay","InPlayNoOut","InPlayOut"]).sum()
            wh_    = gpc.eq("StrikeSwinging").sum()
            bev    = gev[gpr.isin(["Single","Double","Triple","HomeRun",
                                    "Out","Error","FieldersChoice"]) & (gev > 45)]
            rows.append({
                "Pitch":  str(pitch)[:3],
                "N":      len(g),
                "BA":     round(float(hits)/ab_, 3)           if ab_  else np.nan,
                "Whiff%": round(float(wh_)/float(sw_)*100, 1) if sw_  else np.nan,
                "Avg EV": round(float(bev.mean()), 1)         if len(bev) else np.nan,
            })
        if not rows:
            ax.text(0.5, 0.5, "Insufficient data", color=TXT2,
                    ha="center", va="center", transform=ax.transAxes)
            return

        tbl_df = pd.DataFrame(rows).sort_values("N", ascending=False).head(8)

        def _f(v, col):
            if pd.isna(v): return "—"
            if col == "Pitch": return str(v)
            if col == "BA": return f"{float(v):.3f}".replace("0.", ".")
            if col == "Whiff%": return f"{float(v):.1f}%"
            if col == "N":  return str(int(v))
            return f"{float(v):.1f}"

        view = tbl_df.copy()
        for col in view.columns:
            view[col] = view[col].apply(lambda v, c=col: _f(v, c))

        tbl = ax.table(cellText=view.values, colLabels=view.columns,
                       loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)

        col_names = list(view.columns)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#2B3442")
            cell.set_linewidth(0.7)
            if r == 0:
                cell.set_facecolor(primary)
                cell.set_text_props(color=txt_on, weight="bold", size=11.5)
            else:
                row_idx  = r - 1
                col_name = col_names[c] if c < len(col_names) else ""
                raw_val  = tbl_df.iloc[row_idx][col_name] if row_idx < len(tbl_df) else np.nan
                if col_name == "Pitch":
                    pt = str(raw_val)[:2].upper()
                    cell.set_facecolor(pc(pt))
                    cell.set_text_props(color="white", weight="bold", size=13)
                elif col_name == "N":
                    cell.set_facecolor(PANEL2)
                    cell.set_text_props(color=TXT2, weight="normal", size=12)
                elif col_name in ("BA", "Whiff%", "Avg EV"):
                    fc, tc = _hitter_color(col_name, raw_val)
                    cell.set_facecolor(fc); cell.set_text_props(color=tc, weight="bold", size=12)
                else:
                    cell.set_facecolor(PANEL if r % 2 else PANEL2)
                    cell.set_text_props(color=TXT2, weight="normal", size=12)
    except Exception:
        ax.text(0.5, 0.5, "Chart unavailable", color=TXT2,
                ha="center", va="center", transform=ax.transAxes)
    _panel_title(ax, "Hitter Performance vs Pitch Type", "BA, whiff and exit velo")


def _draw_pct_card(fig_title: str, subtitle: str, rows_data: list,
                   pct_fn, stops: list, team_colors: tuple,
                   logo: "Path | None" = None) -> bytes:
    """Generic horizontal-bar percentile card used by both pitcher and hitter cards."""
    primary, accent = team_colors
    BG = "#13151c"; BAR_BG = "#1c1f2a"
    n = len(rows_data)
    fig_h = 2.6 + n * 0.56
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG); ax.axis("off")

    HDR=0.96; SEP=0.875; TOP=0.855; BOT=0.055
    row_h = (TOP - BOT) / n
    BX=0.18; BW=0.54

    # Header bar
    ax.add_patch(plt.Rectangle((0, HDR-0.085), 1, 0.10, facecolor=primary, zorder=0))
    txt_c = readable_text_color(primary)
    ax.text(0.015, HDR-0.015, fig_title, color=txt_c,
            fontsize=22, fontweight="bold", va="top")
    ax.text(0.015, HDR-0.068, subtitle, color=accent,
            fontsize=10.5, fontweight="bold", va="top")

    # Team logo — transparent on header, white bg only if low contrast
    _place_logo(ax, logo, primary, accent,
                (0.865, HDR-0.083, 0.10, 0.078), use_inset=True)

    ax.plot([0.04, 0.96], [SEP, SEP], color="#333344", lw=0.8)
    ax.text(0.735, SEP-0.008, "Value",  color=TXT2, fontsize=9, ha="left",  va="top", fontweight="bold")
    ax.text(0.965, SEP-0.008, "Pct",    color=TXT2, fontsize=9, ha="right", va="top", fontweight="bold")

    for i, (key, label, fmt_s, val) in enumerate(rows_data):
        cy    = TOP - (i + 0.5) * row_h
        pct   = pct_fn(key, val)
        color = _pct_to_hex_cbb(pct, stops)
        bh    = row_h * 0.54

        ax.text(BX-0.01, cy, label, color=TXT2, fontsize=11.5,
                fontweight="bold", ha="right", va="center")
        ax.add_patch(plt.Rectangle((BX, cy-bh/2), BW, bh, facecolor=BAR_BG, zorder=2))
        if pct is not None and pct > 0.005:
            ax.add_patch(plt.Rectangle((BX, cy-bh/2), BW*pct, bh, facecolor=color, zorder=3))
        ax.plot([BX+BW*0.5]*2, [cy-bh/2, cy+bh/2], color="#555566", lw=1.0, zorder=4)

        val_s = fmt_s.format(float(val)) if val is not None and not pd.isna(val) else "—"
        ax.text(BX+BW+0.015, cy, val_s, color=TXT, fontsize=11, ha="left", va="center", fontweight="bold")
        ax.text(0.975, cy, _pct_label_cbb(pct), color=color,
                fontsize=11.5, fontweight="bold", ha="right", va="center")

    ax.text(BX,         BOT-0.015, "Poor",           color="#4F9BFF", fontsize=9.5, ha="left",   va="top", fontweight="bold")
    ax.text(BX+BW*0.5, BOT-0.015, "50th pct (avg)",  color=TXT2, fontsize=9.5, ha="center", va="top", fontweight="bold")
    ax.text(BX+BW,     BOT-0.015, "Elite",           color="#FF5A5A", fontsize=9.5, ha="right",  va="top", fontweight="bold")

    out = BytesIO()
    fig.savefig(out, format="png", dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


@st.cache_data(ttl=3600, show_spinner=False)
def build_pitcher_pct_card_cbb(df: pd.DataFrame, pitcher: str, team_code: str) -> bytes:
    """Pitcher percentile card for CBB Plus — red=elite, blue=poor."""
    pc  = df.get("PitchCall","").fillna("").astype(str)
    pr  = df.get("PlayResult","").fillna("").astype(str)
    kbb = df.get("KorBB","").fillna("").astype(str)
    ht  = df.get("TaggedHitType","").fillna("").astype(str)
    ev  = pd.to_numeric(df.get("EV", df.get("ExitSpeed", pd.Series(dtype=float))), errors="coerce")

    swing = pc.isin(["StrikeSwinging","FoulBall","FoulBallNotFieldable","InPlay","InPlayNoOut","InPlayOut"])
    whiff = pc.eq("StrikeSwinging")
    zone  = (_numeric_series(df, "PlateLocSide", 0).between(-0.83,0.83) &
             _numeric_series(df, "PlateLocHeight", 0).between(1.5,3.5))
    csw   = pc.isin(["StrikeCalled","StrikeSwinging"])
    pa_m  = (kbb.isin(["Walk","Strikeout"]) |
             pr.isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice","Sacrifice"]))
    pa_n  = pa_m.sum()
    sw_n  = swing.sum()
    bip_types = ["GroundBall","FlyBall","LineDrive","PopUp","Popup"]
    bip_n = ht.isin(bip_types).sum()
    bip_ev = ev[pr.isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice"]) & (ev>45)]

    # FB velo
    fb_col = df.get("Velo", df.get("RelSpeed", pd.Series(dtype=float)))
    pa_abbr = df.get("pitch_abbr", df.get("Pitch", pd.Series("", index=df.index))).fillna("")
    fb_velo = pd.to_numeric(fb_col, errors="coerce")[pa_abbr.isin(["FB","SI"])]

    def _s(col): return float(df[col].mean()) if col in df.columns and df[col].notna().any() else None

    stats = {
        "Stuff+":   _s("Stuff+"),
        "Loc+":     _s("Loc+"),
        "Velo":     float(fb_velo.mean()) if fb_velo.notna().any() else None,
        "CSW%":     float(csw.mean()*100) if len(df) else None,
        "Zone%":    float(zone.mean()*100) if len(df) else None,
        "Whiff%P":  float(whiff.sum()/sw_n*100) if sw_n else None,
        "K%P":      float(kbb.eq("Strikeout").sum()/pa_n*100) if pa_n else None,
        "BB%P":     float(kbb.eq("Walk").sum()/pa_n*100) if pa_n else None,
        "GB%P":     float(ht.eq("GroundBall").sum()/bip_n*100) if bip_n >= 5 else None,
        "Avg EV A": float(bip_ev.mean()) if len(bip_ev) >= 5 else None,
    }

    ROWS = [
        ("Stuff+",   "Stuff+",    "{:.0f}"),
        ("Loc+",     "Loc+",      "{:.0f}"),
        ("Velo",     "FB Velo",   "{:.1f} mph"),
        ("Whiff%P",  "Whiff%",    "{:.1f}%"),
        ("CSW%",     "CSW%",      "{:.1f}%"),
        ("Zone%",    "Zone%",     "{:.1f}%"),
        ("K%P",      "K%",        "{:.1f}%"),
        ("BB%P",     "BB%",       "{:.1f}%"),
        ("GB%P",     "GB%",       "{:.1f}%"),
        ("Avg EV A", "Avg EV vs", "{:.1f} mph"),
    ]
    rows_data = [(k, lbl, fmt, stats.get(k)) for k, lbl, fmt in ROWS]

    conf = TEAM_CONFERENCES.get(team_code, "")
    _throws = df["PitcherThrows"].dropna().astype(str).mode() if "PitcherThrows" in df.columns else pd.Series()
    _hand   = ("LHP" if _throws.iloc[0].upper().startswith("L") else "RHP") if not _throws.empty else ""
    subtitle = (safe_team_name(team_code) + (f"  ·  {conf}" if conf else "") +
                (f"  ·  {_hand}" if _hand else "") +
                "  ·  D1 Percentile Rankings  ·  2026")
    return _draw_pct_card(pitcher, subtitle, rows_data,
                          _pitcher_pct_rank_cbb, _HITTER_STOPS,
                          get_team_colors(team_code),
                          logo=logo_path_for_team(team_code))


@st.cache_data(ttl=3600, show_spinner=False)
def build_hitter_pct_card_cbb(df: pd.DataFrame, batter: str, team_code: str) -> bytes:
    """Hitter percentile card for CBB Plus — red=elite, blue=poor."""
    card = hitter_stats_cbb(df)

    ROWS = [
        ("wRC+",   "wRC+",    "{:.0f}"),
        ("wOBA",   "wOBA",    "{:.3f}"),
        ("BA",     "BA",      "{:.3f}"),
        ("OBP",    "OBP",     "{:.3f}"),
        ("SLG",    "SLG",     "{:.3f}"),
        ("OPS",    "OPS",     "{:.3f}"),
        ("K%",     "K%",      "{:.1f}%"),
        ("BB%",    "BB%",     "{:.1f}%"),
        ("Whiff%", "Whiff%",  "{:.1f}%"),
        ("Chase%", "Chase%",  "{:.1f}%"),
        ("Avg EV", "Avg EV",  "{:.1f} mph"),
        ("HH%",    "HH%",     "{:.1f}%"),
    ]
    rows_data = [(k, lbl, fmt, card.get(k)) for k, lbl, fmt in ROWS]

    conf = TEAM_CONFERENCES.get(team_code, "")
    subtitle = (safe_team_name(team_code) + (f"  ·  {conf}" if conf else "") +
                "  ·  Hitter Percentile Rankings  ·  2026")

    def _h_rank(stat, val):
        pct = _pct_rank(stat, val)
        return pct

    return _draw_pct_card(batter, subtitle, rows_data,
                          _h_rank, _HITTER_STOPS,
                          get_team_colors(team_code),
                          logo=logo_path_for_team(team_code))


def _draw_batted_ball_profile(ax, df):
    """Stacked horizontal bars: GB%/LD%/FB%/PU  and  Pull%/Center%/Oppo%."""
    ax.set_facecolor(PANEL); ax.axis("off")
    try:
        ht  = df.get("TaggedHitType","").fillna("").astype(str)
        dir_ = pd.to_numeric(df.get("Direction", pd.Series(dtype=float)), errors="coerce")
        pr  = df.get("PlayResult","").fillna("").astype(str)
        bip_types = ["GroundBall","FlyBall","LineDrive","PopUp","Popup"]
        bip_mask  = ht.isin(bip_types)
        bip_n     = bip_mask.sum()

        # ── Row 1 — Hit type breakdown ────────────────────────────────────────
        if bip_n >= 5:
            gb  = ht.eq("GroundBall").sum() / bip_n
            ld  = ht.eq("LineDrive").sum()  / bip_n
            fb  = ht.eq("FlyBall").sum()    / bip_n
            pu  = ht.isin(["PopUp","Popup"]).sum() / bip_n

            segs1 = [("GB",  gb,  "#8B6914"), ("LD", ld, "#44dd55"),
                     ("FB",  fb,  "#44aaff"), ("PU", pu, "#aaaaaa")]
            y1 = 0.62; bh = 0.14
            x_ = 0.02
            for lbl, pct, col in segs1:
                w = pct * 0.96
                if w > 0.005:
                    ax.add_patch(mpatches.FancyBboxPatch((x_, y1), w, bh, facecolor=col,
                                               boxstyle="round,pad=0.002,rounding_size=0.012",
                                               edgecolor=PANEL, linewidth=0.8,
                                               zorder=2, transform=ax.transAxes))
                    if w > 0.06:
                        ax.text(x_ + w/2, y1 + bh/2,
                                f"{lbl}\n{pct*100:.0f}%",
                                color="white", fontsize=10.5, fontweight="bold",
                                ha="center", va="center", transform=ax.transAxes, zorder=3)
                x_ += w
            ax.text(0.5, y1 + bh + 0.04, "Batted Ball Type",
                    color=TXT2, fontsize=11.5, fontweight="bold",
                    ha="center", va="bottom", transform=ax.transAxes)

        # ── Row 2 — Pull/Center/Oppo ──────────────────────────────────────────
        bip_dir_mask = bip_mask & dir_.notna()
        bip_dir_n    = bip_dir_mask.sum()
        if bip_dir_n >= 5:
            bip_dir = dir_[bip_dir_mask]
            # Derive handedness — LHH: pull = RF (positive), oppo = LF (negative)
            _bs     = df.get("BatterSide", pd.Series(dtype=str)).dropna().astype(str)
            _is_lhh = (not _bs.empty) and _bs.mode().iloc[0].upper().startswith("L")
            if _is_lhh:
                pull = (bip_dir > 15).sum()  / bip_dir_n
                oppo = (bip_dir < -15).sum() / bip_dir_n
            else:
                pull = (bip_dir < -15).sum() / bip_dir_n
                oppo = (bip_dir > 15).sum()  / bip_dir_n
            ctr = (bip_dir.between(-15, 15)).sum() / bip_dir_n

            segs2 = [("Pull", pull, "#e05555"), ("Center", ctr, "#ccaa44"), ("Oppo", oppo, "#5599dd")]
            y2 = 0.36; x_ = 0.02
            for lbl, pct, col in segs2:
                w = pct * 0.96
                if w > 0.005:
                    ax.add_patch(mpatches.FancyBboxPatch((x_, y2), w, bh, facecolor=col,
                                               boxstyle="round,pad=0.002,rounding_size=0.012",
                                               edgecolor=PANEL, linewidth=0.8,
                                               zorder=2, transform=ax.transAxes))
                    if w > 0.06:
                        ax.text(x_ + w/2, y2 + bh/2,
                                f"{lbl}\n{pct*100:.0f}%",
                                color="white", fontsize=10.5, fontweight="bold",
                                ha="center", va="center", transform=ax.transAxes, zorder=3)
                x_ += w
            ax.text(0.5, y2 + bh + 0.04, "Direction",
                    color=TXT2, fontsize=11.5, fontweight="bold",
                    ha="center", va="bottom", transform=ax.transAxes)

        # ── Row 3 — Key rate summary ──────────────────────────────────────────
        ev  = pd.to_numeric(df.get("EV", pd.Series(dtype=float)), errors="coerce")
        bip_ev = ev[pr.isin(["Single","Double","Triple","HomeRun",
                              "Out","Error","FieldersChoice"]) & (ev > 45)]
        stats3 = []
        if bip_n >= 5:
            stats3.append(("GB%", f"{ht.eq('GroundBall').sum()/bip_n*100:.0f}%"))
            stats3.append(("LD%", f"{ht.eq('LineDrive').sum()/bip_n*100:.0f}%"))
            stats3.append(("FB%", f"{ht.eq('FlyBall').sum()/bip_n*100:.0f}%"))
        _gb_dir = ht.eq("GroundBall") & dir_.notna()
        if _gb_dir.sum() >= 5:
            _gbn = _gb_dir.sum()
            stats3.append(("GB Left",  f"{(dir_[_gb_dir] < 0).sum()/_gbn*100:.0f}%"))
            stats3.append(("GB Right", f"{(dir_[_gb_dir] > 0).sum()/_gbn*100:.0f}%"))
        if len(bip_ev) >= 5:
            stats3.append(("Avg EV", f"{bip_ev.mean():.1f}"))
            stats3.append(("HH%",   f"{(bip_ev>=95).mean()*100:.0f}%"))
        _n = max(len(stats3), 1)
        _step = 0.96 / _n
        for j, (lbl, val) in enumerate(stats3):
            _cx = 0.02 + (j + 0.5) * _step
            ax.text(_cx, 0.20, val,
                    color=TXT, fontsize=12, fontweight="bold",
                    ha="center", va="center", transform=ax.transAxes)
            ax.text(_cx, 0.07, lbl,
                    color=TXT2, fontsize=9.5,
                    ha="center", va="center", transform=ax.transAxes)
    except Exception:
        ax.text(0.5, 0.5, "Profile data\nunavailable", color=TXT2,
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    _panel_title(ax, "Batted Ball Profile", "contact shape")


@st.cache_data(ttl=3600, show_spinner=False)
def build_hitter_summary_png(df: pd.DataFrame, batter: str, team_code: str) -> bytes:  # noqa: C901
    primary, accent = get_team_colors(team_code)
    txt_on = readable_text_color(primary)
    card   = hitter_stats_cbb(df)

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = fig.add_axes([0, 0.905, 1, 0.095])
    hdr.set_facecolor(primary)
    hdr.axis("off")
    hdr.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hdr.transAxes,
                                facecolor=primary, edgecolor="none", zorder=0))
    hdr.add_patch(plt.Rectangle((0, 0), 1, 0.08, transform=hdr.transAxes,
                                facecolor=accent, edgecolor="none", alpha=0.95, zorder=1))

    logo     = logo_path_for_team(team_code)
    has_logo = _place_logo(fig, logo, primary, accent, (0.925, 0.925, 0.052, 0.055), opacity=1.0)

    # Player name + subtitle
    hdr.text(0.018, 0.72, batter, color=txt_on, fontsize=27, fontweight="bold",
             transform=hdr.transAxes, va="center")
    conf = TEAM_CONFERENCES.get(team_code, "")
    _sides = df["BatterSide"].dropna().astype(str) if "BatterSide" in df.columns else pd.Series(dtype=str)
    _hand  = ("LHH" if _sides.mode().iloc[0] == "Left" else "RHH") if not _sides.empty else ""
    sub  = safe_team_name(team_code) + (f"  ·  {conf}" if conf else "") + (f"  ·  {_hand}" if _hand else "") + "  ·  Hitter Report  ·  2026"
    hdr.text(0.018, 0.25, sub, color=_muted_text_on(primary), fontsize=12, fontweight="bold",
             transform=hdr.transAxes, va="center")

    # Header stat boxes — Baseball Savant-style percentile coloring
    stat_keys = ["PA","AB","H","HR","xHB","BA","OBP","SLG","wOBA","wRC+",
                 "K%","BB%","Avg EV","HH%","Whiff%"]
    x_end = 0.545 if has_logo else 0.655
    step  = x_end / len(stat_keys)
    try:
        for i, key in enumerate(stat_keys):
            x = 0.315 + i * step + step / 2
            v = card.get(key, np.nan)
            if key in {"BA","OBP","SLG","OPS","wOBA"}:
                disp = f"{float(v):.3f}".replace("0.", ".") if not pd.isna(v) else "—"
            elif key in {"PA","AB","H","HR","xHB","BB","K","BIP","Games","wRC+"}:
                disp = str(int(v)) if not pd.isna(v) else "—"
            else:
                disp = f"{float(v):.1f}" if not pd.isna(v) else "—"
            # Percentile stats get a savant-colored pill via text bbox;
            # count stats use plain txt_on
            if key in _HITTER_PCTS:
                bg_c, val_c = _hitter_color(key, v)
                hdr.text(x, 0.72, disp, color=val_c, fontsize=14, fontweight="bold",
                         ha="center", va="center", transform=hdr.transAxes,
                         bbox=dict(facecolor=bg_c, edgecolor="none",
                                   boxstyle="round,pad=0.18", alpha=0.90))
            else:
                hdr.text(x, 0.72, disp, color=txt_on, fontsize=14, fontweight="bold",
                         ha="center", va="center", transform=hdr.transAxes)
            hdr.text(x, 0.26, key, color=accent, fontsize=9.1, fontweight="bold",
                     ha="center", va="center", transform=hdr.transAxes)
    except Exception:
        pass  # never let header crash prevent panel drawing

    # ── Panels — 4-panel layout ───────────────────────────────────────────────
    # Left:  spray chart (full height below header)
    # Right top:    EV zone heatmap
    # Right mid:    batted ball profile (GB/LD/FB + Pull/Center/Oppo)
    # Right bottom: pitch breakdown table
    ax_spray  = fig.add_axes([0.018, 0.035, 0.47, 0.835])
    ax_zone   = fig.add_axes([0.52, 0.505, 0.455, 0.365])
    ax_prof   = fig.add_axes([0.52, 0.280, 0.455, 0.185])
    ax_tbl    = fig.add_axes([0.52, 0.045, 0.455, 0.195])

    try:
        _draw_spray(ax_spray, df, color_by_ev=True, team_code=team_code)
    except Exception:
        ax_spray.axis("off")
        ax_spray.text(0.5, 0.5, "Spray chart unavailable", color=TXT2,
                      ha="center", va="center", transform=ax_spray.transAxes)
    try:
        _draw_hitter_zone(ax_zone, df)
    except Exception:
        ax_zone.axis("off")
        ax_zone.text(0.5, 0.5, "Zone data unavailable", color=TXT2,
                     ha="center", va="center", transform=ax_zone.transAxes)
    try:
        _draw_batted_ball_profile(ax_prof, df)
    except Exception:
        ax_prof.axis("off")
        ax_prof.text(0.5, 0.5, "Profile unavailable", color=TXT2,
                     ha="center", va="center", transform=ax_prof.transAxes)
    try:
        _draw_pitch_breakdown(ax_tbl, df, primary, txt_on)
    except Exception:
        ax_tbl.axis("off")
        ax_tbl.text(0.5, 0.5, "Breakdown unavailable", color=TXT2,
                    ha="center", va="center", transform=ax_tbl.transAxes)

    out = BytesIO()
    fig.savefig(out, format="png", dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


# ── HR Distance leaderboard (national) ───────────────────────────────────────

def _hr_leaderboard_national(folder: str, team_codes: tuple, source_sig: tuple) -> pd.DataFrame:
    team_set = set(team_codes)

    # ── Parquet / cloud mode ─────────────────────────────────────────────────
    if active_data_source(folder) == "parquet":
        bulk = _parquet_load_teams_bulk(tuple(sorted(team_set)), source_sig, role="batter")
        if bulk.empty:
            return pd.DataFrame()
        bt2 = bulk.get("BatterTeam", pd.Series("", index=bulk.index)).astype(str).str.strip()
        pr2 = bulk.get("PlayResult",  pd.Series("", index=bulk.index)).astype(str)
        raw = bulk[bt2.isin(team_set) & pr2.eq("HomeRun")].copy()
        if raw.empty:
            return pd.DataFrame()
        for col, alias in [("Distance","Distance"),("ExitSpeed","ExitSpeed"),("Angle","Angle")]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        if "Distance" in raw.columns:
            raw = raw[raw["Distance"].gt(200)]
        rows = []
        for _, row in raw.iterrows():
            rows.append({
                "Batter":    str(row.get("Batter","—")),
                "Team":      safe_team_name(str(row.get("BatterTeam",""))),
                "Date":      str(row.get("Date","—")),
                "Pitcher":   str(row.get("Pitcher","—")),
                "Dist (ft)": round(float(row["Distance"]),1) if pd.notna(row.get("Distance")) else np.nan,
                "EV (mph)":  round(float(row["ExitSpeed"]),1) if pd.notna(row.get("ExitSpeed")) else np.nan,
                "LA (°)":    round(float(row["Angle"]),1)     if pd.notna(row.get("Angle"))     else np.nan,
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Dist (ft)", ascending=False).reset_index(drop=True)

    # ── CSV / local mode ─────────────────────────────────────────────────────
    rows = []
    for path in _unique_csv_files(folder):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not {"PlayResult","BatterTeam"}.issubset(df.columns):
            continue
        if not df["BatterTeam"].astype(str).str.strip().isin(team_set).any():
            continue
        hr = df[(df["PlayResult"].astype(str).eq("HomeRun")) &
                (df["BatterTeam"].astype(str).str.strip().isin(team_set))]
        if hr.empty:
            continue
        for col in ["Distance","ExitSpeed","Angle"]:
            if col in hr.columns:
                hr = hr.copy()
                hr[col] = pd.to_numeric(hr[col], errors="coerce")
        hr = hr[hr.get("Distance", pd.Series(0,index=hr.index)).gt(200)]
        for _, row in hr.iterrows():
            rows.append({
                "Batter":    str(row.get("Batter","—")),
                "Team":      safe_team_name(str(row.get("BatterTeam",""))),
                "Date":      str(row.get("Date","—")),
                "Pitcher":   str(row.get("Pitcher","—")),
                "Dist (ft)": round(float(row["Distance"]),1),
                "EV (mph)":  round(float(row["ExitSpeed"]),1) if pd.notna(row.get("ExitSpeed")) else np.nan,
                "LA (°)":    round(float(row["Angle"]),1)     if pd.notna(row.get("Angle"))     else np.nan,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Dist (ft)", ascending=False).reset_index(drop=True)


# ── Hitting leaderboard ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building hitting leaderboard...")
def build_hitting_leaderboard(folder: str, team_codes: tuple, source_sig: tuple, min_pa: int = 30) -> pd.DataFrame:
    """Aggregate season hitting stats for all batters on the given teams."""
    team_set = set(team_codes)
    player_chunks: dict[tuple, list] = {}

    # ── Parquet / cloud mode — 2 reads total for entire team set ────────────
    if active_data_source(folder) == "parquet":
        bulk = _parquet_load_teams_bulk(tuple(sorted(team_set)), source_sig, role="batter")
        if not bulk.empty:
            bt_col = bulk.get("BatterTeam", pd.Series("",index=bulk.index)).astype(str).str.strip()
            src_df = bulk[bt_col.isin(team_set)].copy()
            ren = {"ExitSpeed":"EV","Angle":"LA","Distance":"Dist"}
            src_df = src_df.rename(columns={k:v for k,v in ren.items() if k in src_df.columns})
            src_df["Batter"]     = src_df.get("Batter",     pd.Series("",index=src_df.index)).astype(str).str.strip()
            src_df["BatterTeam"] = src_df.get("BatterTeam", pd.Series("",index=src_df.index)).astype(str).str.strip()
            for (tc, batter), g in src_df.groupby(["BatterTeam","Batter"]):
                if not batter: continue
                player_chunks[(str(tc), str(batter))] = [g]
    else:
        # ── CSV / local mode ─────────────────────────────────────────────────
        for path in _unique_csv_files(folder):
            try:
                df = pd.read_csv(path, low_memory=False)
            except Exception:
                continue
            if not {"Batter","BatterTeam"}.issubset(df.columns):
                continue
            df["BatterTeam"] = df["BatterTeam"].astype(str).str.strip()
            sub = df[df["BatterTeam"].isin(team_set)]
            if sub.empty:
                continue
            sub = sub.copy()
            ren = {"ExitSpeed":"EV","Angle":"LA","Distance":"Dist"}
            sub = sub.rename(columns={k: v for k, v in ren.items() if k in sub.columns})
            sub["Batter"] = sub["Batter"].astype(str).str.strip()
            for (tc, batter), g in sub.groupby(["BatterTeam","Batter"]):
                if not batter:
                    continue
                key = (str(tc), str(batter))
                if key not in player_chunks:
                    player_chunks[key] = []
                player_chunks[key].append(g)

    results = []
    for (tc, batter), chunks in player_chunks.items():
        try:
            merged = pd.concat(chunks, ignore_index=True)
            stats  = hitter_stats_cbb(merged)
            if stats.get("PA", 0) < min_pa:
                continue
            results.append({
                "Batter":     batter,
                "Team":       safe_team_name(tc),
                "TeamCode":   tc,
                "Conference": TEAM_CONFERENCES.get(tc, ""),
                **{k: stats.get(k, np.nan) for k in
                   ["PA","H","HR","xHB","BA","OBP","SLG","OPS","wOBA","wRC+",
                    "K%","BB%","Avg EV","HH%","Whiff%","Chase%"]},
            })
        except Exception:
            continue
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("OPS", ascending=False).reset_index(drop=True)


def hitting_leaderboard_section(folder: str, all_known: pd.DataFrame, source_sig: tuple):
    st.caption("Season batting stats — filter by scope and minimum plate appearances.")
    d1 = all_known[all_known["Division"] == "D1"]

    lc1, lc2, lc3, lc4, lc5 = st.columns([1, 1.2, 1.5, 0.8, 1.2])
    with lc1:
        scope = st.radio("Scope", ["Conference","Team","All D1"], key="hb_scope")
    with lc2:
        confs = sorted(d1["Conference"].replace("","—").dropna().unique())
        sel_conf = st.selectbox("Conference", confs, key="hb_conf")
    with lc3:
        conf_teams = (d1[d1["Conference"] == sel_conf][["TeamCode","Team"]]
                      .drop_duplicates().sort_values("Team"))
        if scope == "Team":
            sel_team  = st.selectbox("Team", conf_teams["TeamCode"].tolist(),
                                     format_func=safe_team_name, key="hb_team")
            team_codes = (sel_team,)
        elif scope == "Conference":
            team_codes = tuple(sorted(conf_teams["TeamCode"].unique()))
            sel_team   = None
        else:
            team_codes = None; sel_team = None
    with lc4:
        min_pa = st.number_input("Min PA", min_value=1, max_value=300,
                                 value=30, step=5, key="hb_min_pa")
    with lc5:
        sort_by = st.selectbox("Sort by",
                               ["wRC+","wOBA","OPS","BA","OBP","SLG","HR","Avg EV","HH%","K%","BB%","Whiff%"],
                               key="hb_sort")

    if scope == "All D1":
        if st.button("Load Full D1 Hitting Leaderboard", use_container_width=True):
            st.session_state["hb_d1_confirmed"] = True
        if not st.session_state.get("hb_d1_confirmed"):
            st.info("Loads all tracked batters in D1. Click above to proceed.")
            return
        team_codes = tuple(sorted(d1["TeamCode"].unique()))

    if not team_codes:
        st.warning("No teams found for this selection.")
        return

    with st.spinner("Computing hitting leaderboard…"):
        lb = build_hitting_leaderboard(folder, team_codes, source_sig, min_pa=int(min_pa))

    if lb.empty:
        st.warning("No batters meet the minimum PA threshold.")
        return

    asc = sort_by in {"K%","Whiff%","Chase%"}
    lb  = lb.sort_values(sort_by, ascending=asc).reset_index(drop=True)
    lb.index = lb.index + 1

    show_cols = ["#","Batter","Team"]
    if scope in ("All D1","Conference"):
        show_cols.append("Conference")
    for c in ["PA","H","HR","wOBA","wRC+","BA","OBP","SLG","OPS","K%","BB%","Avg EV","HH%","Whiff%","Chase%"]:
        if c in lb.columns:
            show_cols.append(c)

    view = lb.copy()
    _hb_medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    view["#"] = [_hb_medals.get(i, str(i)) for i in range(1, len(view) + 1)]
    view = view[show_cols]
    for col in show_cols:
        if col in {"PA","H","HR","xHB","wRC+"}:
            view[col] = view[col].apply(lambda v: str(int(v)) if not pd.isna(v) else "—")
        elif col == "wOBA":
            view[col] = view[col].apply(lambda v: f"{float(v):.3f}".replace("0.",".")
                                        if not pd.isna(v) else "—")
        elif col not in ("#","Batter","Team","Conference"):
            view[col] = view[col].apply(lambda v: fmt(v, col))

    # Savant-style cell coloring
    def _style_hb(row):
        styles = [""] * len(row)
        idx = int(row.name) - 1 if row.name else 0
        for ci, col in enumerate(show_cols):
            if col in _HITTER_PCTS and idx < len(lb):
                try:
                    bg, tc = _hitter_color(col, lb.iloc[idx][col])
                    styles[ci] = f"background-color:{bg};color:{tc};font-weight:bold"
                except Exception:
                    pass
        return styles

    st.dataframe(
        view.style.apply(_style_hb, axis=1),
        use_container_width=True,
        height=min(700, 38 + len(view) * 35),
    )
    st.caption(f"{len(view)} batter(s)  ·  minimum {min_pa} PA  ·  sorted by {sort_by}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _deep_team_leaderboard(team_code: str, folder: str, source_sig: tuple) -> None:
    """Show the pitching leaderboard pre-filtered to a team from favorites."""
    st.markdown(f"## {safe_team_name(team_code)} — Leaderboard")
    try:
        all_known = build_index(folder, source_sig)
        all_known["Conference"] = all_known["TeamCode"].map(TEAM_CONFERENCES).fillna("")
        all_known["Team"]       = all_known["TeamCode"].apply(safe_team_name)
        all_known["Division"]   = all_known["TeamCode"].apply(
            lambda c: "D1" if c in TEAM_CONFERENCES else "Other Teams")

        lb_tab = st.radio("", ["Pitching Leaderboard", "Hitting Leaderboard"],
                          horizontal=True, label_visibility="collapsed",
                          key="dl_lb_sub")
        st.markdown("---")
        # Pre-filter to just this team's data
        team_rows = all_known[all_known["TeamCode"] == team_code]
        if team_rows.empty:
            st.info(f"No data found for {safe_team_name(team_code)}.")
            return

        if lb_tab == "Pitching Leaderboard":
            leaderboard_page(folder, all_known, source_sig)
        else:
            hitting_leaderboard_section(folder, all_known, source_sig)
    except Exception as e:
        st.error(f"Could not load leaderboard: {e}")


def _deep_player_report(player_name: str, folder: str, source_sig: tuple) -> None:
    """Detect pitcher vs hitter then show the appropriate report."""
    st.markdown(f"## {player_name}")
    try:
        # Check pitcher index first
        p_idx = build_index(folder, source_sig)
        is_pitcher = (not p_idx.empty and
                      "Pitcher" in p_idx.columns and
                      p_idx["Pitcher"].eq(player_name).any())

        h_idx = build_hitter_index(folder, source_sig)
        is_hitter = (not h_idx.empty and
                     "Batter" in h_idx.columns and
                     h_idx["Batter"].eq(player_name).any())

        if is_pitcher and is_hitter:
            role = st.radio("View as", ["Pitcher", "Hitter"], horizontal=True,
                            key="dl_role")
        elif is_pitcher:
            role = "Pitcher"
        elif is_hitter:
            role = "Hitter"
        else:
            st.warning("No data found for this player.")
            return

        st.markdown("---")

        if role == "Pitcher":
            row      = p_idx[p_idx["Pitcher"] == player_name].iloc[0]
            team     = row["TeamCode"]
            files    = tuple(row["Files"])
            df       = load_pitcher_data(folder, team, player_name, files, source_sig)
            if df.empty:
                st.warning("No pitch data found.")
                return
            card  = pitcher_stats(df)
            primary, _ = get_team_colors(team)
            badge = (f'<span class="conf-badge" style="background:{primary};'
                     f'color:{readable_text_color(primary)}">'
                     f'{TEAM_CONFERENCES.get(team,"")}</span>')
            st.markdown(
                f'<div class="pitcher-card"><p class="pitcher-name">'
                f'{player_name}{badge}</p>'
                f'<p class="pitcher-meta">{safe_team_name(team)}  ·  2026</p></div>',
                unsafe_allow_html=True)
            view = st.radio("Report", ["Percentile Card", "Stat Summary"],
                            horizontal=True, key="dl_p_view")
            if view == "Percentile Card":
                st.image(build_pitcher_pct_card_cbb(df, player_name, team),
                         use_container_width=True)
            else:
                st.image(build_stat_card_png(df, player_name, team),
                         use_container_width=True)

        else:  # Hitter
            row   = h_idx[h_idx["Batter"] == player_name].iloc[0]
            team  = row["TeamCode"]
            files = tuple(row["Files"])
            hdf   = load_hitter_data(folder, team, player_name, files, source_sig)
            if hdf.empty:
                st.warning("No hit data found.")
                return
            view = st.radio("Report", ["Spray Chart Report", "Percentile Card"],
                            horizontal=True, key="dl_h_view")
            if view == "Percentile Card":
                st.image(build_hitter_pct_card_cbb(hdf, player_name, team),
                         use_container_width=True)
            else:
                st.image(build_hitter_summary_png(hdf, player_name, team),
                         use_container_width=True)
    except Exception as e:
        st.error(f"Could not load report: {e}")


def _render_profile() -> None:
    """Collect all teams and players then render the profile page."""
    if st.sidebar.button("← Back to App", key="sb_back_profile", use_container_width=True):
        st.session_state.pop("cbb_show_profile", None)
        st.rerun()

    # Teams come from the in-memory TEAM_CONFERENCES dict — no data loading needed
    all_teams = sorted(TEAM_CONFERENCES.keys(), key=lambda c: safe_team_name(c).lower())

    folder     = str(data_dir())
    source_sig = data_source_signature(folder)
    try:
        p_idx     = build_index(folder, source_sig)
        pitchers  = p_idx["Pitcher"].dropna().unique().tolist() if "Pitcher" in p_idx.columns else []
        h_idx     = build_hitter_index(folder, source_sig)
        batters   = h_idx["Batter"].dropna().unique().tolist() if not h_idx.empty else []
        all_players = sorted(set(pitchers) | set(batters))
    except Exception:
        all_players = []

    render_profile_page(
        safe_team_name_fn=safe_team_name,
        all_team_codes=all_teams,
        all_player_names=all_players,
    )


def _compute_fps_cbb(df: pd.DataFrame) -> float:
    b  = df.get("Balls",   pd.Series(dtype=str)).astype(str).str.strip()
    s  = df.get("Strikes", pd.Series(dtype=str)).astype(str).str.strip()
    fp = df[(b == "0") & (s == "0")]
    if fp.empty: return float("nan")
    strike_calls = {"StrikeCalled","StrikeSwinging","FoulBall","FoulBallNotFieldable",
                    "FoulBallFieldable","FoulTip","InPlay","InPlayNoOut","InPlayOut","InPlayRun"}
    return fp.get("PitchCall", pd.Series(dtype=str)).astype(str).isin(strike_calls).mean() * 100


def _grade_from_score(score: float) -> tuple:
    if   score >= 118: return "A+", "#16a34a"
    elif score >= 113: return "A",  "#22c55e"
    elif score >= 108: return "A-", "#4ade80"
    elif score >= 105: return "B+", "#86efac"
    elif score >= 102: return "B",  "#bef264"
    elif score >= 99:  return "B-", "#fde047"
    elif score >= 96:  return "C+", "#fb923c"
    elif score >= 92:  return "C",  "#f97316"
    elif score >= 87:  return "C-", "#ef4444"
    elif score >= 80:  return "D",  "#dc2626"
    else:              return "F",  "#991b1b"


def _outing_grade_cbb(stuff, loc, fps=float("nan"), csw=float("nan"),
                      whiff=float("nan"), bb_pct=float("nan"),
                      avg_ev=float("nan"), hh_pct=float("nan"),
                      barrel=float("nan")) -> tuple:
    ok = lambda v: not (isinstance(v, float) and np.isnan(v))
    components = []
    # Contact quality — D1 anchors: avg EV 86 mph, HH% 34%, Barrel% 15% (92mph threshold)
    if ok(avg_ev):  components.append((100+(86.0-avg_ev)*3.0,  0.16))
    if ok(hh_pct):  components.append((100+(34.0-hh_pct)*2.0,  0.16))
    if ok(barrel):  components.append((100+(15.0-barrel)*3.0,   0.08))
    # Swing & miss
    if ok(csw):     components.append((100+(csw  -27.0)*2.5,    0.13))
    if ok(whiff):   components.append((100+(whiff-22.0)*2.5,    0.12))
    # Command — D1 avg BB% ~12%
    if ok(bb_pct):  components.append((100+(12.0-bb_pct)*3.0,   0.10))
    if ok(fps):     components.append((100+(fps  -58.0)*2.0,    0.10))
    # Models
    if ok(stuff):   components.append((stuff,                   0.08))
    if ok(loc):     components.append((loc,                     0.07))
    if not components: return "—", "#6B7A93"
    tw = sum(w for _, w in components)
    return _grade_from_score(sum(v*w for v,w in components) / tw)


def _pure_stuff_grade_cbb(stuff: float) -> tuple:
    if isinstance(stuff, float) and np.isnan(stuff): return "—", "#6B7A93"
    return _grade_from_score(stuff)


def _pitch_eff_grade_cbb(df: pd.DataFrame) -> tuple:
    k   = df.get("KorBB", pd.Series(dtype=str)).eq("Strikeout").sum()
    oop = pd.to_numeric(df.get("OutsOnPlay", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    tot = int(oop) + int(k)
    if tot == 0: return "—", "#6B7A93"
    pip = len(df) / (tot / 3.0)
    if   pip <= 14.5: return "A",  "#22c55e"
    elif pip <= 16.5: return "A-", "#4ade80"
    elif pip <= 18.5: return "B+", "#86efac"
    elif pip <= 20.5: return "B",  "#bef264"
    elif pip <= 22.5: return "B-", "#fde047"
    elif pip <= 24.5: return "C+", "#fb923c"
    elif pip <= 26.5: return "C",  "#f97316"
    elif pip <= 28.5: return "C-", "#ef4444"
    elif pip <= 30.0: return "D",  "#dc2626"
    else:             return "F",  "#991b1b"


def _hot_right_now(source_sig: tuple) -> list[dict]:
    """Return national leaders — only runs if leaderboard already cached, else returns []."""
    folder = str(data_dir())
    results = []

    # Only query if the leaderboard cache already exists — never block page load
    cache_info = build_leaderboard.cache_info() if hasattr(build_leaderboard, "cache_info") else None
    # Use a small known conference to keep it fast rather than all D1
    # ACC is reliably large and well-tracked
    try:
        sample_codes = tuple(sorted(
            c for c, v in TEAM_CONFERENCES.items()
            if v in ("ACC", "SEC", "Big Ten", "Big 12")
        ))
        if not sample_codes:
            return []
        p_lb = build_leaderboard(folder, sample_codes, source_sig, min_pitches=50)
        if not p_lb.empty:
            for stat, crown, fmt_fn in [
                ("Stuff+", "🎯", lambda v: f"{v:.0f}"),
                ("FB Velo", "🔥", lambda v: f"{v:.1f} mph"),
                ("K%",     "⚡", lambda v: f"{v:.1f}%"),
            ]:
                if stat in p_lb.columns:
                    row = p_lb.dropna(subset=[stat]).sort_values(stat, ascending=False).iloc[0]
                    name = row.get("Pitcher", "—")
                    display = (name.split(",")[1].strip() + " " + name.split(",")[0]
                               if "," in name else name)
                    results.append({
                        "crown": crown, "stat": stat,
                        "val": fmt_fn(row[stat]),
                        "name": display,
                        "team": safe_team_name(row["TeamCode"]),
                    })
    except Exception:
        pass
    try:
        sample_codes = tuple(sorted(
            c for c, v in TEAM_CONFERENCES.items()
            if v in ("ACC", "SEC", "Big Ten", "Big 12")
        ))
        if sample_codes:
            h_lb = build_hitting_leaderboard(folder, sample_codes, source_sig, min_pa=40)
            if not h_lb.empty and "wRC+" in h_lb.columns:
                row = h_lb.dropna(subset=["wRC+"]).sort_values("wRC+", ascending=False).iloc[0]
                name = row.get("Batter", "—")
                display = (name.split(",")[1].strip() + " " + name.split(",")[0]
                           if "," in name else name)
                results.append({
                    "crown": "👑", "stat": "wRC+",
                    "val": f"{row['wRC+']:.0f}",
                    "name": display,
                    "team": safe_team_name(row["TeamCode"]),
                })
    except Exception:
        pass
    return results


@st.cache_data(ttl=3600, show_spinner=False)
def _build_search_index(source_sig: tuple) -> pd.DataFrame:
    """Combined pitcher + hitter name list for global search. Cached 1hr."""
    folder = str(data_dir())
    p_idx  = build_index(folder, source_sig)
    h_idx  = build_hitter_index(folder, source_sig)
    rows = []
    if not p_idx.empty and "Pitcher" in p_idx.columns:
        rows.append(p_idx[["Pitcher","TeamCode"]].rename(columns={"Pitcher":"name"}))
    if not h_idx.empty and "Batter" in h_idx.columns:
        rows.append(h_idx[["Batter","TeamCode"]].rename(columns={"Batter":"name"}))
    if not rows:
        return pd.DataFrame(columns=["name","TeamCode","label"])
    combined = (pd.concat(rows, ignore_index=True)
                  .drop_duplicates(subset=["name","TeamCode"])
                  .sort_values("name")
                  .reset_index(drop=True))
    combined["label"] = (combined["name"] + "  —  " +
                         combined["TeamCode"].map(safe_team_name))
    return combined


def _render_coverage_strip(all_known: pd.DataFrame) -> None:
    n_pitchers = int(all_known["Pitcher"].nunique()) if "Pitcher" in all_known.columns else 0
    n_teams    = int(all_known["TeamCode"].nunique())
    n_d1       = int(all_known.loc[all_known.get("Division","") == "D1", "TeamCode"].nunique()) \
                 if "Division" in all_known.columns else n_teams
    n_confs    = int(all_known["Conference"].replace("", pd.NA).dropna().nunique()) \
                 if "Conference" in all_known.columns else 0
    st.markdown(f"""
    <div class="cov-strip">
        <div class="cov-tile stat-glow">
            <div class="cov-num">{n_pitchers:,}</div>
            <div class="cov-lbl">Pitchers Tracked</div>
        </div>
        <div class="cov-tile">
            <div class="cov-num">{n_d1}</div>
            <div class="cov-lbl">D1 Programs</div>
        </div>
        <div class="cov-tile">
            <div class="cov-num">{n_confs}</div>
            <div class="cov-lbl">Conferences</div>
        </div>
        <div class="cov-tile">
            <div class="cov-num">2026</div>
            <div class="cov-lbl">Season Live</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _free_preview(all_known: pd.DataFrame, folder: str, source_sig: tuple) -> None:
    """Limited stats view for free / expired-trial users."""
    # Upgrade banner
    st.markdown("""
    <div class="upgrade-banner">
        <div class="ub-icon">⚾</div>
        <div>
            <div class="ub-title">You're on the Free Preview</div>
            <div class="ub-desc">
                Basic team stats are available below. Upgrade to unlock pitcher graphics,
                hitter spray charts, percentile cards, Stuff+, Loc+, and full national leaderboards.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Upgrade for Full Access →", type="primary", key="free_upgrade_btn"):
        st.session_state["cbb_show_upgrade"] = True
        st.rerun()

    # Coverage numbers
    _render_coverage_strip(all_known)

    # What you get with Pro
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">📊</div>
            <div class="feat-title">Pitcher Reports</div>
            <div class="feat-desc">Movement, locations, pitch-by-pitch graphics for every arm in D1</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🎯</div>
            <div class="feat-title">Stuff+ & Loc+</div>
            <div class="feat-desc">ML-powered quality scores — how good is each pitch vs. the field?</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🔥</div>
            <div class="feat-title">Hitter Reports</div>
            <div class="feat-desc">Spray charts, zone heat maps, batted-ball profiles & percentile cards</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🏆</div>
            <div class="feat-title">Leaderboards</div>
            <div class="feat-desc">Rank any D1 pitcher or hitter by Stuff+, wRC+, K%, velo, and more</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📥</div>
            <div class="feat-title">Download Graphics</div>
            <div class="feat-desc">Export pro-quality PNG cards for recruiting, social, or film review</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">⭐</div>
            <div class="feat-title">Favorites</div>
            <div class="feat-desc">Save teams and players for instant one-click access from your profile</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="section-hdr">
        <span class="sh-icon">📋</span>
        <span class="sh-title">Free Team Stats</span>
        <span class="sh-badge">PREVIEW</span>
    </div>
    """, unsafe_allow_html=True)

    fa, fb, fc = st.columns([1, 1.5, 1.5])
    with fa:
        role = st.radio("View", ["Pitchers", "Hitters"], horizontal=True, key="free_role")
    with fb:
        divs = ["D1", "Other Teams", "D2 / D3 / JUCO / NAIA"]
        div  = st.radio("Division", divs, horizontal=False, key="free_div", index=0)
    with fc:
        div_pool = all_known[all_known["Division"] == div]
        confs    = sorted(div_pool["Conference"].replace("", "Unknown").dropna().unique())
        conf     = st.selectbox("Conference", ["All"] + confs, key="free_conf")

    team_pool = div_pool if conf == "All" else div_pool[div_pool["Conference"] == conf]
    teams     = team_pool[["TeamCode", "Team"]].drop_duplicates().sort_values("Team")
    team_code = st.selectbox("Team", teams["TeamCode"].tolist(),
                             format_func=safe_team_name, key="free_team")

    try:
        if role == "Pitchers":
            p_idx = build_index(folder, source_sig)
            rows  = p_idx[p_idx["TeamCode"] == team_code].copy()
            if rows.empty:
                st.info("No pitchers found for this team.")
                return
            stats = []
            for _, r in rows.iterrows():
                df = load_pitcher_data(folder, team_code, r["Pitcher"],
                                       tuple(r["Files"]), source_sig)
                if df.empty:
                    continue
                card = pitcher_stats(df)
                pa_m = (df.get("KorBB", pd.Series("")).isin(["Walk", "Strikeout"]) |
                        df.get("PlayResult", pd.Series("")).isin(
                            ["Single","Double","Triple","HomeRun","Out","Error",
                             "FieldersChoice","Sacrifice"]))
                pa_n = pa_m.sum()
                bb   = df.get("KorBB", pd.Series("")).eq("Walk").sum()
                k    = df.get("KorBB", pd.Series("")).eq("Strikeout").sum()
                stats.append({
                    "Pitcher": r["Pitcher"],
                    "Pitches": card.get("Pitches", 0),
                    "PA":  pa_n,
                    "BB%": f"{bb/pa_n*100:.1f}%" if pa_n else "—",
                    "K%":  f"{k/pa_n*100:.1f}%"  if pa_n else "—",
                })
            if stats:
                st.dataframe(pd.DataFrame(stats).sort_values("Pitches", ascending=False)
                             .set_index("Pitcher"), use_container_width=True)
        else:
            h_idx = build_hitter_index(folder, source_sig)
            rows  = h_idx[h_idx["TeamCode"] == team_code].copy()
            if rows.empty:
                st.info("No hitters found for this team.")
                return
            stats = []
            for _, r in rows.iterrows():
                hdf = load_hitter_data(folder, team_code, r["Batter"],
                                       tuple(r["Files"]), source_sig)
                if hdf.empty:
                    continue
                pr  = hdf.get("PlayResult", pd.Series("")).fillna("")
                kbb = hdf.get("KorBB", pd.Series("")).fillna("")
                pc  = hdf.get("PitchCall", pd.Series("")).fillna("")
                pa_m = kbb.isin(["Walk","Strikeout"]) | pr.isin(
                    ["Single","Double","Triple","HomeRun","Out","Error",
                     "FieldersChoice","Sacrifice"])
                pa = pa_m.sum()
                bb = kbb.eq("Walk").sum()
                k  = kbb.eq("Strikeout").sum()
                hbp = pc.eq("HitByPitch").sum()
                sf  = pr.eq("Sacrifice").sum()
                ab  = max(pa - bb - hbp - sf, 0)
                h   = (pr.isin(["Single","Double","Triple","HomeRun"])).sum()
                stats.append({
                    "Batter": r["Batter"],
                    "PA":  pa,
                    "AB":  ab,
                    "AVG": f"{h/ab:.3f}".replace("0.",".")  if ab else "—",
                    "K%":  f"{k/pa*100:.1f}%"  if pa else "—",
                    "BB%": f"{bb/pa*100:.1f}%" if pa else "—",
                })
            if stats:
                st.dataframe(pd.DataFrame(stats).sort_values("PA", ascending=False)
                             .set_index("Batter"), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load stats: {e}")


def main():
    inject_style()

    # ── Auth gate ─────────────────────────────────────────────────────────────
    if not is_logged_in():
        render_auth_page()
        return

    # ── Sidebar user chip + profile/upgrade nav ──────────────────────────────
    render_sidebar_user()

    if st.session_state.get("cbb_show_upgrade"):
        if st.sidebar.button("← Back to App", key="sb_back_upgrade", use_container_width=True):
            st.session_state.pop("cbb_show_upgrade", None)
            st.rerun()
        render_pricing_page()
        return

    if st.session_state.get("cbb_show_profile"):
        _render_profile()
        return

    # ── Deep-link from profile favorites or global search ────────────────────
    deep = st.session_state.get("cbb_deep_link")
    if deep:
        came_from_search = deep.get("from_search", False)
        back_label = "← Back" if came_from_search else "← Back to Profile"
        if st.sidebar.button(back_label, key="sb_back_deep",
                             use_container_width=True):
            st.session_state.pop("cbb_deep_link", None)
            if not came_from_search:
                st.session_state["cbb_show_profile"] = True
            st.rerun()
        folder     = str(data_dir())
        source_sig = data_source_signature(folder)
        _get_models()
        if deep["type"] == "team":
            _deep_team_leaderboard(deep["code"], folder, source_sig)
        elif deep["type"] == "player":
            # Track recently viewed
            rv = st.session_state.get("cbb_recently_viewed", [])
            name = deep["name"]
            rv = [n for n in rv if n != name][:9]
            rv.insert(0, name)
            st.session_state["cbb_recently_viewed"] = rv
            _deep_player_report(name, folder, source_sig)
        return


    st.markdown("""
    <div class="cbb-hero">
        <div class="hero-kicker">⚾ CBBReports &nbsp;·&nbsp; National D1 Platform &nbsp;·&nbsp; 2026 Season</div>
        <h1>College Baseball Plus</h1>
        <p>The only national D1 analytics platform powered by TrackMan. Pro-grade pitcher graphics,
        hitter spray charts, percentile cards, Stuff+, Loc+, and live national leaderboards —
        for every tracked player in the country.</p>
        <div class="hero-chip-row">
            <div class="hero-chip"><b>⚾ Pitcher Reports</b><span>Movement, locations & arsenal</span></div>
            <div class="hero-chip"><b>🏏 Hitter Reports</b><span>Spray charts & zone heat maps</span></div>
            <div class="hero-chip"><b>🏆 Leaderboards</b><span>D1, conference, team rankings</span></div>
            <div class="hero-chip"><b>🎯 Stuff+ & Loc+</b><span>ML-powered pitch quality</span></div>
            <div class="hero-chip"><b>📥 Download</b><span>Export pro-quality PNG cards</span></div>
        </div>
    </div>""", unsafe_allow_html=True)

    folder = data_dir()
    if not folder.exists():
        st.error(f"Data folder not found: {folder}")
        return

    _get_models()  # warm at startup
    source_sig = data_source_signature(str(folder))

    # ── Data source status ────────────────────────────────────────────────────
    source = active_data_source(str(folder))
    if source == "parquet":
        parts = _parquet_parts()
        try:
            import pyarrow.parquet as _pq_meta
            _n_rows = sum(_pq_meta.read_metadata(str(p)).num_rows for p in parts)
            _pq_dates = []
            for p in parts:
                try:
                    _df_d = pd.read_parquet(p, columns=["Date"])
                    _pq_dates.extend(pd.to_datetime(_df_d["Date"], errors="coerce").dropna().tolist())
                except Exception:
                    pass
            _latest = max(_pq_dates).strftime("%b %d, %Y") if _pq_dates else "unknown"
            _src_label = f"Cloud Parquet  ·  {_n_rows:,} pitches  ·  through {_latest}"
        except Exception:
            _src_label = "Cloud Parquet (size unknown)"
    elif source == "csv":
        _src_label = f"Local CSVs ({len(_unique_csv_files(str(folder))):,} games)"
    else:
        st.error("No scouting data found. Parquet files missing from deployment.")
        return

    ds1, ds2 = st.columns([0.80, 0.20])
    with ds1:
        st.markdown(
            f'<div class="data-strip"><span><b>Data source</b> &nbsp; {_src_label}</span>'
            f'<span>National 2026 coverage</span></div>',
            unsafe_allow_html=True,
        )
    with ds2:
        if st.button("Refresh data cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    index = build_index(str(folder), source_sig)
    if index.empty:
        st.error("No pitchers found. Data source reported above — check if Parquet loaded correctly.")
        return

    # Include ALL teams found in the data; classify by what we know about them
    all_known = index.copy()
    all_known["Conference"] = all_known["TeamCode"].map(TEAM_CONFERENCES).fillna("")
    all_known["Team"]       = all_known["TeamCode"].apply(safe_team_name)
    all_known["Division"]   = all_known["TeamCode"].apply(
        lambda c: "D1" if c in TEAM_CONFERENCES else (
            "Other Teams" if c not in TEAM_NAMES else "D2 / D3 / JUCO / NAIA"
        )
    )

    if not has_pro_access():
        _free_preview(all_known, str(folder), source_sig)
        return

    _render_coverage_strip(all_known)

    # ── Global player search (text input — no 23K selectbox) ─────────────────
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown('<div class="search-label">🔍 &nbsp; Search Any Player</div>',
                unsafe_allow_html=True)
    sc1, sc2 = st.columns([5, 1])
    with sc1:
        query = st.text_input("", key="global_player_search",
                              placeholder="Type last name (e.g. Smith, Jones…)",
                              label_visibility="collapsed")
    with sc2:
        search_go = st.button("Search", key="search_go_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if query and (search_go or len(query) >= 3):
        search_df = _build_search_index(source_sig)
        if not search_df.empty:
            hits = search_df[search_df["name"].str.contains(query, case=False, na=False)]
            if hits.empty:
                st.info(f"No players found matching '{query}'.")
            else:
                hits = hits.head(12)
                btn_cols = st.columns(min(len(hits), 4))
                for idx_r, (_, row) in enumerate(hits.iterrows()):
                    display = (row["name"].split(",")[1].strip() + " " + row["name"].split(",")[0]
                               if "," in row["name"] else row["name"])
                    with btn_cols[idx_r % 4]:
                        if st.button(f"{display}\n{safe_team_name(row['TeamCode'])}",
                                     key=f"srch_{idx_r}", use_container_width=True):
                            st.session_state["cbb_deep_link"] = {
                                "type": "player", "name": row["name"], "from_search": True}
                            st.rerun()

    # ── Hot Right Now (loads on demand only) ──────────────────────────────────
    if st.session_state.get("cbb_hot_loaded"):
        try:
            hot = _hot_right_now(source_sig)
            if hot:
                cards_html = '<div class="hot-strip">'
                for h in hot:
                    cards_html += (
                        f'<div class="hot-card">'
                        f'<div class="hc-crown">{h["crown"]}</div>'
                        f'<div class="hc-val">{h["val"]}</div>'
                        f'<div class="hc-stat">{h["stat"]} Leader  ·  Power 4</div>'
                        f'<div class="hc-name">{h["name"]}</div>'
                        f'<div class="hc-team">{h["team"]}</div>'
                        f'</div>'
                    )
                st.markdown(cards_html + '</div>', unsafe_allow_html=True)
        except Exception:
            pass
    else:
        if st.button("🔥  Load Hot Stats", key="load_hot", use_container_width=False):
            st.session_state["cbb_hot_loaded"] = True
            st.rerun()

    # ── Compare Players ───────────────────────────────────────────────────────
    with st.expander("⚔️  Compare Two Players", expanded=False):
        st.markdown('<div class="compare-hdr">Head-to-Head Comparison — type last names</div>',
                    unsafe_allow_html=True)
        cc1, cc_vs, cc2 = st.columns([5, 1, 5])
        with cc1:
            cmp_q1 = st.text_input("Player 1", key="cmp_q1", placeholder="Last name…",
                                   label_visibility="collapsed")
        with cc_vs:
            st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
        with cc2:
            cmp_q2 = st.text_input("Player 2", key="cmp_q2", placeholder="Last name…",
                                   label_visibility="collapsed")

        if cmp_q1 and cmp_q2:
            cmp_df = _build_search_index(source_sig)
            hits1  = cmp_df[cmp_df["name"].str.contains(cmp_q1, case=False, na=False)].head(5)
            hits2  = cmp_df[cmp_df["name"].str.contains(cmp_q2, case=False, na=False)].head(5)

            def _pick(hits, key):
                if hits.empty: return None
                if len(hits) == 1: return hits.iloc[0]
                opts = hits["label"].tolist()
                sel  = st.selectbox("Select", opts, key=key, label_visibility="collapsed")
                return hits[hits["label"] == sel].iloc[0] if sel else None

            sel1 = _pick(hits1, "cmp_sel1")
            sel2 = _pick(hits2, "cmp_sel2")

            if sel1 is not None and sel2 is not None:
                def _load_cmp(row):
                    name = row["name"]
                    p_idx = build_index(str(folder), source_sig)
                    h_idx = build_hitter_index(str(folder), source_sig)
                    if not p_idx.empty and p_idx["Pitcher"].eq(name).any():
                        pr = p_idx[p_idx["Pitcher"] == name].iloc[0]
                        df = load_pitcher_data(str(folder), pr["TeamCode"], name,
                                               tuple(pr["Files"]), source_sig)
                        return "pitcher", name, pr["TeamCode"], df
                    if not h_idx.empty and h_idx["Batter"].eq(name).any():
                        hr = h_idx[h_idx["Batter"] == name].iloc[0]
                        df = load_hitter_data(str(folder), hr["TeamCode"], name,
                                              tuple(hr["Files"]), source_sig)
                        return "hitter", name, hr["TeamCode"], df
                    return None, name, row["TeamCode"], pd.DataFrame()

                r1 = _load_cmp(sel1)
                r2 = _load_cmp(sel2)
                if r1[0] and r2[0] and r1[0] == r2[0]:
                    role = r1[0]
                    col_a, col_b = st.columns(2)
                    for col, (rtype, name, team, df) in [(col_a, r1), (col_b, r2)]:
                        with col:
                            primary, _ = get_team_colors(team)
                            st.markdown(
                                f'<div style="background:{primary}22;border:1px solid {primary}55;'
                                f'border-radius:10px;padding:10px 14px;margin-bottom:8px;text-align:center">'
                                f'<div style="color:#fff;font-weight:800;font-size:1rem">'
                                f'{"⚾" if role=="pitcher" else "🏏"} {name}</div>'
                                f'<div style="color:#9BAABF;font-size:.75rem">{safe_team_name(team)}</div>'
                                f'</div>', unsafe_allow_html=True)
                            if not df.empty:
                                keys = ([("Velo","Velo"),("K%","K%"),("Whiff%","Whiff%"),
                                         ("CSW%","CSW%"),("Stuff+","Stuff+"),("BAA","BAA")]
                                        if role == "pitcher" else
                                        [("BA","BA"),("OBP","OBP"),("SLG","SLG"),
                                         ("wRC+","wRC+"),("K%","K%"),("Avg EV","Avg EV")])
                                s = pitcher_stats(df) if role == "pitcher" else hitter_stats_cbb(df)
                                for k, lbl in keys:
                                    v = s.get(k, float("nan"))
                                    if not (isinstance(v, float) and pd.isna(v)):
                                        st.metric(lbl, fmt(v, k))
                elif r1[0] and r2[0]:
                    st.info("One is a pitcher, one is a hitter — pick two of the same role.")

    section = st.radio("", ["⚾  Pitcher Reports", "🏏  Hitter Reports", "🏆  Leaderboards"],
                        horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if "Leaderboards" in section:
        lb_tab = st.radio("", ["Pitching Leaderboard", "Hitting Leaderboard", "HR Distance"],
                          horizontal=True, label_visibility="collapsed", key="lb_sub")
        if lb_tab == "HR Distance":
            st.markdown("""<div class="section-hdr"><span class="sh-icon">💣</span>
            <span class="sh-title">HR Distance Leaderboard</span>
            <span class="sh-badge">NATIONAL</span></div>""", unsafe_allow_html=True)
            hr_leaderboard_section(str(folder), all_known, source_sig)
        elif lb_tab == "Hitting Leaderboard":
            st.markdown("""<div class="section-hdr"><span class="sh-icon">🏏</span>
            <span class="sh-title">Hitting Leaderboard</span>
            <span class="sh-badge">NATIONAL</span></div>""", unsafe_allow_html=True)
            hitting_leaderboard_section(str(folder), all_known, source_sig)
        else:
            st.markdown("""<div class="section-hdr"><span class="sh-icon">🔥</span>
            <span class="sh-title">Pitching Leaderboard</span>
            <span class="sh-badge">NATIONAL</span></div>""", unsafe_allow_html=True)
            leaderboard_page(str(folder), all_known, source_sig)
        return

    if "Hitter Reports" in section:
        st.markdown("""<div class="section-hdr"><span class="sh-icon">🏏</span>
        <span class="sh-title">Hitter Reports</span>
        <span class="sh-badge">TRACKMAN 2026</span></div>""", unsafe_allow_html=True)
        st.markdown('<div class="filter-row">', unsafe_allow_html=True)
        hfa, hfb, hfc, hfd = st.columns([0.9, 1.2, 1.5, 1.1])
        with hfa:
            h_div = st.radio("Division", ["D1", "Other Teams", "D2 / D3 / JUCO / NAIA"],
                             horizontal=False, key="h_div")
        with hfb:
            if h_div == "D1":
                h_confs = sorted(all_known.loc[all_known["Division"]=="D1","Conference"]
                                 .replace("","Unknown").dropna().unique())
                h_conf = st.selectbox("Conference", ["All D1"] + h_confs, key="h_conf")
            else:
                h_conf = None
        h_div_pool  = all_known[all_known["Division"] == h_div]
        h_conf_pool = h_div_pool if (not h_conf or h_conf == "All D1") else \
                      h_div_pool[h_div_pool["Conference"] == h_conf]
        h_teams = h_conf_pool[["TeamCode","Team"]].drop_duplicates().sort_values("Team")
        with hfc:
            h_team = st.selectbox("Team", h_teams["TeamCode"].tolist(),
                                  format_func=safe_team_name, key="h_team")
        # Build hitter index on demand
        with st.spinner("Building hitter index…"):
            h_idx = build_hitter_index(str(folder), source_sig)
        h_idx = h_idx[h_idx["TeamCode"].eq(h_team)].sort_values(["PA","Batter"], ascending=[False,True])
        with hfd:
            if h_idx.empty:
                st.warning("No hitter data for this team.")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            def _h_label(p, _i=h_idx):
                try: return f"{p}  ({int(_i.loc[_i.Batter==p,'PA'].iloc[0]):,} PA)"
                except Exception: return p
            hitter = st.selectbox("Hitter", h_idx["Batter"].tolist(),
                                  format_func=_h_label, key="h_hitter")
        st.markdown("</div>", unsafe_allow_html=True)

        h_row = h_idx[h_idx["Batter"] == hitter]
        h_files = tuple(h_row["Files"].iloc[0]) if not h_row.empty else ()
        hdf = load_hitter_data(str(folder), h_team, hitter, h_files, source_sig)
        if hdf.empty:
            st.warning("No data found for this hitter.")
            return

        h_primary, h_accent = get_team_colors(h_team)
        h_conf_label = TEAM_CONFERENCES.get(h_team, "")
        h_card = hitter_stats_cbb(hdf)

        # Header card
        h_logo = logo_path_for_team(h_team)
        if h_logo:
            hc1, hc2 = st.columns([0.08, 0.92])
            with hc1:
                try: st.image(str(h_logo), width=64)
                except Exception: pass
        else:
            hc2 = st.container()
        with hc2:
            badge = (f'<span class="conf-badge" style="background:{h_primary};'
                     f'color:{readable_text_color(h_primary)}">{h_conf_label}</span>') if h_conf_label else ""
            st.markdown(
                f'<div class="pitcher-card"><p class="pitcher-name">{hitter}{badge}</p>'
                f'<p class="pitcher-meta">{safe_team_name(h_team)}  ·  2026 Season  ·  '
                f'{h_card.get("PA",0)} PA tracked</p></div>', unsafe_allow_html=True)

        # Key metrics
        h_stat_keys = ["PA","AB","H","HR","xHB","BA","OBP","SLG","wOBA","wRC+",
                       "K%","BB%","Avg EV","HH%","Whiff%"]
        h_mcols = st.columns(len(h_stat_keys))
        for col, key in zip(h_mcols, h_stat_keys):
            v = h_card.get(key, np.nan)
            if key in {"BA","OBP","SLG","OPS","wOBA"}:
                disp = f"{float(v):.3f}".replace("0.",".")  if not pd.isna(v) else "—"
            elif key in {"PA","AB","H","HR","xHB","BB","K","BIP","wRC+"}:
                disp = str(int(v)) if not pd.isna(v) else "—"
            else:
                disp = f"{float(v):.1f}" if not pd.isna(v) else "—"
            col.metric(key, disp)

        _hdf_gb = hdf[hdf.get("TaggedHitType", pd.Series("", index=hdf.index)).astype(str).eq("GroundBall")].copy() if "TaggedHitType" in hdf.columns else pd.DataFrame()
        if not _hdf_gb.empty and "Direction" in _hdf_gb.columns:
            _hdf_gb["Direction"] = pd.to_numeric(_hdf_gb["Direction"], errors="coerce")
            _hdf_gb = _hdf_gb.dropna(subset=["Direction"])
        if len(_hdf_gb) >= 5:
            _gb_n = len(_hdf_gb)
            _gbl = (_hdf_gb["Direction"] < 0).sum() / _gb_n * 100
            _gbr = (_hdf_gb["Direction"] > 0).sum() / _gb_n * 100
            _gb_cols = st.columns([1, 1, 3])
            _gb_cols[0].metric(f"GB Left of 2B  ({int(_gb_n)} GBs)", f"{_gbl:.0f}%", help="SS / 3B side  ·  Direction < 0")
            _gb_cols[1].metric("GB Right of 2B", f"{_gbr:.0f}%", help="2B / 1B side  ·  Direction > 0")

        st.markdown("<br>", unsafe_allow_html=True)
        h_view = st.radio("Report Type", ["Spray Chart Report", "Percentile Card"],
                          horizontal=True, key="h_view")
        st.markdown("---")
        if h_view == "Percentile Card":
            png = build_hitter_pct_card_cbb(hdf, hitter, h_team)
            st.image(png, use_container_width=True)
            st.download_button("Download Percentile Card", png,
                file_name=f"{hitter.replace(', ','_')}_percentile_card.png",
                mime="image/png", use_container_width=True)
        else:
            png = build_hitter_summary_png(hdf, hitter, h_team)
            st.image(png, use_container_width=True)
            st.download_button("Download Hitter Report PNG", png,
                file_name=f"{hitter.replace(', ','_')}_hitter_report.png",
                mime="image/png", use_container_width=True)
        return

    # ── Pitcher Reports ───────────────────────────────────────────────────────
    st.markdown("""<div class="section-hdr"><span class="sh-icon">⚾</span>
    <span class="sh-title">Pitcher Reports</span>
    <span class="sh-badge">TRACKMAN 2026</span></div>""", unsafe_allow_html=True)
    st.markdown('<div class="filter-row">', unsafe_allow_html=True)
    fa, fb, fc, fd = st.columns([0.9, 1.2, 1.5, 1.1])
    with fa:
        division = st.radio("Division", ["D1", "Other Teams", "D2 / D3 / JUCO / NAIA"],
                            horizontal=False)
    with fb:
        if division == "D1":
            conferences = sorted(
                all_known.loc[all_known["Division"]=="D1","Conference"]
                .replace("","Unknown").dropna().unique())
            conference = st.selectbox("Conference", ["All D1"] + conferences)
        else:
            conference = None
            st.markdown("<br>", unsafe_allow_html=True)

    div_pool  = all_known[all_known["Division"] == division].copy()
    conf_pool = div_pool if (not conference or conference == "All D1") else \
                div_pool[div_pool["Conference"] == conference]

    if conf_pool.empty:
        st.warning("No pitchers found for this selection.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    teams = conf_pool[["TeamCode","Team"]].drop_duplicates().sort_values("Team")
    with fc:
        team_code = st.selectbox("Team", teams["TeamCode"].tolist(),
                                 format_func=safe_team_name)
    team_rows = conf_pool[conf_pool["TeamCode"].eq(team_code)].sort_values(
        ["Pitches","Pitcher"], ascending=[False,True])
    with fd:
        def _p_label(p, _r=team_rows):
            try: return f"{p}  ({int(_r.loc[_r.Pitcher==p,'Pitches'].iloc[0]):,})"
            except Exception: return p
        pitcher = st.selectbox(
            "Pitcher", team_rows["Pitcher"].tolist(), format_func=_p_label)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    row = all_known[(all_known["TeamCode"]==team_code) & (all_known["Pitcher"]==pitcher)]
    file_list = tuple(row["Files"].iloc[0]) if not row.empty else ()
    df = load_pitcher_data(str(folder), team_code, pitcher, file_list, source_sig)
    if df.empty:
        st.warning("No tracked pitches found for that pitcher.")
        return

    primary, accent = get_team_colors(team_code)
    conf_label = TEAM_CONFERENCES.get(team_code, "")

    # ── Pitcher header card ───────────────────────────────────────────────────
    logo = logo_path_for_team(team_code)
    if logo:
        hc1, hc2 = st.columns([0.08, 0.92])
        with hc1:
            try:
                st.image(str(logo), width=64)
            except Exception:
                pass
    else:
        hc2 = st.container()
    with hc2:
        badge = (f'<span class="conf-badge" style="background:{primary};color:{readable_text_color(primary)}">'
                 f'{conf_label}</span>') if conf_label else ""
        try:
            _n_pitches = f"{int(team_rows.loc[team_rows.Pitcher==pitcher,'Pitches'].iloc[0]):,}"
        except Exception:
            _n_pitches = "?"
        _pt = df["PitcherThrows"].dropna().astype(str).mode() if "PitcherThrows" in df.columns else pd.Series()
        _ph = ("LHP" if _pt.iloc[0].upper().startswith("L") else "RHP") if not _pt.empty else ""
        st.markdown(
            f'<div class="pitcher-card">'
            f'<p class="pitcher-name">{pitcher}{badge}</p>'
            f'<p class="pitcher-meta">{safe_team_name(team_code)}  ·  {_ph}  ·  2026 Season  ·  '
            f'{_n_pitches} pitches tracked</p>'
            f'</div>', unsafe_allow_html=True)

    # ── Key metrics ───────────────────────────────────────────────────────────
    card = pitcher_stats(df)
    stat_keys = ["Pitches","Games","FB Velo","FB PercVelo","MaxVelo",
                 "RelH","RelExt","Stuff+","Loc+","K%","Whiff%","Zone%","CSW%"]
    mcols = st.columns(len(stat_keys))
    for col, key in zip(mcols, stat_keys):
        col.metric(key, fmt(card.get(key), key))

    has_models = "Stuff+" in df.columns and df["Stuff+"].notna().any()
    if has_models:
        st.markdown("""
        <div class="metric-explainer">
        <b>Stuff+</b> — pitch quality from velocity, movement &amp; spin. 100 = average D1 pitcher. Higher is better. &nbsp;|&nbsp;
        <b>Loc+</b> — command quality based on competitive locations by count. 100 = average. Higher is better. &nbsp;|&nbsp;
        <b>FB PercVelo</b> — fastball perceived velocity adjusted for extension and pitch shape.
        </div>""", unsafe_allow_html=True)

        # ── Outing grades ─────────────────────────────────────────────────────
        stuff_m  = card.get("Stuff+", float("nan"))
        loc_m    = card.get("Loc+",   float("nan"))
        csw_p    = card.get("CSW%",   float("nan"))
        whiff_p  = card.get("Whiff%", float("nan"))
        bb_p     = card.get("BB%",    float("nan"))
        fps_p    = _compute_fps_cbb(df)
        # Contact quality from raw pitch data
        ev_raw   = pd.to_numeric(df.get("ExitSpeed", df.get("EV", pd.Series(dtype=float))), errors="coerce")
        la_raw   = pd.to_numeric(df.get("Angle",     df.get("LA", pd.Series(dtype=float))), errors="coerce")
        pc_raw   = df.get("PitchCall", pd.Series(dtype=str)).astype(str)
        bip_mask = pc_raw.isin(["InPlay","InPlayNoOut","InPlayOut","InPlayRun"])
        bip_ev   = ev_raw[bip_mask & ev_raw.notna()]
        avg_ev_p = float(bip_ev.mean())        if not bip_ev.empty else float("nan")
        hh_p     = float((bip_ev>=95).mean()*100) if not bip_ev.empty else float("nan")
        bip_la   = la_raw[bip_mask & ev_raw.notna()]
        if not bip_ev.empty and not bip_la.empty:
            _brl = ((bip_ev >= 92) & bip_la.between(16, 36))
            barrel_p = float(_brl.mean() * 100)
        else:
            barrel_p = float("nan")

        g1, g2, g3 = st.columns(3)
        for gcol, label, (letter, color) in [
            (g1, "Outing Grade",      _outing_grade_cbb(stuff_m, loc_m, fps_p, csw_p, whiff_p, bb_p, avg_ev_p, hh_p, barrel_p)),
            (g2, "Pure Stuff Grade",  _pure_stuff_grade_cbb(stuff_m)),
            (g3, "Pitch Efficiency",  _pitch_eff_grade_cbb(df)),
        ]:
            tc = "#0f172a" if color in ("#bef264","#fde047","#86efac","#4ade80") else "#ffffff"
            gcol.markdown(
                f'<div style="background:{color}18;border:2px solid {color}55;'
                f'border-radius:10px;padding:12px;text-align:center;margin:4px 0">'
                f'<div style="font-size:2rem;font-weight:900;color:{color}">{letter}</div>'
                f'<div style="color:#9BAABF;font-size:.72rem;text-transform:uppercase;'
                f'letter-spacing:.08em;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Report selector + output ──────────────────────────────────────────────
    view = st.radio("Report Type", ["Season Summary","Postgame Summary","Stat Card","Percentile Card"],
                    horizontal=True)
    st.markdown("---")

    if view == "Percentile Card":
        png = build_pitcher_pct_card_cbb(df, pitcher, team_code)
        st.image(png, use_container_width=True)
        st.download_button("Download Percentile Card", png,
            file_name=f"{pitcher.replace(', ','_')}_percentile_card.png",
            mime="image/png", use_container_width=True)

    elif view == "Stat Card":
        png = build_stat_card_png(df, pitcher, team_code)
        st.image(png, use_container_width=True)
        st.download_button("Download Stat Card PNG", png,
            file_name=f"{pitcher.replace(', ','_')}_stat_card.png",
            mime="image/png", use_container_width=True)

    elif view == "Postgame Summary":
        if "GameID" in df.columns:
            opp_col = "BatterTeam" if "BatterTeam" in df.columns else None
            games = (df.groupby("GameID", observed=True)
                       .agg(Date=("Date","first"),
                            Pitches=("Pitch","count"),
                            Opp=(opp_col,"first") if opp_col else ("Date","first"))
                       .reset_index()
                       .sort_values("Date", ascending=False))
            games["_label"] = games.apply(lambda r: (
                f"{r['Date']}  vs  {safe_team_name(str(r['Opp']))}  "
                f"({int(r['Pitches'])} pitches)"
            ), axis=1)
            gid = st.selectbox(
                "Select Game", games["GameID"].astype(str).tolist(),
                format_func=lambda g: games.loc[
                    games["GameID"].astype(str).eq(g), "_label"].iloc[0])
        else:
            gid = None
        png = build_summary_png(df, pitcher, team_code, gid, "Postgame Summary")
        st.image(png, use_container_width=True)
        st.download_button("Download Postgame PNG", png,
            file_name=f"{pitcher.replace(', ','_')}_postgame.png",
            mime="image/png", use_container_width=True)

    else:
        png = build_summary_png(df, pitcher, team_code, label="Season Summary")
        st.image(png, use_container_width=True)
        st.download_button("Download Season Summary PNG", png,
            file_name=f"{pitcher.replace(', ','_')}_season.png",
            mime="image/png", use_container_width=True)

    # ── Arsenal table expander ────────────────────────────────────────────────
    with st.expander("Full Arsenal Breakdown"):
        arsen = arsenal_table(df)
        if not arsen.empty:
            show_cols = [c for c in ["Pitch","N","Usage%","Velo","IVB","HB","Spin",
                                     "Stuff+","Loc+","Whiff%","Zone%","CSW%",
                                     "RelH","RelS","Ext"] if c in arsen.columns]
            view_df = arsen[show_cols].copy()
            for col in show_cols:
                if col != "Pitch":
                    view_df[col] = view_df[col].apply(lambda v: fmt(v, col))
            st.dataframe(view_df, hide_index=True, use_container_width=True)

    with st.expander("About These Metrics"):
        st.markdown("""
| Metric | What it measures |
|---|---|
| **Stuff+** | Raw pitch quality — velocity, movement, spin, release. 100 = avg D1 pitcher |
| **Loc+** | Command quality — how often the pitcher locates competitively by count. 100 = avg |
| **FB Velo** | Average fastball velocity (4-seam / 2-seam) |
| **FB PercVelo** | Perceived fastball velocity accounting for extension and pitch shape |
| **Whiff%** | Swings and misses ÷ total swings |
| **Zone%** | Pitches thrown in the strike zone |
| **CSW%** | Called strikes + whiffs ÷ total pitches (premium strike metric) |
| **IVB** | Induced vertical break — how much the pitch rises vs. a spinless ball |
| **HB** | Horizontal break — arm-side (+) or glove-side (−) movement |
| **Ext** | Extension — how far in front of the rubber the pitcher releases the ball |
| **RelH / RelS** | Release height and horizontal release point |
        """)


if __name__ == "__main__":
    main()
