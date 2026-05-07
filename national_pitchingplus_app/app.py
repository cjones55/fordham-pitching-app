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

APP_ROOT          = Path(__file__).resolve().parent
PROJECT_ROOT      = APP_ROOT.parent
DEFAULT_DATA_DIR  = (PROJECT_ROOT / "scouting_2026_trackman").resolve()
SCOUTING_PARQUET_1 = PROJECT_ROOT / "scouting_data_1.parquet"
SCOUTING_PARQUET_2 = PROJECT_ROOT / "scouting_data_2.parquet"
LOGO_DIR          = APP_ROOT / "team_logos"
MODELS_DIR        = PROJECT_ROOT / "models"

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
    "HOU_COG":("#C8102E","#FFFFFF"),"HOU_COU":("#C8102E","#FFFFFF"),
    "YAL_BUL":("#00356B","#FFFFFF"),
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
    "CHA_49E":"American","CHA_FOR":"American","ECU_PIR":"American","FAU_OWL":"American","RIC_OWL":"American",
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
    "GEO_GWI":"Atlantic 10","GEO_PAT":"Atlantic 10","JOE_HAW":"Atlantic 10","LAS_EXP":"Atlantic 10","LAS_EXS":"Atlantic 10",
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
    "LOU_BUL":"C-USA","MIS_BEA":"C-USA","MT":"C-USA","MTSU_BLU":"C-USA","NMS_AGG":"C-USA",
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

BG   = "#13151c"
BG2  = "#1a1d26"
TXT  = "#FFFFFF"
TXT2 = "#CCCCCC"

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

st.set_page_config(page_title="College Baseball Plus", page_icon="⚾", layout="wide")


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
    df["Balls"]   = pd.to_numeric(df.get("Balls",   0), errors="coerce").fillna(0).astype(int)
    df["Strikes"] = pd.to_numeric(df.get("Strikes", 0), errors="coerce").fillna(0).astype(int)
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
    for suffix in [".png",".jpg",".jpeg"]:
        p = LOGO_DIR / f"{str(code or '').strip()}{suffix}"
        if p.exists():
            return p
    return None


def _place_logo(fig_or_ax, logo: "Path | None", primary: str, accent: str,
                bounds: tuple, use_inset: bool = False) -> bool:
    """Render a team logo PNG transparently on the header color.
    White background only when logo content and header have low contrast.
    No border/outline. Returns True if logo was placed."""
    if not logo:
        return False
    try:
        img = Image.open(logo).convert("RGBA")
        arr = np.array(img)
        alpha_mask = arr[:, :, 3] > 50
        h = primary.lstrip("#")
        hdr_lum = (0.299*int(h[0:2],16)+0.587*int(h[2:4],16)+0.114*int(h[4:6],16))/255
        if alpha_mask.any():
            rgb = arr[alpha_mask, :3]
            logo_lum = (0.299*rgb[:,0]+0.587*rgb[:,1]+0.114*rgb[:,2]).mean()/255
            bg = "#FFFFFF" if abs(logo_lum - hdr_lum) < 0.20 else primary
        else:
            bg = primary
        if use_inset:
            lax = fig_or_ax.inset_axes(list(bounds))
        else:
            lax = fig_or_ax.add_axes(list(bounds))
        lax.set_facecolor(bg)
        lax.imshow(arr, aspect="equal")
        lax.set_xticks([]); lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_visible(False)
        return True
    except Exception:
        return False


def pc(pt: str) -> str:
    return PITCH_COLORS.get(str(pt).upper()[:2], "#888888")


def fmt(v, stat="") -> str:
    if pd.isna(v):
        return "—"
    if stat in {"BAA","SLG","BA","OBP","OPS"}:
        return f"{float(v):.3f}".replace("0.",".")
    if stat in {"Pitches","Games","K","BB","N"}:
        return f"{int(round(float(v))):,}"
    if stat == "Usage%":
        return f"{float(v):.1f}%"
    return f"{float(v):.1f}"


# ── Baseball Savant–style percentile coloring ────────────────────────────────
# Breakpoints (p10, p30, p50, p70, p90) for D1 college hitters, 2025-26
_HITTER_PCTS: dict[str, tuple] = {
    # All breakpoints computed from 1,876 D1 hitters ≥50 PA, 7,116 TrackMan
    # games, collegiate wOBA weights, with soft-single error filter (EV<60 mph)
    "BA":     (.204, .241, .282, .321, .354, True),
    "OBP":    (.305, .343, .385, .427, .462, True),
    "SLG":    (.289, .357, .442, .535, .623, True),
    "OPS":    (.614, .713, .827, .951, 1.060, True),
    "wOBA":   (.267, .303, .341, .381, .417, True),   # lgwOBA = .325
    "wRC+":   ( 82,   93, 105,  117,  128, True),
    "K%":     (11.5, 15.4, 19.7, 24.7, 30.1, False),
    "BB%":    ( 6.4,  8.7, 11.5, 14.4, 17.2, True),
    "Whiff%": (14.5, 18.2, 23.0, 28.2, 33.1, False),
    "Chase%": (25.0, 28.2, 31.7, 35.1, 38.5, False),
    "Avg EV": (84.0, 86.1, 88.3, 90.6, 92.5, True),
    "HH%":    (20.7, 29.0, 38.2, 46.2, 52.9, True),
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
        pct = min(0.99, 0.90 + 0.09 * (fv - bps[-1]) / max(bps[-1] * 0.12, 1))
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
    p10, p30, p50, p70, p90, high_good = _HITTER_PCTS[stat]
    bps  = [p10, p30, p50, p70, p90]
    pcts = [0.10, 0.30, 0.50, 0.70, 0.90]
    fv   = float(value)
    if fv <= bps[0]:
        pct = 0.0
    elif fv >= bps[-1]:
        pct = 1.0
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
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    .stApp{background:#0f1117;color:#f0f2f6;font-family:'Inter',sans-serif}
    div[data-testid="stHeader"]{background:transparent}
    div[data-testid="stMetricValue"]{font-size:1.35rem!important;font-weight:700!important}
    div[data-testid="stMetricLabel"]{font-size:0.72rem!important;color:#94a3b8!important;text-transform:uppercase;letter-spacing:.04em}

    .cbb-hero{
        background:linear-gradient(135deg,#1a1f2e 0%,#161b27 60%,#1e2436 100%);
        border:1px solid #2d3748;border-radius:12px;
        padding:24px 28px;margin-bottom:20px;
    }
    .cbb-hero h1{margin:0 0 4px;font-size:26px;font-weight:800;color:#fff;letter-spacing:-.02em}
    .cbb-hero p{margin:0;color:#94a3b8;font-size:14px;line-height:1.5}

    .filter-row{
        background:#161b27;border:1px solid #2d3748;border-radius:10px;
        padding:14px 18px;margin-bottom:12px;
    }
    .pitcher-card{
        background:linear-gradient(135deg,#1a1f2e,#161b27);
        border:1px solid #2d3748;border-radius:10px;
        padding:16px 20px;margin-bottom:16px;
    }
    .pitcher-name{font-size:22px;font-weight:800;color:#fff;margin:0 0 2px}
    .pitcher-meta{font-size:13px;color:#94a3b8;margin:0}
    .conf-badge{
        display:inline-block;padding:2px 10px;border-radius:20px;
        font-size:11px;font-weight:700;letter-spacing:.05em;
        margin-left:8px;vertical-align:middle;
    }

    .metric-explainer{
        background:#161b27;border:1px solid #2d3748;border-radius:8px;
        padding:12px 16px;font-size:12px;color:#94a3b8;line-height:1.6;
        margin-top:4px;
    }
    .metric-explainer b{color:#e2e8f0}

    .paywall{max-width:680px;margin:40px auto;padding:32px;border-radius:12px;
             border:1px solid #2d3748;background:#161b27}

    .stDownloadButton>button{
        background:#2563eb!important;color:#fff!important;
        border:none!important;border-radius:8px!important;
        font-weight:700!important;letter-spacing:.02em!important;
    }
    .stDownloadButton>button:hover{background:#1d4ed8!important}

    hr{border-color:#2d3748!important}
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner="Loading scouting database…")
def _load_scouting_parquet() -> pd.DataFrame:
    """Load scouting_data_1/2.parquet — the compiled cloud-ready scouting database."""
    parts = [p for p in (SCOUTING_PARQUET_1, SCOUTING_PARQUET_2) if p.exists()]
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _csv_folder_has_data(folder: str) -> bool:
    """True if the local CSV folder has any scouting files."""
    return bool(csv_files(folder))


def _get_scouting_df(folder: str) -> pd.DataFrame:
    """Return the full scouting DataFrame — from CSVs locally or Parquet on cloud."""
    if _csv_folder_has_data(folder):
        return pd.DataFrame()   # signal: use CSV path
    return _load_scouting_parquet()


@st.cache_data(show_spinner="Building pitcher index…")
def build_index(folder: str) -> pd.DataFrame:
    """Returns (TeamCode, Pitcher, Pitches, Files) where Files is a list of paths."""
    # ── Parquet / cloud mode ─────────────────────────────────────────────────
    if not _csv_folder_has_data(folder):
        pq = _load_scouting_parquet()
        if pq.empty or "Pitcher" not in pq.columns or "PitcherTeam" not in pq.columns:
            return pd.DataFrame(columns=["TeamCode","Team","Pitcher","Pitches","Files"])
        grp = (pq.dropna(subset=["Pitcher","PitcherTeam"])
                 .assign(Pitcher=lambda d: d["Pitcher"].str.strip(),
                         PitcherTeam=lambda d: d["PitcherTeam"].str.strip())
                 .groupby(["PitcherTeam","Pitcher"], as_index=False)
                 .size().rename(columns={"PitcherTeam":"TeamCode","size":"Pitches"}))
        grp["Files"] = [[] for _ in range(len(grp))]   # no individual files in cloud mode
        grp["Team"]  = grp["TeamCode"].map(safe_team_name)
        return grp

    # ── CSV / local mode ─────────────────────────────────────────────────────
    usecols = ["Pitcher","PitcherTeam"]
    rows = []
    for path in _unique_csv_files(folder):
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


@st.cache_data(show_spinner="Loading pitcher data…")
def load_pitcher_data(folder: str, team_code: str, pitcher: str,
                      file_list: tuple) -> pd.DataFrame:
    """Load data for a specific pitcher — from Parquet (cloud) or CSVs (local)."""
    # ── Parquet / cloud mode ─────────────────────────────────────────────────
    if not _csv_folder_has_data(folder):
        pq = _load_scouting_parquet()
        if pq.empty:
            return pd.DataFrame()
        mask = (pq.get("PitcherTeam", pd.Series("", index=pq.index)).astype(str).str.strip() == str(team_code)) & \
               (pq.get("Pitcher",     pd.Series("", index=pq.index)).astype(str).str.strip() == str(pitcher))
        sub = pq[mask].copy()
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
    for col in ["RelH","RelS","Ext"]:
        if col in df.columns:
            agg[col] = df.groupby("Pitch")[col].mean().reindex(agg["Pitch"]).values
    return agg.sort_values("N", ascending=False).reset_index(drop=True)


# ── Graphics (mirroring Fordham postgame_or_season_card layout) ───────────────

def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TXT2, which="both", labelsize=11)
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
    txt_on  = readable_text_color(primary)
    card    = pitcher_stats(game_df)
    arsen   = arsenal_table(game_df)

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(BG)

    # ── Header bar ────────────────────────────────────────────────────────────
    hdr = fig.add_axes([0, 0.915, 1, 0.085])
    hdr.set_facecolor(primary); hdr.axis("off")

    # Logo: white background so dark-colored logos always show; accent border for polish
    logo = logo_path_for_team(team_code)
    has_logo = _place_logo(fig, logo, primary, accent, (0.893, 0.915, 0.090, 0.082))

    # Truncate very long names so they don't run into the stat columns
    pitcher_display = pitcher if len(pitcher) <= 24 else pitcher[:23] + "…"
    hdr.text(0.015, 0.80, pitcher_display, color=txt_on, fontsize=26, fontweight="bold",
             transform=hdr.transAxes, va="center")
    conf = TEAM_CONFERENCES.get(team_code, "")
    subtitle = f"{safe_team_name(team_code)}"
    if conf:
        subtitle += f"  ·  {conf}"
    subtitle += f"  ·  {label}"
    if date_str and date_str != label:
        subtitle += f"  ·  {date_str}"
    hdr.text(0.015, 0.24, subtitle, color=accent, fontsize=10, fontweight="bold",
             transform=hdr.transAxes, va="center")

    # Stats: values at y=0.70, labels at y=0.14 — clear vertical separation from name/subtitle
    stat_keys = ["Pitches","IP","K","BB","FB Velo","FB PercVelo","MaxVelo","Stuff+","Loc+","K%","Whiff%","Zone%","CSW%"]
    n_s = len(stat_keys)
    x_end = 0.57 if has_logo else 0.68
    for i, key in enumerate(stat_keys):
        x = 0.30 + i * (x_end / n_s) + (x_end / n_s) / 2
        hdr.text(x, 0.70, fmt(card.get(key), key), color=txt_on,
                 fontsize=14, fontweight="bold", ha="center", va="center",
                 transform=hdr.transAxes)
        hdr.text(x, 0.14, key, color=accent, fontsize=9, fontweight="bold",
                 ha="center", va="center", transform=hdr.transAxes)

    # ── Grid: (6,4) — release col split into release (rows 0-1) + ext (row 2) ──
    ax_move = plt.subplot2grid((6,4), (0,0), rowspan=3, fig=fig)
    ax_lhh  = plt.subplot2grid((6,4), (0,1), rowspan=3, fig=fig)
    ax_rhh  = plt.subplot2grid((6,4), (0,2), rowspan=3, fig=fig)
    ax_rel  = plt.subplot2grid((6,4), (0,3), rowspan=2, fig=fig)
    ax_ext  = plt.subplot2grid((6,4), (2,3), rowspan=1, fig=fig)
    ax_tbl  = plt.subplot2grid((6,4), (3,0), colspan=4, rowspan=2, fig=fig)
    ax_foot = plt.subplot2grid((6,4), (5,0), colspan=4, fig=fig)

    fig.subplots_adjust(top=0.91, bottom=0.02, left=0.04, right=0.97,
                        hspace=0.44, wspace=0.32)

    # Movement
    _style_ax(ax_move)
    throws = game_df["PitcherThrows"].iloc[0] if "PitcherThrows" in game_df.columns else "Right"
    arm_x  = (0,25)  if throws.upper().startswith("R") else (-25,0)
    glv_x  = (-25,0) if throws.upper().startswith("R") else (0,25)
    ax_move.axvspan(*arm_x, facecolor=(0.10,0.30,0.60,0.10))
    ax_move.axvspan(*glv_x, facecolor=(0.60,0.10,0.10,0.10))
    ax_move.axhline(0, color="white", linestyle=":", linewidth=1.2)
    ax_move.axvline(0, color="white", linestyle=":", linewidth=1.2)
    ax_move.set_xlim(-25,25); ax_move.set_ylim(-25,25)
    ax_move.set_aspect("equal", adjustable="box")
    for _, row in game_df.iterrows():
        ax_move.scatter(row.get("HB"), row.get("IVB"), s=45,
                        color=pc(row["Pitch"]), edgecolor="white", linewidth=0.4)
    for pt, g in game_df.groupby("Pitch"):
        cx, cy = g["HB"].mean(), g["IVB"].mean()
        ax_move.scatter(cx, cy, s=260, color=pc(pt), edgecolor="white", linewidth=1.5)
        ax_move.text(cx, cy, pt, color="white", fontsize=12, weight="bold",
                     ha="center", va="center")
    ax_move.set_title("Pitch Movement", color="white", fontsize=15, fontweight="bold")
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
                       s=75, color=pc(row["Pitch"]), edgecolor="white", linewidth=0.4)
    ax_lhh.set_title("vs LHH", color="white", fontsize=15, fontweight="bold")

    # vs RHH
    _draw_zone(ax_rhh)
    rhh = game_df[game_df["BatterSide"].eq("Right")] if "BatterSide" in game_df.columns else game_df
    for _, row in rhh.iterrows():
        ax_rhh.scatter(row.get("PlateLocSide"), row.get("PlateLocHeight"),
                       s=75, color=pc(row["Pitch"]), edgecolor="white", linewidth=0.4)
    ax_rhh.set_title("vs RHH", color="white", fontsize=15, fontweight="bold")

    # Release point
    _style_ax(ax_rel)
    ax_rel.set_xlim(-4,4); ax_rel.set_ylim(3,7)
    ax_rel.set_aspect("equal", adjustable="box")
    if "RelS" in game_df.columns and "RelH" in game_df.columns:
        for _, row in game_df.iterrows():
            ax_rel.scatter(row.get("RelS"), row.get("RelH"),
                           s=30, color=pc(row["Pitch"]), edgecolor="white", linewidth=0.25)
        # centroids
        for pt, g in game_df.groupby("Pitch"):
            ax_rel.scatter(g["RelS"].mean(), g["RelH"].mean(), s=200,
                           color=pc(pt), edgecolor="white", linewidth=1.2, marker="D", zorder=5)
    ax_rel.set_title("Release Point", color="white", fontsize=15, fontweight="bold")
    ax_rel.set_xlabel("Horiz Release", color=TXT2, fontsize=11, fontweight="bold")
    ax_rel.set_ylabel("Height (ft)", color=TXT2, fontsize=11, fontweight="bold")
    ax_rel.invert_xaxis()

    # Extension bar chart
    _style_ax(ax_ext)
    ax_ext.set_facecolor(BG)
    if "Ext" in game_df.columns and "Pitch" in game_df.columns:
        ext_means = game_df.groupby("Pitch")["Ext"].mean().dropna().sort_values(ascending=True)
        pitches   = ext_means.index.tolist()
        vals      = ext_means.values
        y_pos     = range(len(pitches))
        bars = ax_ext.barh(list(y_pos), vals, color=[pc(p) for p in pitches],
                           edgecolor="white", linewidth=0.5, height=0.6)
        for bar, val, pt in zip(bars, vals, pitches):
            ax_ext.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f} ft", color="white", fontsize=11,
                        va="center", fontweight="bold")
        ax_ext.set_yticks(list(y_pos))
        ax_ext.set_yticklabels(pitches, color="white", fontsize=11, fontweight="bold")
        ax_ext.set_xlim(0, max(vals)*1.22 if len(vals) else 8)
        ax_ext.tick_params(colors=TXT2, labelsize=11)
        ax_ext.set_title("Extension", color="white", fontsize=15, fontweight="bold", pad=3)
        ax_ext.set_xlabel("ft", color=TXT2, fontsize=11, fontweight="bold")
        ax_ext.spines[:].set_color("#444")
        ax_ext.grid(axis="x", color="#2a2a2a", linewidth=0.5, alpha=0.7)
    else:
        ax_ext.axis("off")

    # Arsenal table — includes release metrics
    ax_tbl.axis("off")
    if not arsen.empty:
        cols_show = ["Pitch","N","Usage%","Velo","IVB","HB","Spin"]
        for x in ["Stuff+","Loc+","Whiff%","Zone%","CSW%","RelH","RelS","Ext"]:
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
            cell.set_edgecolor("#2a2a2a")
            if r == 0:
                cell.set_facecolor(primary)
                cell.set_text_props(color=txt_on, weight="bold", size=12)
            else:
                pt = view.iloc[r-1]["Pitch"] if r-1 < len(view) else ""
                cell.set_facecolor(pc(pt))
                cell.set_text_props(color="white", weight="bold", size=13)

    # Footer: pitch mix bar
    ax_foot.axis("off")
    if not arsen.empty and "Usage%" in arsen.columns:
        x_cur = 0.0
        for _, row in arsen.iterrows():
            w = row["Usage%"] / 100.0
            if w < 0.001: continue
            ax_foot.add_patch(plt.Rectangle((x_cur,0.45), w, 0.50,
                facecolor=pc(row["Pitch"]), edgecolor=BG, linewidth=0.6,
                transform=ax_foot.transAxes))
            if w > 0.04:
                ax_foot.text(x_cur + w/2, 0.70,
                    f"{row['Pitch']}  {row['Usage%']:.0f}%",
                    transform=ax_foot.transAxes, color="white",
                    ha="center", va="center", fontsize=12, fontweight="bold")
            x_cur += w
    ax_foot.text(0.5, 0.12, "CBBReports  ·  College Baseball Pitching Plus  ·  2026 TrackMan",
                 transform=ax_foot.transAxes, ha="center", color=TXT2, fontsize=10)

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

    # Header band
    ax.add_patch(plt.Rectangle((0,0.78),1,0.22, transform=ax.transAxes,
                                color=primary, zorder=2))

    # Logo: white background so dark-colored logos always show; accent border for polish
    logo = logo_path_for_team(team_code)
    _place_logo(fig, logo, primary, accent, (0.845, 0.808, 0.130, 0.170))

    # Pitcher name + team — stay left, clear of logo
    ax.text(0.03, 0.90, pitcher, transform=ax.transAxes,
            color=txt_on, fontsize=24, fontweight="bold", va="center", zorder=3)
    ax.text(0.03, 0.825, f"{safe_team_name(team_code)}  ·  2026 Season",
            transform=ax.transAxes, color=accent, fontsize=11, fontweight="bold",
            va="center", zorder=3)

    # 16 stat tiles in 2 rows of 8 — sized to fit within x=0.025 to x=0.835
    stat_keys = ["Pitches","Games","IP","K","BB","K%","BB%","BAA",
                 "SLG","Velo","MaxVelo","Stuff+","Loc+","Whiff%","Zone%","CSW%"]
    tw, th = 0.097, 0.145
    for i, key in enumerate(stat_keys):
        ci, ri = i % 8, i // 8
        x = 0.025 + ci*(tw+0.007)
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


# ── Leaderboard ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building leaderboard…")
def build_leaderboard(folder: str, team_codes: tuple, min_pitches: int = 25) -> pd.DataFrame:
    """Load all files for the given teams at once, run models once, aggregate by pitcher."""
    idx = build_index(folder)
    pool = idx[idx["TeamCode"].isin(set(team_codes))]
    if pool.empty:
        return pd.DataFrame()

    # Collect unique deduplicated files for all pitchers in the pool
    all_files = set()
    for files in pool["Files"]:
        all_files.update(files)

    chunks = []
    for path in sorted(all_files):
        try:
            df = pd.read_csv(path, low_memory=False)
            if not {"Pitcher","PitcherTeam"}.issubset(df.columns):
                continue
            mask = df["PitcherTeam"].astype(str).str.strip().isin(set(team_codes))
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


def leaderboard_page(folder: str, all_known: pd.DataFrame):
    st.markdown("### Pitching Leaderboard")
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

    lb = build_leaderboard(folder, team_codes, min_pitches=int(min_p))
    if lb.empty:
        st.warning("No pitchers meet the minimum pitch threshold.")
        return

    # Sort and rank
    if sort_by in lb.columns:
        lb = lb.sort_values(sort_by, ascending=False)
    lb = lb.reset_index(drop=True)
    lb.index = lb.index + 1  # 1-based rank

    show_cols = ["Pitcher","Team"]
    if scope in ("All D1","Conference"):
        show_cols.append("Conference")
    for c in ["Pitches","Games","Stuff+","Loc+","FB Velo","FB PercVelo",
              "MaxVelo","K%","Whiff%","Zone%","CSW%","BAA"]:
        if c in lb.columns:
            show_cols.append(c)

    view = lb[show_cols].copy()
    for col in show_cols:
        if col not in ("Pitcher","Team","Conference"):
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


def hr_leaderboard_section(folder: str, all_known: pd.DataFrame):
    st.markdown("### HR Distance Leaderboard")
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
        hr_df = _hr_leaderboard_national(folder, team_codes)

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

@st.cache_data(show_spinner="Building hitter index…")
def build_hitter_index(folder: str) -> pd.DataFrame:
    # ── Parquet / cloud mode ─────────────────────────────────────────────────
    if not _csv_folder_has_data(folder):
        pq = _load_scouting_parquet()
        if pq.empty or "Batter" not in pq.columns or "BatterTeam" not in pq.columns:
            return pd.DataFrame(columns=["TeamCode","Team","Batter","PA","Files"])
        grp = (pq.dropna(subset=["Batter","BatterTeam"])
                 .assign(Batter=lambda d: d["Batter"].str.strip(),
                         BatterTeam=lambda d: d["BatterTeam"].str.strip())
                 .groupby(["BatterTeam","Batter"], as_index=False)
                 .size().rename(columns={"BatterTeam":"TeamCode","size":"PA"}))
        grp["Files"] = [[] for _ in range(len(grp))]
        grp["Team"]  = grp["TeamCode"].map(safe_team_name)
        return grp

    # ── CSV / local mode ─────────────────────────────────────────────────────
    usecols = ["Batter", "BatterTeam"]
    rows = []
    for path in _unique_csv_files(folder):
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
    return idx


@st.cache_data(show_spinner="Loading hitter data…")
def load_hitter_data(folder: str, team_code: str, batter: str,
                     file_list: tuple) -> pd.DataFrame:
    # ── Parquet / cloud mode ─────────────────────────────────────────────────
    if not _csv_folder_has_data(folder):
        pq = _load_scouting_parquet()
        if pq.empty:
            return pd.DataFrame()
        mask = (pq.get("BatterTeam", pd.Series("", index=pq.index)).astype(str).str.strip() == str(team_code)) & \
               (pq.get("Batter",     pd.Series("", index=pq.index)).astype(str).str.strip() == str(batter))
        chunks = [pq[mask].copy()]
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
    in_z   = (pd.to_numeric(df.get("PlateLocSide",   0), errors="coerce").between(-0.83, 0.83) &
              pd.to_numeric(df.get("PlateLocHeight", 0), errors="coerce").between(1.5, 3.5))
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


def _draw_spray(ax, df, color_by_ev: bool = True):  # noqa: C901
    ax.set_facecolor(BG)
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
                f"{ring}'", color="#bbbbbb", fontsize=9.5, ha="left", va="center", zorder=8)

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
                lbl, color="#aaaaaa", fontsize=11, ha="center", va="center",
                fontweight="bold", alpha=0.80, zorder=7)

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
                    color="#cccccc", fontsize=9.5, ha="left", va="bottom", zorder=11)

    except Exception:
        pass

    ax.set_xlim(-400, 400)
    ax.set_ylim(-40, 450)
    ax.set_title("Spray Chart  —  dots colored by Exit Velocity",
                 color=TXT, fontsize=14, fontweight="bold", pad=6)

    # EV colorbar beneath the spray chart
    try:
        import matplotlib.colors as mcolors2
        ev_cmap2 = mcolors2.LinearSegmentedColormap.from_list("ev2", [
            (0.00, "#0a2e6e"), (0.35, "#5ea3d0"), (0.50, "#787878"),
            (0.70, "#f5a17a"), (1.00, "#8b0000")
        ])
        cax = ax.inset_axes([0.04, -0.05, 0.72, 0.03])
        cb  = plt.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(50, 105), cmap=ev_cmap2),
            cax=cax, orientation="horizontal")
        cb.set_ticks([50, 70, 87, 95, 105])
        cb.ax.set_xticklabels(["50", "70", "87 (avg)", "95 (HH)", "105+"],
                               color=TXT2, fontsize=8.5)
        cb.outline.set_edgecolor("#333333")
        cb.ax.tick_params(colors=TXT2, size=2)
        ax.text(0.78, -0.035, "★ = HR", color=TXT2, fontsize=9,
                transform=ax.transAxes, va="center")
    except Exception:
        pass


def _draw_hitter_zone(ax, df):
    """3×3 strike-zone heatmap: Avg EV colored via savant percentile scale."""
    ax.set_facecolor(BG)
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
                             facecolor=bg_c, edgecolor=BG, lw=2.5, zorder=2))
                ax.text(cx, cy + 0.08, f"{val:.0f}",
                        fontsize=16, fontweight="bold",
                        ha="center", va="center", color=txt_c, zorder=3)
                ax.text(cx, cy - 0.20, f"n={n}",
                        fontsize=8.5, ha="center", va="center",
                        color=txt_c, alpha=0.65, zorder=3)
            else:
                ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                             facecolor="#1a1a2a", edgecolor=BG, lw=2.5, zorder=2))
                ax.text(cx, cy, f"n={n}" if n > 0 else "—",
                        fontsize=9, ha="center", va="center",
                        color="#444466", zorder=3)

    # Strike zone border + grid lines
    ax.plot([-0.83, 0.83, 0.83, -0.83, -0.83],
            [1.5,  1.5,  3.5,  3.5,  1.5],
            color="white", lw=2.5, zorder=4)
    for xg in zx[1:-1]:
        ax.plot([xg, xg], [1.5, 3.5], color="white", lw=0.7, alpha=0.35, zorder=4)
    for yg in zy[1:-1]:
        ax.plot([-0.83, 0.83], [yg, yg], color="white", lw=0.7, alpha=0.35, zorder=4)

    # Zone position labels
    for ci, lbl in enumerate(["Inside", "Middle", "Outside"]):
        ax.text((zx[ci]+zx[ci+1])/2, 3.75, lbl,
                color=TXT2, fontsize=10, ha="center", va="bottom", alpha=0.7)
    for ri, lbl in enumerate(["Low", "Mid", "High"]):
        ax.text(-1.08, (zy[ri]+zy[ri+1])/2, lbl,
                color=TXT2, fontsize=10, ha="right", va="center", alpha=0.7)

    # Colour scale legend (blue=low → red=high, hitter perspective)
    try:
        import matplotlib.colors as mcolors
        cmap_h = mcolors.LinearSegmentedColormap.from_list(
            "hitter", [(p, tuple(c/255 for c in rgb)) for p, rgb in _HITTER_STOPS], N=256)
        cax = ax.inset_axes([0.05, -0.10, 0.90, 0.055])
        norm = plt.Normalize(0, 1)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_h),
                          cax=cax, orientation="horizontal")
        cb.set_ticks([0, 0.5, 1])
        cb.ax.set_xticklabels(["Poor EV", "Avg", "Hard Hit"],
                               color=TXT2, fontsize=9)
        cb.outline.set_edgecolor("#333333")
        cb.ax.tick_params(colors=TXT2, size=2)
    except Exception:
        pass

    ax.set_xlim(-1.65, 1.5)
    ax.set_ylim(0.8, 4.3)
    ax.set_title("Avg Exit Velocity by Zone", color=TXT, fontsize=13,
                 fontweight="bold", pad=6)


def _draw_pitch_breakdown(ax, df, primary, txt_on):  # noqa: C901
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
            if col == "BA": return f"{float(v):.3f}".replace("0.", ".")
            if col == "N":  return str(int(v))
            return f"{float(v):.1f}"

        view = tbl_df.copy()
        for col in view.columns:
            view[col] = view[col].apply(lambda v, c=col: _f(v, c))

        tbl = ax.table(cellText=view.values, colLabels=view.columns,
                       loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)

        col_names = list(view.columns)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#2a2a2a")
            if r == 0:
                cell.set_facecolor(primary)
                cell.set_text_props(color=txt_on, weight="bold", size=10)
            else:
                row_idx  = r - 1
                col_name = col_names[c] if c < len(col_names) else ""
                raw_val  = tbl_df.iloc[row_idx][col_name] if row_idx < len(tbl_df) else np.nan
                if col_name == "Pitch":
                    pt = str(raw_val)[:2].upper()
                    cell.set_facecolor(pc(pt))
                    cell.set_text_props(color="white", weight="bold", size=12)
                elif col_name == "N":
                    cell.set_facecolor("#222222")
                    cell.set_text_props(color=TXT2, weight="normal", size=11)
                elif col_name in ("BA", "Whiff%", "Avg EV"):
                    fc, tc = _hitter_color(col_name, raw_val)
                    cell.set_facecolor(fc); cell.set_text_props(color=tc, weight="bold", size=11)
                else:
                    cell.set_facecolor("#1f1f1f")
                    cell.set_text_props(color=TXT2, weight="normal", size=11)
    except Exception:
        ax.text(0.5, 0.5, "Chart unavailable", color=TXT2,
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Hitter Performance vs Pitch Type", color=TXT,
                 fontsize=13, fontweight="bold", pad=8)


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
            fontsize=20, fontweight="bold", va="top")
    ax.text(0.015, HDR-0.068, subtitle, color=accent,
            fontsize=8.5, fontweight="bold", va="top")

    # Team logo — transparent on header, white bg only if low contrast
    _place_logo(ax, logo, primary, accent,
                (0.865, HDR-0.083, 0.10, 0.078), use_inset=True)

    ax.plot([0.04, 0.96], [SEP, SEP], color="#333344", lw=0.8)
    ax.text(0.735, SEP-0.008, "Value",  color="#666677", fontsize=7.5, ha="left",  va="top")
    ax.text(0.965, SEP-0.008, "Pct",    color="#666677", fontsize=7.5, ha="right", va="top")

    for i, (key, label, fmt_s, val) in enumerate(rows_data):
        cy    = TOP - (i + 0.5) * row_h
        pct   = pct_fn(key, val)
        color = _pct_to_hex_cbb(pct, stops)
        bh    = row_h * 0.54

        ax.text(BX-0.01, cy, label, color="#cccccc", fontsize=10,
                fontweight="bold", ha="right", va="center")
        ax.add_patch(plt.Rectangle((BX, cy-bh/2), BW, bh, facecolor=BAR_BG, zorder=2))
        if pct is not None and pct > 0.005:
            ax.add_patch(plt.Rectangle((BX, cy-bh/2), BW*pct, bh, facecolor=color, zorder=3))
        ax.plot([BX+BW*0.5]*2, [cy-bh/2, cy+bh/2], color="#555566", lw=1.0, zorder=4)

        val_s = fmt_s.format(float(val)) if val is not None and not pd.isna(val) else "—"
        ax.text(BX+BW+0.015, cy, val_s, color="white", fontsize=9.5, ha="left", va="center")
        ax.text(0.975, cy, _pct_label_cbb(pct), color=color,
                fontsize=10, fontweight="bold", ha="right", va="center")

    ax.text(BX,         BOT-0.015, "◀ Poor",        color="#1956a0", fontsize=8, ha="left",   va="top")
    ax.text(BX+BW*0.5, BOT-0.015, "50th pct (avg)", color="#888888", fontsize=8, ha="center", va="top")
    ax.text(BX+BW,     BOT-0.015, "Elite ▶",         color="#8b0000", fontsize=8, ha="right",  va="top")

    out = BytesIO()
    fig.savefig(out, format="png", dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


def build_pitcher_pct_card_cbb(df: pd.DataFrame, pitcher: str, team_code: str) -> bytes:
    """Pitcher percentile card for CBB Plus — red=elite, blue=poor."""
    pc  = df.get("PitchCall","").fillna("").astype(str)
    pr  = df.get("PlayResult","").fillna("").astype(str)
    kbb = df.get("KorBB","").fillna("").astype(str)
    ht  = df.get("TaggedHitType","").fillna("").astype(str)
    ev  = pd.to_numeric(df.get("EV", df.get("ExitSpeed", pd.Series(dtype=float))), errors="coerce")

    swing = pc.isin(["StrikeSwinging","FoulBall","FoulBallNotFieldable","InPlay","InPlayNoOut","InPlayOut"])
    whiff = pc.eq("StrikeSwinging")
    zone  = (pd.to_numeric(df.get("PlateLocSide",0), errors="coerce").between(-0.83,0.83) &
             pd.to_numeric(df.get("PlateLocHeight",0), errors="coerce").between(1.5,3.5))
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
    subtitle = (safe_team_name(team_code) + (f"  ·  {conf}" if conf else "") +
                "  ·  D1 Percentile Rankings  ·  2026")
    return _draw_pct_card(pitcher, subtitle, rows_data,
                          _pitcher_pct_rank_cbb, _HITTER_STOPS,
                          get_team_colors(team_code),
                          logo=logo_path_for_team(team_code))


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
    ax.set_facecolor(BG); ax.axis("off")
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
                    ax.add_patch(plt.Rectangle((x_, y1), w, bh, facecolor=col,
                                               zorder=2, transform=ax.transAxes))
                    if w > 0.06:
                        ax.text(x_ + w/2, y1 + bh/2,
                                f"{lbl}\n{pct*100:.0f}%",
                                color="white", fontsize=9, fontweight="bold",
                                ha="center", va="center", transform=ax.transAxes, zorder=3)
                x_ += w
            ax.text(0.5, y1 + bh + 0.04, "Batted Ball Type",
                    color=TXT2, fontsize=10, fontweight="bold",
                    ha="center", va="bottom", transform=ax.transAxes)

        # ── Row 2 — Pull/Center/Oppo ──────────────────────────────────────────
        bip_dir_mask = bip_mask & dir_.notna()
        bip_dir_n    = bip_dir_mask.sum()
        if bip_dir_n >= 5:
            bip_dir = dir_[bip_dir_mask]
            # For right-handed hitters: Pull < -15°, Center ±15°, Oppo > 15°
            # But we don't have hand info here so use absolute direction
            pull  = (bip_dir < -15).sum() / bip_dir_n
            ctr   = (bip_dir.between(-15, 15)).sum() / bip_dir_n
            oppo  = (bip_dir > 15).sum() / bip_dir_n

            segs2 = [("Pull", pull, "#e05555"), ("Center", ctr, "#ccaa44"), ("Oppo", oppo, "#5599dd")]
            y2 = 0.36; x_ = 0.02
            for lbl, pct, col in segs2:
                w = pct * 0.96
                if w > 0.005:
                    ax.add_patch(plt.Rectangle((x_, y2), w, bh, facecolor=col,
                                               zorder=2, transform=ax.transAxes))
                    if w > 0.06:
                        ax.text(x_ + w/2, y2 + bh/2,
                                f"{lbl}\n{pct*100:.0f}%",
                                color="white", fontsize=9, fontweight="bold",
                                ha="center", va="center", transform=ax.transAxes, zorder=3)
                x_ += w
            ax.text(0.5, y2 + bh + 0.04, "Direction",
                    color=TXT2, fontsize=10, fontweight="bold",
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
        if len(bip_ev) >= 5:
            stats3.append(("Avg EV", f"{bip_ev.mean():.1f}"))
            stats3.append(("HH%",   f"{(bip_ev>=95).mean()*100:.0f}%"))
        for j, (lbl, val) in enumerate(stats3):
            xp = 0.02 + j * (0.96 / max(len(stats3), 1))
            ax.text(xp + 0.096/max(len(stats3),1), 0.15, val,
                    color=TXT, fontsize=12, fontweight="bold",
                    ha="center", va="center", transform=ax.transAxes)
            ax.text(xp + 0.096/max(len(stats3),1), 0.05, lbl,
                    color=TXT2, fontsize=9,
                    ha="center", va="center", transform=ax.transAxes)
    except Exception:
        ax.text(0.5, 0.5, "Profile data\nunavailable", color=TXT2,
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_title("Batted Ball Profile", color=TXT, fontsize=13, fontweight="bold", pad=4)


def build_hitter_summary_png(df: pd.DataFrame, batter: str, team_code: str) -> bytes:  # noqa: C901
    primary, accent = get_team_colors(team_code)
    txt_on = readable_text_color(primary)
    card   = hitter_stats_cbb(df)

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(BG)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = fig.add_axes([0, 0.910, 1, 0.090])
    hdr.set_facecolor(primary)
    hdr.axis("off")

    logo     = logo_path_for_team(team_code)
    has_logo = _place_logo(fig, logo, primary, accent, (0.891, 0.912, 0.092, 0.085))

    # Player name + subtitle
    hdr.text(0.013, 0.73, batter, color=txt_on, fontsize=26, fontweight="bold",
             transform=hdr.transAxes, va="center")
    conf = TEAM_CONFERENCES.get(team_code, "")
    sub  = safe_team_name(team_code) + (f"  ·  {conf}" if conf else "") + "  ·  Hitter Report  ·  2026"
    hdr.text(0.013, 0.22, sub, color=accent, fontsize=10.5, fontweight="bold",
             transform=hdr.transAxes, va="center")

    # Header stat boxes — Baseball Savant-style percentile coloring
    stat_keys = ["PA","AB","H","HR","xHB","BA","OBP","SLG","wOBA","wRC+",
                 "K%","BB%","Avg EV","HH%","Whiff%"]
    x_end = 0.555 if has_logo else 0.665
    step  = x_end / len(stat_keys)
    try:
        for i, key in enumerate(stat_keys):
            x = 0.305 + i * step + step / 2
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
                hdr.text(x, 0.72, disp, color=val_c, fontsize=13, fontweight="bold",
                         ha="center", va="center", transform=hdr.transAxes,
                         bbox=dict(facecolor=bg_c, edgecolor="none",
                                   boxstyle="round,pad=0.18", alpha=0.90))
            else:
                hdr.text(x, 0.72, disp, color=txt_on, fontsize=13, fontweight="bold",
                         ha="center", va="center", transform=hdr.transAxes)
            hdr.text(x, 0.20, key, color=accent, fontsize=8.5, fontweight="bold",
                     ha="center", va="center", transform=hdr.transAxes)
    except Exception:
        pass  # never let header crash prevent panel drawing

    # ── Panels — 4-panel layout ───────────────────────────────────────────────
    # Left:  spray chart (full height below header)
    # Right top:    EV zone heatmap
    # Right mid:    batted ball profile (GB/LD/FB + Pull/Center/Oppo)
    # Right bottom: pitch breakdown table
    ax_spray  = fig.add_axes([0.01, 0.03, 0.48, 0.86])
    ax_zone   = fig.add_axes([0.52, 0.50, 0.46, 0.39])
    ax_prof   = fig.add_axes([0.52, 0.27, 0.46, 0.20])
    ax_tbl    = fig.add_axes([0.52, 0.03, 0.46, 0.21])

    try:
        _draw_spray(ax_spray, df, color_by_ev=True)
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

def _hr_leaderboard_national(folder: str, team_codes: tuple) -> pd.DataFrame:
    rows = []
    for path in _unique_csv_files(folder):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not {"PlayResult","BatterTeam"}.issubset(df.columns):
            continue
        if not df["BatterTeam"].astype(str).str.strip().isin(set(team_codes)).any():
            continue
        hr = df[(df["PlayResult"].astype(str).eq("HomeRun")) &
                (df["BatterTeam"].astype(str).str.strip().isin(set(team_codes)))]
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

@st.cache_data(show_spinner="Building hitting leaderboard…")
def build_hitting_leaderboard(folder: str, team_codes: tuple, min_pa: int = 30) -> pd.DataFrame:
    """Aggregate season hitting stats for all batters on the given teams."""
    team_set = set(team_codes)
    player_chunks: dict[tuple, list] = {}

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


def hitting_leaderboard_section(folder: str, all_known: pd.DataFrame):
    st.markdown("### Hitting Leaderboard")
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
        lb = build_hitting_leaderboard(folder, team_codes, min_pa=int(min_pa))

    if lb.empty:
        st.warning("No batters meet the minimum PA threshold.")
        return

    asc = sort_by in {"K%","Whiff%","Chase%"}
    lb  = lb.sort_values(sort_by, ascending=asc).reset_index(drop=True)
    lb.index = lb.index + 1

    show_cols = ["Batter","Team"]
    if scope in ("All D1","Conference"):
        show_cols.append("Conference")
    for c in ["PA","H","HR","wOBA","wRC+","BA","OBP","SLG","OPS","K%","BB%","Avg EV","HH%","Whiff%","Chase%"]:
        if c in lb.columns:
            show_cols.append(c)

    view = lb[show_cols].copy()
    for col in show_cols:
        if col in {"PA","H","HR","xHB","wRC+"}:
            view[col] = view[col].apply(lambda v: str(int(v)) if not pd.isna(v) else "—")
        elif col == "wOBA":
            view[col] = view[col].apply(lambda v: f"{float(v):.3f}".replace("0.",".")
                                        if not pd.isna(v) else "—")
        elif col not in ("Batter","Team","Conference"):
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

def main():
    inject_style()
    if not check_paywall():
        return

    st.markdown("""
    <div class="cbb-hero">
        <h1>College Baseball Plus</h1>
        <p>Advanced analytics for every pitcher and hitter in the 2026 TrackMan database —
        postgame graphics, season summaries, stat cards, and leaderboards powered by machine learning.</p>
    </div>""", unsafe_allow_html=True)

    folder = data_dir()
    if not folder.exists():
        st.error(f"Data folder not found: {folder}")
        return

    _get_models()  # warm at startup

    index = build_index(str(folder))
    if index.empty:
        st.error("No pitchers found in the TrackMan folder.")
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

    section = st.radio("", ["Pitcher Reports", "Hitter Reports", "Leaderboards"], horizontal=True,
                        label_visibility="collapsed")
    st.markdown("---")

    if section == "Leaderboards":
        lb_tab = st.radio("", ["Pitching Leaderboard", "Hitting Leaderboard", "HR Distance"],
                          horizontal=True, label_visibility="collapsed", key="lb_sub")
        if lb_tab == "HR Distance":
            hr_leaderboard_section(str(folder), all_known)
        elif lb_tab == "Hitting Leaderboard":
            hitting_leaderboard_section(str(folder), all_known)
        else:
            leaderboard_page(str(folder), all_known)
        return

    if section == "Hitter Reports":
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
            h_idx = build_hitter_index(str(folder))
        h_idx = h_idx[h_idx["TeamCode"].eq(h_team)].sort_values(["PA","Batter"], ascending=[False,True])
        with hfd:
            if h_idx.empty:
                st.warning("No hitter data for this team.")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            hitter = st.selectbox("Hitter", h_idx["Batter"].tolist(),
                                  format_func=lambda p: f"{p}  ({int(h_idx.loc[h_idx.Batter==p,'PA'].iloc[0]):,} PA)",
                                  key="h_hitter")
        st.markdown("</div>", unsafe_allow_html=True)

        h_row = h_idx[h_idx["Batter"] == hitter]
        h_files = tuple(h_row["Files"].iloc[0]) if not h_row.empty else ()
        hdf = load_hitter_data(str(folder), h_team, hitter, h_files)
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

    # ── Filters ───────────────────────────────────────────────────────────────
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
        pitcher = st.selectbox(
            "Pitcher", team_rows["Pitcher"].tolist(),
            format_func=lambda p: f"{p}  ({int(team_rows.loc[team_rows.Pitcher==p,'Pitches'].iloc[0]):,})")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    row = all_known[(all_known["TeamCode"]==team_code) & (all_known["Pitcher"]==pitcher)]
    file_list = tuple(row["Files"].iloc[0]) if not row.empty else ()
    df = load_pitcher_data(str(folder), team_code, pitcher, file_list)
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
        st.markdown(
            f'<div class="pitcher-card">'
            f'<p class="pitcher-name">{pitcher}{badge}</p>'
            f'<p class="pitcher-meta">{safe_team_name(team_code)}  ·  2026 Season  ·  '
            f'{int(team_rows.loc[team_rows.Pitcher==pitcher,"Pitches"].iloc[0]):,} pitches tracked</p>'
            f'</div>', unsafe_allow_html=True)

    # ── Key metrics ───────────────────────────────────────────────────────────
    card = pitcher_stats(df)
    stat_keys = ["Pitches","Games","FB Velo","FB PercVelo","MaxVelo",
                 "Stuff+","Loc+","K%","Whiff%","Zone%","CSW%"]
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
            games = (df.groupby("GameID")
                       .agg(Date=("Date","first"), Pitches=("Pitch","count"))
                       .reset_index().sort_values("Date"))
            gid = st.selectbox(
                "Select Game", games["GameID"].astype(str).tolist(),
                format_func=lambda g: (
                    f"{games.loc[games['GameID'].astype(str).eq(g),'Date'].iloc[0]}  ·  "
                    f"{int(games.loc[games['GameID'].astype(str).eq(g),'Pitches'].iloc[0])} pitches"))
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
