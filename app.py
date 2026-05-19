#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import base64
import requests
from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import streamlit as st
import ftplib
import tempfile
import re
import textwrap
from matplotlib.backends.backend_pdf import PdfPages

def figure_to_pdf_bytes(fig):
    """Convert a Matplotlib figure to PDF bytes for download."""
    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()



# ------------------------------------------------------------
# PATHS / IMPORTS
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SCOUTING_DATA_DIR = ROOT / "scouting_2026_trackman"
SCOUTING_PARQUET_1 = ROOT / "scouting_data_1.parquet"
SCOUTING_PARQUET_2 = ROOT / "scouting_data_2.parquet"
PRACTICE_DATA_DIR = ROOT / "practice_data"
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

PITCH_TYPE_COLORS = {
    "FB": "#1f77b4",
    "FF": "#1f77b4",
    "SI": "#17becf",
    "FT": "#17becf",
    "FC": "#ff9f1c",
    "SL": "#e63946",
    "SW": "#b56576",
    "CU": "#7b2cbf",
    "CB": "#7b2cbf",
    "CH": "#2a9d8f",
    "SP": "#2a9d8f",
    "KN": "#CDBFAF",
}

TEAM_CODE_NAME_OVERRIDES = {
    "FOR_RAM": "Fordham Rams",
    "FOR_RAM1": "Fordham Rams",
    "FLA__GAT": "Florida Gators",
    "FLA_GAT": "Florida Gators",
    "TEN_VOL": "Tennessee Volunteers",
    "HAW_WAR": "Hawaii Warriors",
    "VIR_CAV": "Virginia Cavaliers",
    "STE_HAT": "Stetson Hatters",
    "UCLA": "UCLA Bruins",
    "ARK_RAZ": "Arkansas Razorbacks",
    "OLE_REB": "Ole Miss Rebels",
    "ORE_BEA": "Oregon State Beavers",
    "NEB": "Nebraska Cornhuskers",
    "OKL_SOO": "Oklahoma Sooners",
    "MIZ_TIG": "Missouri Tigers",
    "BAY_BEA": "Baylor Bears",
    "BIN_BEA": "Binghamton Bearcats",
    "RUT_SCA": "Rutgers Scarlet Knights",
    "SPU_PEA": "Saint Peter's Peacocks",
    "GON_BUL": "Gonzaga Bulldogs",
    "KAN_JAY": "Kansas Jayhawks",
    "MIN_GOL": "Minnesota Golden Gophers",
    "SAN_DON": "San Francisco Dons",
    "SAN_AZT": "San Diego State Aztecs",
    "ECU_PIR": "East Carolina Pirates",
    "FAU_OWL": "Florida Atlantic Owls",
    "FGCU": "Florida Gulf Coast Eagles",
    "UNC_SEA": "UNCW Seahawks",
    "UNCW": "UNCW Seahawks",
    "VCU_RAM": "VCU Rams",
    "JMU_DUK": "James Madison Dukes",
    "UIC_FLA": "UIC Flames",
    "UAB_BLA": "UAB Blazers",
    "MTSU_BLU": "Middle Tennessee Blue Raiders",
    "ULM_WAR": "ULM Warhawks",
    "WIC_SHO": "Wichita State Shockers",
    "PEP_WAV": "Pepperdine Waves",
    "NOT_IRI": "Notre Dame Fighting Irish",
    "TOW_TIG": "Towson Tigers",
    "TRO_T": "Troy Trojans",
    "CSU_BAK": "Cal State Bakersfield Roadrunners",
    "CAL_BEA": "California Golden Bears",
    "USC_UPS": "USC Upstate Spartans",
    "USC_BEA": "USC Beaufort Sand Sharks",
    "OLD_MON": "Old Dominion Monarchs",
    "IND_SYC": "Indiana State Sycamores",
    "NMS_AGG": "New Mexico State Aggies",
    "UTR_VAQ": "UT Rio Grande Valley Vaqueros",
    "MUR_RAC": "Murray State Racers",
    "SIU_SAL": "Southern Illinois Salukis",
    "SAC_HOR": "Sacramento State Hornets",
    "TAR_TEX": "Tarleton State Texans",
    "DAV_WIL": "Davidson Wildcats",
    "WRI_RAI": "Wright State Raiders",
    "WIN_BUL": "Winthrop Eagles",
    "QUI_BOB": "Quinnipiac Bobcats",
    "COR_BRE": "Cornell Big Red",
    "FLA_COL": "Florida College Falcons",
    "GEO_BUL": "Georgia Bulldogs",
    "GEO_COL": "George Washington Revolutionaries",
    "GEO_PAT": "George Mason Patriots",
    "GEO_COL1": "George Washington Revolutionaries",
    "GEO_COL2": "George Washington Revolutionaries",
    "GEO_EAG": "Georgia Southern Eagles",
    "GEO_FOX": "George Fox Bruins",
    "GEO_GWI": "George Washington Revolutionaries",
    "GEO_HOY": "Georgetown Hoyas",
    "GEO_PAN": "Georgia State Panthers",
    "GEO_SOU": "Georgia Southern Eagles",
    "SAC_PIO": "Sacred Heart Pioneers",
    "ION_GAL": "Iona Gaels",
    "ION_GAE": "Iona Gaels",
    "STM_GAE": "Saint Mary's Gaels",
    "MAR_RED": "Marist Red Foxes",
    "MAR_LIO": "Marymount Lions",
    "LOY_LIO": "Loyola Marymount Lions",
    "WAG_SEA": "Wagner Seahawks",
    "FAI_STA": "Fairfield Stags",
    "MAN_JAS": "Manhattan Jaspers",
    "NIA_EAG": "Niagara Purple Eagles",
    "CAN_GRI": "Canisius Golden Griffins",
    "SIE_SAI": "Siena Saints",
    "SIE_HEI": "Siena Heights Saints",
    "MON_HAW": "Monmouth Hawks",
    "QUI_HAW": "Quincy Hawks",
    "TUS_PIO": "Tusculum Pioneers",
    "STM_RAT": "St. Thomas Rattlers",
    "RIC_SPI": "Richmond Spiders",
    "RHO_RAM": "Rhode Island Rams",
    "DAY_FLY": "Dayton Flyers",
    "LAS_EXP": "La Salle Explorers",
    "LAS_EXS": "La Salle Explorers",
    "SAI_BIL": "Saint Louis Billikens",
    "STL_BIL": "Saint Louis Billikens",
    "SLU_BILL": "Saint Louis Billikens",
    "SBU_BON": "St. Bonaventure Bonnies",
    "STB_BON": "St. Bonaventure Bonnies",
    "JOE_HAW": "Saint Joseph's Hawks",
    "SAI_JOE": "Saint Joseph's Hawks",
    "STJ_HAW": "Saint Joseph's Hawks",
    "DUQ_DUK": "Duquesne Dukes",
    "LOY_RAM": "Loyola Chicago Ramblers",
    "UMASS": "Massachusetts Minutemen",
    "MAS_MIN": "Massachusetts Minutemen",
    "RHI_RAM": "Rhode Island Rams",
    "COL_LION": "Columbia Lions",
    "PIE_LIO": "Piedmont Lions",
    "ALA_LIO": "North Alabama Lions",
    "ARK_LIO": "Arkansas-Fort Smith Lions",
    "SOU_LIO": "Southeastern Louisiana Lions",
    "SAC_DON": "Santa Clara Broncos",
    "LOY_UNI": "Loyola University",
    "MAR_HAW": "Maryland Eastern Shore Hawks",
    "MAR_HAR": "Marian Knights",
    "MAR_BAL": "Mars Hill Lions",
    "MAR_SAI": "Marymount Saints",
    "MAR_TER": "Marian Terriers",
    "MAR_UNI3": "Marian University",
    "SBU_SEA": "Stony Brook Seawolves",
    "STJ_RED": "St. John's Red Storm",
    "DUK_BLU": "Duke Blue Devils",
    "UMASS_RIV": "UMass Lowell River Hawks",
    "VAN_COM": "Vanderbilt Commodores",
    "AKR_ZIP": "Akron Zips",
    "ALA_CRI": "Alabama Crimson Tide",
    "BOC_EAG": "Boston College Eagles",
    "OHIO_BOB": "Ohio Bobcats",
    "PUR_BOI": "Purdue Boilermakers",
    "RIC_OWL": "Rice Owls",
    "SET_PIR": "Seton Hall Pirates",
    "PIT_PAN": "Pitt Panthers",
    "BYU_COU": "BYU Cougars",
    "CIN_BEA": "Cincinnati Bearcats",
    "CLE_TIG": "Clemson Tigers",
    "MIC_SPA": "Michigan State Spartans",
    "MIC_WOL": "Michigan Wolverines",
    "CEN_MIC": "Central Michigan Chippewas",
    "LOU_CAJ": "Louisiana Ragin' Cajuns",
    "LOU_CAR": "Louisville Cardinals",
    "FDU_KNI": "Fairleigh Dickinson Knights",
    "FRE_BUL": "Fresno State Bulldogs",
    "LSU_TIG": "LSU Tigers",
    "MIA_HUR": "Miami Hurricanes",
    "MIL_UNI": "Milwaukee Panthers",
    "MSU_BDG": "Mississippi State Bulldogs",
    "NEW_HAV": "New Haven Chargers",
    "NIU_HUS": "NIU Huskies",
    "ARI_SUN": "Arizona State Sun Devils",
    "AIR_FOR": "Air Force Falcons",
    "ARM_BLA": "Army Black Knights",
    "NAV_MID": "Navy Midshipmen",
    "MEX_LOB": "New Mexico Lobos",
    "XAV_MUS": "Xavier Musketeers",
    "AUB_TIG": "Auburn Tigers",
    "NEV_WOL": "Nevada Wolf Pack",
    "UCF_KNI": "UCF Knights",
    "HOU_COG": "Houston Cougars",
    "YAL_BUL": "Yale Bulldogs",
    "NOR_TAR": "North Carolina Tar Heels",
    "NOR_WOL": "NC State Wolfpack",
    "ORE_DUC": "Oregon Ducks",
    "CCU_BLD": "Central Connecticut Blue Devils",
    "UCO_HUS": "UConn Huskies",
    "CON_HUS": "UConn Huskies",
    "BOS_COL": "Boston College Eagles",
    "DEL_BLU": "Delaware Blue Hens",
    "DEL_STA": "Delaware State Hornets",
    "HOF_PRI": "Hofstra Pride",
    "DRE_DRA": "Drexel Dragons",
    "NOR_HUS": "Northeastern Huskies",
    "WIL_SEA": "UNCW Seahawks",
    "ELON_PHO": "Elon Phoenix",
    "CAM_CAM": "Campbell Camels",
    "CHS_COU": "Charleston Cougars",
    "BRY_BUL": "Bryant Bulldogs",
    "LIU_SHA": "LIU Sharks",
    "MER_WAR": "Merrimack Warriors",
    "MAI_BLA": "Maine Black Bears",
    "ALB_GRE": "UAlbany Great Danes",
    "ALB_DAN": "UAlbany Great Danes",
    "UML_RIV": "UMass Lowell River Hawks",
    "LOW_RIV": "UMass Lowell River Hawks",
    "LEH_MOU": "Lehigh Mountain Hawks",
    "LAF_LEO": "Lafayette Leopards",
    "BUC_BIS": "Bucknell Bison",
    "HOL_CRO": "Holy Cross Crusaders",
    "COL_GAT": "Colgate Raiders",
    "RIDER_BRO": "Rider Broncs",
    "RID_BRO": "Rider Broncs",
    "CAN_GOL": "Canisius Golden Griffins",
    "NIA_PUR": "Niagara Purple Eagles",
    "QUIN_BOB": "Quinnipiac Bobcats",
    "ION_GAE1": "Iona Gaels",
    "SAC_HEA": "Sacred Heart Pioneers",
    "STP_PEA": "Saint Peter's Peacocks",
    "STP_PCO": "Saint Peter's Peacocks",
    "SIE_SAI1": "Siena Saints",
    "OLD_DOM": "Old Dominion Monarchs",
    "CHA_49E": "Charlotte 49ers",
    "LIB_FLA": "Liberty Flames",
    "APP_MOU": "App State Mountaineers",
    "WES_MOU": "West Virginia Mountaineers",
    "WVU_MOU": "West Virginia Mountaineers",
    "GEO_STA": "Georgia State Panthers",
    "COA_CHA": "Coastal Carolina Chanticleers",
    "UMA_AMH": "UMass Amherst Minutemen",
    "UMA_BOS": "UMass Boston Beacons",
    "UMBC_RET": "UMBC Retrievers",
    "UNC_SPA": "UNC Greensboro Spartans",
    "UNL_REB": "UNLV Rebels",
    "USF_BUL": "South Florida Bulls",
    "FLO_SEM": "Florida State Seminoles",
    "HIG_PAN": "High Point Panthers",
    "ABI_WIL": "Abilene Christian Wildcats",
    "ALB_STA": "Alabama State Hornets",
    "ALA_HOR": "Alabama State Hornets",
    "ARA_WIL": "Arizona Wildcats",
    "ARI_WIL": "Arizona Wildcats",
    "ARL_MAV": "UT Arlington Mavericks",
    "ASU_RED": "Arkansas State Red Wolves",
    "AUS_GOV": "Austin Peay Governors",
    "BEL_BRU": "Belmont Bruins",
    "BGS_FAL": "Bowling Green Falcons",
    "BRA_BRA": "Bradley Braves",
    "BUT_BUL": "Butler Bulldogs",
    "CAL_AGO": "UC Davis Aggies",
    "CAL_ANT": "UC Irvine Anteaters",
    "CAL_FUL": "Cal State Fullerton Titans",
    "CAL_LAN": "California Baptist Lancers",
    "CAL_MAT": "CSUN Matadors",
    "CAL_MUS": "Cal Poly Mustangs",
    "CHA_FOR": "Charlotte 49ers",
    "CIT_BUL": "The Citadel Bulldogs",
    "COL_CHA": "Charleston Cougars",
    "CRE_BLU": "Creighton Bluejays",
    "DAL_PAT": "Dallas Baptist Patriots",
    "DAR_GRE": "Dartmouth Big Green",
    "DIX_STE": "Utah Tech Trailblazers",
    "ELO_PHO": "Elon Phoenix",
    "LAF_LEP": "Lafayette Leopards",
    "EMU_EAG": "Eastern Michigan Eagles",
    "ETS_BUC": "ETSU Buccaneers",
    "EVA_ACE": "Evansville Purple Aces",
    "GIT_YEL": "Georgia Tech Yellow Jackets",
    "GRA_CAN": "Grand Canyon Lopes",
    "HBU_HUS": "Houston Christian Huskies",
    "HOU_COU": "Houston Cougars",
    "ILL_ILL": "Illinois Fighting Illini",
    "ILL_RED": "Illinois State Redbirds",
    "IU": "Indiana Hoosiers",
    "KAN_WIL": "Kansas State Wildcats",
    "KEN_OWL": "Kennesaw State Owls",
    "KEN_WIL": "Kentucky Wildcats",
    "LAM_CAR": "Lamar Cardinals",
    "LIP_BIS": "Lipscomb Bisons",
    "LIT_TRO": "Little Rock Trojans",
    "LOU_BUL": "Louisiana Tech Bulldogs",
    "MAR_THU": "Marshall Thundering Herd",
    "MCN_COW": "McNeese Cowboys",
    "MIS_BEA": "Missouri State Bears",
    "MT": "Middle Tennessee Blue Raiders",
    "NEW_PRI": "New Orleans Privateers",
    "NIC_COL": "Nicholls Colonels",
    "NOF_OSP": "North Florida Ospreys",
    "NOR_DEM": "Northwestern State Demons",
    "NOR_AGG": "North Carolina A&T Aggies",
    "OAK_GOL": "Oakland Golden Grizzlies",
    "OKL_COW": "Oklahoma State Cowboys",
    "ORA_GOL": "Oral Roberts Golden Eagles",
    "OSU_BUC": "Ohio State Buckeyes",
    "PAC_TIG": "Pacific Tigers",
    "PEN_NIT": "Penn State Nittany Lions",
    "PEN_QUA": "Penn Quakers",
    "POR_PIL": "Portland Pilots",
    "PRE_BLH": "Presbyterian Blue Hose",
    "PRI_TIG": "Princeton Tigers",
    "QUN_RYL": "Queens Royals",
    "RAD_HIG": "Radford Highlanders",
    "SAL_JAG": "South Alabama Jaguars",
    "SAN_BAR1": "UC Santa Barbara Gauchos",
    "SAN_BRO": "Santa Clara Broncos",
    "SAN_GAU": "UC Santa Barbara Gauchos",
    "SAN_TOR": "San Diego Toreros",
    "SOU_COU": "SIUE Cougars",
    "SOU_GOL": "Southern Miss Golden Eagles",
    "SOU_IND16": "Southern Indiana Screaming Eagles",
    "SOU_JAG": "Southern Jaguars",
    "SOU_MIS": "Southern Miss Golden Eagles",
    "SOU_RED": "Southeast Missouri Redhawks",
    "STA_CAR": "Stanford Cardinal",
    "TCU_HFG": "TCU Horned Frogs",
    "TEX_BOB": "Texas State Bobcats",
    "TEX_LON": "Texas Longhorns",
    "TEX_RAI": "Texas Tech Red Raiders",
    "TUL_GRE": "Tulane Green Wave",
    "UTS_ROA": "UTSA Roadrunners",
    "VAL_BLA": "Valparaiso Beacons",
    "VAL_CRU": "Valparaiso Beacons",
    "VIL_WIL": "Villanova Wildcats",
    "VIR_KEY": "VMI Keydets",
    "VIR_TEC": "Virginia Tech Hokies",
    "WAK_DEA": "Wake Forest Demon Deacons",
    "WAS_HUS": "Washington Huskies",
    "WM_TRI": "William & Mary Tribe",
    "WOF_TER": "Wofford Terriers",
    "YOU_HAR": "Youngstown State Penguins",
    "YSU_PEN": "Youngstown State Penguins",
    # ── D1 alt codes & newly identified ──────────────────────────────────────
    "BEL_COL":"Bellarmine Knights","BEL_KNI":"Bellarmine Knights",
    "BRO_COL":"Brown Bears","BRO_BEA":"Brown Bears",
    "BUT_COL1":"Butler Bulldogs","BRY_STR1":"Bryant Bulldogs",
    "CAM_UNI":"Campbell Camels","CAM_UNI1":"Campbell Camels","CAL_POL1":"Cal Poly Mustangs",
    "CCU_BLD":"Central Connecticut State Blue Devils","CEN_COL1":"Central Connecticut State Blue Devils",
    "COP_STA":"Coppin State Eagles","DEL_HOR":"Delaware State Hornets",
    "EC":"East Carolina Pirates","GAR_RUN":"Gardner-Webb Runnin' Bulldogs",
    "HOW_HAW":"Howard Bison","KEN_STA1":"Kennesaw State Owls",
    "LIN_UNI":"Lindenwood Lions","LON_ISL22":"Long Island University Sharks",
    "LON_DIR":"Long Beach State Dirtbags","MER_UNI":"Merrimack Warriors",
    "MIS_BEA":"Missouri State Bears","MIS_DEL1":"Mississippi Valley State Delta Devils",
    "MIS_ST.":"Missouri State Bears","MT":"Memphis Tigers",
    "NCA_BUL":"North Carolina A&T Aggies","NIC_COL1":"Nicholls Colonels",
    "NOR_CAT":"Northwestern Wildcats","NOR_IOW2":"Northern Iowa Panthers",
    "PRA_ACA":"Prairie View A&M Panthers","PRA_PAN":"Prairie View A&M Panthers","PRA_PRA1":"Prairie View A&M Panthers",
    "SAD_GAU":"UC Santa Barbara Gauchos","SOU_ILL":"Southern Illinois Salukis",
    "SOU_SOU8":"Southern Jaguars","STE_MUS":"Stephen F. Austin Lumberjacks",
    "STO_COL":"Stonehill Skyhawks",
    "TEX_A&M":"Texas A&M Aggies","TEX_A&M1":"Texas A&M Aggies","WMI_BRO":"Western Michigan Broncos",
    # ── D2 / D3 / NAIA / JUCO ────────────────────────────────────────────────
    "AND_TRO":"Anderson University Trojans","AVE_MAR":"Ave Maria University Gyrenes",
    "BAR_COL":"Barry University Buccaneers","BIO_UNI":"Biola University Eagles",
    "CAR_EAG":"Carson-Newman Eagles","CED_UNI":"Cedarville University Yellow Jackets",
    "CEN_COL":"Central College Dutch","EAS_TEX":"East Texas Baptist University Tigers",
    "ERS_COL":"Erskine College Flying Fleet","FER_COL":"Ferris State University Bulldogs",
    "FRA_MAR1":"Franklin & Marshall Diplomats","FRE_PAC":"Fresno Pacific University Sunbirds",
    "GAS_COL":"Gadsden State Fighting Cardinals","GAD_STA":"Gadsden State Fighting Cardinals",
    "GEO_FOX":"George Fox University Bruins","GUL_COM":"Gulf Coast State College Commodores",
    "HEN_COL":"Henderson State Reddies","HIL_COL2":"Hill College Rebels",
    "HUT_COM":"Hutchinson CC Blue Dragons","IOW_CEN":"Iowa Central CC Tritons",
    "ILL_WES":"Illinois Wesleyan University Titans",
    "ITA_ITA":"Italy National Team","JOH_LOG":"John A. Logan College Volunteers",
    "JOH_UNI":"Johns Hopkins University Blue Jays","JON_COL":"Jones County Junior College Bobcats",
    "JUD":"Judson University Eagles","KIN_UNI":"King University Tornado",
    "LAN_BEA":"Lane College Dragons","LEE_UNI":"Lee University Flames",
    "LEN_BEA":"Lenoir-Rhyne University Bears","LIN_MEM":"Lincoln Memorial University Railsplitters",
    "LIN_UNI2":"Limestone University Saints","LOW_COL":"Lower Columbia College Red Devils",
    "MAR_LIO":"Marion University Flying Knights","MIS_COL1":"Missouri S&T Miners",
    "MOU_OLV":"Mount Olive University Trojans","NAV_COL":"Navarro College Bulldogs",
    "NCB":"Northwestern College Eagles","NEW_HAV":"New Haven Chargers",
    "NOR_GEO3":"North Georgia University Nighthawks","NOR_GRE":"North Greenville University Crusaders",
    "NOV_SOU":"Nova Southeastern University Sharks","ODE_COL":"Odessa College Wranglers",
    "ORA_COA":"Orange Coast College Pirates","PAN_COL":"Panola College Ponies",
    "PAR_JUN":"Paris Junior College Dragons","PEA_RIV":"Pearl River Community College Wildcats",
    "POI_LOM":"Point Loma Nazarene Sea Lions","QUI_HAW":"Quincy University Hawks",
    "REI_UNI":"Reinhardt University Eagles","ROG_WIL":"Roger Williams University Hawks",
    "SAG_VAL":"Saginaw Valley State Cardinals","SAN_JAC":"San Jacinto College Ravens",
    "SET_HIL":"Seton Hill University Griffins",
    "SHE_UNI":"Shenandoah University Hornets","SHE_UNI1":"Shenandoah University Hornets",
    "SHI_UNI":"Shippensburg University Raiders","SLC_CCB":"Salt Lake Community College Bruins",
    "SLI_ROC":"Slippery Rock University The Rock",
    "SOU_ARK":"Southern Arkansas Muleriders","SOU_ARK2":"Southern Arkansas Muleriders",
    "SOU_ORE":"Southern Oregon University Raiders",
    "SOU_WES":"Southwestern University Pirates","SOU_WES1":"Southwestern Adventist Eagles",
    "STM_RAT":"St. Mary's University Rattlers","TEM_LEO":"Temple College Leopards",
    "TEN_WES":"Tennessee Wesleyan University Bulldogs","TEX_LUT":"Texas Lutheran University Bulldogs",
    "TJC_APA":"Tyler Junior College Apaches","TNU":"Trevecca Nazarene University Trojans",
    "TRI_TIG":"Trinity University Tigers","TUF_UNI":"Tufts University Jumbos",
    "TUS_PIO":"Tusculum University Pioneers","TUS_TUS":"Tusculum University Pioneers",
    "UNI_FIN":"University of Findlay Oilers","UNI_MON":"University of Montevallo Falcons",
    "UNC_PEM":"UNC Pembroke Braves","VIR_WIS":"Virginia Wesleyan University Marlins",
    "Wal_Sen":"Walters State CC Senators","WAL_WAL4":"Walters State CC Senators",
    "WAR_UNI1":"Wartburg College Knights","WES_COL":"Westminster College Titans",
    "WES_FLO5":"West Florida Argonauts",
    "WES_TEX":"West Texas A&M Buffs","WES_TEX1":"West Texas A&M Buffs",
    "WIL_CAR":"William Carey University Crusaders","WIL_JEW":"William Jewell College Cardinals",
    "WOR_POL":"Worcester Polytechnic Institute Engineers","WOU_WOL":"Western Oregon University Wolves",
    # ── Synced from CBB+ ─────────────────────────────────────────────────────
    "ALA_ANM":"Alabama A&M Bulldogs","CHI_STA":"Chicago State Cougars",
    "DEN_UNI":"Denver Pioneers","EIU_PAN":"Eastern Illinois Panthers",
    "EKU_COL":"Eastern Kentucky Colonels","IOW_HAW":"Iowa Hawkeyes",
    "JAC_STA":"Jacksonville State Gamecocks","JAC_TIG":"Jackson State Tigers",
    "LON_BEA":"Long Beach State Dirtbags","NOR_TEX":"North Texas Mean Green",
    "OHI_BOB":"Ohio Bobcats","SOU_GAM":"South Carolina Gamecocks",
    "TEX_AGG":"Texas A&M Aggies","UTA_STA":"Utah State Aggies",
    "UTA_UTE":"Utah Utes","UTM_SKY":"UT Martin Skyhawks",
}

TEAM_COLOR_OVERRIDES = {
    "FOR_RAM": ("#8C1515", "#C7A45D"),
    "FOR_RAM1": ("#8C1515", "#C7A45D"),
    "OSU_BUC": ("#BB0000", "#D4D4D4"),   # Ohio State Scarlet & Silver
    "FLA__GAT": ("#0021A5", "#FA4616"),
    "FLO_SEM": ("#782F40", "#CEB888"),       # Florida State Garnet & Gold
    "FLA_GAT": ("#0021A5", "#FA4616"),
    "TEN_VOL": ("#FF8200", "#58595B"),
    "VIR_CAV": ("#232D4B", "#F84C1E"),
    "UCLA": ("#2774AE", "#FFD100"),
    "VAN_COM": ("#000000", "#B3A369"),   # Vanderbilt Black & Gold
    "AKR_ZIP": ("#041E42", "#A89968"),   # Akron Navy & Gold
    "ALA_CRI": ("#9E1B32", "#FFFFFF"),   # Alabama Crimson & White
    "BOC_EAG": ("#8A0000", "#C9A84C"),
    "ARK_RAZ": ("#9D2235", "#FFFFFF"),
    "OLE_REB": ("#CE1126", "#14213D"),
    "ORE_BEA": ("#DC4405", "#000000"),
    "NEB": ("#E41C38", "#FFFFFF"),
    "MIZ_TIG": ("#F1B82D", "#000000"),
    "BAY_BEA": ("#154734", "#FFB81C"),
    "BIN_BEA": ("#005A43", "#FFFFFF"),
    "RUT_SCA": ("#CC0033", "#5F6A72"),
    "SPU_PEA": ("#003DA5", "#FFFFFF"),
    "GON_BUL": ("#041E42", "#C8102E"),
    "KAN_JAY": ("#0051BA", "#E8000D"),
    "MIN_GOL": ("#7A0019", "#FFCC33"),
    "SAN_DON": ("#00543C", "#FDBB30"),
    "SAN_AZT": ("#A6192E", "#000000"),
    "ECU_PIR": ("#592A8A", "#FDC82F"),
    "FAU_OWL": ("#003366", "#CC0000"),
    "FGCU": ("#002D72", "#007A33"),
    "UNC_SEA": ("#006666", "#CBA052"),
    "STE_HAT": ("#006747", "#B9975B"),
    "UNCW": ("#006666", "#CBA052"),
    "VCU_RAM": ("#FFB300", "#000000"),
    "JMU_DUK": ("#450084", "#CBB677"),
    "UIC_FLA": ("#001E62", "#D50032"),
    "UAB_BLA": ("#1E6B52", "#FFC845"),
    "MTSU_BLU": ("#0066CC", "#C0C0C0"),
    "ULM_WAR": ("#840029", "#F1B82D"),
    "WIC_SHO": ("#FFCD00", "#000000"),
    "PEP_WAV": ("#00205C", "#F37021"),
    "NOT_IRI": ("#0C2340", "#C99700"),
    "TOW_TIG": ("#FFBB00", "#000000"),
    "TRO_T": ("#8A2432", "#B3A369"),
    "HAW_WAR": ("#024731", "#A5ACAF"),
    "CSU_BAK": ("#005DAA", "#FFC72C"),
    "CAL_BEA": ("#003262", "#FDB515"),
    "OLD_MON": ("#003057", "#7C878E"),
    "IND_SYC": ("#0142BC", "#FFFFFF"),
    "NMS_AGG": ("#861F41", "#000000"),
    "MUR_RAC": ("#002144", "#ECAC00"),
    "SIU_SAL": ("#720000", "#000000"),
    "SAC_HOR": ("#043927", "#C4B581"),
    "TAR_TEX": ("#4B116F", "#FFFFFF"),
    "DAV_WIL": ("#AC1A2F", "#000000"),
    "WRI_RAI": ("#026937", "#FFCC33"),
    "WIN_BUL": ("#660000", "#FFD200"),
    "QUI_BOB": ("#00205B", "#C8102E"),
    "COR_BRE": ("#B31B1B", "#FFFFFF"),
    "FLA_COL": ("#B9975B", "#002855"),
    "GEO_BUL": ("#BA0C2F", "#000000"),
    "GEO_COL": ("#033C5A", "#AA9868"),
    "GEO_PAT": ("#006633", "#FFCC33"),
    "GEO_COL1": ("#033C5A", "#AA9868"),
    "GEO_COL2": ("#033C5A", "#AA9868"),
    "GEO_EAG": ("#011E41", "#A99260"),
    "GEO_FOX": ("#002F6C", "#C8102E"),
    "GEO_GWI": ("#033C5A", "#AA9868"),
    "GEO_HOY": ("#041E42", "#8D817B"),
    "GEO_PAN": ("#0039A6", "#C60C30"),
    "GEO_SOU": ("#011E41", "#A99260"),
    "GIT_YEL": ("#003057", "#B3A369"),
    "SAC_PIO": ("#CE1141", "#FFFFFF"),
    "ION_GAL": ("#891C2C", "#C8960C"),
    "ION_GAE": ("#891C2C", "#C8960C"),
    "STM_GAE": ("#D80024", "#003A70"),
    "MAR_RED": ("#B31B1B", "#FFFFFF"),
    "MAR_LIO": ("#002F6C", "#C8102E"),
    "LOY_LIO": ("#A50034", "#003B5C"),
    "WAG_SEA": ("#006747", "#FFFFFF"),
    "FAI_STA": ("#C8102E", "#003A70"),
    "MAN_JAS": ("#00703C", "#FFFFFF"),
    "NIA_EAG": ("#4B116F", "#C99700"),
    "CAN_GRI": ("#0C2340", "#FFCC00"),
    "SIE_SAI": ("#006747", "#FFB81C"),
    "SIE_HEI": ("#003A70", "#C99700"),
    "MON_HAW": ("#041E42", "#A7A9AC"),
    "QUI_HAW": ("#800000", "#FFFFFF"),
    "TUS_PIO": ("#F58220", "#000000"),
    "STM_RAT": ("#660000", "#FFD100"),
    "RIC_SPI": ("#990000", "#000066"),
    "RHO_RAM": ("#68ABE8", "#002147"),
    "DAY_FLY": ("#CE1141", "#00539B"),
    "LAS_EXP": ("#00205B", "#FDB515"),
    "LAS_EXS": ("#00205B", "#FDB515"),
    "SAI_BIL": ("#003DA5", "#C8C9C7"),
    "STL_BIL": ("#003DA5", "#C8C9C7"),
    "SLU_BILL": ("#003DA5", "#C8C9C7"),
    "SBU_BON": ("#54261A", "#FDB515"),
    "STB_BON": ("#54261A", "#FDB515"),
    "JOE_HAW": ("#9E1B32", "#A7A8AA"),
    "SAI_JOE": ("#9E1B32", "#A7A8AA"),
    "STJ_HAW": ("#9E1B32", "#A7A8AA"),
    "DUQ_DUK": ("#041E42", "#BA0C2F"),
    "LOY_RAM": ("#8D0034", "#FFC72C"),
    "UMASS": ("#971B2F", "#FFFFFF"),
    "MAS_MIN": ("#971B2F", "#FFFFFF"),
    "RHI_RAM": ("#68ABE8", "#002147"),
    "COL_LION": ("#75AADB", "#FFFFFF"),
    "PIE_LIO": ("#002F6C", "#C8102E"),
    "ALA_LIO": ("#46166B", "#DB9F11"),
    "ARK_LIO": ("#007A33", "#FFD700"),   # Arkansas-Pine Bluff Golden Lions
    "SOU_LIO": ("#006747", "#F1B82D"),
    "SAC_DON": ("#862633", "#FFFFFF"),
    "LOY_UNI": ("#8D0034", "#FFC72C"),
    "MAR_HAW": ("#7A0019", "#000000"),
    "MAR_HAR": ("#003A70", "#C8102E"),
    "MAR_BAL": ("#0033A0", "#FFFFFF"),
    "MAR_SAI": ("#002F6C", "#C8102E"),
    "MAR_TER": ("#E03A3E", "#FFCC00"),   # Maryland Terrapins
    "MAR_THU": ("#00B140", "#000000"),
    "MAR_UNI3": ("#003A70", "#C8102E"),
    "SBU_SEA": ("#990000", "#1F1F1F"),
    "STJ_RED": ("#BA0C2F", "#FFFFFF"),
    "DUK_BLU": ("#012169", "#FFFFFF"),
    "UMASS_RIV": ("#003DA5", "#C0C0C0"),
    "OHIO_BOB": ("#00694E", "#FFFFFF"),      # Ohio Green & White
    "PUR_BOI": ("#CEB888", "#000000"),       # Purdue Gold & Black
    "RIC_OWL": ("#00205B", "#FFFFFF"),       # Rice Blue & White
    "SET_PIR": ("#003366", "#FFFFFF"),       # Seton Hall Blue & White
    "PIT_PAN": ("#003594", "#FFB81C"),       # Pitt Royal Blue & Gold
    "BYU_COU": ("#002255", "#FFFFFF"),       # BYU Navy & White
    "CIN_BEA": ("#E00122", "#000000"),       # Cincinnati Red & Black
    "CLE_TIG": ("#F66733", "#522D80"),       # Clemson Orange & Purple
    "MIC_SPA": ("#18453B", "#FFFFFF"),       # Michigan State Green & White
    "MIC_WOL": ("#00274C", "#FFCB05"),       # Michigan Blue & Maize
    "CEN_MIC": ("#6A0032", "#FFCB05"),       # CMU Maroon & Gold
    "LOU_CAJ": ("#CE181E", "#000000"),       # Louisiana Ragin' Cajuns Red & Black
    "LOU_CAR": ("#AD0000", "#000000"),       # Louisville Red & Black
    "FDU_KNI": ("#0033A0", "#C8102E"),       # FDU Blue & Red
    "FRE_BUL": ("#003A70", "#C41230"),  
    "LSU_TIG": ("#461D7C", "#FDD023"),        # LSU Purple & Gold
    "MIA_HUR": ("#F47321", "#005030"),       # Miami Orange & Green
    "MIL_UNI": ("#000000", "#FFC72C"),       # Milwaukee Black & Gold
    "MSU_BDG": ("#660000", "#FFFFFF"),       # Mississippi State Maroon & White
    "NEW_HAV": ("#0033A0", "#FFCD00"),       # New Haven Blue & Gold
    "NIU_HUS": ("#BA0C2F", "#000000"),       # NIU Red & Black
    "ARI_SUN": ("#8C1D40", "#FFC627"),
    "AIR_FOR": ("#003087", "#8A8D8F"),      # Air Force Blue & Silver
    "ARM_BLA": ("#000000", "#D4AF37"),      # Army Black & Gold
    "NAV_MID": ("#00205B", "#C5B783"),      # Navy Blue & Gold
    "MEX_LOB": ("#BA0C2F", "#000000"),      # New Mexico Red & Black
    "XAV_MUS": ("#0C2340", "#9EA2A2"),      # Xavier Navy & Silver
    "AUB_TIG": ("#0C2340", "#E87722"),
    "NEV_WOL": ("#003366", "#A7A9AC"),        # Nevada Blue & Silver
    "UCF_KNI": ("#000000", "#BA9B37"),        # UCF Black & Gold
    "HOU_COG": ("#C8102E", "#FFFFFF"),        # Houston Red & White
    "HOU_COU": ("#C8102E", "#FFFFFF"),        # Houston Red & White (alt code)
    "YAL_BUL": ("#00356B", "#FFFFFF"),        # Yale Blue & White
    "NOR_TAR": ("#7BAFD4", "#13294B"),        # UNC Carolina Blue & Navy
    "NOR_WOL": ("#CC0000", "#000000"),        # NC State Red & Black
    "ORE_DUC": ("#154733", "#FEE123"),        # Oregon Green & Yellow
    "CCU_BLD": ("#0033A0", "#A7A9AC"),        # CCSU Blue & Silver
    "UCO_HUS": ("#000E2F", "#FFFFFF"),  
    "CON_HUS": ("#000E2F", "#FFFFFF"),
    "BOS_COL": ("#8A0000", "#C9A84C"),
    "DEL_BLU": ("#00539B", "#FFD200"),
    "DEL_STA": ("#EE3124", "#00539B"),
    "HOF_PRI": ("#003591", "#FFB81C"),
    "DRE_DRA": ("#07294D", "#FFC600"),
    "NOR_HUS": ("#CC0000", "#000000"),
    "WIL_SEA": ("#006666", "#CBA052"),
    "ELON_PHO": ("#73000A", "#B59A57"),
    "CAM_CAM": ("#F47920", "#000000"),
    "CHS_COU": ("#73000A", "#000000"),
    "BRY_BUL": ("#000000", "#C8A415"),
    "LIU_SHA": ("#002D6C", "#69BE28"),
    "MER_WAR": ("#002D72", "#FDB515"),
    "MAI_BLA": ("#003263", "#B9975B"),
    "ALB_GRE": ("#46166B", "#EEB211"),
    "ALB_DAN": ("#46166B", "#EEB211"),
    "UML_RIV": ("#003DA5", "#C0C0C0"),
    "LOW_RIV": ("#003DA5", "#C0C0C0"),
    "LEH_MOU": ("#653600", "#FFFFFF"),
    "LAF_LEO": ("#800000", "#FFFFFF"),
    "BUC_BIS": ("#E87722", "#002F6C"),
    "HOL_CRO": ("#602D89", "#FFFFFF"),
    "COL_GAT": ("#821019", "#FFFFFF"),
    "RIDER_BRO": ("#981E32", "#FFFFFF"),
    "RID_BRO": ("#981E32", "#FFFFFF"),
    "CAN_GOL": ("#0C2340", "#FFCC00"),
    "NIA_PUR": ("#4B116F", "#C99700"),
    "QUIN_BOB": ("#00205B", "#C8102E"),
    "ION_GAE1": ("#891C2C", "#C8960C"),
    "SAC_HEA": ("#CE1141", "#FFFFFF"),
    "STP_PEA": ("#003DA5", "#FFFFFF"),
    "STP_PCO": ("#003DA5", "#FFFFFF"),
    "SIE_SAI1": ("#006747", "#FFB81C"),
    "OLD_DOM": ("#003057", "#7C878E"),
    "CHA_49E": ("#005035", "#A49665"),
    "LIB_FLA": ("#002D62", "#C41230"),
    "APP_MOU": ("#000000", "#FFCC00"),
    "WES_MOU": ("#002855", "#EAAA00"),
    "WVU_MOU": ("#002855", "#EAAA00"),
    "GEO_STA": ("#0039A6", "#C60C30"),
    "COA_CHA": ("#006F71", "#A17A2C"),
    "UMA_AMH": ("#971B2F", "#FFFFFF"),       # UMass Maroon & White
    "UMA_BOS": ("#0033A0", "#FFFFFF"),       # UMass Boston Blue & White
    "UMBC_RET": ("#FFCC00", "#000000"),      # UMBC Gold & Black
    "UNC_SPA": ("#003366", "#FFC72C"),       # UNCG Navy & Gold
    "UNL_REB": ("#BA0C2F", "#000000"),       # UNLV Scarlet & Black
    "USF_BUL": ("#006747", "#CFC493"),
    # ── D1 newly identified colors ────────────────────────────────────────────
    "BEL_COL":("#002D62","#C8102E"),"BEL_KNI":("#002D62","#C8102E"),
    "BRO_COL":("#4E3629","#ED1C24"),"BRO_BEA":("#4E3629","#ED1C24"),
    "BUT_COL1":("#13294B","#747F7F"),"BRY_STR1":("#000000","#C8A415"),
    "CAM_UNI":("#F47920","#000000"),"CAM_UNI1":("#F47920","#000000"),
    "CAL_POL1":("#154734","#C8B560"),
    "CCU_BLD":("#003DA5","#FFFFFF"),"CEN_COL1":("#003DA5","#FFFFFF"),
    "COP_STA":("#00205B","#B3A369"),"DEL_HOR":("#002F6C","#EAA221"),
    "EC":("#592A8A","#FDC82F"),"GAR_RUN":("#750000","#000000"),
    "HOW_HAW":("#003A70","#E31837"),"KEN_STA1":("#FDB515","#000000"),
    "LIN_UNI":("#002D6C","#FFD700"),"LON_ISL22":("#002D6C","#69BE28"),
    "LON_DIR":("#000000","#FFD700"),"MER_UNI":("#002D72","#FDB515"),
    "MIS_BEA":("#5E0009","#F1B82D"),"MIS_DEL1":("#006747","#C5B783"),
    "MIS_ST.":("#5E0009","#F1B82D"),"MT":("#002147","#8C8C8C"),
    "NCA_BUL":("#004684","#FFD700"),"NIC_COL1":("#C8102E","#A7A8AA"),
    "NOR_CAT":("#4E2A84","#FFFFFF"),"NOR_IOW2":("#4B116F","#FFCC00"),
    "PRA_ACA":("#4F2D7F","#FFD700"),"PRA_PAN":("#4F2D7F","#FFD700"),"PRA_PRA1":("#4F2D7F","#FFD700"),
    "SAD_GAU":("#003660","#FDD023"),"SOU_ILL":("#720000","#000000"),
    "SOU_SOU8":("#003DA5","#F0C528"),"STE_MUS":("#512888","#FFFFFF"),
    "STO_COL":("#003DA5","#C8A415"),
    "TEX_A&M":("#500000","#FFFFFF"),"TEX_A&M1":("#500000","#FFFFFF"),
    "WMI_BRO":("#6C4023","#B5A167"),
    # ── Synced from CBB+ ─────────────────────────────────────────────────────
    "ALA_ANM": ("#63003C","#F5C518"),   # Alabama A&M Bulldogs
    "ALA_HOR": ("#7F2633","#FFAD00"),   # Alabama State Hornets
    "ARL_MAV": ("#003087","#FF8200"),   # UT Arlington Mavericks
    "AUS_GOV": ("#C8102E","#FFFFFF"),   # Austin Peay Governors
    "BUT_BUL": ("#13294B","#747F7F"),   # Butler Bulldogs
    "CAL_AGO": ("#022851","#DAAA00"),   # UC Davis Aggies
    "CAL_ANT": ("#003764","#FFD200"),   # UC Irvine Anteaters
    "CAL_FUL": ("#00274C","#F47920"),   # Cal State Fullerton Titans
    "CAL_MUS": ("#154734","#C8B560"),   # Cal Poly Mustangs
    "CHI_STA": ("#006747","#FFFFFF"),   # Chicago State Cougars
    "CRE_BLU": ("#005CA9","#FFFFFF"),   # Creighton Bluejays
    "DAL_PAT": ("#003087","#C8102E"),   # Dallas Baptist Patriots
    "DEN_UNI": ("#912727","#C8A032"),   # Denver Pioneers
    "DIX_STE": ("#00853E","#FFFFFF"),   # Utah Tech Trailblazers
    "EIU_PAN": ("#004B98","#9B9EA4"),   # Eastern Illinois Panthers
    "EKU_COL": ("#7D0028","#B59A57"),   # Eastern Kentucky Colonels
    "ELO_PHO": ("#73000A","#B59A57"),   # Elon Phoenix
    "ETS_BUC": ("#041E42","#FFCC00"),   # ETSU Buccaneers
    "GRA_CAN": ("#492E7F","#B59A57"),   # Grand Canyon Lopes
    "HBU_HUS": ("#002D62","#C8102E"),   # Houston Christian Huskies
    "HIG_PAN": ("#6B2D8B","#FFFFFF"),   # High Point Panthers
    "ILL_ILL": ("#E84A27","#13294B"),   # Illinois Fighting Illini
    "IOW_HAW": ("#FFCD00","#000000"),   # Iowa Hawkeyes
    "IU":      ("#990000","#DFBBBB"),   # Indiana Hoosiers
    "JAC_STA": ("#002D62","#C9A240"),   # Jacksonville State Gamecocks
    "JAC_TIG": ("#004B8D","#FFFFFF"),   # Jackson State Tigers
    "KAN_WIL": ("#512888","#D1A82D"),   # Kansas State Wildcats
    "KEN_WIL": ("#0033A0","#FFFFFF"),   # Kentucky Wildcats
    "LIP_BIS": ("#00205B","#C8A84B"),   # Lipscomb Bisons
    "LON_BEA": ("#000000","#FFD700"),   # Long Beach State Dirtbags
    "NOR_AGG": ("#004684","#FFD966"),   # North Carolina A&T Aggies
    "NOR_TEX": ("#00853E","#FFFFFF"),   # North Texas Mean Green
    "OHI_BOB": ("#00694E","#FFFFFF"),   # Ohio Bobcats
    "OKL_COW": ("#FF6600","#000000"),   # Oklahoma State Cowboys
    "PEN_NIT": ("#1E407C","#FFFFFF"),   # Penn State Nittany Lions
    "PEN_QUA": ("#011F5B","#990000"),   # Penn Quakers
    "POR_PIL": ("#6E0E19","#C5B783"),   # Portland Pilots
    "PRI_TIG": ("#E87722","#000000"),   # Princeton Tigers
    "SAL_JAG": ("#00205B","#B9975B"),   # South Alabama Jaguars
    "SAN_GAU": ("#003660","#FDD023"),   # UC Santa Barbara Gauchos
    "SAN_TOR": ("#002147","#A5843B"),   # San Diego Toreros
    "SOU_GAM": ("#73000A","#000000"),   # South Carolina Gamecocks
    "SOU_JAG": ("#003087","#FFD700"),   # Southern Jaguars
    "SOU_MIS": ("#FFC72C","#000000"),   # Southern Miss Golden Eagles
    "STA_CAR": ("#8C1515","#FFFFFF"),   # Stanford Cardinal
    "TCU_HFG": ("#4D1979","#A3A9AC"),   # TCU Horned Frogs
    "TEX_AGG": ("#500000","#FFFFFF"),   # Texas A&M Aggies
    "TEX_BOB": ("#501214","#AC9155"),   # Texas State Bobcats
    "TEX_LON": ("#BF5700","#FFFFFF"),   # Texas Longhorns
    "TEX_RAI": ("#CC0000","#000000"),   # Texas Tech Red Raiders
    "TUL_GRE": ("#006747","#418FDE"),   # Tulane Green Wave
    "UTA_STA": ("#003263","#9EADB5"),   # Utah State Aggies
    "UTA_UTE": ("#CC0000","#000000"),   # Utah Utes
    "UTM_SKY": ("#FF8200","#002147"),   # UT Martin Skyhawks
    "VIR_TEC": ("#630031","#CF4420"),   # Virginia Tech Hokies
    "WAK_DEA": ("#9E7E38","#000000"),   # Wake Forest Demon Deacons
    "WAS_HUS": ("#4B2E83","#E8D3A2"),   # Washington Huskies
    "WOF_TER": ("#CEB888","#000000"),   # Wofford Terriers
}

TEAM_LEAGUE_OVERRIDES = {
    "ABI_WIL": "WAC",
    "ALB_STA": "SWAC",
    "FOR_RAM": "Atlantic 10",
    "FOR_RAM1": "Atlantic 10",
    "VCU_RAM": "Atlantic 10",
    "DAV_WIL": "Atlantic 10",
    "GEO_PAT": "Atlantic 10",
    "GEO_COL": "Atlantic 10",
    "GEO_COL1": "Atlantic 10",
    "GEO_COL2": "Atlantic 10",
    "GEO_GWI": "Atlantic 10",
    "RIC_SPI": "Atlantic 10",
    "RHO_RAM": "Atlantic 10",
    "RHI_RAM": "Atlantic 10",
    "DAY_FLY": "Atlantic 10",
    "LAS_EXP": "Atlantic 10",
    "LAS_EXS": "Atlantic 10",
    "SAI_BIL": "Atlantic 10",
    "STL_BIL": "Atlantic 10",
    "SLU_BILL": "Atlantic 10",
    "SBU_BON": "Atlantic 10",
    "STB_BON": "Atlantic 10",
    "JOE_HAW": "Atlantic 10",
    "SAI_JOE": "Atlantic 10",
    "STJ_HAW": "Atlantic 10",
    "DUQ_DUK": "Atlantic 10",
    "LOY_RAM": "Atlantic 10",
    "UMASS": "MAC",
    "MAS_MIN": "MAC",
    "UMA_AMH": "MAC",
    "UNC_SEA": "CAA",
    "UNCW": "CAA",
    "WIL_SEA": "CAA",
    "HOF_PRI": "CAA",
    "DRE_DRA": "CAA",
    "NOR_HUS": "CAA",
    "ELON_PHO": "CAA",
    "ELO_PHO": "CAA",
    "CAM_CAM": "CAA",
    "CHS_COU": "CAA",
    "TOW_TIG": "CAA",
    "MON_HAW": "CAA",
    "SBU_SEA": "CAA",
    "NIA_EAG": "MAAC",
    "NIA_PUR": "MAAC",
    "CAN_GRI": "MAAC",
    "CAN_GOL": "MAAC",
    "ION_GAL": "MAAC",
    "ION_GAE": "MAAC",
    "ION_GAE1": "MAAC",
    "FAI_STA": "MAAC",
    "MAR_RED": "MAAC",
    "MAN_JAS": "MAAC",
    "QUI_BOB": "MAAC",
    "QUIN_BOB": "MAAC",
    "RIDER_BRO": "MAAC",
    "RID_BRO": "MAAC",
    "SIE_SAI": "MAAC",
    "SIE_SAI1": "MAAC",
    "SAC_PIO": "MAAC",
    "SAC_HEA": "MAAC",
    "SPU_PEA": "MAAC",
    "STP_PEA": "MAAC",
    "STP_PCO": "MAAC",
    "LEH_MOU": "Patriot League",
    "LAF_LEO": "Patriot League",
    "LAF_LEP": "Patriot League",
    "BUC_BIS": "Patriot League",
    "HOL_CRO": "Patriot League",
    "COL_GAT": "Patriot League",
    "ARM_BLA": "Patriot League",
    "NAV_MID": "Patriot League",
    "BIN_BEA": "America East",
    "ALB_GRE": "America East",
    "ALB_DAN": "America East",
    "MAI_BLA": "America East",
    "UMASS_RIV": "America East",
    "UML_RIV": "America East",
    "LOW_RIV": "America East",
    "UMBC_RET": "America East",
    "BRY_BUL": "America East",
    "NJI_HIG": "America East",
    "MER_WAR": "MAAC",
    "MSM_MTN": "MAAC",
    "LIU_SHA": "America East",
    "WAG_SEA": "NEC",
    "FDU_KNI": "NEC",
    "CCU_BLD": "NEC",
    "NEW_HAV": "NEC",
    "MAR_HAW": "NEC",
    "DEL_BLU": "C-USA",
    "DEL_STA": "NEC",
    "OLD_MON": "Sun Belt",
    "OLD_DOM": "Sun Belt",
    "APP_MOU": "Sun Belt",
    "COA_CHA": "Sun Belt",
    "GEO_SOU": "Sun Belt",
    "GEO_EAG": "Sun Belt",
    "GEO_STA": "Sun Belt",
    "GEO_PAN": "Sun Belt",
    "MAR_THU": "Sun Belt",
    "SOU_GOL": "Sun Belt",
    "TEX_BOB": "Sun Belt",
    "SAL_JAG": "Sun Belt",
    "ASU_RED": "Sun Belt",
    "SOU_MIS": "Sun Belt",
    "ULM_WAR": "Sun Belt",
    "TRO_T": "Sun Belt",
    "TRO_TRJ": "Sun Belt",
    "LOU_CAJ": "Sun Belt",
    "CHA_49E": "American",
    "ECU_PIR": "American",
    "UAB_BLA": "American",
    "WIC_SHO": "American",
    "FAU_OWL": "American",
    "USF_BUL": "American",
    "RIC_OWL": "American",
    "CHA_FOR": "American",
    "TUL_GRE": "American",
    "UTS_ROA": "American",
    "HOU_COG": "Big 12",
    "HOU_COU": "Big 12",
    "BAY_BEA": "Big 12",
    "BYU_COU": "Big 12",
    "CIN_BEA": "Big 12",
    "UCF_KNI": "Big 12",
    "KAN_JAY": "Big 12",
    "TCU_HFG": "Big 12",
    "OKL_COW": "Big 12",
    "TEX_RAI": "Big 12",
    "ARI_WIL": "Big 12",
    "UTA_UTE": "Big 12",
    "WES_MOU": "Big 12",
    "WVU_MOU": "Big 12",
    "KAN_WIL": "Big 12",
    "OKL_SOO": "SEC",
    "FLA__GAT": "SEC",
    "FLA_GAT": "SEC",
    "TEN_VOL": "SEC",
    "ARK_RAZ": "SEC",
    "OLE_REB": "SEC",
    "TEX_LON": "SEC",
    "TEX_AGG": "SEC",
    "TEX_A&M": "SEC",
    "TEX_A&M1": "SEC",
    "KEN_WIL": "SEC",
    "MIZ_TIG": "SEC",
    "ALA_CRI": "SEC",
    "AUB_TIG": "SEC",
    "LSU_TIG": "SEC",
    "MSU_BDG": "SEC",
    "VAN_COM": "SEC",
    "GEO_BUL": "SEC",
    "VIR_CAV": "ACC",
    "NOT_IRI": "ACC",
    "BOC_EAG": "ACC",
    "BOS_COL": "ACC",
    "PIT_PAN": "ACC",
    "CLE_TIG": "ACC",
    "DUK_BLU": "ACC",
    "NOR_TAR": "ACC",
    "NOR_WOL": "ACC",
    "FLO_SEM": "ACC",
    "MIA_HUR": "ACC",
    "WAK_DEA": "ACC",
    "VIR_TEC": "ACC",
    "GIT_YEL": "ACC",
    "LOU_CAR": "ACC",
    "STA_CAR": "ACC",
    "RUT_SCA": "Big Ten",
    "NEB": "Big Ten",
    "MIN_GOL": "Big Ten",
    "MIC_SPA": "Big Ten",
    "MIC_WOL": "Big Ten",
    "PUR_BOI": "Big Ten",
    "UCLA": "Big Ten",
    "ORE_DUC": "Big Ten",
    "USC_UPS": "Big Ten",
    "SOU_TRO": "Big Ten",
    "ILL_ILL": "Big Ten",
    "IOW_HAW": "Big Ten",
    "PEN_NIT": "Big Ten",
    "WAS_HUS": "Big Ten",
    "IU": "Big Ten",
    "OSU_BUC": "Big Ten",
    "ORE_BEA": "Mountain West",
    "GON_BUL": "WCC",
    "SAN_DON": "WCC",
    "SAC_DON": "WCC",
    "PEP_WAV": "WCC",
    "STM_GAE": "WCC",
    "LOY_LIO": "WCC",
    "SAN_BRO": "WCC",
    "PAC_TIG": "WCC",
    "POR_PIL": "WCC",
    "SAN_TOR": "WCC",
    "SAN_DIE22": "WCC",
    "SAN_DIE23": "WCC",
    "JMU_DUK": "Sun Belt",
    "MTSU_BLU": "C-USA",
    "LIB_FLA": "C-USA",
    "NMS_AGG": "C-USA",
    "DAL_PAT": "C-USA",
    "KEN_OWL": "C-USA",
    "LOU_BUL": "C-USA",
    "FLO_PAN": "C-USA",
    "JAC_GAM": "C-USA",
    "MIS_BEA": "C-USA",
    "MT": "C-USA",
    "UTR_VAQ": "WAC",
    "TAR_TEX": "WAC",
    "ARL_MAV": "WAC",
    "DIX_STE": "WAC",
    "SEA_RED": "WAC",
    "UTA_WOL": "WAC",
    "GRA_CAN": "WAC",
    "CAL_LAN": "WAC",
    "SAC_HOR": "WAC",
    "STE_HAT": "ASUN",
    "FGCU": "ASUN",
    "QUN_RYL": "ASUN",
    "EKU_COL": "ASUN",
    "NOF_OSP": "ASUN",
    "NOR_FLO": "ASUN",
    "LIP_BIS": "ASUN",
    "ALA_LIO": "ASUN",
    "AUS_GOV": "ASUN",
    "BEL_ABB": "Conference Carolinas",
    "FLA_COL": "NAIA",
    "HIG_PAN": "Big South",
    "WIN_BUL": "Big South",
    "WIN_EAG": "Big South",
    "CHA_BUC": "Big South",
    "LON_LAN": "Big South",
    "PRE_BLH": "Big South",
    "RAD_HIG": "Big South",
    "WRI_RAI": "Horizon League",
    "OAK_GOL": "Horizon League",
    "YOU_HAR": "Horizon League",
    "YSU_PEN": "Horizon League",
    "UIC_FLA": "Missouri Valley",
    "IND_SYC": "Missouri Valley",
    "MUR_RAC": "Missouri Valley",
    "SIU_SAL": "Missouri Valley",
    "BEL_BRU": "Missouri Valley",
    "BRA_BRA": "Missouri Valley",
    "EVA_ACE": "Missouri Valley",
    "ILL_RED": "Missouri Valley",
    "VAL_BLA": "Missouri Valley",
    "VAL_CRU": "Missouri Valley",
    "CEN_MIC": "MAC",
    "AKR_ZIP": "MAC",
    "OHIO_BOB": "MAC",
    "OHI_BOB": "MAC",
    "NIU_HUS": "MAC",
    "BGS_FAL": "MAC",
    "EMU_EAG": "MAC",
    "MIA_RED": "MAC",
    "TOL_ROC": "MAC",
    "MIL_UNI": "Horizon League",
    "UWM_PAN": "Horizon League",
    "COL_LION": "Ivy League",
    "COR_BRE": "Ivy League",
    "YAL_BUL": "Ivy League",
    "DAR_GRE": "Ivy League",
    "PEN_QUA": "Ivy League",
    "PRI_TIG": "Ivy League",
    "STJ_RED": "Big East",
    "SET_PIR": "Big East",
    "GEO_HOY": "Big East",
    "XAV_MUS": "Big East",
    "UCO_HUS": "Big East",
    "CON_HUS": "Big East",
    "BUT_BUL": "Big East",
    "CRE_BLU": "Big East",
    "VIL_WIL": "Big East",
    "SAN_AZT": "Mountain West",
    "NEV_WOL": "Mountain West",
    "MEX_LOB": "Mountain West",
    "AIR_FOR": "Mountain West",
    "FRE_BUL": "Mountain West",
    "UNL_REB": "Mountain West",
    "CSU_BAK": "Big West",
    "HAW_WAR": "Big West",
    "CAL_AGO": "Big West",
    "CAL_ANT": "Big West",
    "CAL_FUL": "Big West",
    "CAL_MAT": "Big West",
    "CAL_MUS": "Big West",
    "SAN_BAR1": "Big West",
    "SAN_GAU": "Big West",
    "CAL_BEA": "ACC",
    "ARI_SUN": "Big 12",
    "NIC_COL": "Southland",
    "LAM_CAR": "Southland",
    "MCN_COW": "Southland",
    "NOR_DEM": "Southland",
    "HBU_HUS": "Southland",
    "TEX_ISL": "Southland",
    "SOU_LIO": "Southland",
    "NEW_PRI": "Southland",
    "UTM_SKY": "Ohio Valley",
    "TEN_TEC": "Ohio Valley",
    "EIU_PAN": "Ohio Valley",
    "LIT_TRO": "Ohio Valley",
    "MOR_EAG": "Ohio Valley",
    "WIU_LEA": "Ohio Valley",
    "UTS_EAG": "Ohio Valley",
    "SOU_IND16": "Ohio Valley",
    "SOU_COU": "Ohio Valley",
    "SOU_RED": "Ohio Valley",
    "ORA_GOL": "Summit League",
    "UNO_MAV": "Summit League",
    "STM_BOB": "Summit League",
    "STU_BOB": "Summit League",
    "CIT_BUL": "SoCon",
    "ETS_BUC": "SoCon",
    "MER_BEA": "SoCon",
    "SAM_BUL": "SoCon",
    "VIR_KEY": "SoCon",
    "WOF_TER": "SoCon",
    "UNC_SPA": "SoCon",
    "ALA_HOR": "SWAC",
    "ALC_BRA": "SWAC",
    "FLO_RAT": "SWAC",
    "GRA_TIG": "SWAC",
    "MIS_DEL": "SWAC",
    "SOU_JAG": "SWAC",
    "SOU_GAM": "SWAC",
    "JAC_TIG": "SWAC",
    "NOR_AGG": "CAA",
    "COL_CHA": "CAA",
    "COA_COU": "CAA",
    "WM_TRI": "CAA",
    # ── Newly identified D1 conferences ──────────────────────────────────────
    "BEL_COL":"ASUN","BEL_KNI":"ASUN",
    "BRO_COL":"Ivy League","BRO_BEA":"Ivy League",
    "BUT_COL1":"Big East","BRY_STR1":"America East",
    "CAM_UNI":"Big South","CAM_UNI1":"Big South",
    "CAL_POL1":"Big West",
    "CCU_BLD":"NEC","CEN_COL1":"NEC",
    "COP_STA":"MEAC","DEL_HOR":"MEAC","HOW_HAW":"MEAC",
    "EC":"American","EMU_EAG":"MAC","WMI_BRO":"MAC",
    "GAR_RUN":"Big South",
    "KEN_STA1":"C-USA",
    "LIN_UNI":"Ohio Valley",
    "LON_ISL22":"America East",
    "MER_UNI":"NEC",
    "MIS_BEA":"Missouri Valley","MIS_ST.":"Missouri Valley",
    "MIS_DEL1":"SWAC","PRA_ACA":"SWAC","PRA_PAN":"SWAC","PRA_PRA1":"SWAC",
    "SOU_SOU8":"SWAC","TEX_A&M":"SEC","TEX_A&M1":"SEC",
    "NCA_BUL":"CAA",
    "NIC_COL1":"Southland","STE_MUS":"Southland",
    "NOR_CAT":"Big Ten","NOR_IOW2":"Missouri Valley","SOU_ILL":"Missouri Valley",
    "SAD_GAU":"Big West",
    "STO_COL":"NEC",
}

NCAA_D1_BASEBALL_LEAGUES = {
    "ACC", "American", "America East", "ASUN", "Atlantic 10", "Big 12", "Big East",
    "Big South", "Big Ten", "Big West", "CAA", "C-USA", "Horizon League",
    "Independent", "Ivy League", "MAAC", "MAC", "Missouri Valley", "Mountain West",
    "NEC", "Ohio Valley", "Patriot League", "SEC", "SoCon", "Southland", "Summit League",
    "Sun Belt", "SWAC", "WAC", "WCC", "MEAC",
}

TEAM_PREFIX_OVERRIDES = {
    "ALA": "Alabama", "ARK": "Arkansas", "AUS": "Austin", "BAY": "Baylor",
    "CAL": "California", "CSU": "Cal State", "ECU": "East Carolina",
    "FAU": "Florida Atlantic", "FGCU": "Florida Gulf Coast", "FLA": "Florida", "FLO": "Florida",
    "FOR": "Fordham", "GEO": "Georgia", "GON": "Gonzaga", "HOU": "Houston", "IND": "Indiana",
    "IOW": "Iowa", "JMU": "James Madison", "KAN": "Kansas", "KEN": "Kentucky", "MAR": "Marshall",
    "MIN": "Minnesota", "MIS": "Mississippi", "MIZ": "Missouri",
    "MTSU": "Middle Tennessee", "NEB": "Nebraska", "NEV": "Nevada", "NMS": "New Mexico State",
    "NOF": "North Florida", "NOR": "Northern", "NOT": "Notre Dame", "OKL": "Oklahoma",
    "OLE": "Ole Miss", "ORE": "Oregon State", "OSU": "Oklahoma State", "SAM": "Samford",
    "SAN": "San", "SOU": "Southern", "STA": "Stanford", "TEN": "Tennessee", "TEX": "Texas",
    "TRO": "Troy", "UAB": "UAB", "UCLA": "UCLA", "UIC": "UIC", "ULM": "ULM", "USC": "USC", "USF": "South Florida",
    "UTA": "Utah Tech", "UTR": "UT Rio Grande Valley", "VCU": "VCU", "VIR": "Virginia", "UMA": "UMass",
    "WAS": "Washington", "WIC": "Wichita State", "OHIO": "Ohio", "PUR": "Purdue", "RIC": "Rice", "SET": "Seton Hall", "PIT": "Pitt", 
    "BYU": "BYU", "CIN": "Cincinnati", "CLE": "Clemson", "MIC": "Michigan", "ANM": "A&M", "LOU": "Louisiana", "FDU": "Fairleigh Dickinson", "FRE": "Fresno State", "LSU": "LSU", 
    "MIA": "Miami","MIL": "Milwaukee", "MSU": "Mississippi State", "NEW": "New Haven", "NIU": "Northern Illinois", "ARI": "Arizona State", "AIR": "Air Force", "ARM": "Army", "NAV": "Navy",
    "MEX": "New Mexico", "XAV": "Xavier", "AUB": "Auburn", "ORG": "Oregon", "CCU": "Central Connecticut", "UCO": "UConn",
    "CON": "UConn", "BOS": "Boston College", "DEL": "Delaware", "HOF": "Hofstra", "DRE": "Drexel",
    "WIL": "UNCW", "ELON": "Elon", "CAM": "Campbell", "CHS": "Charleston", "BRY": "Bryant",
    "LIU": "LIU", "MER": "Merrimack", "MAI": "Maine", "ALB": "UAlbany", "UML": "UMass Lowell",
    "LOW": "UMass Lowell", "LEH": "Lehigh", "LAF": "Lafayette", "BUC": "Bucknell", "HOL": "Holy Cross",
    "COL": "Colgate", "RID": "Rider", "RIDER": "Rider", "CAN": "Canisius", "NIA": "Niagara",
    "QUIN": "Quinnipiac", "ION": "Iona", "STP": "Saint Peter's", "SAC": "Sacred Heart", "OLD": "Old Dominion",
    "CHA": "Charlotte", "LIB": "Liberty", "APP": "App State", "WVU": "West Virginia", "WES": "West Virginia",



}

TEAM_MASCOT_OVERRIDES = {
    "RAM": "Rams", "GAT": "Gators", "VOL": "Volunteers", "CAV": "Cavaliers", "RAZ": "Razorbacks",
    "REB": "Rebels", "BEA": "Bears", "BUL": "Bulldogs", "TIG": "Tigers", "EAG": "Eagles",
    "OWL": "Owls", "PIR": "Pirates", "BLU": "Blue Raiders", "WAR": "Warriors", "DON": "Dons",
    "AZT": "Aztecs", "DUK": "Dukes", "FLA": "Flames", "SHO": "Shockers",
    "WAV": "Waves", "IRI": "Fighting Irish", "T": "Trojans", "MON": "Monarchs", "SYC": "Sycamores",
    "AGG": "Aggies", "VAQ": "Vaqueros", "RAC": "Racers", "SAL": "Salukis", "HOR": "Hornets", "DUC": "Ducks", "BLD": "Blue Devils", "HUS": "Huskies",
    "TEX": "Texans", "WIL": "Wildcats", "RAI": "Raiders", "GOL": "Golden Eagles", "COM": "Commodores",
    "ZIP": "Zips", "CRI": "Crimson Tide", "BOB": "Bobcats", "BOI": "Boilermakers", "PAN": "Panthers", "COU": "Cougars", "SPA": "Spartans", "WOL": "Wolverines",
    "MIC": "Chippewas", "CAJ": "Ragin' Cajuns", "CAR": "Cardinals", "KNI": "Knights", "HUR": "Hurricanes", "UNI": "Panthers", "BDG": "Bulldogs","HAV": "Chargers",
    "SUN": "Sun Devils", "FOR": "Falcons", "BLA": "Black Knights", "MID": "Midshipmen", "LOB": "Lobos", "MUS": "Musketeers",
    "PRI": "Pride", "DRA": "Dragons", "SEA": "Seahawks", "PHO": "Phoenix", "CAM": "Camels", "SHA": "Sharks",
    "GRE": "Great Danes", "DAN": "Great Danes", "RIV": "River Hawks", "MOU": "Mountain Hawks",
    "LEO": "Leopards", "BIS": "Bison", "CRO": "Crusaders", "BRO": "Broncs",
    "PUR": "Purple Eagles", "PCO": "Peacocks", "49E": "49ers",

}

_TEAM_TAG_D1_NAME_UPDATES = {
    "NEW_PRI": "New Orleans Privateers",
    "TRO_TRJ": "Troy Trojans",
    "LOU_BUL": "Louisiana Tech Bulldogs",
    "UTR_VAQ": "UT Rio Grande Valley Vaqueros",
    "MER_BEA": "Mercer Bears",
    "MT": "Middle Tennessee Blue Raiders",
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
}

_TEAM_TAG_D1_LEAGUE_UPDATES = {
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
}

_TEAM_TAG_D1_COLOR_UPDATES = {
    "NEW_PRI": ("#005EB8", "#C99700"),
    "TRO_TRJ": ("#8A2432", "#B3A369"),
    "LOU_BUL": ("#E31B23", "#003DA5"),
    "UTR_VAQ": ("#F15A22", "#005CB9"),
    "MER_BEA": ("#F76800", "#000000"),
    "MT": ("#0066CC", "#C0C0C0"),
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
}

TEAM_CODE_NAME_OVERRIDES.update(_TEAM_TAG_D1_NAME_UPDATES)
TEAM_LEAGUE_OVERRIDES.update(_TEAM_TAG_D1_LEAGUE_UPDATES)
TEAM_COLOR_OVERRIDES.update(_TEAM_TAG_D1_COLOR_UPDATES)

SCOUT_LOGO_ALIASES = {
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
    "MT": "MTSU_BLU",
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


def team_display_name(team_code: str, include_code=False) -> str:
    code = str(team_code or "").strip()
    if not code:
        return "Unknown Team"
    normalized = re.sub(r"_+", "_", code.upper())
    name = TEAM_CODE_NAME_OVERRIDES.get(code.upper()) or TEAM_CODE_NAME_OVERRIDES.get(normalized)
    if not name:
        parts = [p for p in normalized.split("_") if p]
        translated = []
        for idx, part in enumerate(parts):
            if idx == 0:
                translated.append(TEAM_PREFIX_OVERRIDES.get(part, part.title()))
            else:
                translated.append(TEAM_MASCOT_OVERRIDES.get(part, TEAM_PREFIX_OVERRIDES.get(part, part.title())))
        name = " ".join(translated) if translated else code
    return f"{name} ({code})" if include_code and code not in name else name


def _normalize_team_code(team_code: str) -> str:
    return re.sub(r"_+", "_", str(team_code or "").strip().upper())


def _generated_team_colors(team_code: str) -> tuple[str, str]:
    palette = [
        ("#8C1515", "#C7A45D"), ("#003262", "#FDB515"), ("#0057B8", "#FFB81C"),
        ("#552583", "#FDB927"), ("#154734", "#FFB81C"), ("#00205B", "#BA0C2F"),
        ("#006747", "#F2A900"), ("#4B116F", "#A7A8AA"), ("#9D2235", "#000000"),
        ("#0C2340", "#C99700"), ("#004B8D", "#EF3E42"), ("#512D6D", "#FFCD00"),
    ]
    key = sum((i + 1) * ord(ch) for i, ch in enumerate(str(team_code or "")))
    return palette[key % len(palette)]


def team_colors(team_code: str) -> tuple[str, str]:
    code = str(team_code or "").strip().upper()
    normalized = _normalize_team_code(code)
    return TEAM_COLOR_OVERRIDES.get(code) or TEAM_COLOR_OVERRIDES.get(normalized) or _generated_team_colors(code)


def team_league_name(team_code: str) -> str:
    code = str(team_code or "").strip().upper()
    normalized = _normalize_team_code(code)
    return TEAM_LEAGUE_OVERRIDES.get(code) or TEAM_LEAGUE_OVERRIDES.get(normalized) or "Unmapped"


def is_ncaa_d1_baseball_team(team_code: str) -> bool:
    return team_league_name(team_code) in NCAA_D1_BASEBALL_LEAGUES


def readable_text_color(bg_hex: str) -> str:
    bg = str(bg_hex or "#000000").lstrip("#")
    if len(bg) != 6:
        return "#FFFFFF"
    r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#100D0C" if luminance > 0.62 else "#FFFFFF"


def _muted_text_on(hex_color: str) -> str:
    """Muted secondary text color guaranteed readable against hex_color background."""
    try:
        h = str(hex_color or "#000000").lstrip("#")
        lum = (0.299*int(h[0:2],16) + 0.587*int(h[2:4],16) + 0.114*int(h[4:6],16)) / 255
        return "#444444" if lum > 0.62 else "#CDBFAF"
    except Exception:
        return "#CDBFAF"


def render_team_badge(team_code: str):
    primary, accent = team_colors(team_code)
    text = readable_text_color(primary)
    label = team_display_name(team_code)
    st.markdown(
        f"""
        <div style="
            border:1px solid {accent};
            border-left:8px solid {accent};
            background:{primary};
            color:{text};
            padding:0.75rem 0.9rem;
            border-radius:8px;
            font-weight:800;
            letter-spacing:0;
            margin-top:0.35rem;
            box-shadow:0 10px 26px rgba(0,0,0,0.22);
        ">
            {label}
            <span style="opacity:0.78;font-weight:650;margin-left:0.45rem;">{team_code}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# VISUAL THEME
# ------------------------------------------------------------
def get_logo_b64():
    paths = [
        ROOT / "static" / "rams.png",
        ROOT / "assets" / "rams.png",
        ROOT / "national_pitchingplus_app" / "team_logos" / "FOR_RAM.png",
    ]
    for logo_path in paths:
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

            .fordham-hero-logo {{
                width: 78px;
                height: 78px;
                object-fit: contain;
                flex-shrink: 0;
                filter: drop-shadow(0 4px 12px rgba(0,0,0,0.35));
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
            div[data-testid="stTextInput"] label,
            div[data-testid="stMultiSelect"] label,
            div[data-testid="stCheckbox"] label {{
                color: #F7E9D0;
                font-weight: 750;
            }}

            div[data-testid="stCheckbox"] label span,
            div[data-testid="stCheckbox"] p {{
                color: #FFF8E9 !important;
                font-weight: 720;
            }}

            div[data-baseweb="select"] > div,
            div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
            div[data-testid="stTextInput"] input {{
                background-color: #211C1A !important;
                color: #FFF8E9 !important;
                border-color: rgba(199,164,93,0.38) !important;
            }}

            div[data-testid="stSlider"] label {{
                color: #F7E9D0;
                font-weight: 750;
            }}

            div[data-testid="stMultiSelect"] div,
            div[data-testid="stMultiSelect"] span,
            div[data-testid="stMultiSelect"] input,
            div[data-baseweb="select"] span,
            div[data-baseweb="select"] input {{
                color: #FFF8E9 !important;
                caret-color: #FFF8E9 !important;
            }}

            div[data-testid="stMultiSelect"] div[data-baseweb="select"] {{
                background-color: #211C1A !important;
            }}

            div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div > div {{
                background-color: #211C1A !important;
                color: #FFF8E9 !important;
            }}

            div[data-baseweb="tag"] {{
                background-color: var(--fordham-maroon) !important;
                color: #FFF8E9 !important;
                border: 1px solid rgba(199,164,93,0.58) !important;
            }}

            div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
            div[data-testid="stMultiSelect"] div[data-baseweb="tag"] > span,
            div[data-testid="stMultiSelect"] div[data-baseweb="tag"] > div {{
                background-color: var(--fordham-maroon) !important;
                border-color: rgba(199,164,93,0.58) !important;
            }}

            div[data-baseweb="tag"] *,
            div[data-testid="stMultiSelect"] div[data-baseweb="tag"] *,
            div[data-testid="stMultiSelect"] div[data-baseweb="tag"] span,
            div[data-testid="stMultiSelect"] div[data-baseweb="tag"] svg {{
                color: #FFF8E9 !important;
                fill: #FFF8E9 !important;
                font-weight: 720;
            }}

            div[data-testid="stMultiSelect"] [data-baseweb="tag"],
            div[data-testid="stMultiSelect"] [data-baseweb="tag"] > div,
            div[data-testid="stMultiSelect"] [data-baseweb="tag"] > span,
            div[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
            div[data-testid="stMultiSelect"] [data-baseweb="tag"] button {{
                background: var(--fordham-maroon) !important;
                background-color: var(--fordham-maroon) !important;
                color: #FFF8E9 !important;
                border-color: rgba(199,164,93,0.68) !important;
                opacity: 1 !important;
            }}

            div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
            div[data-testid="stMultiSelect"] [data-baseweb="tag"] path {{
                color: #FFF8E9 !important;
                fill: #FFF8E9 !important;
                opacity: 1 !important;
            }}

            div[data-baseweb="popover"] ul,
            div[role="listbox"] {{
                background-color: #171514 !important;
                border: 1px solid rgba(199,164,93,0.38) !important;
            }}

            div[role="option"] {{
                background-color: #171514 !important;
                color: #FFF8E9 !important;
            }}

            div[role="option"] *,
            div[data-baseweb="popover"] * {{
                color: #FFF8E9 !important;
            }}

            div[role="option"]:hover,
            div[aria-selected="true"] {{
                background-color: var(--fordham-maroon) !important;
                color: #FFF8E9 !important;
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
                background: linear-gradient(180deg, rgba(33,28,26,0.95), rgba(20,17,16,0.95));
                border: 1px solid rgba(199,164,93,0.48);
                border-radius: 12px;
                padding: 0.85rem 1rem 0.65rem 1rem;
                margin-bottom: 1.15rem;
                box-shadow: 0 12px 28px rgba(0,0,0,0.28);
                transition: border-color 0.2s ease, box-shadow 0.2s ease;
            }}

            .nav-panel:hover {{
                border-color: rgba(199,164,93,0.70);
                box-shadow: 0 14px 36px rgba(0,0,0,0.34);
            }}

            /* Chart container — gives figures a Savant-style framed look */
            .chart-frame {{
                border: 1px solid rgba(199,164,93,0.30);
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 8px 24px rgba(0,0,0,0.28);
                margin-bottom: 0.75rem;
            }}

            /* Metric cards — subtle hover lift */
            div[data-testid="metric-container"] {{
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}
            div[data-testid="metric-container"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.32);
            }}

            /* Section header labels */
            .app-section-label {{
                font-size: 0.68rem;
                font-weight: 800;
                letter-spacing: 0.10em;
                text-transform: uppercase;
                color: var(--fordham-gold);
                margin: 0.1rem 0 0.55rem 0.1rem;
                border-left: 3px solid var(--fordham-maroon);
                padding-left: 0.55rem;
            }}

            /* Dataframe table headers */
            thead tr th {{
                background: var(--fordham-maroon) !important;
                color: #FFF7E8 !important;
                font-weight: 700 !important;
                letter-spacing: 0.03em !important;
                font-size: 0.78rem !important;
            }}

            /* Dataframe row hover */
            tbody tr:hover td {{
                filter: brightness(1.15);
                transition: filter 0.12s ease;
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
    st.title("Umpire Scorecard")
    st.caption("Review called-ball and called-strike accuracy from TrackMan plate-location data, with TrackMan team tags translated into readable team names.")

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

    try:
        preview = build_umpire_scorecard_data(selected_game)
    except Exception as exc:
        st.error(f"Unable to read scorecard data: {exc}")
        return

    metrics = preview["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Called Pitches", int(metrics["called_pitches"]))
    m2.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.1f}%")
    m3.metric("Missed Calls", int(metrics["missed_calls"]))
    m4.metric("Net Fordham Benefit", int(metrics["fordham_net"]))
    tag_cols = st.columns(2)
    tag_cols[0].metric("Home Team", team_tag_label(metrics.get("home_team")))
    tag_cols[1].metric("Away Team", team_tag_label(metrics.get("away_team")))

    missed_preview = preview["missed"].copy()
    if missed_preview.empty:
        st.success("No missed called pitches detected with the current zone rule.")
    else:
        st.dataframe(missed_preview.head(25), use_container_width=True, hide_index=True)

    if st.button("Generate Scorecard"):
        out_path, _ = generate_umpire_scorecard(selected_game)
        st.image(str(out_path), caption="Umpire Scorecard", use_container_width=True)
        st.download_button(
            "Download Scorecard PNG",
            data=Path(out_path).read_bytes(),
            file_name=Path(out_path).name,
            mime="image/png",
        )


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
@st.cache_resource(show_spinner=False)
def _load_models_cached():
    """Load ML models once per process — shared across all sessions and pages."""
    return load_models()


def _read_csv_fast(path) -> pd.DataFrame:
    """Read a TrackMan CSV with the fast C engine (comma-separated, latin1)."""
    try:
        return pd.read_csv(path, encoding="latin1")
    except Exception:
        # Fallback for rare malformed files
        return pd.read_csv(path, encoding="latin1", sep=None, engine="python")


@st.cache_data(show_spinner="Loading season data…")
def prepare_data():
    """Load and process all Fordham game CSVs.

    Cached per Streamlit session — first load processes every file once;
    every subsequent call (pitcher switch, PDF, etc.) returns instantly.
    Cache clears automatically on app restart / Streamlit Cloud redeploy.
    """
    DATA_DIR = ROOT / "data"
    csvs = sorted(
        f for f in DATA_DIR.glob("*.csv")
        if f.name.lower() != "pitching_stats.csv"
    )
    if not csvs:
        return pd.DataFrame()

    # Load ML models ONCE for all files
    stuff_model, stuff_league, loc_model, loc_league = _load_models_cached()

    processed = []
    for f in csvs:
        try:
            df = _read_csv_fast(f)
            if "Pitcher" not in df.columns:
                continue
            df = basic_clean(df)
            df = add_flags(df)
            df = add_perceived_velocity(df)
            df = compute_stuffplus(df, stuff_model, stuff_league)
            df = compute_locationplus(df, loc_model, loc_league)
            processed.append(df)
        except Exception:
            continue

    if not processed:
        return pd.DataFrame()
    return pd.concat(processed, ignore_index=True)


def _safe_upload_name(name):
    stem = Path(str(name)).stem
    suffix = Path(str(name)).suffix.lower() or ".csv"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "trackman_upload"
    if suffix != ".csv":
        suffix = ".csv"
    return f"{safe_stem}{suffix}"


def _practice_session_type_from_name(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("bullpen__") or "bullpen" in name:
        return "Bullpen"
    if name.startswith("intersquad__") or "intersquad" in name or "scrimmage" in name:
        return "Intersquad"
    if name.startswith("batting_practice__") or "batting_practice" in name or "bp__" in name:
        return "Batting Practice"
    if name.startswith("practice__") or "practice" in name:
        return "Batting Practice"
    return "Batting Practice"


def _practice_file_label(path: Path) -> str:
    label = path.stem
    for prefix in ["bullpen__", "practice__", "batting_practice__", "bp__", "intersquad__"]:
        if label.lower().startswith(prefix):
            label = label[len(prefix):]
            break
    return label.replace("_", " ")


def _coerce_trackman_dates(df: pd.DataFrame) -> pd.Series:
    for col in ["GameDate", "Date", "LocalDate", "UTCDate", "PitchUID"]:
        if col in df.columns:
            dates = pd.to_datetime(df[col], errors="coerce")
            if dates.notna().any():
                return dates
    return pd.Series(pd.NaT, index=df.index)


def _nonempty_trackman_text(series) -> pd.Series:
    if series is None:
        return pd.Series(False)
    text = series.astype(str).str.strip()
    return text.ne("") & ~text.str.lower().isin(["nan", "none", "null", "undefined"])


def filter_real_trackman_pitch_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Bullpen exports often include numbered blank/setup rows. Keep only actual tracked pitches.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    pitcher_ok = _nonempty_trackman_text(df["Pitcher"]) if "Pitcher" in df.columns else pd.Series(False, index=df.index)
    if "PitcherId" in df.columns:
        pitcher_ok = pitcher_ok | _nonempty_trackman_text(df["PitcherId"])
    batter_ok = _nonempty_trackman_text(df["Batter"]) if "Batter" in df.columns else pd.Series(False, index=df.index)
    if "BatterId" in df.columns:
        batter_ok = batter_ok | _nonempty_trackman_text(df["BatterId"])

    pitch_type_ok = _nonempty_trackman_text(df["TaggedPitchType"]) if "TaggedPitchType" in df.columns else pd.Series(False, index=df.index)
    if "AutoPitchType" in df.columns:
        pitch_type_ok = pitch_type_ok | _nonempty_trackman_text(df["AutoPitchType"])
    if "pitch_abbr" in df.columns:
        pitch_type_ok = pitch_type_ok | _nonempty_trackman_text(df["pitch_abbr"])

    speed_ok = pd.Series(False, index=df.index)
    for col in ["RelSpeed", "Velo", "ZoneSpeed", "Velocity"]:
        if col in df.columns:
            speed_ok = speed_ok | pd.to_numeric(df[col], errors="coerce").between(20, 110)

    location_ok = pd.Series(False, index=df.index)
    if {"PlateLocSide", "PlateLocHeight"}.issubset(df.columns):
        location_ok = (
            pd.to_numeric(df["PlateLocSide"], errors="coerce").notna() &
            pd.to_numeric(df["PlateLocHeight"], errors="coerce").notna()
        )

    call_ok = _nonempty_trackman_text(df["PitchCall"]) if "PitchCall" in df.columns else pd.Series(False, index=df.index)
    contact_ok = pd.Series(False, index=df.index)
    for col in ["ExitSpeed", "EV", "Angle", "LA", "Distance"]:
        if col in df.columns:
            contact_ok = contact_ok | pd.to_numeric(df[col], errors="coerce").notna()

    real_pitch = (pitcher_ok | batter_ok) & (pitch_type_ok | speed_ok | contact_ok) & (speed_ok | location_ok | call_ok | contact_ok)
    return df[real_pitch].copy()


def filter_live_practice_pitches(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only competitive/live practice pitches. Warmups often have pitch traits
    but no hitter, pitch call, or result, so they should not drive reports.
    """
    df = filter_real_trackman_pitch_rows(raw)
    if df.empty:
        return df

    if "PitchSession" in df.columns:
        session = df["PitchSession"].astype(str).str.strip().str.lower()
        live_mask = session.eq("live")
        if live_mask.any():
            return df[live_mask].copy()
        warmup_mask = session.eq("warmup")
        if warmup_mask.any():
            df = df[~warmup_mask].copy()
            if df.empty:
                return df

    batter_ok = pd.Series(False, index=df.index)
    for col in ["Batter", "BatterId", "BatterID"]:
        if col in df.columns:
            batter_ok = batter_ok | _nonempty_trackman_text(df[col])

    live_call_ok = pd.Series(False, index=df.index)
    if "PitchCall" in df.columns:
        pitch_call = df["PitchCall"].astype(str).str.strip()
        non_live_calls = {
            "", "nan", "none", "null", "undefined", "nopitch", "no pitch",
            "automaticball", "automaticstrike", "warmup", "warmuppitch",
        }
        live_call_ok = ~pitch_call.str.lower().isin(non_live_calls)

    play_result_ok = pd.Series(False, index=df.index)
    if "PlayResult" in df.columns:
        play_result_ok = _nonempty_trackman_text(df["PlayResult"])

    korbb_ok = pd.Series(False, index=df.index)
    if "KorBB" in df.columns:
        korbb_ok = _nonempty_trackman_text(df["KorBB"])

    live_pitch = batter_ok | live_call_ok | play_result_ok | korbb_ok
    return df[live_pitch].copy()


def filter_batting_practice_rows(raw: pd.DataFrame) -> pd.DataFrame:
    df = filter_real_trackman_pitch_rows(raw)
    if df.empty:
        return df
    if "PitchSession" in df.columns:
        session = df["PitchSession"].astype(str).str.strip().str.lower()
        live_mask = session.eq("live")
        if live_mask.any():
            df = df[live_mask].copy()
        else:
            df = df[~session.eq("warmup")].copy()
        if df.empty:
            return df
    batter_ok = pd.Series(False, index=df.index)
    for col in ["Batter", "BatterId", "BatterID"]:
        if col in df.columns:
            batter_ok = batter_ok | _nonempty_trackman_text(df[col])
    contact_ok = pd.Series(False, index=df.index)
    for col in ["ExitSpeed", "EV", "Angle", "LA", "Distance", "Direction", "Bearing"]:
        if col in df.columns:
            contact_ok = contact_ok | pd.to_numeric(df[col], errors="coerce").notna()
    call_ok = pd.Series(False, index=df.index)
    if "PitchCall" in df.columns:
        call_ok = df["PitchCall"].astype(str).str.strip().str.lower().isin(["inplay", "inplaynoout", "inplayout"])
    return df[batter_ok & (contact_ok | call_ok)].copy()


def normalize_practice_team_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["PitcherTeam", "BatterTeam", "HomeTeam", "AwayTeam"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().replace({"FOR_RAM1": "FOR_RAM"})
    return out


def _ensure_practice_trackman_columns(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    defaults = {
        "Pitcher": "Batting Practice",
        "PitcherTeam": "Practice",
        "BatterTeam": "Practice",
        "BatterSide": "",
        "PitchCall": "",
        "PlayResult": "",
        "KorBB": "",
        "Balls": 0,
        "Strikes": 0,
        "PlateLocSide": np.nan,
        "PlateLocHeight": np.nan,
        "TaggedPitchType": "",
        "RelSpeed": np.nan,
        "InducedVertBreak": np.nan,
        "HorzBreak": np.nan,
        "SpinRate": np.nan,
        "RelHeight": np.nan,
        "RelSide": np.nan,
        "Extension": np.nan,
        "VertApprAngle": np.nan,
        "HorzApprAngle": np.nan,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    if "TaggedPitchType" in df.columns and "AutoPitchType" in df.columns:
        tagged = df["TaggedPitchType"].astype(str).str.strip()
        df.loc[tagged.eq("") | tagged.str.lower().isin(["nan", "none"]), "TaggedPitchType"] = df["AutoPitchType"]
    for col in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "RelHeight", "RelSide", "Extension", "VertApprAngle", "HorzApprAngle", "PlateLocSide", "PlateLocHeight", "Balls", "Strikes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return normalize_practice_team_tags(df)


def filter_intersquad_at_bats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Batter" not in out.columns:
        return pd.DataFrame()
    batter_ok = _nonempty_trackman_text(out["Batter"])
    action_ok = pd.Series(True, index=out.index)
    if "PitchCall" in out.columns:
        pitch_call = out["PitchCall"].astype(str).str.strip()
        action_ok = action_ok & ~pitch_call.str.lower().isin(["", "nan", "undefined", "nopitch"])
    result_ok = pd.Series(False, index=out.index)
    for col in ["PlayResult", "KorBB"]:
        if col in out.columns:
            result_ok = result_ok | _nonempty_trackman_text(out[col])
    contact_ok = pd.Series(False, index=out.index)
    for col in ["EV", "ExitSpeed", "LA", "Angle", "Distance"]:
        if col in out.columns:
            contact_ok = contact_ok | pd.to_numeric(out[col], errors="coerce").notna()
    action_ok = action_ok | result_ok | contact_ok
    return out[batter_ok & action_ok].copy()


def _practice_hitter_contact_leaderboard(df: pd.DataFrame, group_col="Batter") -> pd.DataFrame:
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()

    work = normalize_hitter_columns(df.copy())
    if "EV" not in work.columns and "ExitSpeed" in work.columns:
        work["EV"] = work["ExitSpeed"]
    if "LA" not in work.columns and "Angle" in work.columns:
        work["LA"] = work["Angle"]
    if "EV" not in work.columns:
        work["EV"] = np.nan
    if "LA" not in work.columns:
        work["LA"] = np.nan

    work["EV"] = pd.to_numeric(work["EV"], errors="coerce")
    work["LA"] = pd.to_numeric(work["LA"], errors="coerce")
    if "Distance" in work.columns:
        work["Distance"] = pd.to_numeric(work["Distance"], errors="coerce")

    contact = work[work["EV"].notna() | work["LA"].notna()].copy()
    rows = []
    for name, g in work.groupby(group_col):
        contact_g = contact[contact[group_col] == name]
        ev = pd.to_numeric(contact_g.get("EV", pd.Series(dtype=float)), errors="coerce")
        la = pd.to_numeric(contact_g.get("LA", pd.Series(dtype=float)), errors="coerce")
        row = {
            group_col: name,
            "Pitches": len(g),
            "BIP": len(contact_g),
            "AvgEV": ev.mean(),
            "MaxEV": ev.max(),
            "HardHit%": (ev >= 95).mean() * 100 if len(ev.dropna()) else np.nan,
            "Barrel%": barrel_mask(ev, la).mean() * 100 if len(contact_g) else np.nan,
            "SweetSpot%": la.between(8, 32).mean() * 100 if len(la.dropna()) else np.nan,
            "AvgLA": la.mean(),
        }
        if "Distance" in contact_g.columns:
            row["AvgDist"] = pd.to_numeric(contact_g["Distance"], errors="coerce").mean()
            row["MaxDist"] = pd.to_numeric(contact_g["Distance"], errors="coerce").max()
        if "pitch_abbr" in g.columns:
            row["Most Seen"] = g["pitch_abbr"].dropna().astype(str).mode().iloc[0] if not g["pitch_abbr"].dropna().empty else ""
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    numeric_cols = [c for c in out.columns if c not in {group_col, "Most Seen"}]
    out[numeric_cols] = out[numeric_cols].round(1)
    return out.sort_values(["BIP", "AvgEV"], ascending=False)


def _practice_pitcher_tracking_leaderboard(df: pd.DataFrame, min_pitches=1) -> pd.DataFrame:
    if df is None or df.empty or "Pitcher" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    for col in ["Velo", "PerceivedVelo", "IVB", "HB", "Spin", "Ext", "RelH", "Stuff+", "Loc+", "PlateLocSide", "PlateLocHeight"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "in_zone" not in work.columns and {"PlateLocSide", "PlateLocHeight"}.issubset(work.columns):
        work["in_zone"] = work["PlateLocSide"].between(-0.83, 0.83) & work["PlateLocHeight"].between(1.5, 3.5)

    agg_map = {
        "Pitches": ("Pitcher", "count"),
        "Velo": ("Velo", "mean") if "Velo" in work.columns else ("Pitcher", "count"),
        "MaxVelo": ("Velo", "max") if "Velo" in work.columns else ("Pitcher", "count"),
        "IVB": ("IVB", "mean") if "IVB" in work.columns else ("Pitcher", "count"),
        "HB": ("HB", "mean") if "HB" in work.columns else ("Pitcher", "count"),
        "Ext": ("Ext", "mean") if "Ext" in work.columns else ("Pitcher", "count"),
    }
    if "Stuff+" in work.columns:
        agg_map["Stuff+"] = ("Stuff+", "mean")
    if "Loc+" in work.columns:
        agg_map["Loc+"] = ("Loc+", "mean")
    if "in_zone" in work.columns:
        agg_map["Zone%"] = ("in_zone", lambda x: x.mean() * 100)

    out = work.groupby("Pitcher").agg(**agg_map).reset_index()
    if "Batter" in work.columns:
        faced = work.groupby("Pitcher")["Batter"].nunique().reset_index(name="Batters")
        out = out.merge(faced, on="Pitcher", how="left")
    if "pitch_abbr" in work.columns:
        primary = (
            work.groupby(["Pitcher", "pitch_abbr"]).size()
            .reset_index(name="PitchN")
            .sort_values(["Pitcher", "PitchN"], ascending=[True, False])
            .drop_duplicates("Pitcher")
            .rename(columns={"pitch_abbr": "Primary Pitch"})
        )
        out = out.merge(primary[["Pitcher", "Primary Pitch"]], on="Pitcher", how="left")

    out = out[out["Pitches"] >= min_pitches].sort_values(["Pitches", "Velo"], ascending=False).reset_index(drop=True)
    if out.empty:
        return out
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out.round(1)


def _practice_pitcher_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Pitcher" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    for col in ["KorBB", "PitchCall", "PlayResult", "OutsOnPlay"]:
        if col not in work.columns:
            work[col] = "" if col != "OutsOnPlay" else 0

    pa = get_pa_endings(work).copy()
    if pa.empty:
        return pd.DataFrame()

    has_outcomes = (
        pa["KorBB"].isin(["Walk", "Strikeout"]).any()
        or pa["PitchCall"].isin(["HitByPitch"]).any()
        or pa["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice", "Error", "Sacrifice"]).any()
    )
    run_col = next((c for c in ["EarnedRuns", "RunsScored", "RunsOnPlay", "Runs"] if c in pa.columns), None)
    if run_col:
        pa[run_col] = pd.to_numeric(pa[run_col], errors="coerce").fillna(0)
    outs_source = pa["OutsOnPlay"] if "OutsOnPlay" in pa.columns else pd.Series(0, index=pa.index)
    pa["OutsOnPlay"] = pd.to_numeric(outs_source, errors="coerce").fillna(0)

    rows = []
    for pitcher, g in pa.groupby("Pitcher"):
        bf = len(g)
        if not has_outcomes:
            rows.append({
                "Pitcher": pitcher, "BF": bf, "IP": np.nan, "ERA": np.nan, "H": np.nan,
                "K": np.nan, "BB": np.nan, "K%": np.nan, "BB%": np.nan,
                "BA": np.nan, "OBP": np.nan, "SLG": np.nan, "OPS": np.nan,
            })
            continue
        bb = g["KorBB"].eq("Walk").sum()
        k = g["KorBB"].eq("Strikeout").sum()
        hbp = g["PitchCall"].eq("HitByPitch").sum()
        sf = g["PlayResult"].eq("Sacrifice").sum()
        singles = g["PlayResult"].eq("Single").sum()
        doubles = g["PlayResult"].eq("Double").sum()
        triples = g["PlayResult"].eq("Triple").sum()
        homers = g["PlayResult"].eq("HomeRun").sum()
        hits = singles + doubles + triples + homers
        tb = singles + 2 * doubles + 3 * triples + 4 * homers
        ab = bf - bb - hbp - sf
        obp_den = ab + bb + hbp + sf
        outs = g["OutsOnPlay"].sum() + k
        ip = outs / 3 if outs else np.nan
        runs = g[run_col].sum() if run_col else np.nan

        rows.append({
            "Pitcher": pitcher,
            "BF": bf,
            "IP": round(ip, 1) if ip == ip else np.nan,
            "ERA": round((runs * 9 / ip), 2) if run_col and ip and ip == ip else np.nan,
            "H": hits,
            "K": k,
            "BB": bb,
            "K%": round(k / bf * 100, 1) if bf else np.nan,
            "BB%": round(bb / bf * 100, 1) if bf else np.nan,
            "BA": round(hits / ab, 3) if ab > 0 else np.nan,
            "OBP": round((hits + bb + hbp) / obp_den, 3) if obp_den > 0 else np.nan,
            "SLG": round(tb / ab, 3) if ab > 0 else np.nan,
            "OPS": round(((hits + bb + hbp) / obp_den) + (tb / ab), 3) if obp_den > 0 and ab > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def _practice_hitter_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Batter" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    for col in ["KorBB", "PitchCall", "PlayResult"]:
        if col not in work.columns:
            work[col] = ""

    pa = get_pa_endings(work).copy()
    if pa.empty:
        return pd.DataFrame()

    has_outcomes = (
        pa["KorBB"].isin(["Walk", "Strikeout"]).any()
        or pa["PitchCall"].isin(["HitByPitch"]).any()
        or pa["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice", "Error", "Sacrifice"]).any()
    )

    rows = []
    for batter, g in pa.groupby("Batter"):
        pa_count = len(g)
        if not has_outcomes:
            rows.append({
                "Batter": batter, "PA": pa_count, "AB": np.nan, "H": np.nan,
                "K": np.nan, "BB": np.nan, "K%": np.nan, "BB%": np.nan,
                "BA": np.nan, "OBP": np.nan, "SLG": np.nan, "OPS": np.nan,
            })
            continue

        bb = g["KorBB"].eq("Walk").sum()
        k = g["KorBB"].eq("Strikeout").sum()
        hbp = g["PitchCall"].eq("HitByPitch").sum()
        sf = g["PlayResult"].eq("Sacrifice").sum()
        singles = g["PlayResult"].eq("Single").sum()
        doubles = g["PlayResult"].eq("Double").sum()
        triples = g["PlayResult"].eq("Triple").sum()
        homers = g["PlayResult"].eq("HomeRun").sum()
        hits = singles + doubles + triples + homers
        tb = singles + 2 * doubles + 3 * triples + 4 * homers
        ab = pa_count - bb - hbp - sf
        obp_den = ab + bb + hbp + sf

        rows.append({
            "Batter": batter,
            "PA": pa_count,
            "AB": ab,
            "H": hits,
            "K": k,
            "BB": bb,
            "K%": round(k / pa_count * 100, 1) if pa_count else np.nan,
            "BB%": round(bb / pa_count * 100, 1) if pa_count else np.nan,
            "BA": round(hits / ab, 3) if ab > 0 else np.nan,
            "OBP": round((hits + bb + hbp) / obp_den, 3) if obp_den > 0 else np.nan,
            "SLG": round(tb / ab, 3) if ab > 0 else np.nan,
            "OPS": round(((hits + bb + hbp) / obp_den) + (tb / ab), 3) if obp_den > 0 and ab > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def get_practice_csv_files():
    PRACTICE_DATA_DIR.mkdir(exist_ok=True)
    return sorted(PRACTICE_DATA_DIR.glob("*.csv"))


def _push_practice_file_to_github(file_path: Path, content_bytes: bytes) -> bool:
    """Push a practice CSV to GitHub so it survives app restarts on Streamlit Cloud."""
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo  = st.secrets.get("GITHUB_REPO",  "")
        if not token or not repo:
            return False
        relative = f"practice_data/{file_path.name}"
        url = f"https://api.github.com/repos/{repo}/contents/{relative}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        existing = requests.get(url, headers=headers, timeout=10)
        sha = existing.json().get("sha") if existing.status_code == 200 else None
        payload = {
            "message": f"Add practice data: {file_path.name}",
            "content": base64.b64encode(content_bytes).decode(),
        }
        if sha:
            payload["sha"] = sha
        resp = requests.put(url, json=payload, headers=headers, timeout=15)
        return resp.status_code in (200, 201)
    except Exception:
        return False


def save_practice_uploads(uploaded_files, session_type: str, session_label: str = ""):
    PRACTICE_DATA_DIR.mkdir(exist_ok=True)
    saved = []
    prefix = {
        "Bullpen": "bullpen",
        "Batting Practice": "batting_practice",
        "Intersquad": "intersquad",
    }.get(session_type, "batting_practice")
    label_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(session_label).strip()).strip("._-")

    for uploaded in uploaded_files:
        safe_name = _safe_upload_name(uploaded.name)
        base_name = f"{prefix}__{label_slug}__{safe_name}" if label_slug else f"{prefix}__{safe_name}"
        out_path = PRACTICE_DATA_DIR / base_name
        counter = 2
        while out_path.exists():
            out_path = PRACTICE_DATA_DIR / f"{Path(base_name).stem}_{counter}.csv"
            counter += 1
        content = uploaded.getvalue()
        out_path.write_bytes(content)
        _push_practice_file_to_github(out_path, content)
        saved.append(out_path)
    return saved


def summarize_practice_files():
    rows = []
    for path in get_practice_csv_files():
        try:
            df = _read_csv_fast(path)
        except Exception:
            continue
        live_df = filter_live_practice_pitches(df)
        pitcher_count = live_df["Pitcher"].nunique() if "Pitcher" in live_df.columns else 0
        date_series = _coerce_trackman_dates(df)
        date_range = "No date"
        if date_series.notna().any():
            lo = date_series.min().strftime("%Y-%m-%d")
            hi = date_series.max().strftime("%Y-%m-%d")
            date_range = lo if lo == hi else f"{lo} to {hi}"
        rows.append({
            "Session": _practice_file_label(path),
            "Type": _practice_session_type_from_name(path),
            "File": path.name,
            "Rows": len(df),
            "Live Pitches": len(live_df),
            "Pitchers": pitcher_count,
            "Date Range": date_range,
        })
    return pd.DataFrame(rows)


def prepare_practice_data(selected_files=None):
    files = [Path(f) for f in (selected_files or get_practice_csv_files())]
    if not files:
        return pd.DataFrame()

    processed = []
    stuff_model, stuff_league, loc_model, loc_league = _load_models_cached()

    for path in files:
        try:
            raw = _read_csv_fast(path)
            raw = filter_real_trackman_pitch_rows(raw)
            if raw.empty:
                continue
            raw = _ensure_practice_trackman_columns(raw)
            df = basic_clean(raw)
            df = add_flags(df)
            df = add_perceived_velocity(df)
            df = compute_stuffplus(df, stuff_model, stuff_league)
            df = compute_locationplus(df, loc_model, loc_league)
            df["PracticeFile"] = path.name
            df["PracticeSession"] = _practice_file_label(path)
            df["SessionType"] = _practice_session_type_from_name(path)
            if "GameDate" not in df.columns:
                dates = _coerce_trackman_dates(raw)
                if dates.notna().any():
                    df["GameDate"] = dates.dt.strftime("%Y-%m-%d")
            processed.append(df)
        except Exception:
            continue

    if not processed:
        return pd.DataFrame()
    return pd.concat(processed, ignore_index=True)


def get_scouting_csv_files():
    # Sort descending so newest FTP-import date comes first — dedup keeps latest
    return sorted(SCOUTING_DATA_DIR.glob("*.csv"), reverse=True)


def get_unique_scouting_files():
    """Deduplicated scouting files — one per game."""
    seen_games: set = set()
    unique: list = []
    for p in get_scouting_csv_files():
        m = re.match(r"v3__\d{4}__\d{2}__\d{2}__CSV__(.+)", p.name)
        if m:
            game_key = m.group(1)
            if game_key in seen_games:
                continue
            seen_games.add(game_key)
        unique.append(p)
    return unique


def get_scouting_csv_count():
    return len(get_unique_scouting_files())


def _scouting_parquet_parts() -> list[Path]:
    return [p for p in (SCOUTING_PARQUET_1, SCOUTING_PARQUET_2) if p.exists()]


def scouting_data_source_signature() -> tuple:
    """Version stamp for Scouting Zone CSV/Parquet data.

    Cached Streamlit loaders only refresh when their arguments change. This
    signature changes whenever the local scouting CSV folder or split Parquet
    files change, so new TrackMan imports show up without a manual cache clear.
    """
    parquet_sig = []
    for p in _scouting_parquet_parts():
        try:
            stat = p.stat()
        except OSError:
            continue
        parquet_sig.append((p.name, stat.st_size, stat.st_mtime_ns))
    if parquet_sig:
        return ("parquet", tuple(parquet_sig))

    csvs = get_scouting_csv_files()
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


def _scouting_source() -> str:
    """Return the Scouting Zone source, preferring rebuilt Parquet when present."""
    if _scouting_parquet_parts():
        return "parquet"
    if get_scouting_csv_files():
        return "csv"
    return "none"


@st.cache_data(show_spinner="Loading team index…")
def _scouting_parquet_index(source_sig: tuple) -> pd.DataFrame:
    """Read only PitcherTeam+BatterTeam columns — tiny, for building team lists."""
    parts = _scouting_parquet_parts()
    if not parts:
        return pd.DataFrame()
    import pyarrow.parquet as _pq
    chunks = []
    for p in parts:
        try:
            tbl = _pq.read_table(str(p), columns=["PitcherTeam","BatterTeam"])
            chunks.append(tbl.to_pandas())
        except Exception:
            pass
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    for col in out.columns:
        out[col] = out[col].astype("category")
    return out


@st.cache_data(show_spinner="Loading scouting data…", ttl=300)
def _scouting_parquet_for_team(team: str, source_sig: tuple) -> pd.DataFrame:
    """Load only rows for one team using pyarrow predicate pushdown.
    Peak memory: one team's pitches only — never the full 2M-row dataset."""
    parts = _scouting_parquet_parts()
    if not parts:
        return pd.DataFrame()
    import pyarrow.parquet as _pq
    chunks = []
    for p in parts:
        try:
            tbl = _pq.read_table(str(p), filters=[
                [("PitcherTeam", "=", team)],
                [("BatterTeam",  "=", team)],
            ])
            if len(tbl):
                chunks.append(tbl.to_pandas())
        except Exception:
            pass
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# Disk-cache location (gitignored folder — local only, not committed)
_SCOUTING_INDEX_FILE = SCOUTING_DATA_DIR / ".scouting_index.pkl"


def _read_teams_from_file(path) -> tuple[str, set]:
    """Read BatterTeam + PitcherTeam from a single CSV. Used by thread pool."""
    try:
        df = pd.read_csv(
            path,
            usecols=["BatterTeam", "PitcherTeam"],
            dtype=str,
            encoding="latin1",
            low_memory=False,
        )
        teams: set = set()
        for col in df.columns:
            teams |= set(df[col].dropna().str.strip())
        teams.discard("")
        return str(path), teams
    except Exception:
        return str(path), set()


@st.cache_data(show_spinner="Building scouting index…")
def build_scouting_team_index(source_sig: tuple):
    """Inverted index: team_code → [file_paths].

    Strategy (fastest first):
    1. Parquet mode (cloud): build index from in-memory Parquet — fast.
    2. Load from disk cache if file count matches — instant.
    3. Otherwise build with a thread pool (parallel I/O) then save to disk.
    """
    import concurrent.futures
    import pickle

    # ── Parquet / cloud mode — column projection only ────────────────────────
    if _scouting_source() == "parquet":
        idx = _scouting_parquet_index(source_sig)
        if idx.empty:
            return {}, []
        inverted: dict[str, list[str]] = {}
        for col in ["PitcherTeam", "BatterTeam"]:
            if col in idx.columns:
                for team in idx[col].dropna().unique():
                    inverted.setdefault(str(team).strip(), [])
        return inverted, sorted(inverted.keys())

    csvs = get_unique_scouting_files()
    n = len(csvs)

    # ── 1. Try disk cache ────────────────────────────────────────────────────
    if _SCOUTING_INDEX_FILE.exists():
        try:
            with open(_SCOUTING_INDEX_FILE, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("source_sig") == source_sig:
                return saved["index"], saved["teams"]
        except Exception:
            pass

    # ── 2. Build in parallel ─────────────────────────────────────────────────
    inverted = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for file_path, teams in pool.map(_read_teams_from_file, csvs):
            for team in teams:
                inverted.setdefault(team, []).append(file_path)

    teams_list = sorted(inverted.keys())

    # ── 3. Persist to disk ───────────────────────────────────────────────────
    try:
        with open(_SCOUTING_INDEX_FILE, "wb") as fh:
            pickle.dump({"source_sig": source_sig, "index": inverted, "teams": teams_list}, fh)
    except Exception:
        pass

    return inverted, teams_list


def _scouting_files_for_team(team: str, source_sig: tuple) -> list:
    """O(1) lookup via the inverted index."""
    index, _ = build_scouting_team_index(source_sig)
    return sorted(Path(p) for p in index.get(str(team).strip(), []))


def _coerce_scouting_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "Velo", "IVB", "HB", "Spin", "RelH", "RelS", "Ext", "VAA", "HAA",
        "PlateLocSide", "PlateLocHeight", "ExitSpeed", "Angle", "Direction",
        "Distance", "Balls", "Strikes",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def prepare_scouting_data(team=None, source_sig=None):
    if source_sig is None:
        source_sig = scouting_data_source_signature()
    source = _scouting_source()

    # ── Parquet / cloud mode — team-filtered pyarrow read ────────────────────
    if source == "parquet":
        if not team:
            # No specific team → can't load 2M rows; return empty and let UI guide
            return pd.DataFrame()
        raw = _scouting_parquet_for_team(str(team), source_sig)
        if raw.empty:
            return pd.DataFrame()
        df = basic_clean(raw)
        df = _coerce_scouting_numeric(df)
        df = add_flags(df)
        df = add_perceived_velocity(df)
        try:
            sm, sl, lm, ll = _load_models_cached()
            df = compute_stuffplus(df, sm, sl)
            df = compute_locationplus(df, lm, ll)
        except Exception:
            pass
        return df

    # ── CSV / local mode ─────────────────────────────────────────────────────
    csvs = get_unique_scouting_files()
    if not csvs:
        return prepare_data()

    if team:
        csvs = _scouting_files_for_team(team, source_sig)
        if not csvs:
            return pd.DataFrame()

    processed = []
    stuff_model, stuff_league, loc_model, loc_league = _load_models_cached()
    for path in csvs:
        try:
            raw = _read_csv_fast(path)
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
            df = _coerce_scouting_numeric(df)
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
        sample = pd.read_csv(local_path, nrows=25, encoding="latin1")
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
                    copy_fordham_csv_to_data(target, target.name)
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

    _SCOUTING_INDEX_FILE.unlink(missing_ok=True)
    get_scouting_csv_count.clear()
    get_unique_scouting_files.clear()
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
                    copy_fordham_csv_to_data(target, target.name)
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

    _SCOUTING_INDEX_FILE.unlink(missing_ok=True)
    get_scouting_csv_count.clear()
    get_unique_scouting_files.clear()
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
# OPPONENT DETECTION (FOR_RAM to BatterTeam)
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


def first_nonempty_value(df: pd.DataFrame, columns, default=""):
    for col in columns:
        if col in df.columns:
            values = df[col].dropna().astype(str).str.strip()
            values = values[values.ne("") & values.ne("nan")]
            if not values.empty:
                return values.iloc[0]
    return default


def compute_contact_quality(df: pd.DataFrame) -> tuple:
    """
    Return (avg_ev, hh_pct, barrel_pct) from true BIP pitches.
    All three are pitcher-perspective: lower = better (will be inverted in grade).
    """
    # Normalize EV and LA column names
    ev_raw = df.get("EV", df.get("ExitSpeed", pd.Series(dtype=float)))
    la_raw = df.get("LA", df.get("Angle",     pd.Series(dtype=float)))
    ev = pd.to_numeric(ev_raw, errors="coerce")
    la = pd.to_numeric(la_raw, errors="coerce")

    # True BIP = ball put in play (not a foul)
    pc = df.get("PitchCall", pd.Series(dtype=str)).astype(str)
    bip_mask = pc.isin(["InPlay", "InPlayNoOut", "InPlayOut", "InPlayRun"])
    bip_ev = ev[bip_mask & ev.notna()]

    if bip_ev.empty:
        return float("nan"), float("nan"), float("nan")

    avg_ev  = float(bip_ev.mean())
    hh_pct  = float((bip_ev >= 95).mean() * 100)

    bip_la  = la[bip_mask & ev.notna()]
    if not bip_la.empty:
        barrels    = barrel_mask(bip_ev, bip_la)
        barrel_pct = float(barrels.mean() * 100)
    else:
        barrel_pct = float("nan")

    return avg_ev, hh_pct, barrel_pct


def compute_fps(df: pd.DataFrame) -> float:
    """First-pitch strike % from TrackMan data (Balls='0', Strikes='0')."""
    b = df.get("Balls",   pd.Series(dtype=str)).astype(str).str.strip()
    s = df.get("Strikes", pd.Series(dtype=str)).astype(str).str.strip()
    first = df[(b == "0") & (s == "0")]
    if first.empty:
        return float("nan")
    strike_calls = {
        "StrikeCalled","StrikeSwinging","FoulBall","FoulBallNotFieldable",
        "FoulBallFieldable","FoulTip","InPlay","InPlayNoOut","InPlayOut","InPlayRun",
    }
    pc = first.get("PitchCall", pd.Series(dtype=str)).astype(str)
    return pc.isin(strike_calls).mean() * 100


def _norm(val: float, avg: float, scale: float) -> float:
    """Normalize a raw % to a Stuff+-style 100-centered score."""
    return 100.0 + (val - avg) * scale


def outing_grade(stuff: float, loc: float,
                 fps: float    = float("nan"),
                 csw: float    = float("nan"),
                 whiff: float  = float("nan"),
                 bb_pct: float = float("nan"),
                 avg_ev: float = float("nan"),
                 hh_pct: float = float("nan"),
                 barrel: float = float("nan")) -> tuple:
    """
    Comprehensive A-F outing grade — 9 metrics, 4 categories.

    CONTACT QUALITY (40%) — pitcher-perspective, all inverted (lower = better):
      Avg EV   15%  D1 avg ≈ 84.5 mph, 3.0 pts per mph
      HH%      15%  D1 avg ≈ 32%,      2.0 pts per %
      Barrel%  10%  D1 avg ≈ 8%,       4.0 pts per %

    SWING & MISS (25%):
      CSW%     13%  D1 avg ≈ 27%,      2.5 pts per %
      Whiff%   12%  D1 avg ≈ 22%,      2.5 pts per %

    COMMAND (20%):
      BB%      10%  D1 avg ≈ 10.5%,    3.0 pts per % (inverted)
      FPS%     10%  D1 avg ≈ 58%,      2.0 pts per %

    PITCH MODELS (15%):
      Stuff+    8%  (already 100-centered)
      Loc+      7%  (already 100-centered)

    Missing metrics are dropped and remaining weights are rescaled.
    This design is run-independent — a pitcher who allows hard-hit balls
    that find gaps still grades poorly; soft contact that finds holes grades well.
    """
    def _ok(v): return not (isinstance(v, float) and np.isnan(v))

    components = []

    # Contact quality (inverted — lower raw value → higher score)
    # D1 anchors: avg EV ~86 mph, HH% ~34%, Barrel% ~15% (using 92 mph threshold)
    # Barrel weight reduced (8%) due to small-sample volatility per outing
    if _ok(avg_ev):  components.append((100 + (86.0 - avg_ev)  * 3.0, 0.16))
    if _ok(hh_pct):  components.append((100 + (34.0 - hh_pct)  * 2.0, 0.16))
    if _ok(barrel):  components.append((100 + (15.0 - barrel)   * 3.0, 0.08))

    # Swing & miss
    if _ok(csw):     components.append((_norm(csw,   27.0, 2.5),        0.13))
    if _ok(whiff):   components.append((_norm(whiff, 22.0, 2.5),        0.12))

    # Command — D1 avg BB% ~12% (higher than MLB 8.2%); FPS% ~58%
    if _ok(bb_pct):  components.append((100 + (12.0 - bb_pct)  * 3.0,  0.10))
    if _ok(fps):     components.append((_norm(fps,   58.0, 2.0),        0.10))

    # Pitch models
    if _ok(stuff):   components.append((stuff,                          0.08))
    if _ok(loc):     components.append((loc,                            0.07))

    if not components:
        return "—", "#6B7A93", "No data", None

    total_w   = sum(w for _, w in components)
    combined  = sum(v * w for v, w in components) / total_w

    if   combined >= 118: return "A+", "#16a34a", "Elite",         combined
    elif combined >= 113: return "A",  "#22c55e", "Excellent",     combined
    elif combined >= 108: return "A-", "#4ade80", "Very Good",     combined
    elif combined >= 105: return "B+", "#86efac", "Good",          combined
    elif combined >= 102: return "B",  "#bef264", "Above Average", combined
    elif combined >= 99:  return "B-", "#fde047", "Average",       combined
    elif combined >= 96:  return "C+", "#fb923c", "Below Average", combined
    elif combined >= 92:  return "C",  "#f97316", "Struggling",    combined
    elif combined >= 87:  return "C-", "#ef4444", "Poor",          combined
    elif combined >= 80:  return "D",  "#dc2626", "Very Poor",     combined
    else:                 return "F",  "#991b1b", "Rough",         combined


def _render_outing_grade(stuff: float, loc: float,
                         fps: float    = float("nan"),
                         csw: float    = float("nan"),
                         whiff: float  = float("nan"),
                         bb_pct: float = float("nan"),
                         avg_ev: float = float("nan"),
                         hh_pct: float = float("nan"),
                         barrel: float = float("nan")) -> None:
    """Render a styled outing grade badge with all component scores."""
    letter, color, desc, combined = outing_grade(stuff, loc, fps, csw, whiff, bb_pct, avg_ev, hh_pct, barrel)
    if combined is None:
        return

    def _s(v, fmt=".1f", suffix=""):
        return f"{v:{fmt}}{suffix}" if not (isinstance(v, float) and np.isnan(v)) else "—"

    text_color = "#0f172a" if color in ("#bef264","#fde047","#86efac","#4ade80") else "#ffffff"

    def _ok(v): return not (isinstance(v, float) and np.isnan(v))
    parts = []
    if _ok(avg_ev):  parts.append(f"Avg EV {_s(avg_ev)}")
    if _ok(hh_pct):  parts.append(f"HH% {_s(hh_pct, suffix='%')}")
    if _ok(barrel):  parts.append(f"Barrel% {_s(barrel, suffix='%')}")
    if _ok(csw):     parts.append(f"CSW% {_s(csw, suffix='%')}")
    if _ok(whiff):   parts.append(f"Whiff% {_s(whiff, suffix='%')}")
    if _ok(bb_pct):  parts.append(f"BB% {_s(bb_pct, suffix='%')}")
    if _ok(fps):     parts.append(f"FPS% {_s(fps, suffix='%')}")
    if _ok(stuff):   parts.append(f"Stuff+ {_s(stuff)}")
    if _ok(loc):     parts.append(f"Loc+ {_s(loc)}")
    detail = "&nbsp;·&nbsp;".join(parts)

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;
        background:linear-gradient(135deg,#1a2535,#111827);
        border:2px solid {color}55;border-radius:12px;
        padding:14px 20px;margin:10px 0 16px">
      <div style="width:72px;height:72px;border-radius:50%;
          background:{color};display:flex;align-items:center;justify-content:center;
          font-size:2rem;font-weight:900;color:{text_color};flex-shrink:0;
          box-shadow:0 0 20px {color}55">{letter}</div>
      <div>
        <div style="color:#9BAABF;font-size:.7rem;font-weight:700;
            text-transform:uppercase;letter-spacing:.10em">Outing Grade</div>
        <div style="color:#F7F2E8;font-size:1.25rem;font-weight:800;
            margin:.15rem 0">{desc} &nbsp;
          <span style="font-size:.9rem;font-weight:600;color:{color}">{combined:.1f}</span>
        </div>
        <div style="color:#9BAABF;font-size:.80rem">{detail}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def compute_pitch_efficiency(df: pd.DataFrame) -> tuple:
    """Return (pitches_per_true_inning, total_outs).
    Uses integer outs throughout — never divides by baseball IP notation."""
    total      = len(df)
    k          = df.get("KorBB", pd.Series(dtype=str)).eq("Strikeout").sum()
    oop        = pd.to_numeric(
        df.get("OutsOnPlay", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).sum()
    total_outs = int(oop) + int(k)
    if total_outs == 0:
        return float("nan"), 0
    # true innings = outs / 3  (1.2 baseball IP = 5 outs = 5/3 = 1.667 true innings)
    true_ip = total_outs / 3.0
    return total / true_ip, total_outs


def pitch_efficiency_grade(p_per_ip: float) -> tuple:
    """
    A-F grade for pitch efficiency.
    15 pitches/IP = average (C).  Lower is better.
    """
    if isinstance(p_per_ip, float) and np.isnan(p_per_ip):
        return "—", "#6B7A93", "No data"
    if   p_per_ip <= 14.5: return "A",  "#22c55e", "Excellent"
    elif p_per_ip <= 16.5: return "A-", "#4ade80", "Very Efficient"
    elif p_per_ip <= 18.5: return "B+", "#86efac", "Efficient"
    elif p_per_ip <= 20.5: return "B",  "#bef264", "Above Average"
    elif p_per_ip <= 22.5: return "B-", "#fde047", "Average"
    elif p_per_ip <= 24.5: return "C+", "#fb923c", "Below Average"
    elif p_per_ip <= 26.5: return "C",  "#f97316", "Inefficient"
    elif p_per_ip <= 28.5: return "C-", "#ef4444", "Struggling"
    elif p_per_ip <= 30.0: return "D",  "#dc2626", "Very Inefficient"
    else:                  return "F",  "#991b1b", "Rough"


def _render_pitch_efficiency_grade(df: pd.DataFrame) -> None:
    """Render a styled pitch efficiency badge below the metrics strip."""
    p_per_ip, total_outs = compute_pitch_efficiency(df)
    letter, color, desc  = pitch_efficiency_grade(p_per_ip)
    if letter == "—":
        return
    text_color = "#0f172a" if color in ("#bef264","#fde047","#86efac","#4ade80") else "#ffffff"
    total      = len(df)
    # Baseball IP display: e.g. 17 outs → "5.2" (5 full innings + 2 outs)
    full_inn   = total_outs // 3
    rem_outs   = total_outs % 3          # always 0, 1, or 2
    ip_display = f"{full_inn}.{rem_outs}"   # e.g. "5.2"
    true_inn   = total_outs / 3.0        # e.g. 5.667 — used only for display
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;
        background:linear-gradient(135deg,#1a2535,#111827);
        border:2px solid {color}55;border-radius:12px;
        padding:14px 20px;margin:6px 0 16px">
      <div style="width:72px;height:72px;border-radius:50%;
          background:{color};display:flex;align-items:center;justify-content:center;
          font-size:2rem;font-weight:900;color:{text_color};flex-shrink:0;
          box-shadow:0 0 20px {color}55">{letter}</div>
      <div>
        <div style="color:#9BAABF;font-size:.7rem;font-weight:700;
            text-transform:uppercase;letter-spacing:.10em">Pitch Efficiency</div>
        <div style="color:#F7F2E8;font-size:1.25rem;font-weight:800;
            margin:.15rem 0">{desc} &nbsp;
          <span style="font-size:.9rem;font-weight:600;color:{color}">{p_per_ip:.1f} P/IP</span>
        </div>
        <div style="color:#9BAABF;font-size:.80rem">
          {total} pitches &nbsp;·&nbsp; {total_outs} outs ({ip_display} IP = {true_inn:.2f} true innings)
          &nbsp;·&nbsp; A ≤14.5 · A- ≤16.5 · B+ ≤18.5 · B ≤20.5
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def pure_stuff_grade(stuff: float) -> tuple:
    """A-F grade based solely on Stuff+ — isolates raw pitch quality."""
    if isinstance(stuff, float) and np.isnan(stuff):
        return "—", "#6B7A93", "No data"
    if   stuff >= 118: return "A+", "#16a34a", "Elite Stuff"
    elif stuff >= 113: return "A",  "#22c55e", "Excellent"
    elif stuff >= 108: return "A-", "#4ade80", "Very Good"
    elif stuff >= 105: return "B+", "#86efac", "Good"
    elif stuff >= 102: return "B",  "#bef264", "Above Average"
    elif stuff >= 99:  return "B-", "#fde047", "Average"
    elif stuff >= 96:  return "C+", "#fb923c", "Below Average"
    elif stuff >= 92:  return "C",  "#f97316", "Struggling"
    elif stuff >= 87:  return "C-", "#ef4444", "Poor"
    elif stuff >= 80:  return "D",  "#dc2626", "Very Poor"
    else:              return "F",  "#991b1b", "Rough"


def _render_stuff_grade(stuff: float, loc: float = float("nan")) -> None:
    """Render a pure Stuff+ grade badge."""
    letter, color, desc = pure_stuff_grade(stuff)
    if letter == "—":
        return
    text_color = "#0f172a" if color in ("#bef264","#fde047","#86efac","#4ade80") else "#ffffff"
    stuff_str  = f"{stuff:.1f}" if not (isinstance(stuff, float) and np.isnan(stuff)) else "—"
    loc_str    = f"{loc:.1f}"   if not (isinstance(loc,   float) and np.isnan(loc))   else "—"
    loc_part   = f"&nbsp;·&nbsp; Loc+ {loc_str}" if loc_str != "—" else ""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;
        background:linear-gradient(135deg,#1a2535,#111827);
        border:2px solid {color}55;border-radius:12px;
        padding:14px 20px;margin:6px 0 16px">
      <div style="width:72px;height:72px;border-radius:50%;
          background:{color};display:flex;align-items:center;justify-content:center;
          font-size:2rem;font-weight:900;color:{text_color};flex-shrink:0;
          box-shadow:0 0 20px {color}55">{letter}</div>
      <div>
        <div style="color:#9BAABF;font-size:.7rem;font-weight:700;
            text-transform:uppercase;letter-spacing:.10em">Pure Stuff Grade</div>
        <div style="color:#F7F2E8;font-size:1.25rem;font-weight:800;
            margin:.15rem 0">{desc} &nbsp;
          <span style="font-size:.9rem;font-weight:600;color:{color}">Stuff+ {stuff_str}</span>
        </div>
        <div style="color:#9BAABF;font-size:.80rem">
          Based on Stuff+ only{loc_part} &nbsp;·&nbsp; 100 = D1 average
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def pitch_efficiency_leaderboard(df: pd.DataFrame, min_pitches: int = 25) -> pd.DataFrame:
    """Build a per-pitcher pitch efficiency leaderboard from raw pitch data."""
    if df.empty or "Pitcher" not in df.columns:
        return pd.DataFrame()
    rows = []
    for pitcher, g in df.groupby("Pitcher"):
        n = len(g)
        if n < min_pitches:
            continue
        p_per_ip, total_outs = compute_pitch_efficiency(g)
        if isinstance(p_per_ip, float) and np.isnan(p_per_ip):
            continue
        full_inn = total_outs // 3
        rem_outs = total_outs % 3
        ip_disp  = f"{full_inn}.{rem_outs}"
        letter, _, _ = pitch_efficiency_grade(p_per_ip)
        rows.append({
            "Pitcher":    pitcher,
            "Pitches":    n,
            "IP":         ip_disp,
            "P/IP":       round(p_per_ip, 1),
            "Eff Grade":  letter,
        })
    if not rows:
        return pd.DataFrame()
    lb = (pd.DataFrame(rows)
            .sort_values("P/IP", ascending=True)
            .reset_index(drop=True))
    lb.insert(0, "Rank", np.arange(1, len(lb) + 1))
    return lb


def trackman_game_metadata(df: pd.DataFrame) -> dict:
    home_team = first_nonempty_value(df, ["HomeTeam"], "")
    away_team = first_nonempty_value(df, ["AwayTeam"], "")
    return {
        "game_id": first_nonempty_value(df, ["GameID"], ""),
        "game_uid": first_nonempty_value(df, ["GameUID"], ""),
        "game_foreign_id": first_nonempty_value(df, ["GameForeignID"], ""),
        "home_team": home_team,
        "away_team": away_team,
        "home_team_id": first_nonempty_value(df, ["HomeTeamForeignID", "HomeTeamId", "HomeTeamID"], ""),
        "away_team_id": first_nonempty_value(df, ["AwayTeamForeignID", "AwayTeamId", "AwayTeamID"], ""),
    }


def team_tag_label(team_code: str) -> str:
    code = str(team_code or "").strip()
    if not code:
        return "-"
    return team_display_name(code, include_code=True)


def trackman_team_tag_lines(df: pd.DataFrame) -> list[str]:
    meta = trackman_game_metadata(df)
    lines = []
    team_parts = []
    if meta["home_team"]:
        team_parts.append(f"Home {team_tag_label(meta['home_team'])}")
    if meta["away_team"]:
        team_parts.append(f"Away {team_tag_label(meta['away_team'])}")
    if team_parts:
        lines.append(" | ".join(team_parts))
    return lines


def build_postgame_figure(pdf, pitcher, game_date, opponent, trackman_lines=None):
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

    def draw_pitch_usage_panel(ax, source_df):
        ax.set_facecolor(PANEL)
        ax.text(0.0, 0.985, "Pitch Usage", transform=ax.transAxes,
                color=TEXT, fontsize=13, weight="bold",
                ha="left", va="top")
        ax.set_xlim(0, 112)
        ax.set_ylim(-0.6, 3.05)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["Overall", "vs LHH", "vs RHH"], color=TEXT, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Usage %", color=MUTED, fontsize=9, fontweight="bold")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="x", color=GRID, alpha=0.12, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_color(GRID)

        if source_df.empty or "pitch_abbr" not in source_df.columns:
            ax.text(0.5, 0.5, "No pitch mix", transform=ax.transAxes,
                    color=MUTED, ha="center", va="center", fontsize=11, fontweight="bold")
            return

        pitch_order = source_df["pitch_abbr"].value_counts().index.tolist()

        def subset(label):
            if label == "Overall" or "BatterSide" not in source_df.columns:
                return source_df
            want = "Left" if label == "vs LHH" else "Right"
            return source_df[source_df["BatterSide"].astype(str).eq(want)]

        for label, y in [("Overall", 2), ("vs LHH", 1), ("vs RHH", 0)]:
            sub = subset(label)
            if sub.empty:
                ax.text(2, y, "No data", color=MUTED, fontsize=9, va="center", fontweight="bold")
                continue
            counts = sub["pitch_abbr"].value_counts().reindex(pitch_order).fillna(0)
            total = float(counts.sum())
            left = 0.0
            for pitch, count in counts.items():
                if count <= 0 or total <= 0:
                    continue
                width = float(count) / total * 100
                ax.barh(y, width, left=left, height=0.46,
                        color=pitch_colors.get(pitch, "white"),
                        edgecolor=BACKGROUND, linewidth=0.8)
                if width >= 16:
                    ax.text(left + width / 2, y, f"{pitch}\n{width:.0f}%",
                            color="white", ha="center", va="center",
                            fontsize=8.5, fontweight="bold")
                elif width >= 10:
                    ax.text(left + width / 2, y, pitch,
                            color="white", ha="center", va="center",
                            fontsize=8, fontweight="bold")
                left += width
            ax.text(106.0, y, f"{int(total)}", color=MUTED, fontsize=9,
                    va="center", ha="left", fontweight="bold")
        ax.text(106.0, 2.53, "N", color=MUTED, fontsize=9,
                va="center", ha="left", fontweight="bold")

    # -----------------------------
    # GAME TOTALS
    # -----------------------------
    total_pitches = len(pdf)
    whiffs = pdf["is_whiff"].sum()
    swings = pdf["is_swing"].sum() if "is_swing" in pdf.columns else 0
    walks = pdf["KorBB"].eq("Walk").sum()
    strikeouts = pdf["KorBB"].eq("Strikeout").sum()
    hits = pdf["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()

    outs_on_play = pdf["OutsOnPlay"].sum() if "OutsOnPlay" in pdf.columns else 0
    total_outs = outs_on_play + strikeouts
    ip = total_outs // 3 + (total_outs % 3) / 10 if total_outs else 0.0

    strike_pct = round(pdf["is_strike"].mean() * 100, 1)
    zone_pct = round(pdf["in_zone"].mean() * 100, 1) if "in_zone" in pdf.columns else np.nan
    whiff_pct = round(whiffs / swings * 100, 1) if swings else 0.0
    stuff_avg = round(pdf["Stuff+"].mean(), 1) if "Stuff+" in pdf.columns and len(pdf) else np.nan
    loc_avg = round(pdf["Loc+"].mean(), 1) if "Loc+" in pdf.columns and len(pdf) else np.nan

    LHH_pdf = pdf[pdf["is_LHH"]]
    RHH_pdf = pdf[pdf["is_RHH"]]

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

    # Avg EV, HH%, Barrel% per pitch type — BIP only (EV > 45 mph)
    _ev_bip = pdf.copy()
    if "EV" not in _ev_bip.columns and "ExitSpeed" in _ev_bip.columns:
        _ev_bip["EV"] = _ev_bip["ExitSpeed"]
    if "LA" not in _ev_bip.columns and "Angle" in _ev_bip.columns:
        _ev_bip["LA"] = _ev_bip["Angle"]
    if "EV" in _ev_bip.columns and "PlayResult" in _ev_bip.columns:
        bip_pr = {"Single","Double","Triple","HomeRun","Out","Error","FieldersChoice"}
        _ev_bip["EV"] = pd.to_numeric(_ev_bip["EV"], errors="coerce")
        _ev_bip["LA"] = pd.to_numeric(_ev_bip.get("LA", pd.Series(dtype=float)), errors="coerce")
        _ev_bip = _ev_bip[_ev_bip["PlayResult"].isin(bip_pr) & (_ev_bip["EV"] > 45)]
        if not _ev_bip.empty:
            _ev_bip["_barrel"] = (_ev_bip["EV"] >= BARREL_EV_MIN) & _ev_bip["LA"].between(BARREL_LA_MIN, BARREL_LA_MAX)
            _ev_agg = _ev_bip.groupby("pitch_abbr").agg(
                AvgEV   =("EV",      "mean"),
                HardHit =("EV",      lambda x: (x >= 95).mean() * 100),
                Barrel  =("_barrel", "mean"),
                BIP     =("EV",      "count"),
            ).reset_index()
            _ev_agg["AvgEV"]   = _ev_agg["AvgEV"].round(1)
            _ev_agg["HardHit"] = _ev_agg["HardHit"].round(1)
            _ev_agg["Barrel"]  = (_ev_agg["Barrel"] * 100).round(1)
            _ev_agg.loc[_ev_agg["BIP"] < 3, ["AvgEV","HardHit","Barrel"]] = np.nan
            _ev_agg = _ev_agg.rename(columns={"pitch_abbr":"Pitch"})
            agg = agg.merge(_ev_agg[["Pitch","AvgEV","HardHit","Barrel"]], on="Pitch", how="left")
            agg.rename(columns={"AvgEV":"Avg EV","HardHit":"HH%","Barrel":"Barrel%"}, inplace=True)
        else:
            agg["Avg EV"] = np.nan; agg["HH%"] = np.nan; agg["Barrel%"] = np.nan

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
    logo_path = ROOT / "static" / "rams.png"
    if not logo_path.exists():
        logo_path = ROOT / "assets" / "rams.png"
    if logo_path.exists():
        logo_img = mpimg.imread(logo_path)
        logo_ax = fig.add_axes([0.035, 0.855, 0.095, 0.105], zorder=50)
        logo_ax.set_facecolor((0, 0, 0, 0))
        logo_ax.patch.set_alpha(0)
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
    if trackman_lines is None:
        trackman_lines = trackman_team_tag_lines(pdf)
    for i, line in enumerate(trackman_lines[:2]):
        fig.text(
            0.5, 0.902 - i * 0.022,
            line,
            ha="center",
            va="center",
            fontsize=9.5,
            color=MUTED,
            fontweight="bold",
        )

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
    # PITCH USAGE
    # -----------------------------
    ax_usage = fig.add_subplot(gs[0, 3])
    draw_pitch_usage_panel(ax_usage, pdf)

    # -----------------------------
    # TABLE
    # -----------------------------
    ax_table = fig.add_subplot(gs[1:, :])
    ax_table.axis("off")

    _table_cols = ["Pitch","N","Usage%","Velo","PerceivedVelo","IVB","HB",
                   "Spin","Stuff+","Loc+","CSW%","Whiff%","Strike%","Zone%"]
    for _c in ["Avg EV","HH%","Barrel%"]:
        if _c in agg.columns:
            _table_cols.append(_c)
    table_df = agg[[c for c in _table_cols if c in agg.columns]].rename(
        columns={"PerceivedVelo": "PerVelo"})
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
# PAGE 1 - POSTGAME SUMMARY (Pitcher to Game Selector)
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

    game_keys = ["Date", "BatterTeam"]
    for optional_key in ["GameID", "GameUID", "GameForeignID", "HomeTeamForeignID", "AwayTeamForeignID"]:
        if optional_key in pdf.columns:
            game_keys.append(optional_key)
    games = pdf.groupby(game_keys, dropna=False).size().reset_index(name="Pitches")
    games["_DateSort"] = pd.to_datetime(games["Date"], errors="coerce")
    sort_cols = ["_DateSort", "BatterTeam"]
    if "GameID" in games.columns:
        sort_cols.append("GameID")
    elif "GameUID" in games.columns:
        sort_cols.append("GameUID")
    games = games.sort_values(sort_cols, ascending=False, na_position="last").reset_index(drop=True)
    games["GameNumber"] = games.groupby(["Date", "BatterTeam"], dropna=False).cumcount() + 1
    games["GamesThatDay"] = games.groupby(["Date", "BatterTeam"], dropna=False)["GameNumber"].transform("max")
    games["GameLabel"] = np.where(
        games["GamesThatDay"].gt(1),
        " (Game " + games["GameNumber"].astype(str) + ")",
        ""
    )
    games["label"] = (
        games["Date"].astype(str)
        + " vs "
        + games["BatterTeam"].astype(str).map(lambda code: team_display_name(code, include_code=True))
        + games["GameLabel"]
    )
    selected_game = st.selectbox("Select Game", games["label"], key="pg_game")

    game_row = games[games["label"].eq(selected_game)].iloc[0]
    g_date = str(game_row["Date"])
    g_opp = str(game_row["BatterTeam"])

    g_pdf = pdf[
        (pdf["Date"].astype(str) == g_date) &
        (pdf["BatterTeam"] == g_opp)
    ].copy()
    if "GameID" in pdf.columns and str(game_row.get("GameID", "")).strip():
        g_pdf = g_pdf[g_pdf["GameID"].astype(str).eq(str(game_row["GameID"]))].copy()
    elif "GameUID" in pdf.columns and str(game_row.get("GameUID", "")).strip():
        g_pdf = g_pdf[g_pdf["GameUID"].astype(str).eq(str(game_row["GameUID"]))].copy()

    if g_pdf.empty:
        st.error("No data found for that game.")
        return

    meta     = trackman_game_metadata(g_pdf)
    total    = len(g_pdf)
    swings   = g_pdf.get("is_swing", pd.Series(False, index=g_pdf.index)).sum()
    whiffs   = g_pdf.get("is_whiff", pd.Series(False, index=g_pdf.index)).sum()
    strike_p = g_pdf.get("is_strike", pd.Series(False, index=g_pdf.index)).mean() * 100
    zone_p   = g_pdf.get("in_zone",   pd.Series(False, index=g_pdf.index)).mean() * 100
    csw_p    = g_pdf.get("is_csw",    pd.Series(False, index=g_pdf.index)).mean() * 100
    whiff_p  = whiffs / swings * 100 if swings else float("nan")
    stuff_m  = g_pdf["Stuff+"].mean() if "Stuff+" in g_pdf.columns else float("nan")
    loc_m    = g_pdf["Loc+"].mean()   if "Loc+"   in g_pdf.columns else float("nan")

    fps_p              = compute_fps(g_pdf)
    kbb_col            = g_pdf.get("KorBB", pd.Series(dtype=str))
    pa_ends            = kbb_col.isin(["Walk","Strikeout"]) | g_pdf.get("PlayResult", pd.Series(dtype=str)).isin(
        ["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice","Sacrifice"])
    bb_p               = kbb_col.eq("Walk").sum() / pa_ends.sum() * 100 if pa_ends.sum() else float("nan")
    avg_ev_p, hh_p, barrel_p = compute_contact_quality(g_pdf)
    mc       = st.columns(12)
    mc[0].metric("Pitches",  f"{total:,}")
    mc[1].metric("Opponent", team_display_name(g_opp))
    mc[2].metric("FPS%",     f"{fps_p:.1f}%"    if not pd.isna(fps_p)    else "—", help="First-pitch strike %")
    mc[3].metric("CSW%",     f"{csw_p:.1f}%"    if not pd.isna(csw_p)    else "—")
    mc[4].metric("Whiff%",   f"{whiff_p:.1f}%"  if not pd.isna(whiff_p)  else "—")
    mc[5].metric("BB%",      f"{bb_p:.1f}%"     if not pd.isna(bb_p)     else "—")
    mc[6].metric("Avg EV",   f"{avg_ev_p:.1f}"  if not pd.isna(avg_ev_p) else "—")
    mc[7].metric("HH%",      f"{hh_p:.1f}%"     if not pd.isna(hh_p)     else "—")
    mc[8].metric("Barrel%",  f"{barrel_p:.1f}%" if not pd.isna(barrel_p) else "—")
    mc[9].metric("Stuff+",   f"{stuff_m:.1f}"   if not pd.isna(stuff_m)  else "—")
    mc[10].metric("Loc+",    f"{loc_m:.1f}"     if not pd.isna(loc_m)    else "—")
    mc[11].metric("Home",    team_tag_label(meta["home_team"]))

    _render_outing_grade(stuff_m, loc_m, fps_p, csw_p, whiff_p, bb_p, avg_ev_p, hh_p, barrel_p)
    _render_stuff_grade(stuff_m, loc_m)
    _render_pitch_efficiency_grade(g_pdf)

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
# PAGE 2 - SEASON SUMMARY
# ------------------------------------------------------------
def season_page():
    st.title("Season Summary - Stuff+ & Location+")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    pitchers = get_pitcher_list(df)
    pitcher = st.selectbox("Select pitcher", pitchers, key="season_pitcher")

    pdf = df[df["Pitcher"] == pitcher].copy()

    total   = len(pdf)
    games   = pdf["Date"].nunique() if "Date" in pdf.columns else float("nan")
    swings  = pdf.get("is_swing", pd.Series(False, index=pdf.index)).sum()
    whiffs  = pdf.get("is_whiff", pd.Series(False, index=pdf.index)).sum()
    strike_p = pdf.get("is_strike", pd.Series(False, index=pdf.index)).mean() * 100
    zone_p   = pdf.get("in_zone",   pd.Series(False, index=pdf.index)).mean() * 100
    csw_p    = pdf.get("is_csw",    pd.Series(False, index=pdf.index)).mean() * 100
    whiff_p  = whiffs / swings * 100 if swings else float("nan")
    stuff_m  = pdf["Stuff+"].mean()  if "Stuff+" in pdf.columns else float("nan")
    loc_m    = pdf["Loc+"].mean()    if "Loc+"   in pdf.columns else float("nan")
    velo_m   = pdf["Velo"].mean()    if "Velo"   in pdf.columns else float("nan")

    fps_m                  = compute_fps(pdf)
    kbb_s                  = pdf.get("KorBB", pd.Series(dtype=str))
    pa_s                   = kbb_s.isin(["Walk","Strikeout"]) | pdf.get("PlayResult", pd.Series(dtype=str)).isin(
        ["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice","Sacrifice"])
    bb_m                   = kbb_s.eq("Walk").sum() / pa_s.sum() * 100 if pa_s.sum() else float("nan")
    avg_ev_m, hh_m, brl_m = compute_contact_quality(pdf)
    mc = st.columns(13)
    mc[0].metric("Pitches",  f"{total:,}")
    mc[1].metric("Games",    f"{int(games)}" if not pd.isna(games) else "—")
    mc[2].metric("Avg Velo", f"{velo_m:.1f}" if not pd.isna(velo_m) else "—")
    mc[3].metric("FPS%",     f"{fps_m:.1f}%"    if not pd.isna(fps_m)    else "—", help="First-pitch strike %")
    mc[4].metric("CSW%",     f"{csw_p:.1f}%"    if not pd.isna(csw_p)    else "—")
    mc[5].metric("Whiff%",   f"{whiff_p:.1f}%"  if not pd.isna(whiff_p)  else "—")
    mc[6].metric("BB%",      f"{bb_m:.1f}%"     if not pd.isna(bb_m)     else "—")
    mc[7].metric("Avg EV",   f"{avg_ev_m:.1f}"  if not pd.isna(avg_ev_m) else "—")
    mc[8].metric("HH%",      f"{hh_m:.1f}%"     if not pd.isna(hh_m)     else "—")
    mc[9].metric("Barrel%",  f"{brl_m:.1f}%"    if not pd.isna(brl_m)    else "—")
    mc[10].metric("Stuff+",  f"{stuff_m:.1f}"   if not pd.isna(stuff_m)  else "—")
    mc[11].metric("Loc+",    f"{loc_m:.1f}"     if not pd.isna(loc_m)    else "—")
    mc[12].metric("Pitchers",str(pdf["Pitcher"].nunique()) if "Pitcher" in pdf.columns else "1")

    _render_outing_grade(stuff_m, loc_m, fps_m, csw_p, whiff_p, bb_m, avg_ev_m, hh_m, brl_m)
    _render_stuff_grade(stuff_m, loc_m)
    _render_pitch_efficiency_grade(pdf)

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

# ─────────────────────────────────────────────────────────────────────────────
# D1 PERCENTILE BENCHMARKS — all computed from actual season-level TrackMan
# data: 2,088 D1 pitchers with ≥30 PA, 7,116 deduplicated games, 2026.
# Format: (p10, p25, p50, p75, p90, high_is_good)
# ─────────────────────────────────────────────────────────────────────────────
_D1_PITCHER_PCTS = {
    # 5-point breakpoints (p10,p25,p50,p75,p90) from 7,399 D1 pitchers ≥50 pitches, 2026
    "Stuff+":     ( 75.0,  87.0, 100.0, 113.0, 125.0, True),
    "Loc+":       ( 75.0,  87.0, 100.0, 113.0, 125.0, True),
    "Velo":       ( 86.2,  88.0,  89.7,  91.5,  93.2, True),
    "CSW%":       ( 23.5,  25.7,  28.0,  30.7,  32.9, True),
    "Zone%":      ( 38.6,  41.4,  44.5,  47.2,  49.6, True),
    "Whiff%":     ( 16.1,  19.6,  23.6,  28.2,  32.7, True),
    "K%":         ( 12.2,  16.2,  20.5,  25.7,  30.5, True),
    "BB%":        (  5.9,   8.2,  11.0,  14.7,  19.4, False),
    "GB%":        ( 31.4,  36.8,  42.6,  49.2,  55.1, True),
    "Avg EV":     ( 84.8,  86.6,  88.3,  89.7,  91.1, False),  # lower = better
    "Barrel%A":   (  5.9,   9.6,  14.0,  18.2,  22.8, False),
    "Swing%I":    ( 33.6,  37.6,  41.8,  45.2,  48.1, True),
    "ZSwing%I":   ( 56.5,  61.7,  66.4,  70.5,  75.0, True),
    "OSwing%I":   ( 15.6,  19.5,  23.5,  27.3,  30.7, True),
    "ZContact%A": ( 77.1,  81.8,  86.0,  90.0,  94.0, False),
    "OContact%A": ( 44.1,  52.6,  61.3,  70.3,  80.0, False),
    "FPS%":       ( 42.9,  50.0,  56.2,  61.5,  66.2, True),
}

# Savant-style gradient: blue (poor) → mid-gray (avg) → red (elite)
# Mid-gray instead of near-white keeps text readable on both sides of avg
_PCT_STOPS = [
    (0.00, ( 10,  46, 110)),  # #0a2e6e  deep blue    0th pct
    (0.20, ( 25,  86, 160)),  # #1956a0  blue        20th pct
    (0.40, ( 94, 163, 208)),  # #5ea3d0  light blue  40th pct
    (0.50, (120, 120, 120)),  # #787878  mid-gray    50th pct
    (0.60, (209, 100,  70)),  # #d16446  light red   60th pct
    (0.80, (209,  60,  40)),  # #d13c28  red         80th pct
    (1.00, (139,   0,   0)),  # #8b0000  dark red   100th pct
]


def _pitcher_pct_rank(stat: str, val) -> float | None:
    """0–1 percentile rank (1.0 = best). Extrapolates smoothly outside p10–p90."""
    if val is None or pd.isna(val) or stat not in _D1_PITCHER_PCTS:
        return None
    p10, p25, p50, p75, p90, high = _D1_PITCHER_PCTS[stat]
    bps  = [p10, p25, p50, p75, p90]
    pcts = [0.10, 0.25, 0.50, 0.75, 0.90]
    fv   = float(val)
    # Smooth extrapolation below p10 and above p90 to avoid cliffs at 0/1
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


def _pct_hex(pct: float | None) -> str:
    if pct is None:
        return "#2a2a3a"
    r, g, b = 165, 165, 165
    for i in range(len(_PCT_STOPS) - 1):
        p0, c0 = _PCT_STOPS[i]
        p1, c1 = _PCT_STOPS[i + 1]
        if p0 <= pct <= p1:
            t = (pct - p0) / (p1 - p0) if p1 > p0 else 0
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            break
    return f"#{r:02x}{g:02x}{b:02x}"


def _readable_on(hex_bg: str) -> str:
    h = hex_bg.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#000000" if (0.299*r + 0.587*g + 0.114*b) / 255 > 0.52 else "#ffffff"


def _pct_label(pct: float | None) -> str:
    if pct is None:
        return "—"
    n = max(1, int(round(pct * 100)))  # floor at 1st — never show "0th"
    if n >= 90:
        return f"{n}th ★"
    if 11 <= n <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _compute_pitcher_pct_stats(pdf: pd.DataFrame) -> dict:
    """Aggregate pitcher stats for the percentile card."""
    if pdf.empty:
        return {}
    out = {}
    for col in ("Stuff+", "Loc+"):
        if col in pdf.columns:
            v = pd.to_numeric(pdf[col], errors="coerce").mean()
            out[col] = float(v) if not pd.isna(v) else None
    # FB velo (FB/SI only)
    if "Velo" in pdf.columns and "pitch_abbr" in pdf.columns:
        fb = pd.to_numeric(pdf.loc[pdf["pitch_abbr"].isin(["FB","SI"]), "Velo"], errors="coerce")
        out["Velo"] = float(fb.mean()) if fb.notna().any() else None
    elif "Velo" in pdf.columns:
        v = pd.to_numeric(pdf["Velo"], errors="coerce").mean()
        out["Velo"] = float(v) if not pd.isna(v) else None
    if "is_csw"  in pdf.columns: out["CSW%"]  = float(pdf["is_csw"].mean()  * 100)
    if "in_zone" in pdf.columns: out["Zone%"] = float(pdf["in_zone"].mean() * 100)
    if "is_swing" in pdf.columns and pdf["is_swing"].sum() > 0:
        out["Whiff%"] = float(pdf["is_whiff"].sum() / pdf["is_swing"].sum() * 100)
    kbb = pdf.get("KorBB",     pd.Series("", index=pdf.index)).fillna("").astype(str)
    pr  = pdf.get("PlayResult",pd.Series("", index=pdf.index)).fillna("").astype(str)
    pa_n = (kbb.isin(["Walk","Strikeout"]) |
            pr.isin(["Single","Double","Triple","HomeRun",
                     "Out","Error","FieldersChoice","Sacrifice"])).sum()
    if pa_n > 0:
        out["K%"]  = float(kbb.eq("Strikeout").sum() / pa_n * 100)
        out["BB%"] = float(kbb.eq("Walk").sum()       / pa_n * 100)
    # GB%
    ht_col = "TaggedHitType" if "TaggedHitType" in pdf.columns else None
    if ht_col:
        ht = pdf[ht_col].fillna("").astype(str)
        bip_types = ["GroundBall","FlyBall","LineDrive","PopUp","Popup"]
        bip_n = ht.isin(bip_types).sum()
        if bip_n >= 5:
            out["GB%"] = float(ht.eq("GroundBall").sum() / bip_n * 100)
    # Contact quality against
    ev_col = "EV" if "EV" in pdf.columns else ("ExitSpeed" if "ExitSpeed" in pdf.columns else None)
    la_col = "LA" if "LA" in pdf.columns else ("Angle" if "Angle" in pdf.columns else None)
    if ev_col:
        ev     = pd.to_numeric(pdf[ev_col], errors="coerce")
        bip_ev = ev[pr.isin(["Single","Double","Triple","HomeRun",
                              "Out","Error","FieldersChoice"]) & (ev > 45)]
        if len(bip_ev) >= 5:
            out["Avg EV"] = float(bip_ev.mean())
        if len(bip_ev) >= 10 and la_col:
            la_    = pd.to_numeric(pdf[la_col], errors="coerce")
            bip_la = la_[pr.isin(["Single","Double","Triple","HomeRun",
                                   "Out","Error","FieldersChoice"]) & (ev > 45)]
            barrels = ((bip_ev >= 92) & bip_la.between(16, 36)).sum()
            out["Barrel%A"] = float(barrels / len(bip_ev) * 100)
    # Swing / zone discipline (pitcher induces)
    if "is_swing" in pdf.columns and "in_zone" in pdf.columns:
        n_tot  = len(pdf)
        sw     = pdf["is_swing"].astype(bool)
        in_z   = pdf["in_zone"].astype(bool)
        pc_p   = pdf.get("PitchCall", pd.Series("", index=pdf.index)).astype(str)
        contact_calls = {"FoulBall","FoulBallNotFieldable","FoulBallFieldable","FoulTip",
                         "InPlay","InPlayNoOut","InPlayOut","InPlayRun"}
        is_ct  = pc_p.isin(contact_calls)
        z_sw   = (sw & in_z).sum();   o_sw = (sw & ~in_z).sum()
        z_ct   = (is_ct & in_z).sum(); o_ct = (is_ct & ~in_z).sum()
        z_pit  = in_z.sum();           o_pit = (~in_z).sum()
        if n_tot:  out["Swing%I"]    = float(sw.sum() / n_tot * 100)
        if z_pit:  out["ZSwing%I"]   = float(z_sw / z_pit * 100)
        if o_pit:  out["OSwing%I"]   = float(o_sw / o_pit * 100)
        if z_sw:   out["ZContact%A"] = float(z_ct / z_sw * 100)
        if o_sw:   out["OContact%A"] = float(o_ct / o_sw * 100)
    # FPS%
    if "Balls" in pdf.columns and "Strikes" in pdf.columns:
        b_s   = pdf["Balls"].astype(str).str.strip()
        s_s   = pdf["Strikes"].astype(str).str.strip()
        first = (b_s == "0") & (s_s == "0")
        strike_calls = {"StrikeCalled","StrikeSwinging","FoulBall","FoulBallNotFieldable",
                        "FoulBallFieldable","FoulTip","InPlay","InPlayNoOut","InPlayOut","InPlayRun"}
        fps_n = first.sum()
        if fps_n:
            pc_fp = pdf.get("PitchCall", pd.Series("", index=pdf.index)).astype(str)
            out["FPS%"] = float((pc_fp.isin(strike_calls) & first).sum() / fps_n * 100)
    return out


def build_percentile_card_png(pdf: pd.DataFrame, pitcher: str) -> bytes:  # noqa: C901
    """Savant-style horizontal percentile bar card — fixed coordinate system."""
    stats = _compute_pitcher_pct_stats(pdf)
    ROWS = [
        ("Stuff+",     "Stuff+",        "{:.0f}"),
        ("Loc+",       "Loc+",          "{:.0f}"),
        ("Velo",       "FB Velo",       "{:.1f} mph"),
        ("Whiff%",     "Whiff%",        "{:.1f}%"),
        ("CSW%",       "CSW%",          "{:.1f}%"),
        ("Zone%",      "Zone%",         "{:.1f}%"),
        ("Swing%I",    "Swing%",    "{:.1f}%"),
        ("ZSwing%I",   "Z-Swing%",  "{:.1f}%"),
        ("OSwing%I",   "O-Swing%",  "{:.1f}%"),
        ("K%",         "K%",            "{:.1f}%"),
        ("BB%",        "BB%",           "{:.1f}%"),
        ("FPS%",       "FPS%",          "{:.1f}%"),
        ("GB%",        "GB%",           "{:.1f}%"),
        ("Avg EV",     "Avg EV",     "{:.1f} mph"),
        ("Barrel%A",   "Barrel%",    "{:.1f}%"),
        ("ZContact%A", "Z-Contact%", "{:.1f}%"),
        ("OContact%A", "O-Contact%", "{:.1f}%"),
    ]

    BG  = "#13151c"
    BAR_BG = "#1c1f2a"
    n   = len(ROWS)

    # Figure height scales with number of rows
    fig_h = 2.6 + n * 0.58
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor(BG)
    # Use data coords 0–1 × 0–1 so Rectangle coords need no transform
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG); ax.axis("off")

    HDR  = 0.96   # top of header text
    SEP  = 0.875  # separator line y
    TOP  = 0.855  # top of bar rows
    BOT  = 0.055  # bottom of bar rows
    row_h = (TOP - BOT) / n

    # ── Header ────────────────────────────────────────────────────────────────
    _MAROON = "#8C1515"; _GOLD = "#C7A45D"
    ax.add_patch(plt.Rectangle((0, HDR-0.09), 1, 0.105, facecolor=_MAROON, zorder=0))
    ax.text(0.015, HDR-0.012, pitcher, color="white", fontsize=20, fontweight="bold", va="top")
    ax.text(0.015, HDR-0.065, "Fordham Rams  ·  D1 Percentile Rankings  ·  2026",
            color=_GOLD, fontsize=8.5, fontweight="bold", va="top")

    # Fordham logo — top-right of header
    _logo_p = ROOT / "static" / "rams.png"
    if not _logo_p.exists():
        _logo_p = ROOT / "national_pitchingplus_app" / "team_logos" / "FOR_RAM.png"
    if not _logo_p.exists():
        _logo_p = ROOT / "national_pitchingplus_app" / "team_logos" / "FOR_RAM1.png"
    if _logo_p.exists():
        try:
            from PIL import Image as _PIL
            _img = _PIL.open(_logo_p).convert("RGBA")
            _arr = np.array(_img)
            _arr[:, :, 3] = (_arr[:, :, 3].astype(float) * 0.92).clip(0, 255).astype(np.uint8)
            _la = ax.inset_axes([0.865, HDR-0.088, 0.10, 0.082])
            _la.set_facecolor((0, 0, 0, 0)); _la.patch.set_alpha(0)
            _la.imshow(_arr, aspect="equal")
            _la.set_xticks([]); _la.set_yticks([])
            for _sp in _la.spines.values():
                _sp.set_visible(False)
        except Exception:
            pass

    ax.plot([0.04, 0.96], [SEP, SEP], color="#333344", lw=0.8)

    # Column headers
    ax.text(0.735, SEP - 0.008, "Value",    color="#666677", fontsize=7.5, ha="left",  va="top")
    ax.text(0.965, SEP - 0.008, "Pct",      color="#666677", fontsize=7.5, ha="right", va="top")

    # ── Bar rows ──────────────────────────────────────────────────────────────
    BX  = 0.18   # bar left edge
    BW  = 0.54   # bar width
    for i, (key, label, fmt_s) in enumerate(ROWS):
        cy    = TOP - (i + 0.5) * row_h
        val   = stats.get(key)
        pct   = _pitcher_pct_rank(key, val)
        color = _pct_hex(pct)
        bh    = row_h * 0.54

        # Stat label
        ax.text(BX - 0.01, cy, label, color="#cccccc", fontsize=10,
                fontweight="bold", ha="right", va="center")

        # Track (background bar)
        ax.add_patch(plt.Rectangle((BX, cy - bh/2), BW, bh,
                                   facecolor=BAR_BG, zorder=2))

        # Fill
        if pct is not None and pct > 0.005:
            ax.add_patch(plt.Rectangle((BX, cy - bh/2), BW * pct, bh,
                                       facecolor=color, zorder=3))

        # 50th-pct tick
        ax.plot([BX + BW*0.5]*2, [cy - bh/2, cy + bh/2],
                color="#555566", lw=1.0, zorder=4)

        # Value
        val_s = fmt_s.format(float(val)) if val is not None and not pd.isna(val) else "—"
        ax.text(BX + BW + 0.015, cy, val_s, color="white",
                fontsize=9.5, ha="left", va="center")

        # Percentile
        ax.text(0.975, cy, _pct_label(pct), color=color,
                fontsize=10, fontweight="bold", ha="right", va="center")

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.text(BX,           BOT - 0.015, "◀ Poor",          color="#1956a0", fontsize=8, ha="left",   va="top")
    ax.text(BX + BW*0.5,  BOT - 0.015, "50th pct (avg)",  color="#888888", fontsize=8, ha="center", va="top")
    ax.text(BX + BW,      BOT - 0.015, "Elite ▶",          color="#8b0000", fontsize=8, ha="right",  va="top")

    out = BytesIO()
    fig.savefig(out, format="png", dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


def percentile_card_page():
    st.title("Pitcher Percentile Cards")
    st.caption("Stats ranked vs the 2026 D1 population — 19,435 pitcher-game records, "
               "7,116 TrackMan games.  Red = elite · Blue = poor.")

    df = prepare_data()
    df = filter_fordham_only(df)
    if df.empty:
        st.error("No Fordham pitcher data found.")
        return

    pitchers = get_pitcher_list(df)
    c1, _ = st.columns([2, 3])
    with c1:
        pitcher = st.selectbox("Select Pitcher", pitchers, key="pct_pitcher")
    pdf   = df[df["Pitcher"] == pitcher].copy()
    stats = _compute_pitcher_pct_stats(pdf)

    # ── Two rows of 5 coloured pills ─────────────────────────────────────────
    ALL_PILLS = [
        ("Stuff+", "Stuff+",   "{:.0f}"),
        ("Loc+",   "Loc+",     "{:.0f}"),
        ("Velo",   "FB Velo",  "{:.1f}"),
        ("Whiff%", "Whiff%",   "{:.1f}%"),
        ("CSW%",   "CSW%",     "{:.1f}%"),
        ("Zone%",  "Zone%",    "{:.1f}%"),
        ("K%",     "K%",       "{:.1f}%"),
        ("BB%",    "BB%",      "{:.1f}%"),
        ("GB%",    "GB%",      "{:.1f}%"),
        ("Avg EV", "Avg EV","{:.1f}"),
    ]
    for row_slice in (ALL_PILLS[:5], ALL_PILLS[5:]):
        cols = st.columns(5)
        for col, (key, label, fmt_s) in zip(cols, row_slice):
            val = stats.get(key)
            pct = _pitcher_pct_rank(key, val)
            bg  = _pct_hex(pct)
            tc  = _readable_on(bg)
            v_s = fmt_s.format(float(val)) if val is not None and not pd.isna(val) else "—"
            col.markdown(
                f'<div style="background:{bg};border-radius:8px;padding:10px 4px;'
                f'text-align:center;margin:2px 0">'
                f'<div style="font-size:19px;font-weight:bold;color:{tc}">{v_s}</div>'
                f'<div style="font-size:11px;color:{tc};opacity:.9">{label}</div>'
                f'<div style="font-size:10px;color:{tc};opacity:.7">{_pct_label(pct)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PNG card ──────────────────────────────────────────────────────────────
    card_png = build_percentile_card_png(pdf, pitcher)
    st.image(card_png, use_container_width=True)
    st.download_button(
        "Download Percentile Card",
        card_png,
        file_name=f"{pitcher.replace(', ','_')}_percentile_card.png",
        mime="image/png",
        key="pct_dl",
    )


# ─────────────────────────────────────────────────────────────────────────────
# D1 HITTER PERCENTILE BENCHMARKS — season-level stats from 1,876 D1 hitters
# ≥50 PA, 7,116 deduplicated D1-vs-D1 TrackMan games, 2026.
# Format: (p10, p25, p50, p75, p90, high_is_good)
# ─────────────────────────────────────────────────────────────────────────────
# 2026 D1 collegiate wOBA weights + league average
# Weights from D1 run-expectancy research (lower than FanGraphs MLB weights)
_WOBA_BB, _WOBA_HBP, _WOBA_1B = 0.64, 0.66, 0.80
_WOBA_2B, _WOBA_3B, _WOBA_HR  = 1.12, 1.41, 1.76
_LG_WOBA_APP   = 0.325    # 2026 D1 lgwOBA — calibrated value
_WOBA_SCALE_APP = 0.873   # lgwOBA / lgOBP = 0.338 / 0.387
_LG_R_PA_APP    = 0.387   # lgOBP = lgR/PA by construction

_D1_HITTER_PCTS_APP = {
    # 7-point breakpoints (p2,p10,p25,p50,p75,p90,p98) from 2026 D1 TrackMan parquet
    # ≥50 PA / ≥15 BIP where noted. Last field: True = higher is better.
    "Bat+":      ( 66.0,  81.0,  93.0, 106.0, 119.0, 132.0, 152.0, True),
    "wOBA":      (0.213, 0.263, 0.300, 0.341, 0.385, 0.427, 0.497, True),
    "BA":        (0.159, 0.219, 0.254, 0.296, 0.341, 0.383, 0.440, True),
    "OBP":       (0.248, 0.309, 0.350, 0.398, 0.440, 0.484, 0.562, True),
    "SLG":       (0.220, 0.296, 0.373, 0.448, 0.551, 0.642, 0.800, True),
    "OPS":       (0.487, 0.629, 0.734, 0.851, 0.983, 1.109, 1.320, True),
    "K%":        (  6.5,  10.6,  14.3,  19.0,  24.4,  30.0,  38.2, False),
    "BB%":       (  3.8,   6.3,   8.6,  11.3,  14.3,  17.5,  22.2, True),
    "Whiff%":    (  9.0,  13.4,  17.4,  22.5,  27.8,  33.0,  40.2, False),
    "Chase%":    ( 21.0,  25.3,  28.6,  32.2,  35.8,  39.3,  44.1, False),
    "OSwing%":   ( 12.1,  16.5,  20.1,  24.4,  29.0,  33.9,  40.9, False),
    "ZSwing%":   ( 50.0,  56.4,  61.4,  66.7,  72.0,  76.8,  83.3, True),
    "Swing%":    ( 30.7,  35.3,  38.7,  42.7,  46.7,  50.9,  56.4, None),
    "Contact%":  ( 54.2,  63.7,  70.4,  76.7,  82.6,  87.3,  93.0, True),
    "ZContact%": ( 63.6,  73.3,  80.0,  85.6,  90.2,  93.9, 100.0, True),
    "OContact%": ( 25.0,  40.0,  50.0,  59.5,  69.2,  78.3,  90.7, True),
    "Avg EV":    ( 80.1,  83.1,  85.2,  87.6,  89.9,  91.9,  94.3, True),
    "Max EV":    ( 94.9,  99.2, 102.2, 105.5, 108.8, 112.0, 118.3, True),
    "EV90":      ( 91.6,  95.5,  98.0, 100.8, 103.5, 105.7, 108.6, True),
    "HH%":       (  6.3,  17.0,  25.9,  35.2,  43.9,  51.1,  58.8, True),
    "Barrel%":   (  0.0,   4.3,   9.1,  14.9,  20.6,  26.1,  33.3, True),
}


def _hitter_pct_rank(stat: str, val) -> float | None:
    """0–1 percentile rank for a hitter stat (1.0 = best for hitter)."""
    if val is None or pd.isna(val) or stat not in _D1_HITTER_PCTS_APP:
        return None
    p2, p10, p25, p50, p75, p90, p98, high = _D1_HITTER_PCTS_APP[stat]
    bps  = [p2, p10, p25, p50, p75, p90, p98]
    pcts = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    fv   = float(val)
    if fv < bps[0]:
        pct = max(0.01, 0.02 * fv / bps[0]) if bps[0] > 0 else 0.01
    elif fv > bps[-1]:
        pct = min(0.99, 0.98 + 0.01 * (fv - bps[-1]) / max(abs(bps[-1]) * 0.15, 1))
    else:
        pct = 0.50
        for i in range(len(bps) - 1):
            if bps[i] <= fv <= bps[i + 1]:
                t = (fv - bps[i]) / (bps[i + 1] - bps[i])
                pct = pcts[i] + t * (pcts[i + 1] - pcts[i])
                break
    return (1.0 - pct) if not high else pct


def _compute_hitter_pct_stats(bdf: pd.DataFrame) -> dict:
    """Aggregate season batting stats for one Fordham hitter."""
    if bdf.empty:
        return {}
    pr  = bdf.get("PlayResult", pd.Series("", index=bdf.index)).fillna("").astype(str)
    kbb = bdf.get("KorBB",      pd.Series("", index=bdf.index)).fillna("").astype(str)
    pc  = bdf.get("PitchCall",  pd.Series("", index=bdf.index)).fillna("").astype(str)
    ht  = bdf.get("TaggedHitType", pd.Series("", index=bdf.index)).fillna("").astype(str)
    ev  = pd.to_numeric(bdf.get("ExitSpeed", pd.Series(dtype=float)), errors="coerce")
    pls = pd.to_numeric(bdf.get("PlateLocSide",   pd.Series(dtype=float)), errors="coerce")
    plh = pd.to_numeric(bdf.get("PlateLocHeight", pd.Series(dtype=float)), errors="coerce")
    s=pr.eq("Single").sum(); d2=pr.eq("Double").sum()
    t=pr.eq("Triple").sum(); hr=pr.eq("HomeRun").sum()
    H=s+d2+t+hr; TB=s+2*d2+3*t+4*hr
    walks=kbb.eq("Walk").sum(); ks=kbb.eq("Strikeout").sum()
    hbp=pc.eq("HitByPitch").sum(); sf=pr.eq("Sacrifice").sum()
    pa_m=(kbb.isin(["Walk","Strikeout"]) |
          pr.isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice","Sacrifice"]))
    pa=pa_m.sum(); ab=max(pa-walks-hbp-sf,0); obd=ab+walks+hbp+sf

    swing_calls_h = {"StrikeSwinging","FoulBall","FoulBallNotFieldable","FoulBallFieldable",
                     "FoulTip","InPlay","InPlayNoOut","InPlayOut","InPlayRun"}
    contact_calls_h = {"FoulBall","FoulBallNotFieldable","FoulBallFieldable","FoulTip",
                       "InPlay","InPlayNoOut","InPlayOut","InPlayRun"}
    swing_m  = pc.isin(swing_calls_h)
    contact_m= pc.isin(contact_calls_h)
    zone_m   = pls.between(-0.83,0.83) & plh.between(1.5,3.5)
    sw_n     = swing_m.sum()
    wh_n     = pc.eq("StrikeSwinging").sum()
    ch_n     = (swing_m & ~zone_m).sum()
    z_sw     = (swing_m & zone_m).sum()
    o_sw     = (swing_m & ~zone_m).sum()
    z_ct     = (contact_m & zone_m).sum()
    o_ct     = (contact_m & ~zone_m).sum()
    z_pit    = zone_m.sum()
    o_pit    = (~zone_m).sum()
    tot_p    = len(bdf)

    la  = pd.to_numeric(bdf.get("LA", bdf.get("Angle", pd.Series(dtype=float))), errors="coerce")
    ev2 = pd.to_numeric(bdf.get("EV", bdf.get("ExitSpeed", pd.Series(dtype=float))), errors="coerce")
    bip_mask = pr.isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice"]) & (ev2>45)
    bip_ev   = ev2[bip_mask]
    bip_la   = la[bip_mask]
    bip_n    = ht.isin(["GroundBall","FlyBall","LineDrive","PopUp","Popup"]).sum()

    _woba_calc = compute_woba(bdf)
    woba     = _woba_calc if _woba_calc > 0 else None
    bat_plus = compute_wrc_plus(_woba_calc) if _woba_calc > 0 else None

    return {
        "PA":int(pa), "AB":int(ab), "H":int(H), "HR":int(hr),
        "BA":   H/ab             if ab   else None,
        "OBP":  (H+walks+hbp)/obd if obd else None,
        "SLG":  TB/ab            if ab   else None,
        "OPS":  ((H+walks+hbp)/obd+TB/ab) if (ab and obd) else None,
        "wOBA":    woba,
        "Bat+":    bat_plus,
        "K%":      float(ks/pa*100)     if pa    else None,
        "BB%":     float(walks/pa*100)  if pa    else None,
        "Whiff%":  float(wh_n/sw_n*100) if sw_n  else None,
        "Chase%":  float(ch_n/sw_n*100) if sw_n  else None,
        "Swing%":  float(sw_n/tot_p*100) if tot_p else None,
        "ZSwing%": float(z_sw/z_pit*100) if z_pit else None,
        "OSwing%": float(o_sw/o_pit*100) if o_pit else None,
        "Contact%":  float((z_ct+o_ct)/sw_n*100) if sw_n else None,
        "ZContact%": float(z_ct/z_sw*100) if z_sw else None,
        "OContact%": float(o_ct/o_sw*100) if o_sw else None,
        "Avg EV": float(bip_ev.mean()) if len(bip_ev)>=5 else None,
        "Max EV": float(bip_ev.max())  if len(bip_ev)>=5 else None,
        "EV90":   float(bip_ev.quantile(0.90)) if len(bip_ev)>=10 else None,
        "HH%":    float((bip_ev>=95).mean()*100) if len(bip_ev)>=5 else None,
        "Barrel%":float(((bip_ev>=92)&bip_la.between(16,36)).mean()*100) if len(bip_ev)>=10 else None,
    }


def build_hitter_percentile_card_png(bdf: pd.DataFrame, batter: str) -> bytes:
    """Savant-style hitter percentile bar card for one Fordham hitter."""
    stats = _compute_hitter_pct_stats(bdf)
    ROWS = [
        ("Bat+",      "Bat+",       "{:.0f}"),
        ("wOBA",      "wOBA",       "{:.3f}"),
        ("BA",        "BA",         "{:.3f}"),
        ("OBP",       "OBP",        "{:.3f}"),
        ("SLG",       "SLG",        "{:.3f}"),
        ("OPS",       "OPS",        "{:.3f}"),
        ("K%",        "K%",         "{:.1f}%"),
        ("BB%",       "BB%",        "{:.1f}%"),
        ("Whiff%",    "Whiff%",     "{:.1f}%"),
        ("Chase%",    "Chase%",     "{:.1f}%"),
        ("OSwing%",   "O-Swing%",   "{:.1f}%"),
        ("ZSwing%",   "Z-Swing%",   "{:.1f}%"),
        ("Contact%",  "Contact%",   "{:.1f}%"),
        ("ZContact%", "Z-Contact%", "{:.1f}%"),
        ("OContact%", "O-Contact%", "{:.1f}%"),
        ("Avg EV",    "Avg EV",     "{:.1f} mph"),
        ("Max EV",    "Max EV",     "{:.1f} mph"),
        ("EV90",      "EV 90th%",   "{:.1f} mph"),
        ("HH%",       "HH%",        "{:.1f}%"),
        ("Barrel%",   "Barrel%",    "{:.1f}%"),
    ]

    BG = "#13151c"; BAR_BG = "#1c1f2a"
    n   = len(ROWS)
    fig_h = 2.6 + n * 0.52
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_facecolor(BG); ax.axis("off")

    HDR=0.96; SEP=0.875; TOP=0.855; BOT=0.055
    row_h = (TOP - BOT) / n
    BX=0.18; BW=0.54

    _MAROON2 = "#8C1515"; _GOLD2 = "#C7A45D"
    ax.add_patch(plt.Rectangle((0, HDR-0.09), 1, 0.105, facecolor=_MAROON2, zorder=0))
    ax.text(0.015, HDR-0.012, batter, color="white", fontsize=20, fontweight="bold", va="top")
    ax.text(0.015, HDR-0.065, "Fordham Rams  ·  D1 Percentile Rankings  ·  2026",
            color=_GOLD2, fontsize=8.5, fontweight="bold", va="top")

    _logo_p2 = ROOT / "static" / "rams.png"
    if not _logo_p2.exists():
        _logo_p2 = ROOT / "national_pitchingplus_app" / "team_logos" / "FOR_RAM.png"
    if not _logo_p2.exists():
        _logo_p2 = ROOT / "national_pitchingplus_app" / "team_logos" / "FOR_RAM1.png"
    if _logo_p2.exists():
        try:
            from PIL import Image as _PIL2
            _img2 = _PIL2.open(_logo_p2).convert("RGBA")
            _arr2 = np.array(_img2)
            _arr2[:, :, 3] = (_arr2[:, :, 3].astype(float) * 0.92).clip(0, 255).astype(np.uint8)
            _la2 = ax.inset_axes([0.865, HDR-0.088, 0.10, 0.082])
            _la2.set_facecolor((0, 0, 0, 0)); _la2.patch.set_alpha(0)
            _la2.imshow(_arr2, aspect="equal")
            _la2.set_xticks([]); _la2.set_yticks([])
            for _sp2 in _la2.spines.values():
                _sp2.set_visible(False)
        except Exception:
            pass

    ax.plot([0.04,0.96], [SEP,SEP], color="#333344", lw=0.8)
    ax.text(0.735, SEP-0.008, "Value",  color="#666677", fontsize=7.5, ha="left",  va="top")
    ax.text(0.965, SEP-0.008, "Pct",    color="#666677", fontsize=7.5, ha="right", va="top")

    for i, (key, label, fmt_s) in enumerate(ROWS):
        cy    = TOP - (i+0.5)*row_h
        val   = stats.get(key)
        pct   = _hitter_pct_rank(key, val)
        color = _pct_hex(pct)
        bh    = row_h * 0.54

        ax.text(BX-0.01, cy, label, color="#cccccc", fontsize=10,
                fontweight="bold", ha="right", va="center")
        ax.add_patch(plt.Rectangle((BX, cy-bh/2), BW, bh, facecolor=BAR_BG, zorder=2))
        if pct is not None and pct > 0.005:
            ax.add_patch(plt.Rectangle((BX, cy-bh/2), BW*pct, bh, facecolor=color, zorder=3))
        ax.plot([BX+BW*0.5]*2, [cy-bh/2, cy+bh/2], color="#555566", lw=1.0, zorder=4)

        val_s = fmt_s.format(float(val)) if val is not None and not pd.isna(val) else "—"
        ax.text(BX+BW+0.015, cy, val_s, color="white", fontsize=9.5, ha="left", va="center")
        ax.text(0.975, cy, _pct_label(pct), color=color,
                fontsize=10, fontweight="bold", ha="right", va="center")

    ax.text(BX,          BOT-0.015, "◀ Poor",         color="#1956a0", fontsize=8, ha="left",   va="top")
    ax.text(BX+BW*0.5,  BOT-0.015, "50th pct (avg)",  color="#888888", fontsize=8, ha="center", va="top")
    ax.text(BX+BW,      BOT-0.015, "Elite ▶",          color="#8b0000", fontsize=8, ha="right",  va="top")

    out = BytesIO()
    fig.savefig(out, format="png", dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


def hitter_percentile_card_page(all_df: pd.DataFrame):
    st.title("Hitter Percentile Cards")
    st.caption("Stats ranked vs 1,876 D1 hitters (≥50 PA) from 7,116 TrackMan games.  "
               "Red = elite · Blue = poor.")

    # Fordham batters = rows where BatterTeam starts with FOR_RAM
    if "BatterTeam" not in all_df.columns or "Batter" not in all_df.columns:
        st.error("Batter data not available in the loaded dataset.")
        return

    batter_df = all_df[all_df["BatterTeam"].astype(str).str.startswith("FOR_RAM")].copy()
    if batter_df.empty:
        st.info("No Fordham batter data found (BatterTeam = FOR_RAM).")
        return

    batters = sorted(batter_df["Batter"].dropna().unique().tolist())
    if not batters:
        st.info("No batters found.")
        return

    c1, _ = st.columns([2, 3])
    with c1:
        batter = st.selectbox("Select Hitter", batters, key="hpct_batter")

    bdf   = batter_df[batter_df["Batter"] == batter]
    stats = _compute_hitter_pct_stats(bdf)

    if stats.get("PA", 0) < 10:
        st.warning(f"Only {stats.get('PA',0)} PA found for {batter}. Stats may not be reliable.")

    # ── Stat summary ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, key in [(c1,"PA"),(c2,"H"),(c3,"HR"),(c4,"AB")]:
        v = stats.get(key)
        col.metric(key, str(int(v)) if v is not None else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two rows of coloured pills (6 + 6) ───────────────────────────────────
    ALL_PILLS = [
        ("Bat+",   "Bat+",   "{:.0f}"),
        ("wOBA",   "wOBA",   "{:.3f}"),
        ("BA",     "BA",     "{:.3f}"),
        ("OBP",    "OBP",    "{:.3f}"),
        ("SLG",    "SLG",    "{:.3f}"),
        ("OPS",    "OPS",    "{:.3f}"),
        ("K%",     "K%",     "{:.1f}%"),
        ("BB%",    "BB%",    "{:.1f}%"),
        ("Whiff%", "Whiff%", "{:.1f}%"),
        ("Chase%", "Chase%", "{:.1f}%"),
        ("Avg EV", "Avg EV", "{:.1f}"),
        ("HH%",    "HH%",    "{:.1f}%"),
    ]
    for row_slice in (ALL_PILLS[:6], ALL_PILLS[6:]):
        cols = st.columns(6)
        for col, (key, label, fmt_s) in zip(cols, row_slice):
            val = stats.get(key)
            pct = _hitter_pct_rank(key, val)
            bg  = _pct_hex(pct)
            tc  = _readable_on(bg)
            # Format value
            if key in {"wOBA","BA","OBP","SLG","OPS"}:
                v_s = f"{float(val):.3f}".replace("0.",".") if val is not None and not pd.isna(val) else "—"
            elif key in {"Bat+"}:
                v_s = str(int(round(float(val)))) if val is not None and not pd.isna(val) else "—"
            else:
                v_s = fmt_s.format(float(val)) if val is not None and not pd.isna(val) else "—"
            col.markdown(
                f'<div style="background:{bg};border-radius:8px;padding:10px 4px;'
                f'text-align:center;margin:2px 0">'
                f'<div style="font-size:19px;font-weight:bold;color:{tc}">{v_s}</div>'
                f'<div style="font-size:11px;color:{tc};opacity:.9">{label}</div>'
                f'<div style="font-size:10px;color:{tc};opacity:.7">{_pct_label(pct)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if "TaggedHitType" in bdf.columns and "Direction" in bdf.columns:
        _gb = bdf[bdf["TaggedHitType"].astype(str).eq("GroundBall")].copy()
        _gb["Direction"] = pd.to_numeric(_gb["Direction"], errors="coerce")
        _gb = _gb.dropna(subset=["Direction"])
        if len(_gb) >= 5:
            _gbn = len(_gb)
            _gbl = (_gb["Direction"] < 0).sum() / _gbn * 100
            _gbr = (_gb["Direction"] > 0).sum() / _gbn * 100
            _gc1, _gc2, _ = st.columns([1, 1, 3])
            _gc1.metric(f"GB Left of 2B  ({_gbn} GBs)", f"{_gbl:.0f}%",
                        help="SS / 3B side — Direction < 0")
            _gc2.metric("GB Right of 2B", f"{_gbr:.0f}%",
                        help="2B / 1B side — Direction > 0")

    st.markdown("<br>", unsafe_allow_html=True)

    card_png = build_hitter_percentile_card_png(bdf, batter)
    st.image(card_png, use_container_width=True)
    st.download_button(
        "Download Hitter Percentile Card",
        card_png,
        file_name=f"{batter.replace(', ','_')}_hitter_pct_card.png",
        mime="image/png",
        key="hpct_dl",
    )


# ------------------------------------------------------------
# PAGE 3 - STUFF+ LEADERBOARD
# ------------------------------------------------------------
def pitcher_plus_leaderboard(df: pd.DataFrame, metric_col: str, min_pitches=25) -> pd.DataFrame:
    if df.empty or metric_col not in df.columns or "Pitcher" not in df.columns:
        return pd.DataFrame()

    base = df.copy()
    if "pitch_abbr" not in base.columns:
        base["pitch_abbr"] = "UNK"

    agg_map = {
        metric_col: (metric_col, "mean"),
        "Pitches": (metric_col, "count"),
    }
    if "Stuff+" in base.columns:
        agg_map["Stuff+"] = ("Stuff+", "mean")
    if "Loc+" in base.columns:
        agg_map["Loc+"] = ("Loc+", "mean")
    if "Velo" in base.columns:
        agg_map["Velo"] = ("Velo", "mean")
    if "is_strike" in base.columns:
        agg_map["Strike%"] = ("is_strike", lambda x: x.mean() * 100)
    if "in_zone" in base.columns:
        agg_map["Zone%"] = ("in_zone", lambda x: x.mean() * 100)
    if "is_csw" in base.columns:
        agg_map["CSW%"] = ("is_csw", lambda x: x.mean() * 100)

    out = base.groupby("Pitcher").agg(**agg_map).reset_index()
    top_pitch = (
        base.groupby(["Pitcher", "pitch_abbr"]).size()
        .reset_index(name="PitchN")
        .sort_values(["Pitcher", "PitchN"], ascending=[True, False])
        .drop_duplicates("Pitcher")
        .rename(columns={"pitch_abbr": "Primary Pitch"})
    )
    out = out.merge(top_pitch[["Pitcher", "Primary Pitch"]], on="Pitcher", how="left")
    out = out[pd.to_numeric(out["Pitches"], errors="coerce").fillna(0) >= min_pitches]
    out = out.sort_values(metric_col, ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out.round(1)


def plus_leaderboard_figure(leaderboard: pd.DataFrame, metric_col: str, title: str, top_n=15):
    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor("#100D0C")
    ax.set_facecolor("#100D0C")
    ax.axis("off")

    ax.text(0.03, 0.94, title, color="#FFF7E8", fontsize=22, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(
        0.03, 0.895,
        "Minimum-pitch filtered | 100 is college average baseline",
        color="#CDBFAF",
        fontsize=10,
        transform=ax.transAxes,
        va="top",
    )

    if leaderboard.empty:
        ax.text(0.5, 0.5, "No qualified pitchers", color="#CDBFAF", fontsize=16, ha="center", va="center", transform=ax.transAxes)
        return fig

    view = leaderboard.head(top_n).copy()
    scores = pd.to_numeric(view[metric_col], errors="coerce").fillna(0)
    xmin = min(85, float(scores.min()) - 4)
    xmax = max(125, float(scores.max()) + 6)
    bar_ax = fig.add_axes([0.08, 0.10, 0.84, 0.70])
    bar_ax.set_facecolor("#171514")
    for spine in bar_ax.spines.values():
        spine.set_color("#4E4036")
    bar_ax.axvline(100, color="#CDBFAF", linewidth=1.2, linestyle="--", alpha=0.75)
    bar_ax.text(100, len(view) + 0.1, "100", color="#CDBFAF", fontsize=9, ha="center", va="bottom")

    y = np.arange(len(view))[::-1]
    colors = []
    for score in scores:
        rgb = _value_to_color(score, metric_col, leaderboard[metric_col], context="pitching") or (140, 21, 21)
        colors.append("#{:02x}{:02x}{:02x}".format(*rgb))
    bar_ax.barh(y, scores, color=colors, edgecolor="#C7A45D", linewidth=0.8, height=0.64)
    bar_ax.set_xlim(xmin, xmax)
    bar_ax.set_ylim(-0.7, len(view) - 0.25)
    bar_ax.set_yticks([])
    bar_ax.tick_params(axis="x", colors="#CDBFAF", labelsize=9)
    bar_ax.grid(axis="x", color="#4E4036", alpha=0.35, linewidth=0.8)

    for idx, (_, row) in enumerate(view.iterrows()):
        yy = y[idx]
        pitcher = textwrap.shorten(str(row.get("Pitcher", "-")), width=24, placeholder="...")
        rank = int(row.get("Rank", idx + 1))
        score = row.get(metric_col, np.nan)
        primary = row.get("Primary Pitch", "-")
        pitches = int(row.get("Pitches", 0))
        bar_ax.text(xmin + 0.4, yy, f"{rank:>2}", color=FORDHAM_GOLD, fontsize=10, fontweight="bold", va="center")
        bar_ax.text(xmin + 4.0, yy, pitcher, color="#FFF7E8", fontsize=10.5, fontweight="bold", va="center")
        bar_ax.text(float(score) + 0.8, yy, _fmt_pdf_value(score, metric_col), color="#FFF7E8", fontsize=11, fontweight="bold", va="center")
        bar_ax.text(xmax - 0.5, yy, f"{primary} | {pitches} P", color="#CDBFAF", fontsize=8.5, ha="right", va="center")

    return fig


def stuff_leaderboard_page():
    st.title("Stuff+ Leaderboard")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        min_pitches = st.slider("Minimum pitches", 10, 250, 25, 5, key="stuff_min")
    with c2:
        top_n = st.slider("Show top", 5, 25, 15, 5, key="stuff_top")

    agg = pitcher_plus_leaderboard(df, "Stuff+", min_pitches=min_pitches)
    # Add grade columns
    if not agg.empty:
        agg["Stuff Grade"]  = agg["Stuff+"].apply(lambda v: pure_stuff_grade(v)[0])
        agg["Outing Grade"] = agg.apply(lambda r: outing_grade(
            r.get("Stuff+", float("nan")), r.get("Loc+", float("nan")),
            float("nan"), r.get("CSW%", float("nan")), float("nan"))[0], axis=1)
    fig = plus_leaderboard_figure(agg, "Stuff+", "Fordham Baseball - Stuff+ Leaderboard", top_n=top_n)
    st.pyplot(fig)
    st.dataframe(
        style_scouting_dataframe(_table_columns(agg, ["Rank", "Pitcher", "Stuff Grade", "Outing Grade", "Stuff+", "Loc+", "Pitches", "Primary Pitch", "Velo", "CSW%"]), context="pitching"),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------
# PAGE 4 - LOCATION+ LEADERBOARD
# ------------------------------------------------------------
def location_leaderboard_page():
    st.title("Location+ Leaderboard")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        min_pitches = st.slider("Minimum pitches", 10, 250, 25, 5, key="loc_min")
    with c2:
        top_n = st.slider("Show top", 5, 25, 15, 5, key="loc_top")

    agg = pitcher_plus_leaderboard(df, "Loc+", min_pitches=min_pitches)
    if not agg.empty:
        agg["Stuff Grade"]  = agg["Stuff+"].apply(lambda v: pure_stuff_grade(v)[0])
        agg["Outing Grade"] = agg.apply(lambda r: outing_grade(
            r.get("Stuff+", float("nan")), r.get("Loc+", float("nan")),
            float("nan"), r.get("CSW%", float("nan")), float("nan"))[0], axis=1)
    fig = plus_leaderboard_figure(agg, "Loc+", "Fordham Baseball - Location+ Leaderboard", top_n=top_n)
    st.pyplot(fig)
    st.dataframe(
        style_scouting_dataframe(_table_columns(agg, ["Rank", "Pitcher", "Stuff Grade", "Outing Grade", "Loc+", "Stuff+", "Pitches", "Primary Pitch", "Velo", "CSW%"]), context="pitching"),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------
# PAGE 4b - PITCH EFFICIENCY LEADERBOARD
# ------------------------------------------------------------
def pitch_efficiency_leaderboard_page():
    st.title("Pitch Efficiency Leaderboard")
    st.caption("Ranked by Pitches per Inning (P/IP). Lower = more efficient. "
               "A ≤14.5 · A- ≤16.5 · B+ ≤18.5 · B ≤20.5 · B- ≤22.5")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    min_pitches = st.slider("Minimum pitches", 10, 250, 25, 5, key="eff_min")
    lb = pitch_efficiency_leaderboard(df, min_pitches=min_pitches)

    if lb.empty:
        st.info("No pitchers meet the minimum pitch threshold.")
        return

    # Color the grade column
    grade_colors = {
        "A":  "#22c55e", "A-": "#4ade80",
        "B+": "#86efac", "B":  "#bef264", "B-": "#fde047",
        "C+": "#fb923c", "C":  "#f97316", "C-": "#ef4444",
        "D":  "#dc2626", "F":  "#991b1b",
    }

    def _style_eff(row):
        styles = [""] * len(row)
        if "Eff Grade" in lb.columns:
            gi = list(lb.columns).index("Eff Grade")
            g  = row.get("Eff Grade", "")
            c  = grade_colors.get(g, "")
            if c:
                styles[gi] = f"background-color:{c}22;color:{c};font-weight:900"
        return styles

    st.dataframe(
        lb.style.apply(_style_eff, axis=1),
        use_container_width=True,
        hide_index=True,
        height=min(700, 42 + len(lb) * 36),
    )
    st.caption(f"{len(lb)} pitcher(s) · minimum {min_pitches} pitches · sorted by P/IP (best first)")


# ------------------------------------------------------------
# PAGE 5 - PITCH-TYPE LEADERBOARDS
# ------------------------------------------------------------
def pitch_type_plus_leaderboard(df: pd.DataFrame, min_pitches=10) -> pd.DataFrame:
    if df.empty or "Pitcher" not in df.columns or "pitch_abbr" not in df.columns:
        return pd.DataFrame()

    base = df.copy()
    for col in ["Stuff+", "Loc+", "Velo", "IVB", "HB", "in_zone", "is_swing", "is_whiff"]:
        if col not in base.columns:
            base[col] = np.nan
    if "BatterSide" in base.columns:
        side = base["BatterSide"].astype(str).str.upper()
        df_lhh = base[side.str.startswith("L")]
        df_rhh = base[side.str.startswith("R")]
    else:
        df_lhh = base.iloc[0:0].copy()
        df_rhh = base.iloc[0:0].copy()

    agg = base.groupby(["Pitcher", "pitch_abbr"]).agg(
        Stuff_plus=("Stuff+", "mean"),
        Loc_plus=("Loc+", "mean"),
        Pitches=("pitch_abbr", "count"),
        Velo=("Velo", "mean"),
        IVB=("IVB", "mean"),
        HB=("HB", "mean"),
        Zone=("in_zone", "mean"),
        Swings=("is_swing", "sum"),
        Whiffs=("is_whiff", "sum"),
    ).reset_index()
    agg["Zone%"] = agg["Zone"] * 100
    agg["Whiff%"] = np.where(agg["Swings"] > 0, agg["Whiffs"] / agg["Swings"] * 100, np.nan)

    split_parts = []
    for label, split_df in [("LHH", df_lhh), ("RHH", df_rhh)]:
        if split_df.empty:
            continue
        split = split_df.groupby(["Pitcher", "pitch_abbr"]).agg(
            **{f"Stuff+ {label}": ("Stuff+", "mean"), f"Loc+ {label}": ("Loc+", "mean"), f"N {label}": ("pitch_abbr", "count")}
        ).reset_index()
        split_parts.append(split)
    for split in split_parts:
        agg = agg.merge(split, on=["Pitcher", "pitch_abbr"], how="left")

    agg = agg.rename(columns={"pitch_abbr": "Pitch", "Stuff_plus": "Stuff+", "Loc_plus": "Loc+"})
    for col in ["Stuff+ LHH", "Stuff+ RHH", "Loc+ LHH", "Loc+ RHH", "N LHH", "N RHH"]:
        if col not in agg.columns:
            agg[col] = np.nan
    agg = agg[pd.to_numeric(agg["Pitches"], errors="coerce").fillna(0) >= min_pitches].copy()
    return agg.round(1)


def pitch_type_grid_figure(board: pd.DataFrame, metric_col: str, pitch_types, top_n=8):
    pitch_types = list(pitch_types)
    n = max(1, len(pitch_types))
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.4 * rows), squeeze=False)
    fig.patch.set_facecolor("#100D0C")
    fig.suptitle(f"Fordham Pitch-Type {metric_col} Leaderboards", color="#FFF7E8", fontsize=21, fontweight="bold", y=0.965)

    for ax in axes.flatten():
        ax.set_facecolor("#171514")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#4E4036")
            spine.set_linewidth(1.0)

    for ax, pitch in zip(axes.flatten(), pitch_types):
        color = PITCH_TYPE_COLORS.get(str(pitch), FORDHAM_GOLD)
        sub = board[board["Pitch"].astype(str).eq(str(pitch))].sort_values(metric_col, ascending=False).head(top_n)
        ax.text(0.04, 0.94, str(pitch), color=color, fontsize=20, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.18, 0.94, f"Top {top_n} {metric_col}", color="#FFF7E8", fontsize=12, fontweight="bold", transform=ax.transAxes, va="top")
        ax.plot([0.04, 0.96], [0.875, 0.875], color=color, linewidth=1.4, transform=ax.transAxes)

        if sub.empty:
            ax.text(0.5, 0.48, "No qualified pitches", color="#CDBFAF", fontsize=11, ha="center", va="center", transform=ax.transAxes)
            continue

        split_l = f"{metric_col} LHH"
        split_r = f"{metric_col} RHH"
        y = 0.805
        row_h = 0.087
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            bg = "#211C1A" if rank % 2 else "#181412"
            ax.add_patch(plt.Rectangle((0.035, y - 0.045), 0.93, 0.066, facecolor=bg, edgecolor="#322923", linewidth=0.5, transform=ax.transAxes))
            ax.text(0.055, y, f"{rank}", color=FORDHAM_GOLD, fontsize=10, fontweight="bold", va="center", transform=ax.transAxes)
            ax.text(0.115, y, textwrap.shorten(str(row.get("Pitcher", "-")), width=19, placeholder="..."), color="#FFF7E8", fontsize=10, fontweight="bold", va="center", transform=ax.transAxes)
            ax.text(0.56, y, _fmt_pdf_value(row.get(metric_col), metric_col), color=color, fontsize=12, fontweight="bold", ha="right", va="center", transform=ax.transAxes)
            ax.text(0.62, y + 0.012, f"L {_fmt_pdf_value(row.get(split_l), metric_col)}", color="#9FC7FF", fontsize=7.8, va="center", transform=ax.transAxes)
            ax.text(0.62, y - 0.016, f"R {_fmt_pdf_value(row.get(split_r), metric_col)}", color="#FFB1A8", fontsize=7.8, va="center", transform=ax.transAxes)
            ax.text(0.94, y, f"{int(row.get('Pitches', 0))} P", color="#CDBFAF", fontsize=8.5, ha="right", va="center", transform=ax.transAxes)
            y -= row_h

    for ax in axes.flatten()[len(pitch_types):]:
        ax.axis("off")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.925])
    return fig


def pitchtype_grids_page():
    st.title("Pitch-Type Leaderboards")

    df = prepare_data()
    df = filter_fordham_only(df)

    if df.empty:
        st.error("No FOR_RAM pitcher data found.")
        return

    if "pitch_abbr" not in df.columns:
        st.error("Pitch type data is missing from the loaded TrackMan files.")
        return

    c1, c2 = st.columns([0.9, 0.8])
    with c1:
        metric_col = st.radio("Metric", ["Stuff+", "Loc+"], horizontal=True, key="pt_metric")
    with c2:
        min_pitches = st.slider("Minimum pitches per type", 5, 75, 10, 5, key="pt_min")

    board = pitch_type_plus_leaderboard(df, min_pitches=min_pitches)
    if board.empty:
        st.warning("No qualified pitch-type rows for the selected minimum.")
        return

    all_pitch_types = sorted(board["Pitch"].dropna().astype(str).unique())
    default_pitch_types = all_pitch_types[:6]
    st.markdown("### Pitch Types")
    pitch_cols = st.columns(min(6, max(1, len(all_pitch_types))))
    selected_pitch_types = []
    for i, pitch in enumerate(all_pitch_types):
        with pitch_cols[i % len(pitch_cols)]:
            if st.checkbox(str(pitch), value=str(pitch) in default_pitch_types, key=f"pt_type_{pitch}"):
                selected_pitch_types.append(str(pitch))
    if not selected_pitch_types:
        st.warning("Choose at least one pitch type.")
        return

    top_n = st.slider("Rows per pitch type", 3, 12, 8, 1, key="pt_top_n")
    fig = pitch_type_grid_figure(board, metric_col, selected_pitch_types, top_n=top_n)
    st.pyplot(fig)

    detail_cols = [
        "Pitch", "Pitcher", "Pitches", "Stuff+", "Stuff+ LHH", "Stuff+ RHH",
        "Loc+", "Loc+ LHH", "Loc+ RHH", "Velo", "IVB", "HB", "Zone%", "Whiff%"
    ]
    detail = _table_columns(board[board["Pitch"].astype(str).isin(selected_pitch_types)], detail_cols)
    detail = detail.sort_values(["Pitch", metric_col], ascending=[True, False])
    st.dataframe(style_scouting_dataframe(detail, context="pitching"), use_container_width=True, hide_index=True)
    
# ------------------------------------------------------------
# PAGE 6 - PITCHER PROFILE
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
    df = pitcher_df.dropna(subset=["HB", "IVB"]).copy()
    fig = plt.figure(figsize=(9.2, 6.8))
    fig.patch.set_facecolor("#100D0C")
    gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.15], left=0.08, right=0.96, top=0.90, bottom=0.12, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_key = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#181412")
    ax_key.set_facecolor("#181412")

    if df.empty:
        ax.text(0.5, 0.5, "No movement data", ha="center", va="center", color="#FFF7E8", transform=ax.transAxes)
        ax.set_axis_off()
        ax_key.set_axis_off()
        return fig

    pitch_colors = {
        "FB": "#1f77b4",
        "SI": "#17becf",
        "FC": "#ff9f1c",
        "SL": "#e63946",
        "CU": "#7b2cbf",
        "CH": "#2a9d8f",
        "SW": "#b56576",
    }
    df["HB"] = pd.to_numeric(df["HB"], errors="coerce")
    df["IVB"] = pd.to_numeric(df["IVB"], errors="coerce")
    df = df.dropna(subset=["HB", "IVB"])
    if df.empty:
        ax.text(0.5, 0.5, "No movement data", ha="center", va="center", color="#FFF7E8", transform=ax.transAxes)
        ax.set_axis_off()
        ax_key.set_axis_off()
        return fig

    x_abs = max(25, float(np.nanmax(np.abs(df["HB"]))) + 3)
    y_lo = min(-25, float(df["IVB"].min()) - 3)
    y_hi = max(25, float(df["IVB"].max()) + 3)
    x_lim = float(np.ceil(x_abs / 5) * 5)
    y_min = float(np.floor(y_lo / 5) * 5)
    y_max = float(np.ceil(y_hi / 5) * 5)

    throws = str(df["PitcherThrows"].dropna().iloc[0]) if "PitcherThrows" in df.columns and df["PitcherThrows"].notna().any() else "Right"
    if throws.upper().startswith("R"):
        ax.axvspan(0, x_lim, color="#13365F", alpha=0.18, zorder=0)
        ax.axvspan(-x_lim, 0, color="#5E1814", alpha=0.18, zorder=0)
        ax.text(x_lim * 0.55, y_max - 2, "ARM SIDE", color="#9FC7FF", fontsize=9, fontweight="bold", ha="center")
        ax.text(-x_lim * 0.55, y_max - 2, "GLOVE SIDE", color="#FFB1A8", fontsize=9, fontweight="bold", ha="center")
    else:
        ax.axvspan(-x_lim, 0, color="#13365F", alpha=0.18, zorder=0)
        ax.axvspan(0, x_lim, color="#5E1814", alpha=0.18, zorder=0)
        ax.text(-x_lim * 0.55, y_max - 2, "ARM SIDE", color="#9FC7FF", fontsize=9, fontweight="bold", ha="center")
        ax.text(x_lim * 0.55, y_max - 2, "GLOVE SIDE", color="#FFB1A8", fontsize=9, fontweight="bold", ha="center")

    for pitch, sub in df.groupby("pitch_abbr", sort=False):
        color = pitch_colors.get(pitch, "#CDBFAF")
        ax.scatter(sub["HB"], sub["IVB"], s=32, alpha=0.28, color=color, edgecolor="none", zorder=2)

    centroids = (
        df.groupby("pitch_abbr")
        .agg(
            N=("pitch_abbr", "count"),
            HB=("HB", "mean"),
            IVB=("IVB", "mean"),
            Velo=("Velo", "mean") if "Velo" in df.columns else ("HB", "count"),
        )
        .reset_index()
        .sort_values("N", ascending=False)
    )

    for _, row in centroids.iterrows():
        pitch = row["pitch_abbr"]
        color = pitch_colors.get(pitch, "#CDBFAF")
        ax.scatter(row["HB"], row["IVB"], s=430, color=color, edgecolor="#FFF7E8", linewidth=1.4, zorder=5)
        ax.text(row["HB"], row["IVB"], pitch, color="#FFFFFF", fontsize=13, fontweight="bold", ha="center", va="center", zorder=6)

    ax.axhline(0, color="#FFF7E8", linewidth=1.35, alpha=0.72, zorder=1)
    ax.axvline(0, color="#FFF7E8", linewidth=1.35, alpha=0.72, zorder=1)
    ax.grid(True, color="#C7A45D", alpha=0.15, linewidth=0.8)
    ax.set_title("Pitch Break Plot", color="#FFF7E8", fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("Horizontal Break (inches)", color="#CDBFAF", fontsize=10, fontweight="bold")
    ax.set_ylabel("Induced Vertical Break (inches)", color="#CDBFAF", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#CDBFAF", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#C7A45D")
        spine.set_linewidth(1.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(y_min, y_max)

    ax_key.axis("off")
    ax_key.set_title("Pitch Averages", color="#FFF7E8", fontsize=12, fontweight="bold", loc="left", pad=10)
    y = 0.91
    for _, row in centroids.iterrows():
        pitch = row["pitch_abbr"]
        color = pitch_colors.get(pitch, "#CDBFAF")
        ax_key.add_patch(plt.Rectangle((0.02, y - 0.045), 0.96, 0.072, facecolor="#211C1A", edgecolor="#4E4036", linewidth=0.7, transform=ax_key.transAxes))
        ax_key.scatter(0.09, y - 0.010, s=120, color=color, edgecolor="#FFF7E8", linewidth=0.8, transform=ax_key.transAxes)
        ax_key.text(0.18, y + 0.012, f"{pitch}  N={int(row['N'])}", color="#FFF7E8", fontsize=9.4, fontweight="bold", transform=ax_key.transAxes, va="center")
        ax_key.text(
            0.18, y - 0.022,
            f"HB {_fmt_pdf_value(row['HB'], 'HB')}  IVB {_fmt_pdf_value(row['IVB'], 'IVB')}  V {_fmt_pdf_value(row['Velo'], 'Velo')}",
            color="#CDBFAF", fontsize=7.8, transform=ax_key.transAxes, va="center"
        )
        y -= 0.088
        if y < 0.08:
            break

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
    # PITCH BREAK PLOT
    # -----------------------------
    st.subheader("Pitch Break Plot")
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


def _build_umpire_from_df(df: pd.DataFrame) -> dict:
    """Core umpire scorecard computation from a pre-loaded DataFrame."""

    ZONE_LEFT, ZONE_RIGHT = -0.83, 0.83
    ZONE_BOTTOM, ZONE_TOP = 1.5, 3.5
    TOUCH_MARGIN = 0.15

    def in_zone(row):
        x = pd.to_numeric(row.get("PlateLocSide"), errors="coerce")
        y = pd.to_numeric(row.get("PlateLocHeight"), errors="coerce")
        if pd.isna(x) or pd.isna(y):
            return False
        in_main = ZONE_LEFT <= x <= ZONE_RIGHT and ZONE_BOTTOM <= y <= ZONE_TOP
        touching = (
            (ZONE_LEFT - TOUCH_MARGIN <= x <= ZONE_RIGHT + TOUCH_MARGIN) and
            (ZONE_BOTTOM - TOUCH_MARGIN <= y <= ZONE_TOP + TOUCH_MARGIN)
        )
        return in_main or touching

    for col in ["PlateLocSide", "PlateLocHeight"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["PitchCall", "PitcherTeam", "BatterTeam", "Pitcher", "Batter", "Inning"]:
        if col not in df.columns:
            df[col] = ""

    df["InZone"] = df.apply(in_zone, axis=1)
    called_df = df[df["PitchCall"].isin(["StrikeCalled", "BallCalled"])].copy()
    called_df["Correct"] = (
        (called_df["PitchCall"] == "StrikeCalled") & (called_df["InZone"]) |
        (called_df["PitchCall"] == "BallCalled") & (~called_df["InZone"])
    )
    called_df["MissType"] = np.select(
        [
            called_df["Correct"],
            called_df["PitchCall"].eq("StrikeCalled") & ~called_df["InZone"],
            called_df["PitchCall"].eq("BallCalled") & called_df["InZone"],
        ],
        ["Correct", "Bad Strike", "Bad Ball"],
        default="Missed Call",
    )

    def favor_team(row):
        if row["Correct"]:
            return "None"
        if row["PitchCall"] == "StrikeCalled":
            return row["PitcherTeam"]
        else:
            return row["BatterTeam"]

    called_df["FavoredTeam"] = called_df.apply(favor_team, axis=1)
    called_df["HurtTeam"] = np.where(
        called_df["Correct"],
        "None",
        np.where(called_df["PitchCall"].eq("StrikeCalled"), called_df["BatterTeam"], called_df["PitcherTeam"])
    )

    def pct(mask):
        sample = called_df[mask]
        return float(sample["Correct"].mean() * 100) if len(sample) else 0.0

    fordham_team = "FOR_RAM"
    if "HomeTeam" in df.columns and df["HomeTeam"].notna().any():
        home_team = str(df["HomeTeam"].dropna().iloc[0])
    else:
        home_team = ""
    if "AwayTeam" in df.columns and df["AwayTeam"].notna().any():
        away_team = str(df["AwayTeam"].dropna().iloc[0])
    else:
        away_team = ""
    if home_team == "FOR_RAM":
        opponent_team = away_team
    elif away_team == "FOR_RAM":
        opponent_team = home_team
    elif "BatterTeam" in df.columns:
        opponents = [t for t in df["BatterTeam"].dropna().astype(str).unique() if t != "FOR_RAM"]
        opponent_team = opponents[0] if opponents else away_team or home_team
    else:
        opponent_team = away_team or home_team

    favor_counts = called_df["FavoredTeam"].value_counts()
    hurt_counts = called_df["HurtTeam"].value_counts()
    fordham_net = int(favor_counts.get("FOR_RAM", 0) - hurt_counts.get("FOR_RAM", 0))
    game_date_raw = df["Date"].iloc[0] if "Date" in df.columns and len(df) else ""
    game_date = pd.to_datetime(game_date_raw, errors="coerce")
    game_date_label = game_date.strftime("%B %d, %Y") if pd.notna(game_date) else str(game_date_raw)

    metrics = {
        "home_team": home_team,
        "away_team": away_team,
        "home_team_id": first_nonempty_value(df, ["HomeTeamForeignID", "HomeTeamId", "HomeTeamID"], ""),
        "away_team_id": first_nonempty_value(df, ["AwayTeamForeignID", "AwayTeamId", "AwayTeamID"], ""),
        "game_id": first_nonempty_value(df, ["GameID"], ""),
        "game_uid": first_nonempty_value(df, ["GameUID"], ""),
        "game_foreign_id": first_nonempty_value(df, ["GameForeignID"], ""),
        "fordham_team": fordham_team,
        "opponent_team": opponent_team,
        "game_date": game_date_label,
        "called_pitches": len(called_df),
        "overall_accuracy": float(called_df["Correct"].mean() * 100) if len(called_df) else 0.0,
        "called_strike_accuracy": pct(called_df["PitchCall"].eq("StrikeCalled")),
        "called_ball_accuracy": pct(called_df["PitchCall"].eq("BallCalled")),
        "missed_calls": int((~called_df["Correct"]).sum()) if len(called_df) else 0,
        "bad_strikes": int(called_df["MissType"].eq("Bad Strike").sum()) if len(called_df) else 0,
        "bad_balls": int(called_df["MissType"].eq("Bad Ball").sum()) if len(called_df) else 0,
        "fordham_favor": int(favor_counts.get("FOR_RAM", 0)),
        "fordham_hurt": int(hurt_counts.get("FOR_RAM", 0)),
        "fordham_net": fordham_net,
        "zone": (ZONE_LEFT, ZONE_RIGHT, ZONE_BOTTOM, ZONE_TOP, TOUCH_MARGIN),
    }

    missed_cols = [
        "Inning", "MissType", "PitchCall", "PlateLocSide", "PlateLocHeight",
        "Pitcher", "Batter", "PitcherTeam", "BatterTeam", "FavoredTeam", "HurtTeam"
    ]
    missed = called_df[~called_df["Correct"]][missed_cols].copy()
    missed = missed.rename(columns={
        "MissType": "Miss",
        "PitchCall": "Call",
        "PlateLocSide": "Side",
        "PlateLocHeight": "Height",
        "PitcherTeam": "Pitch Team",
        "BatterTeam": "Bat Team",
        "FavoredTeam": "Favored",
        "HurtTeam": "Hurt",
    })
    for col in ["Side", "Height"]:
        if col in missed.columns:
            missed[col] = missed[col].round(2)

    return {"raw": df, "called": called_df, "missed": missed, "metrics": metrics}


def build_umpire_scorecard_data(csv_path):
    return _build_umpire_from_df(_read_csv_fast(csv_path))


def _scorecard_rate_color(value):
    if value >= 92:
        return "#D62828"
    if value >= 87:
        return "#C7A45D"
    return "#3A5F9B"


def generate_umpire_scorecard(csv_path_or_scorecard):
    if isinstance(csv_path_or_scorecard, dict):
        scorecard = csv_path_or_scorecard
    else:
        scorecard = build_umpire_scorecard_data(csv_path_or_scorecard)
    called_df = scorecard["called"]
    missed = scorecard["missed"]
    metrics = scorecard["metrics"]
    ZONE_LEFT, ZONE_RIGHT, ZONE_BOTTOM, ZONE_TOP, TOUCH_MARGIN = metrics["zone"]

    table_rows = max(1, len(missed))
    fig_height = min(22, max(11, 9.5 + table_rows * 0.28))
    fig = plt.figure(figsize=(14, fig_height), facecolor="#100D0C")
    gs = fig.add_gridspec(3, 4, left=0.045, right=0.965, top=0.89, bottom=0.06, hspace=0.35, wspace=0.28, height_ratios=[0.8, 2.25, 1.35])

    fig.text(0.045, 0.955, "FORDHAM BASEBALL UMPIRE SCORECARD", color="#FFF7E8", fontsize=20, fontweight="bold", ha="left")
    fig.text(
        0.045, 0.925,
        f"{metrics['game_date']} | Fordham vs {team_display_name(metrics['opponent_team'])}",
        color="#CDBFAF",
        fontsize=11,
        fontweight="bold",
        ha="left",
    )
    tag_line_parts = []
    if metrics["home_team"]:
        tag_line_parts.append(f"Home {team_tag_label(metrics['home_team'])}")
    if metrics["away_team"]:
        tag_line_parts.append(f"Away {team_tag_label(metrics['away_team'])}")
    if tag_line_parts:
        fig.text(
            0.045,
            0.902,
            " | ".join(tag_line_parts),
            color="#CDBFAF",
            fontsize=8.8,
            fontweight="bold",
            ha="left",
        )

    card_items = [
        ("Called Pitches", metrics["called_pitches"], "#211C1A"),
        ("Overall Accuracy", f"{metrics['overall_accuracy']:.1f}%", _scorecard_rate_color(metrics["overall_accuracy"])),
        ("Missed Calls", metrics["missed_calls"], "#D62828" if metrics["missed_calls"] else "#211C1A"),
        ("Net Fordham", f"{metrics['fordham_net']:+d}", "#D62828" if metrics["fordham_net"] > 0 else "#3A5F9B" if metrics["fordham_net"] < 0 else "#211C1A"),
    ]
    for i, (label, value, color) in enumerate(card_items):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.08), 1, 0.82, facecolor=color, edgecolor=FORDHAM_GOLD, linewidth=1.2, transform=ax.transAxes))
        ax.text(0.06, 0.62, str(value), color="#FFF7E8", fontsize=22, fontweight="bold", transform=ax.transAxes, ha="left", va="center")
        ax.text(0.06, 0.28, label, color="#F3DFC2", fontsize=9.5, fontweight="bold", transform=ax.transAxes, ha="left", va="center")

    ax_zone = fig.add_subplot(gs[1, :2])
    ax_zone.set_facecolor("#171514")
    ax_zone.set_title("Called Pitch Map", color="#FFF7E8", fontsize=15, fontweight="bold", loc="left", pad=10)
    ax_zone.set_xlim(-2.25, 2.25)
    ax_zone.set_ylim(0, 4.8)
    ax_zone.set_aspect("equal")
    ax_zone.tick_params(colors="#CDBFAF", labelsize=8)
    for spine in ax_zone.spines.values():
        spine.set_color("#4E4036")

    zone_x = [ZONE_LEFT, ZONE_RIGHT, ZONE_RIGHT, ZONE_LEFT, ZONE_LEFT]
    zone_y = [ZONE_BOTTOM, ZONE_BOTTOM, ZONE_TOP, ZONE_TOP, ZONE_BOTTOM]
    ax_zone.plot(zone_x, zone_y, color="#FFF7E8", linewidth=2.2)
    buffer_x = [ZONE_LEFT - TOUCH_MARGIN, ZONE_RIGHT + TOUCH_MARGIN, ZONE_RIGHT + TOUCH_MARGIN, ZONE_LEFT - TOUCH_MARGIN, ZONE_LEFT - TOUCH_MARGIN]
    buffer_y = [ZONE_BOTTOM - TOUCH_MARGIN, ZONE_BOTTOM - TOUCH_MARGIN, ZONE_TOP + TOUCH_MARGIN, ZONE_TOP + TOUCH_MARGIN, ZONE_BOTTOM - TOUCH_MARGIN]
    ax_zone.plot(buffer_x, buffer_y, color=FORDHAM_GOLD, linestyle="--", linewidth=1.1, alpha=0.75)
    plate_x = [-0.85, 0.85, 0.55, 0.0, -0.55]
    plate_y = [0.05, 0.05, 0.25, 0.37, 0.25]
    ax_zone.fill(plate_x, plate_y, facecolor="#FFF7E8", edgecolor="#100D0C", linewidth=1.6, zorder=4)
    ax_zone.grid(color="#4E4036", alpha=0.35, linewidth=0.7)

    styles = {
        "Correct": ("#58B368", "o", 42),
        "Bad Strike": ("#F4A261", "X", 82),
        "Bad Ball": ("#D62828", "o", 78),
    }
    for miss_type, sub in called_df.groupby("MissType"):
        color, marker, size = styles.get(miss_type, ("#CDBFAF", "o", 50))
        ax_zone.scatter(
            sub["PlateLocSide"], sub["PlateLocHeight"], s=size, color=color,
            marker=marker, edgecolor="#FFF7E8", linewidth=0.7, alpha=0.9, label=miss_type
        )
    ax_zone.legend(loc="upper right", facecolor="#211C1A", edgecolor="#4E4036", labelcolor="#FFF7E8", fontsize=8)

    ax_breakdown = fig.add_subplot(gs[1, 2:])
    ax_breakdown.set_facecolor("#171514")
    ax_breakdown.set_title("Call Breakdown", color="#FFF7E8", fontsize=15, fontweight="bold", loc="left", pad=10)
    labels = ["Correct", "Bad Strike", "Bad Ball"]
    values = [
        int(called_df["MissType"].eq("Correct").sum()),
        metrics["bad_strikes"],
        metrics["bad_balls"],
    ]
    colors = ["#58B368", "#F4A261", "#D62828"]
    ax_breakdown.barh(labels, values, color=colors, edgecolor="#FFF7E8", linewidth=0.6)
    ax_breakdown.tick_params(colors="#CDBFAF")
    ax_breakdown.grid(axis="x", color="#4E4036", alpha=0.3)
    for spine in ax_breakdown.spines.values():
        spine.set_color("#4E4036")
    for i, value in enumerate(values):
        ax_breakdown.text(value + max(values + [1]) * 0.02, i, str(value), color="#FFF7E8", va="center", fontweight="bold")
    ax_breakdown.text(
        0.02, -0.22,
        f"Called strike accuracy: {metrics['called_strike_accuracy']:.1f}%\n"
        f"Called ball accuracy: {metrics['called_ball_accuracy']:.1f}%\n"
        f"Fordham favored: {metrics['fordham_favor']} | hurt: {metrics['fordham_hurt']}",
        color="#CDBFAF",
        fontsize=10,
        transform=ax_breakdown.transAxes,
        va="top",
    )

    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    ax_table.set_title(
        f"Missed Calls ({len(missed)})",
        color="#FFF7E8",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=8
    )
    table_view = missed.copy()
    if table_view.empty:
        ax_table.text(0.5, 0.43, "No missed calls detected", color="#CDBFAF", fontsize=18, ha="center", va="center", transform=ax_table.transAxes)
    else:
        keep_cols = ["Inning", "Miss", "Call", "Side", "Height", "Pitcher", "Batter", "Favored", "Hurt"]
        table_view = table_view[[c for c in keep_cols if c in table_view.columns]]
        for col in ["Pitcher", "Batter"]:
            if col in table_view.columns:
                table_view[col] = table_view[col].map(lambda x: textwrap.shorten(str(x), width=24, placeholder="..."))
        table_font = max(5.2, min(8.3, 11.5 - len(table_view) * 0.12))
        tbl = ax_table.table(
            cellText=table_view.values,
            colLabels=table_view.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
            bbox=[0, 0, 1, 0.82],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(table_font)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#4E4036")
            cell.set_linewidth(0.6)
            cell.set_height(0.82 / (len(table_view) + 1))
            if r == 0:
                cell.set_facecolor(FORDHAM_MAROON)
                cell.set_text_props(color="#FFF7E8", weight="bold")
            else:
                cell.set_facecolor("#211C1A" if r % 2 else "#171514")
                cell.set_text_props(color="#F8EFE2")

    output_dir = Path("output/umpire_scorecards")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_date = re.sub(r"[^A-Za-z0-9]+", "_", metrics["game_date"]).strip("_") or "game"
    out = output_dir / f"UmpireScorecard_{safe_date}.png"
    fig.savefig(out, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out, fig

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

COLLEGE_AVG_WOBA = 0.325
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

    # D1 collegiate weights (derived from run-expectancy research, 2026 calibrated)
    wBB  = 0.64
    wHBP = 0.66
    w1B  = 0.80
    w2B  = 1.12
    w3B  = 1.41
    wHR  = 1.76

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
    denominator = AB + BB + HBP  # exclude SF — tagger inconsistency inflates denominator

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
        "InPlayOut", "InPlayRun",
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
# CONTACT QUALITY (LEADERBOARD TAB) - EV/LA FLAGS
# ============================================================

def add_contact_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "EV" not in df.columns and "ExitSpeed" in df.columns:
        df["EV"] = df["ExitSpeed"]
    if "LA" not in df.columns and "Angle" in df.columns:
        df["LA"] = df["Angle"]
    if "EV" not in df.columns:
        df["EV"] = np.nan
    if "LA" not in df.columns:
        df["LA"] = np.nan

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
    if "in_zone" not in df.columns:
        df["in_zone"] = (
            df["PlateLocSide"].between(-0.83, 0.83) &
            df["PlateLocHeight"].between(1.5, 3.5)
        )
    if "in_zone" not in df.columns:
        df["in_zone"] = (
            df["PlateLocSide"].between(-0.83, 0.83) &
            df["PlateLocHeight"].between(1.5, 3.5)
        ).astype(int)

    df["is_swing"] = df["is_swing"].fillna(0).astype(int)
    df["is_whiff"] = df["is_whiff"].fillna(0).astype(int)
    # Chase = any out-of-zone swing (not just whiffs — that was wrong)
    df["is_chase"] = (
        (df["is_swing"] == 1) &
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
            "Bat+": player_wrc_plus,
            "HR":    homers,
            "xHB":   doubles + triples + homers,
            "BABIP": round((H - homers) / (AB - K - homers), 3) if (AB - K - homers) > 0 else np.nan,
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
    card["H"]   = card["1B"] + card["2B"] + card["3B"] + card["HR"]
    card["xHB"] = card["2B"] + card["3B"] + card["HR"]   # extra base hits

    SF = (pa_end.get("PlayResult", "") == "Sacrifice").sum()
    card["AB"] = card["PA"] - card["BB"] - card["HBP"] - SF

    card["BB%"] = round(card["BB"] / card["PA"] * 100, 1) if card["PA"] else 0.0
    card["K%"] = round(card["K"] / card["PA"] * 100, 1) if card["PA"] else 0.0

    player_woba = compute_woba(hdf)
    card["wOBA"] = round(player_woba, 3)
    card["Bat+"] = compute_wrc_plus(player_woba, lgwOBA)

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

    swings        = hdf["is_swing"].sum() if "is_swing" in hdf.columns else 0
    total_pitches = len(hdf)
    pc_h          = hdf.get("PitchCall", pd.Series("", index=hdf.index)).astype(str)
    contact_calls = {"FoulBall","FoulBallNotFieldable","FoulBallFieldable","FoulTip",
                     "InPlay","InPlayNoOut","InPlayOut","InPlayRun"}
    is_contact_h  = pc_h.isin(contact_calls)

    card["Swing%"]  = round(swings / total_pitches * 100, 1) if total_pitches else 0.0
    card["Whiff%"]  = round(hdf["is_whiff"].sum() / swings * 100, 1) if swings else 0.0
    card["Chase%"]  = round(hdf["is_chase"].sum() / swings * 100, 1) if swings else 0.0
    card["Contact%"]= round(is_contact_h.sum() / swings * 100, 1) if swings else 0.0

    if "in_zone" in hdf.columns:
        in_z_h   = hdf["in_zone"].astype(bool)
        is_sw_h  = hdf["is_swing"].astype(bool) if "is_swing" in hdf.columns else pd.Series(False, index=hdf.index)
        z_sw     = (is_sw_h & in_z_h).sum()
        o_sw     = (is_sw_h & ~in_z_h).sum()
        z_ct     = (is_contact_h & in_z_h).sum()
        o_ct     = (is_contact_h & ~in_z_h).sum()
        z_pitch  = in_z_h.sum()
        o_pitch  = (~in_z_h).sum()
        card["ZSwing%"]   = round(z_sw / z_pitch * 100, 1) if z_pitch else np.nan
        card["OSwing%"]   = round(o_sw / o_pitch * 100, 1) if o_pitch else np.nan
        card["ZContact%"] = round(z_ct / z_sw  * 100, 1) if z_sw  else np.nan
        card["OContact%"] = round(o_ct / o_sw  * 100, 1) if o_sw  else np.nan
    else:
        card["ZSwing%"] = card["OSwing%"] = card["ZContact%"] = card["OContact%"] = np.nan

    # EV 90th percentile
    if not bip.empty and len(bip) >= 10:
        card["EV90"] = round(bip["EV"].quantile(0.90), 1)
    else:
        card["EV90"] = np.nan

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
    if "in_zone" not in df.columns:
        df["in_zone"] = (
            df["PlateLocSide"].between(-0.83, 0.83) &
            df["PlateLocHeight"].between(1.5, 3.5)
        )

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

    elif metric == "Zone%":
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        values = grouped["in_zone"].mean() * 100
        samples = grouped["in_zone"].count().reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "Zone%"
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

    import matplotlib.colors as _mc
    _savant_cmap = _mc.LinearSegmentedColormap.from_list("savant", [
        (0.00, "#0a2e6e"), (0.15, "#1956a0"), (0.35, "#5ea3d0"),
        (0.50, "#787878"), (0.65, "#f5a17a"), (0.85, "#d13c28"),
        (1.00, "#8b0000"),
    ])
    if cmap_name in {"RdYlBu_r", "YlOrRd", "Blues", "RdYlGn"}:
        cmap = _savant_cmap
    else:
        cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#2A2420")

    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    fig.patch.set_facecolor("#100D0C")
    ax.set_facecolor("#181412")

    masked_grid = np.ma.masked_invalid(grid)
    im = ax.pcolormesh(
        x_edges, y_edges, masked_grid,
        cmap=cmap, shading="flat",
        edgecolors="#100D0C", linewidth=2.0,
        vmin=vmin, vmax=vmax
    )

    for y_i, y in enumerate(y_centers):
        for x_i, x in enumerate(x_centers):
            val = grid[y_i, x_i]
            n = int(samples[y_i, x_i]) if not np.isnan(samples[y_i, x_i]) else 0
            if np.isnan(val):
                txt = "-\nn=0"
            elif metric == "wOBA":
                txt = f"{val:.3f}\nn={n}"
            else:
                txt = f"{val:.0f}{label_suffix}\nn={n}"
            if np.isnan(val):
                txt_color = "#CDBFAF"
            else:
                norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                norm_val = float(np.clip(norm_val, 0, 1))
                rgba = cmap(norm_val)
                lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                txt_color = "#111111" if lum > 0.50 else "#FFF7E8"
            ax.text(
                x, y, txt,
                ha="center", va="center",
                color=txt_color, fontsize=10.5, fontweight="bold",
                linespacing=1.15
            )

    strike_zone = plt.Rectangle(
        (-0.83, 1.5), 1.66, 2.0,
        fill=False, edgecolor="#C7A45D", linewidth=2.6
    )
    ax.add_patch(strike_zone)

    plate_x = [-0.83, 0.83, 0.83, 0, -0.83, -0.83]
    plate_y = [0, 0, 0.17, 0.34, 0.17, 0]
    ax.plot(plate_x, plate_y, color="#FFF7E8", linewidth=1.8)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.0, 5.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-0.83, 0, 0.83])
    ax.set_yticks([1.5, 2.5, 3.5])
    ax.tick_params(labelsize=8, length=0, colors="#CDBFAF")
    ax.set_xlabel("Plate side - catcher view", fontsize=9, color="#CDBFAF", fontweight="bold")
    ax.set_ylabel("Plate height", fontsize=9, color="#CDBFAF", fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", color="#FFF7E8", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        2.35, 4.75, hitter_side,
        ha="right", va="top",
        fontsize=11, fontweight="bold", color="#FFF7E8",
        bbox=dict(facecolor="#211C1A", edgecolor="#C7A45D", boxstyle="round,pad=0.25")
    )

    if masked_grid.count() > 0:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=9, color="#CDBFAF", fontweight="bold")
        cbar.ax.tick_params(labelsize=8, colors="#CDBFAF")
        cbar.outline.set_edgecolor("#4E4036")

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

    if metric in {"Usage%", "Zone%"}:
        grouped = df.groupby(["y_bin", "x_bin"], observed=False)
        counts = grouped["PitchCall"].count() if "PitchCall" in df.columns else grouped["PlateLocSide"].count()
        values = counts / max(total_pitches, 1) * 100
        samples = counts.reindex(full_index).values.reshape(3, 3)
        grid = values.reindex(full_index).values.reshape(3, 3)
        label_suffix = "%"
        colorbar_label = "Zone%" if metric == "Zone%" else "Pitch%"
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

    import matplotlib.colors as _mc2
    _savant_cmap2 = _mc2.LinearSegmentedColormap.from_list("savant2", [
        (0.00, "#0a2e6e"), (0.15, "#1956a0"), (0.35, "#5ea3d0"),
        (0.50, "#787878"), (0.65, "#f5a17a"), (0.85, "#d13c28"),
        (1.00, "#8b0000"),
    ])
    if cmap_name in {"RdYlBu_r", "YlOrRd", "Blues", "RdYlGn"}:
        cmap = _savant_cmap2
    else:
        cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#2A2420")

    fig, ax = plt.subplots(figsize=(5.15, 5.55))
    fig.patch.set_facecolor("#100D0C")
    ax.set_facecolor("#181412")

    im = ax.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(grid),
        cmap=cmap, shading="flat",
        edgecolors="#100D0C", linewidth=2.6,
        vmin=vmin, vmax=vmax
    )

    for y_i, y in enumerate(y_centers):
        for x_i, x in enumerate(x_centers):
            val = grid[y_i, x_i]
            n = int(samples[y_i, x_i]) if not np.isnan(samples[y_i, x_i]) else 0
            if np.isnan(val):
                text = "-\nn=0"
            elif metric == "wOBA":
                text = f"{val:.3f}\nn={n}"
            else:
                text = f"{val:.0f}{label_suffix}\nn={n}"
            if np.isnan(val):
                txt_color = "#CDBFAF"
            else:
                norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                norm_val = float(np.clip(norm_val, 0, 1))
                rgba = cmap(norm_val)
                lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                txt_color = "#111111" if lum > 0.50 else "#FFF7E8"
            ax.text(
                x, y, text,
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=txt_color, linespacing=1.12
            )

    ax.add_patch(plt.Rectangle((-0.83, 1.5), 1.66, 2.0, fill=False, edgecolor="#C7A45D", linewidth=3.0))
    ax.set_xlim(-0.83, 0.83)
    ax.set_ylim(1.5, 3.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color="#FFF7E8")
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#CDBFAF"
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=9, color="#CDBFAF", fontweight="bold")
    cbar.ax.tick_params(labelsize=8, colors="#CDBFAF")
    cbar.outline.set_edgecolor("#4E4036")

    fig.tight_layout()
    return fig




  

# ============================================================
# HITTER DEVELOPMENT & APPROACH PAGE
# ============================================================

def _legacy_hitter_development_page_basic(all_pitches_df: pd.DataFrame):

    st.title("Hitter Development & Approach")

    df = normalize_hitter_columns(all_pitches_df)
    df = add_contact_quality_local(df)

    # Only FOR_RAM hitters (no pitchers)
    if "BatterTeam" in df.columns:
        df = df[df["BatterTeam"].astype(str).str.upper().str.startswith("FOR_RAM")]

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
        st.metric("xHB", card.get("xHB", card["2B"] + card["3B"] + card["HR"]))

    with c2:
        st.metric("BB%", f"{card['BB%']}%")
        st.metric("K%", f"{card['K%']}%")
        st.metric("Swing%", f"{card['Swing%']}%")
        st.metric("Chase%", f"{card['Chase%']}%")

    with c3:
        st.metric("wOBA", f"{card['wOBA']:.3f}")
        st.metric("Bat+", f"{card.get('Bat+', card.get('wRC+', '-'))}")
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
            contact = "Fly"

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
            "FB%": round((g["ContactType"] == "Fly").mean() * 100, 1) if len(g) else 0,
            "HardHit%": round((g["EV"] >= 95).mean() * 100, 1) if len(g) else 0,
            "AvgEV": round(g["EV"].mean(), 1) if len(g) else np.nan,
        })

    summary = pd.DataFrame(rows)
    spray_summary = summary.sort_values(["BIP", "HardHit%"], ascending=False)
    best = spray_summary.iloc[0]
    second = spray_summary.iloc[1] if len(spray_summary) > 1 else best

    ground = df[df["ContactType"] == "Ground"].copy()
    gb_rate = (df["ContactType"] == "Ground").mean() * 100
    fb_rate = (df["ContactType"] == "Fly").mean() * 100
    hard_rate = (df["EV"] >= 95).mean() * 100
    pull_rate = float(summary.loc[summary["Spray"] == "Pull", "BIP%"].iloc[0])
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

    depth_note = "Outfield no-doubles depth" if fb_rate >= 45 and hard_rate >= 35 else "Normal OF depth"
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
            contact = "Fly"
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
            "FB%": round(g["Contact"].eq("Fly").mean() * 100, 1) if len(g) else 0,
            "HH%": round(g["EV"].ge(95).mean() * 100, 1) if len(g) else 0,
            "AvgEV": round(g["EV"].mean(), 1) if len(g) else np.nan,
        })
    summary = pd.DataFrame(rows)

    ground = df[df["Contact"].eq("GB")]
    gb_rate = df["Contact"].eq("GB").mean() * 100
    fb_rate = df["Contact"].eq("Fly").mean() * 100
    hard_rate = df["EV"].ge(95).mean() * 100
    pull_rate = float(summary.loc[summary["Spray"].eq("Pull"), "BIP%"].iloc[0])
    middle_rate = float(summary.loc[summary["Spray"].eq("Middle"), "BIP%"].iloc[0])
    oppo_rate = float(summary.loc[summary["Spray"].eq("Oppo"), "BIP%"].iloc[0])
    pull_gb = ground["Spray"].eq("Pull").mean() * 100 if len(ground) else 0
    middle_gb = ground["Spray"].eq("Middle").mean() * 100 if len(ground) else 0
    oppo_air = df[df["Contact"].isin(["LD", "Fly"])]["Spray"].eq("Oppo").mean() * 100 if len(df[df["Contact"].isin(["LD", "Fly"])]) else 0

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

    depth = "No-doubles depth" if fb_rate >= 45 and hard_rate >= 35 else "Normal depth"
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


def build_hitter_spray_chart(hdf: pd.DataFrame, hitter: str = "Hitter", annotate_ev: bool = False):
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

    # Fordham logo watermark in center field
    _logo_paths = [
        ROOT / "static" / "rams.png",
        ROOT / "national_pitchingplus_app" / "team_logos" / "FOR_RAM.png",
        ROOT / "national_pitchingplus_app" / "team_logos" / "FOR_RAM1.png",
        ROOT / "assets" / "rams.png",
    ]
    for _lp in _logo_paths:
        if _lp.exists():
            try:
                from PIL import Image as _PILSpray
                _logo_img = np.array(_PILSpray.open(_lp).convert("RGBA"))
                # Centre: x=0, y=1.72 (just above second base, in CF)
                _ext = [-0.42, 0.42, 1.30, 2.14]
                ax.imshow(_logo_img, extent=_ext, aspect="auto", alpha=0.18, zorder=2)
            except Exception:
                pass
            break

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
        ev_label_x, ev_label_y = x, y  # default annotation position
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
            # EV label goes at the end of the line, not at the dot
            ev_label_x, ev_label_y = guide_x, guide_y
        elif la <= 27:
            marker, color = "D", "#F04E45"
        else:
            marker, color = "^", "#8EC5FF"
        size = 38 + max(float(row["EV"]) - 80, 0) * 3.2
        edge = "#FFFFFF" if row["EV"] >= 95 else "#1A1412"
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor=edge, linewidth=0.8, alpha=0.88, zorder=5)
        if annotate_ev and not pd.isna(row["EV"]):
            ax.text(ev_label_x, ev_label_y + 0.07, f"{row['EV']:.0f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#FFF7E8", fontweight="bold", zorder=7,
                    bbox=dict(facecolor="#100D0C", edgecolor="none", alpha=0.55, pad=0.5))

    df.apply(plot_point, axis=1)

    ax.scatter([], [], marker="o", color="#E7C66A", label="Ground ball")
    ax.scatter([], [], marker="D", color="#F04E45", label="Line drive")
    ax.scatter([], [], marker="^", color="#8EC5FF", label="Fly ball")
    ax.scatter([], [], marker="o", color="#222222", edgecolor="#FFFFFF", label="95+ EV")
    ax.legend(loc="lower right", fontsize=8, facecolor="#211C1A", edgecolor="#C7A45D", labelcolor="#FFF7E8")

    ax.text(0, -0.22, "HOME", ha="center", va="center", color="#FFF7E8", fontsize=9, fontweight="bold")

    # Groundball direction annotation
    if "TaggedHitType" in hdf.columns and "Direction" in hdf.columns:
        _gb = hdf[hdf["TaggedHitType"].astype(str).eq("GroundBall")].copy()
        _gb["Direction"] = pd.to_numeric(_gb["Direction"], errors="coerce")
        _gb = _gb.dropna(subset=["Direction"])
        if len(_gb) >= 5:
            _gbn     = len(_gb)
            _gb_ln   = int((_gb["Direction"] < 0).sum())
            _gb_rn   = int((_gb["Direction"] > 0).sum())
            _gbl     = _gb_ln / _gbn * 100
            _gbr     = _gb_rn / _gbn * 100
            ax.text(0, -0.30, f"GROUNDBALL DIRECTION  ({_gbn} total GBs)",
                    ha="center", va="center", color="#BFC7D5", fontsize=7.5,
                    fontweight="bold", zorder=10)
            _box = dict(facecolor="#1A2A1A", edgecolor="#C7A45D", alpha=0.92, boxstyle="round,pad=0.28")
            ax.text(-1.55, -0.50, f"← {_gb_ln} GBs  ({_gbl:.0f}%)\nLeft of 2B  ·  SS / 3B",
                    ha="center", va="center", color="#FFF7E8", fontsize=8.5,
                    fontweight="bold", linespacing=1.35, bbox=_box, zorder=10)
            ax.text(1.55, -0.50, f"{_gb_rn} GBs  ({_gbr:.0f}%) →\nRight of 2B  ·  2B / 1B",
                    ha="center", va="center", color="#FFF7E8", fontsize=8.5,
                    fontweight="bold", linespacing=1.35, bbox=_box, zorder=10)

    ax.set_title(f"Spray Chart - {hitter} ({hitter_side})", color="#FFF7E8", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-0.72, 3.02)
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
        shift_cols = [c for c in ["Spray", "BIP%", "GB%", "FB%", "Shift Read"] if c in shift_view.columns]
        shift_view = shift_view[shift_cols].rename(columns={
            "Spray": "Zone",
            "BIP%": "BIP",
            "GB%": "GB",
            "FB%": "FB",
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
    "BA", "OBP", "SLG", "OPS", "wOBA", "BABIP",
    "BA Allowed", "OBP Allowed", "SLG Allowed", "OPS Allowed"
}
INTEGER_COLS = {
    "N", "PA", "AB", "H", "HR", "xHB", "BB", "K", "BIP", "Swings", "Whiffs", "Chases",
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
        if col_name in INTEGER_COLS or col_name.startswith("N "):
            return f"{int(round(val))}"
        if col_name in RATE_3_DECIMAL_COLS:
            return f"{val:.3f}".replace("0.", ".")
        if col_name.endswith("%") or col_name in {
            "Velo", "PerVelo", "PerceivedVelo", "IVB", "HB", "Spin", "Ext", "RelExt",
            "RelH", "RelHt", "AvgEV", "MaxEV", "AvgLA", "Avg EV", "Max EV", "EV90",
            "Stuff+", "Loc+", "Bat+", "Bat+"
        }:
            return f"{val:.1f}"
        if abs(val) < 1:
            return f"{val:.3f}".replace("0.", ".")
        if abs(val - round(val)) < 0.000001:
            return f"{int(round(val))}"
        return f"{val:.1f}"
    return str(value)


GOOD_HIGH_COLS = {
    "BA", "OBP", "SLG", "OPS", "wOBA", "Bat+", "AvgEV", "MaxEV", "AvgLA",
    "HardHit%", "HH%", "Barrel%", "SweetSpot%", "BABIP", "HR", "xHB",
    "Stuff+", "Loc+", "Strike%", "Zone%", "CSW%", "Whiff%", "K%", "BB%",
    "Swing%", "Usage%", "Velo", "PerVelo", "PerceivedVelo", "IVB", "Ext"
}
GOOD_LOW_COLS = {
    "Chase%", "Avg EV Allowed", "HH% Allowed", "HardHit% Allowed",
    "BA Allowed", "OBP Allowed", "SLG Allowed", "OPS Allowed"
}

# Batting result stats — good when HIGH for hitters, but LOWER is better for pitchers
_BATTING_RESULT_COLS = {
    "BA", "OBP", "SLG", "OPS", "wOBA", "Bat+",
    "AvgEV", "HardHit%", "HH%", "Barrel%", "SweetSpot%", "BABIP"
}


def _metric_direction(col, context=None):
    name = str(col)
    if name.startswith("Stuff+") or name.startswith("Loc+"):
        return 1

    # Pitching context: batting result stats are bad when high (lower BA against = good)
    if context == "pitching":
        if name in {"HR"}:
            return 0                        # show count but don't color-code
        if name in _BATTING_RESULT_COLS:
            return -1                       # lower is better for pitcher
        if name in {"BB%"}:
            return -1
        if name in {"GB%"}:
            return 1                        # more grounders = good for pitcher
        if name in {"K%", "Whiff%", "Strike%", "Zone%", "CSW%"}:
            return 1

    # Hitting context: K%, Whiff%, Chase% are bad for hitters
    if context == "hitting":
        if name in {"K%", "Whiff%", "Chase%"}:
            return -1
        if name in _BATTING_RESULT_COLS:
            return 1

    # Generic fallbacks
    if name == "BB%" and context == "pitching":
        return -1
    if name in {"K%", "Whiff%"} and context == "hitting":
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

    # Savant-style: blue (poor) → neutral mid-gray (avg) → red (elite)
    # Using mid-gray rather than near-white so cells are readable with either text color
    bad  = np.array([ 10,  46, 110])  # #0a2e6e Savant deep blue
    mid  = np.array([120, 120, 120])  # #787878 neutral mid-gray (avg)
    good = np.array([139,   0,   0])  # #8b0000 Savant dark red
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
                r, g, b = rgb
                lum = (0.299*r + 0.587*g + 0.114*b) / 255
                txt = "#111111" if lum > 0.50 else "#ffffff"
                styles.append(f"background-color: rgb{rgb}; color: {txt}; font-weight: 650;")
        return styles

    formatters = {col: (lambda value, c=col: _fmt_pdf_value(value, c)) for col in df.columns}
    return df.style.apply(style_col, axis=0).format(formatters)


def _safe_pdf_name(name):
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")


def _dominant_batter_hand(df: pd.DataFrame) -> str:
    if "BatterSide" not in df.columns or df["BatterSide"].dropna().empty:
        return "Unknown"
    side_raw = str(df["BatterSide"].dropna().mode().iloc[0]).upper()
    if side_raw.startswith("L"):
        return "LHH"
    if side_raw.startswith("R"):
        return "RHH"
    return "Unknown"


def _dominant_pitcher_hand(df: pd.DataFrame) -> str:
    if "PitcherThrows" not in df.columns or df["PitcherThrows"].dropna().empty:
        return "Unknown"
    hand_raw = str(df["PitcherThrows"].dropna().mode().iloc[0]).upper()
    if hand_raw.startswith("L"):
        return "LHP"
    if hand_raw.startswith("R"):
        return "RHP"
    return "Unknown"


def _pdf_table_col_widths(columns) -> list:
    weights = []
    for col in columns:
        name = str(col)
        if name in {"Batter", "Hitter", "Pitcher", "Player"}:
            weights.append(2.35)
        elif name == "Tendency":
            weights.append(2.75)
        elif name in {"Team"}:
            weights.append(2.0)
        elif name in {"Side", "Pitch", "N", "PA", "AB", "BF", "BIP"}:
            weights.append(0.78)
        else:
            weights.append(1.0)
    total = sum(weights) or 1
    return [w / total for w in weights]


def _fmt_pdf_table_cell(value, col=None):
    text = _fmt_pdf_value(value, col)
    col_name = str(col or "")
    if col_name in {"Batter", "Hitter", "Pitcher", "Player"}:
        return textwrap.shorten(text, width=22, placeholder="...")
    if col_name == "Tendency":
        return textwrap.shorten(text, width=36, placeholder="...")
    if col_name == "Team":
        return textwrap.shorten(text, width=24, placeholder="...")
    return text


def _add_report_table(ax, df, title, max_rows=10, font_size=8, context=None, title_size=14):
    ax.axis("off")
    ax.set_title(title, color="#FFF7E8", fontsize=title_size, fontweight="bold", loc="left", pad=6)

    if df is None or df.empty:
        ax.text(0.02, 0.55, "No data available", color="#CDBFAF", fontsize=10, ha="left", va="center")
        return

    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(lambda value, c=col: _fmt_pdf_table_cell(value, c))

    adjusted_font_size = min(font_size, 6.2) if len(view.columns) >= 12 else font_size
    tbl = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=_pdf_table_col_widths(view.columns),
        bbox=[0, 0, 1, 0.86]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(adjusted_font_size)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#4E4036")
        cell.set_linewidth(0.7)
        col_name = view.columns[c] if c < len(view.columns) else ""
        if r == 0:
            cell.set_facecolor(FORDHAM_MAROON)
            cell.set_text_props(color="#FFF7E8", weight="bold", fontsize=max(adjusted_font_size - 0.3, 4.8))
        else:
            face = "#211C1A" if r % 2 else "#171514"
            if col_name in df.columns and r - 1 < len(df):
                rgb = _value_to_color(df.iloc[r - 1][col_name], col_name, df[col_name], context=context)
                if rgb is not None:
                    face = "#{:02x}{:02x}{:02x}".format(*rgb)
            cell.set_facecolor(face)
            # Dynamic text color: dark text on light backgrounds, white on dark
            _fx = face.lstrip("#")
            _lm = (0.299*int(_fx[0:2],16) + 0.587*int(_fx[2:4],16) + 0.114*int(_fx[4:6],16)) / 255
            _tc = "#111111" if _lm > 0.45 else "#F8EFE2"
            text_kwargs = {"color": _tc}
            if col_name in {"Batter", "Hitter", "Pitcher", "Player", "Tendency", "Team"}:
                text_kwargs["ha"] = "left"
                text_kwargs["fontsize"] = max(adjusted_font_size - 0.2, 4.8)
            cell.set_text_props(**text_kwargs)


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
        line_count = wrapped.count("\n") + 1
        needed = 0.070 + 0.037 * line_count
        if y - needed < 0.13:
            remaining = len([n for n in notes if n]) - i + 1
            if remaining > 0:
                ax.text(0.055, y, f"+{remaining} more reads", color="#CDBFAF", fontsize=max(note_size - 0.4, 6), transform=ax.transAxes, va="top")
            break
        ax.text(0.055, y, f"{i}.", color=FORDHAM_GOLD, fontsize=number_size, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.115, y, wrapped, color="#F8EFE2", fontsize=note_size, transform=ax.transAxes, va="top", linespacing=1.25)
        y -= needed
    if footer:
        footer_txt = "\n".join(textwrap.wrap(str(footer), width=wrap_width + 4, break_long_words=False))
        ax.text(0.055, 0.07, footer_txt, color="#CDBFAF", fontsize=footer_size, transform=ax.transAxes, va="bottom", linespacing=1.15)


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

    if "TaggedHitType" in hdf.columns and "Direction" in hdf.columns:
        _gb = hdf[hdf["TaggedHitType"].astype(str).eq("GroundBall")].copy()
        _gb["Direction"] = pd.to_numeric(_gb["Direction"], errors="coerce")
        _gb = _gb.dropna(subset=["Direction"])
        if len(_gb) >= 5:
            _gbn = len(_gb)
            _gbl = (_gb["Direction"] < 0).sum() / _gbn * 100
            _gbr = (_gb["Direction"] > 0).sum() / _gbn * 100
            notes.append(
                f"GB direction ({_gbn} GBs): {_gbl:.0f}% left of 2B (SS/3B side), "
                f"{_gbr:.0f}% right of 2B (2B/1B side)."
            )

    return notes


def pitcher_quick_read_notes(pdf_df, arsenal, splits, allowed, pa_rates):
    notes = []
    total = len(pdf_df)
    notes.append(
        f"Overall: {total} pitches, {_fmt_pdf_value(pdf_df['Stuff+'].mean() if 'Stuff+' in pdf_df.columns else np.nan)} Stuff+, "
        f"{_fmt_pdf_value(pdf_df['Loc+'].mean() if 'Loc+' in pdf_df.columns else np.nan)} Loc+, "
        f"{_fmt_pdf_value(allowed.get('BA'))}/{_fmt_pdf_value(allowed.get('OBP'))}/{_fmt_pdf_value(allowed.get('SLG'))} allowed."
    )

    first_usage = _first_pitch_pitcher_usage(pdf_df)
    if first_usage is not None:
        notes.append(
            f"0-0 plan: {first_usage.get('Pitch', '-')} is the primary first-pitch look "
            f"({_fmt_pdf_value(first_usage.get('Usage%'))}%)."
        )

    two_strike_chase = _two_strike_pitcher_chase(pdf_df)
    if two_strike_chase is not None:
        notes.append(
            f"Put-away: {two_strike_chase.get('Pitch', '-')} leads two-strike chase "
            f"({_fmt_pdf_value(two_strike_chase.get('Chase%'))}% chase, {_fmt_pdf_value(two_strike_chase.get('Whiff%'))}% whiff)."
        )

    stuff = _best_row_note(arsenal, "Stuff+", min_n=8)
    if stuff is not None:
        notes.append(
            f"Best stuff: {stuff.get('Pitch', '-')} at {_fmt_pdf_value(stuff.get('Stuff+'))} Stuff+ "
            f"with {_fmt_pdf_value(stuff.get('Whiff%'))}% whiff."
        )

    command = _best_row_note(arsenal, "Loc+", min_n=8)
    if command is not None:
        notes.append(
            f"Best command: {command.get('Pitch', '-')} at {_fmt_pdf_value(command.get('Loc+'))} Loc+ "
            f"and {_fmt_pdf_value(command.get('Zone%'))}% Zone."
        )

    risk = _best_row_note(arsenal, "HardHit%", min_n=5)
    if risk is not None:
        notes.append(
            f"Contact risk: {risk.get('Pitch', '-')} allowed {_fmt_pdf_value(risk.get('HardHit%'))}% HH "
            f"and {_fmt_pdf_value(risk.get('AvgEV'))} Avg EV."
        )

    if splits is not None and not splits.empty and "SLG" in splits.columns:
        split = _best_row_note(splits, "SLG", min_n=5)
        if split is not None:
            notes.append(
                f"Split watch: {split.get('Side', '-')} vs {split.get('Pitch', '-')} allowed "
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
        return {"BB%": np.nan, "K%": np.nan, "BABIP": np.nan, "GB%": np.nan}
    kbb = pa.get("KorBB", pd.Series("", index=pa.index)).astype(str)
    pr  = pa.get("PlayResult", pd.Series("", index=pa.index)).astype(str)
    bb  = kbb.eq("Walk").sum()
    k   = kbb.eq("Strikeout").sum()
    hr  = pr.eq("HomeRun").sum()
    h   = pr.isin(["Single","Double","Triple","HomeRun"]).sum()
    ab  = len(pa) - bb
    bip = ab - k - hr
    babip = (h - hr) / bip if bip > 0 else np.nan

    # GB% from all pitches (batted balls)
    gb_pct = np.nan
    if "TaggedHitType" in pdf_df.columns:
        bip_df = pdf_df[pdf_df.get("PitchCall", pd.Series("", index=pdf_df.index))
                        .astype(str).isin(["InPlay","InPlayNoOut","InPlayOut"])]
        if not bip_df.empty:
            gb_pct = bip_df["TaggedHitType"].astype(str).str.contains("Ground", case=False).mean() * 100

    return {"BB%": bb / len(pa) * 100, "K%": k / len(pa) * 100,
            "BABIP": babip, "GB%": gb_pct}


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
        "Bat+": card.get("Bat+"),
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
            "Oppo FB%": round(oppo_air, 1),
            "HH%": round(hh, 1),
            "AvgEV": round(bip["EV"].mean(), 1),
            "Tendency": "; ".join(tags) if tags else "Balanced",
        })

    return pd.DataFrame(rows).sort_values(["BIP", "HH%"], ascending=False)


def _table_columns(df: pd.DataFrame, cols):
    if df is None or df.empty:
        return pd.DataFrame()
    return df[[c for c in cols if c in df.columns]].copy()


def apply_date_range_filter(df: pd.DataFrame, key_prefix: str, label="Date Range") -> pd.DataFrame:
    if df is None or df.empty:
        return df

    date_col = "GameDate" if "GameDate" in df.columns else "Date" if "Date" in df.columns else None
    if not date_col:
        st.caption("Date filter unavailable: no Date/GameDate column found.")
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")
    valid_dates = dates.dropna()
    if valid_dates.empty:
        st.caption("Date filter unavailable: no valid dates found.")
        return df

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    selected = st.date_input(
        label,
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}_date_range",
    )

    if isinstance(selected, tuple):
        if len(selected) != 2:
            st.info("Choose a start and end date to apply the date filter.")
            return df
        start_date, end_date = selected
    else:
        start_date = end_date = selected

    if start_date is None or end_date is None:
        return df
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    mask = dates.dt.date.between(start_date, end_date)
    filtered = df[mask].copy()
    st.caption(
        f"Showing {len(filtered):,} of {len(df):,} pitches from "
        f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}."
    )
    return filtered


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


def _append_pitcher_break_plot_page(out_pdf, pdf_df: pd.DataFrame, pitcher: str):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    fig.text(0.05, 0.94, "Pitch Break Plot", color="#FFF7E8", fontsize=22, fontweight="bold", ha="left", va="center")
    fig.text(0.05, 0.905, pitcher, color="#CDBFAF", fontsize=12, fontweight="bold", ha="left", va="center")
    fig.text(0.95, 0.905, "HB x IVB | pitch averages labeled", color="#CDBFAF", fontsize=9, ha="right", va="center")

    break_img = _fig_to_image(build_movement_figure(pdf_df))
    ax = fig.add_axes([0.035, 0.055, 0.93, 0.81])
    ax.imshow(break_img)
    ax.axis("off")

    fig.text(
        0.05, 0.035,
        "Small dots are individual pitches. Large labeled circles are pitch-type averages. Shading separates arm-side and glove-side break.",
        color="#CDBFAF",
        fontsize=8.5,
        ha="left",
        va="center",
    )
    out_pdf.savefig(fig, bbox_inches="tight")
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


# ── Scouting logo / percentile helpers ───────────────────────────────────────
_LOGO_DIR_SCOUT = ROOT / "national_pitchingplus_app" / "team_logos"

def _scout_logo_path(team_code: str):
    """Return Path to a team logo PNG/JPG if available, else None."""
    if not team_code:
        return None
    base_code = str(team_code).strip()
    if base_code.upper() in {"FOR_RAM", "FOR_RAM1"}:
        ram_head = ROOT / "static" / "rams.png"
        if ram_head.exists():
            return ram_head
    candidates = [base_code, base_code + "1"]
    alias = SCOUT_LOGO_ALIASES.get(base_code.upper())
    if alias:
        candidates.extend([alias, alias + "1"])
    for code in candidates:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = _LOGO_DIR_SCOUT / f"{code}{ext}"
            if p.exists():
                return p
    return None


def _add_scout_logo(ax, team_code: str, primary: str, accent: str,
                    bounds=(0.87, 0.872, 0.095, 0.115), opacity: float = 0.62):
    """Embed a trimmed, translucent team logo on a scouting cover/header."""
    logo = _scout_logo_path(team_code)
    if not logo:
        return
    try:
        from PIL import Image as _PImg
        img = _PImg.open(logo).convert("RGBA")
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        pad = max(8, int(max(img.size) * 0.08))
        canvas = _PImg.new("RGBA", (img.size[0] + pad * 2, img.size[1] + pad * 2), (0, 0, 0, 0))
        canvas.alpha_composite(img, (pad, pad))
        img = canvas
        arr = np.array(img)
        arr[:, :, 3] = (arr[:, :, 3].astype(float) * opacity).clip(0, 255).astype(np.uint8)
        la = ax.inset_axes(list(bounds))
        la.set_facecolor((0, 0, 0, 0))
        la.patch.set_alpha(0)
        la.imshow(arr, aspect="equal")
        la.set_xticks([]); la.set_yticks([])
        for sp in la.spines.values():
            sp.set_visible(False)
    except Exception:
        pass


# Pitcher stats → D1 percentile key + direction mapping for cover page color boxes
_COVER_PITCHER_PCTS = {
    "Stuff+": ("Stuff+", True), "Loc+": ("Loc+", True),
    "K%":     ("K%P",    True), "BB%":  ("BB%P", False),
    "Whiff%": ("Whiff%P",True), "Zone%":("Zone%", True),
    "CSW%":   ("CSW%",   True), "GB%":  ("GB%P", True),
    "Avg EV Allowed": ("Avg EV A", False),
}

# Hitter stats → D1 percentile key mapping for header color pills
_COVER_HITTER_PCTS = {
    "BA":"BA","OBP":"OBP","SLG":"SLG","OPS":"OPS",
    "wOBA":"wOBA","Bat+":"Bat+",
    "K%":"K%","BB%":"BB%","Whiff%":"Whiff%","Chase%":"Chase%",
    "Avg EV":"Avg EV","HH%":"HH%",
}


def _pct_box_color(label: str, value, pitcher_context: bool) -> str:
    """Return a hex facecolor for a metric box based on D1 percentile rank."""
    try:
        fv = float(value) if value is not None and not pd.isna(float(str(value).replace("%",""))) else None
    except Exception:
        fv = None
    if fv is None:
        return "#211C1A"
    if pitcher_context:
        mapping = _COVER_PITCHER_PCTS.get(label)
        if not mapping:
            return "#211C1A"
        stat_key, high_good = mapping
        pct = _pitcher_pct_rank(stat_key, fv)
        if pct is None:
            return "#211C1A"
        return _pct_hex(pct)   # blue=elite for pitchers
    else:
        stat_key = _COVER_HITTER_PCTS.get(label)
        if not stat_key:
            return "#211C1A"
        pct = _hitter_pct_rank(stat_key, fv)
        if pct is None:
            return "#211C1A"
        return _pct_hex(pct)   # same ramp, but hitter direction


def _draw_compact_pct_tiles(fig, rect, rows_data, pct_fn):
    """Draw a row of Savant-colored stat tiles inside *rect* [left, bot, w, h] (figure coords).
    rows_data: list of (stat_key, display_label, fmt_str, value)
    pct_fn: function(key, val) → 0-1 float or None
    Fits all stats in a single compact horizontal strip — no separate PDF page needed.
    """
    import matplotlib.patches as _mp2
    left, bot, w, h = rect
    n = len(rows_data)
    if n == 0:
        return
    ax = fig.add_axes([left, bot, w, h])
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#100D0C")

    pad = 0.06
    for i, (key, label, fmt_s, val) in enumerate(rows_data):
        pct  = pct_fn(key, val)
        bg   = _pct_hex(pct) if pct is not None else "#2a2a3a"
        hx   = bg.lstrip("#")
        r_, g_, b_ = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
        lum  = (0.299*r_ + 0.587*g_ + 0.114*b_) / 255
        tc   = "#111111" if lum > 0.50 else "#FFF7E8"
        sc   = "#333333" if lum > 0.50 else "#CDBFAF"
        val_s = fmt_s.format(float(val)) if val is not None and not pd.isna(val) else "—"
        pct_s = _pct_label(pct) if pct is not None else "—"

        ax.add_patch(_mp2.FancyBboxPatch(
            (i + pad, 0.07), 1 - pad*2, 0.86,
            boxstyle="round,pad=0.02", facecolor=bg, edgecolor="none"
        ))
        ax.text(i + 0.5, 0.73, val_s,  ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=tc)
        ax.text(i + 0.5, 0.44, label,  ha="center", va="center",
                fontsize=6.0, fontweight="bold", color=sc)
        ax.text(i + 0.5, 0.18, pct_s,  ha="center", va="center",
                fontsize=5.5, color=sc)


def _add_pct_pdf_page(pdf, png_bytes: bytes):
    """Embed a percentile card PNG as a landscape PDF page."""
    try:
        from PIL import Image as _PImg
        img = _PImg.open(BytesIO(png_bytes))
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#13151c")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.imshow(np.array(img)); ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight", facecolor="#13151c")
        plt.close(fig)
    except Exception:
        pass


def _scouting_cover_fig(title, subtitle, metric_pairs, team_color=None, accent_color=None,
                        team_code=None, pitcher_context=True):
    team_color = team_color or FORDHAM_MAROON
    accent_color = accent_color or FORDHAM_GOLD
    title_text_color = readable_text_color(team_color)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.add_patch(plt.Rectangle((0, 0.86), 1, 0.14, color=team_color, transform=ax.transAxes))
    ax.add_patch(plt.Rectangle((0, 0.845), 1, 0.015, color=accent_color, transform=ax.transAxes))
    ax.text(0.05, 0.93, "FORDHAM BASEBALL SCOUTING ZONE", color=title_text_color,
            fontsize=18, fontweight="bold", transform=ax.transAxes)

    # Team logo — top-right of header
    if team_code:
        _add_scout_logo(ax, team_code, team_color, accent_color,
                        bounds=(0.87, 0.872, 0.095, 0.115))

    title_size = 28 if len(str(title)) <= 28 else 23
    ax.text(0.05, 0.80, title, color="#FFF7E8", fontsize=title_size, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.05, 0.75, subtitle, color="#CDBFAF", fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    cols = 4
    start_x, start_y = 0.05, 0.62
    box_w, box_h = 0.21, 0.105
    for i, (label, value) in enumerate(metric_pairs):
        x = start_x + (i % cols) * 0.235
        y = start_y - (i // cols) * 0.13
        # Percentile-based box color; falls back to flat dark if no mapping
        box_fc = _pct_box_color(str(label), value, pitcher_context)
        ax.add_patch(plt.Rectangle((x, y), box_w, box_h,
                     facecolor=box_fc, edgecolor="none",
                     linewidth=0, transform=ax.transAxes, alpha=0.85))
        display_value = _fmt_pdf_value(value)
        value_size = 16
        if len(display_value) > 18:
            display_value = textwrap.shorten(display_value, width=24, placeholder="...")
            value_size = 11
        elif len(display_value) > 10:
            value_size = 12.5
        # Dynamic text color so light-background boxes stay readable
        _hx = box_fc.lstrip("#")
        _lum = (0.299*int(_hx[0:2],16) + 0.587*int(_hx[2:4],16) + 0.114*int(_hx[4:6],16)) / 255
        _lbl_c = "#333333" if _lum > 0.50 else "#CDBFAF"
        _val_c = "#111111" if _lum > 0.50 else "#FFF7E8"
        ax.text(x + 0.018, y + 0.067, str(label), color=_lbl_c,
                fontsize=8.5, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.018, y + 0.024, display_value, color=_val_c,
                fontsize=value_size, fontweight="bold", transform=ax.transAxes)

    ax.text(0.05, 0.08,
            "Generated from TrackMan pitch-by-pitch data. "
            "Contact metrics use true in-play batted balls with usable EV.  "
            "Stat boxes colored by D1 percentile (blue = poor · red = elite).",
            color="#CDBFAF", fontsize=9, transform=ax.transAxes)
    return fig


def _hitter_pdf_header(fig, hitter, team, hitter_hand, card, slash, primary, accent,
                       y_top=1.0, height=0.13, team_code=None):
    """Draw a full-width team-colored header bar with hitter name, logo, and colour-coded stats."""
    txt_color = readable_text_color(primary)
    hdr = fig.add_axes([0, y_top - height, 1, height])
    hdr.set_facecolor(primary); hdr.axis("off")
    hdr.text(0.015, 0.76, hitter, color=txt_color, fontsize=20, fontweight="bold",
             transform=hdr.transAxes, va="center")
    hdr.text(0.015, 0.24, f"{team}  ·  {hitter_hand}  ·  Hitter Scouting Report",
             color=_muted_text_on(primary), fontsize=9, fontweight="bold", transform=hdr.transAxes, va="center")

    # Logo — top-right of header strip
    if team_code:
        _add_scout_logo(hdr, team_code, primary, accent, bounds=(0.876, 0.06, 0.11, 0.88))

    # Stat values — stop short of logo column
    raw_vals = {
        "BA": slash.get("BA"), "OBP": slash.get("OBP"),
        "SLG": slash.get("SLG"), "OPS": slash.get("OPS"),
        "Bat+": card.get("Bat+"), "wOBA": card.get("wOBA"),
        "K%": card.get("K%"), "BB%": card.get("BB%"),
        "HH%": card.get("HardHit%"), "Chase%": card.get("Chase%"),
    }
    stats = [
        ("PA",    _fmt_pdf_value(card.get("PA"),    "PA"),   None),
        ("BA",    _fmt_pdf_value(slash.get("BA"),   "BA"),   raw_vals.get("BA")),
        ("OBP",   _fmt_pdf_value(slash.get("OBP"),  "OBP"),  raw_vals.get("OBP")),
        ("SLG",   _fmt_pdf_value(slash.get("SLG"),  "SLG"),  raw_vals.get("SLG")),
        ("OPS",   _fmt_pdf_value(slash.get("OPS"),  "OPS"),  raw_vals.get("OPS")),
        ("Bat+",  _fmt_pdf_value(card.get("Bat+"),  "Bat+"), raw_vals.get("Bat+")),
        ("wOBA",  _fmt_pdf_value(card.get("wOBA"),  "wOBA"), raw_vals.get("wOBA")),
        ("K%",    _fmt_pdf_value(card.get("K%"),    "K%"),   raw_vals.get("K%")),
        ("BB%",   _fmt_pdf_value(card.get("BB%"),   "BB%"),  raw_vals.get("BB%")),
        ("HH%",   _fmt_pdf_value(card.get("HardHit%"), "HardHit%"), raw_vals.get("HH%")),
        ("Chase%",_fmt_pdf_value(card.get("Chase%"),"Chase%"),raw_vals.get("Chase%")),
    ]
    n = len(stats)
    step = 0.555 / n   # leave right ~13% for logo
    for i, (label, disp, raw) in enumerate(stats):
        x = 0.305 + i * step + step / 2
        # Savant-style colour for percentile stats
        val_c = txt_color
        if raw is not None and label in _COVER_HITTER_PCTS:
            try:
                pct = _hitter_pct_rank(label, float(raw))
                if pct is not None:
                    val_c = _pct_hex(pct)
            except Exception:
                pass
        hdr.text(x, 0.76, disp,  color=val_c,    fontsize=10.5, fontweight="bold",
                 ha="center", va="center", transform=hdr.transAxes)
        hdr.text(x, 0.22, label, color=_muted_text_on(primary), fontsize=6.8, fontweight="bold",
                 ha="center", va="center", transform=hdr.transAxes)


def _append_hitter_scouting_pages(pdf, hdf: pd.DataFrame, hitter: str, team: str, team_code=None):
    """Two-page hitter scouting report — all stats on page 1, all visuals on page 2."""
    hdf = hdf.copy()
    primary, accent = team_colors(team_code or team)
    lgwoba      = compute_league_woba(hdf)
    card        = compute_hitter_card(hdf, lgwoba)
    hitter_hand = card.get("Side") or _dominant_batter_hand(hdf)

    slash_df = add_ba_slg_by_group(hdf.assign(Player=hitter), ["Player"])
    slash = {}
    if not slash_df.empty:
        slash = {c: slash_df[c].iloc[0] for c in ["BA","OBP","SLG","OPS"] if c in slash_df.columns}

    pitch_table = hitter_pitchtype_effectiveness(hdf)
    if not pitch_table.empty:
        pitch_table = pitch_table[["Pitch","N","BA","SLG","Swing%","Whiff%","Chase%","AvgEV","HardHit%"]]
        pitch_table = _rename_compact_report_cols(pitch_table)

    count_table = count_effectiveness(hdf)
    if not count_table.empty:
        count_table = count_table[["Count","N","BA","SLG","Swing%","Whiff%","AvgEV","HardHit%"]]
        count_table = _rename_compact_report_cols(count_table)

    splits_table = hitter_splits(hdf)
    spray_table  = _rename_compact_report_cols(hitter_spray_profile(hdf))
    quick_notes  = hitter_quick_read_notes(hdf, card, pitch_table, count_table, spray_table, splits_table)
    damage_view  = (pitch_table.sort_values("SLG",    ascending=False)
                    if pitch_table is not None and not pitch_table.empty and "SLG"    in pitch_table.columns
                    else pitch_table)
    chase_view   = (pitch_table.sort_values("Whiff%", ascending=False)
                    if pitch_table is not None and not pitch_table.empty and "Whiff%" in pitch_table.columns
                    else pitch_table)

    # ── PAGE 1 — Stats & Tables ───────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")

    _hitter_pdf_header(fig, hitter, team, hitter_hand, card, slash, primary, accent,
                       y_top=1.0, height=0.13, team_code=team_code)

    # Three columns: each has its own explicit [left, bottom, width, height] rect.
    # title_size=10 keeps the title small enough that it won't bleed into siblings.
    # Tables start at y=0.04 (bottom margin) and reach to y=0.84 (below header).
    # Content height = 0.80 per column; title headroom included in column height.
    col_h   = 0.80   # axes height for table columns
    col_bot = 0.04   # bottom margin

    _add_report_table(
        fig.add_axes([0.03, col_bot, 0.30, col_h]),
        pitch_table, "vs Pitch Type", max_rows=12,
        font_size=6.8, context="hitting", title_size=10)

    _add_report_table(
        fig.add_axes([0.36, col_bot, 0.30, col_h]),
        count_table, "Count Tendencies", max_rows=12,
        font_size=6.8, context="hitting", title_size=10)

    # Right column split: splits (top) + notes (bottom) with clear gap
    splits_h = 0.37
    notes_h  = 0.37
    gap      = 0.06   # explicit gap between splits and notes panels

    _add_report_table(
        fig.add_axes([0.69, col_bot + notes_h + gap, 0.29, splits_h]),
        splits_table, "vs Pitcher Hand", max_rows=7,
        font_size=6.8, context="hitting", title_size=10)

    _add_notes_panel(
        fig.add_axes([0.69, col_bot, 0.29, notes_h]),
        "Quick Read", quick_notes,
        footer="", max_notes=5, wrap_width=35,
        title_size=11, note_size=8.0, number_size=9.5, footer_size=7.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── PAGE 2 — Visuals + compact percentile tiles ───────────────────────────
    fig2 = plt.figure(figsize=(11, 8.5))
    fig2.patch.set_facecolor("#100D0C")

    _hitter_pdf_header(fig2, hitter, team, hitter_hand, card, slash, primary, accent,
                       y_top=1.0, height=0.10, team_code=team_code)

    # Spray chart — tall left column (shrunk slightly to leave room for pct strip)
    spray_img = _fig_to_image(build_hitter_spray_chart(hdf, ""))
    ax_spray = fig2.add_axes([0.03, 0.17, 0.42, 0.71])
    ax_spray.imshow(spray_img); ax_spray.axis("off")
    ax_spray.text(0.5, 1.01, "Spray Chart", transform=ax_spray.transAxes,
                  color="#FFF7E8", fontsize=10, fontweight="bold", ha="center", va="bottom")

    # Zone heatmaps — right side, top half
    ev_fig    = make_savant_zone_heatmap(hdf, "AvgEV",  "Avg EV by Zone",   "True BIP only")
    whiff_fig = make_savant_zone_heatmap(hdf, "Whiff%", "Whiff% by Zone",   "Whiffs per swing")

    zone_top = 0.45
    zone_h   = 0.43
    tbl_bot  = 0.17
    tbl_h    = 0.26
    col_gap  = 0.01

    for i, src_fig in enumerate([ev_fig, whiff_fig]):
        x = 0.48 + i * (0.255 + col_gap)
        if src_fig:
            src_fig.patch.set_facecolor("#100D0C")
            for ax_ in src_fig.axes:
                ax_.title.set_color("#FFF7E8")
                ax_.tick_params(colors="#FFF7E8")
            img = _fig_to_image(src_fig)
            ax_ = fig2.add_axes([x, zone_top, 0.245, zone_h])
            ax_.imshow(img); ax_.axis("off")

    # Damage and Miss/Chase tables — right side, bottom section
    _add_report_table(
        fig2.add_axes([0.48, tbl_bot, 0.245, tbl_h]),
        damage_view, "Most Damage vs", max_rows=5,
        font_size=6.5, context="hitting", title_size=9)

    _add_report_table(
        fig2.add_axes([0.74, tbl_bot, 0.245, tbl_h]),
        chase_view, "Misses / Chases", max_rows=5,
        font_size=6.5, context="hitting", title_size=9)

    # Compact percentile tiles — full-width strip at the very bottom
    try:
        h_stats = _compute_hitter_pct_stats(hdf)
        pct_rows_h = [
            ("Bat+",   "Bat+",   "{:.0f}",    h_stats.get("Bat+")),
            ("wOBA",   "wOBA",   "{:.3f}",    h_stats.get("wOBA")),
            ("BA",     "BA",     "{:.3f}",    h_stats.get("BA")),
            ("OBP",    "OBP",    "{:.3f}",    h_stats.get("OBP")),
            ("SLG",    "SLG",    "{:.3f}",    h_stats.get("SLG")),
            ("K%",     "K%",     "{:.1f}%",   h_stats.get("K%")),
            ("BB%",    "BB%",    "{:.1f}%",   h_stats.get("BB%")),
            ("Whiff%", "Whiff%", "{:.1f}%",   h_stats.get("Whiff%")),
            ("Chase%", "Chase%", "{:.1f}%",   h_stats.get("Chase%")),
            ("Avg EV", "Avg EV", "{:.1f}",    h_stats.get("Avg EV")),
            ("HH%",    "HH%",    "{:.1f}%",   h_stats.get("HH%")),
        ]
        _draw_compact_pct_tiles(fig2, [0.03, 0.02, 0.94, 0.13], pct_rows_h, _hitter_pct_rank)
        fig2.text(0.03, 0.155, "D1 Percentile Rankings  ·  blue = poor  ·  red = elite",
                  color="#CDBFAF", fontsize=7, va="bottom")
    except Exception:
        pass

    pdf.savefig(fig2, bbox_inches="tight")
    plt.close(fig2)


def build_hitter_scouting_pdf(hdf: pd.DataFrame, hitter: str, team: str, team_code=None) -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        _append_hitter_scouting_pages(pdf, hdf, hitter, team, team_code=team_code)

    buf.seek(0)
    return buf.getvalue()


def _append_pitcher_scouting_pages(out_pdf, pdf_df: pd.DataFrame, pitcher: str, team: str, team_code=None):
    pdf_df = pdf_df.copy()
    primary, accent = team_colors(team_code or team)
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

    pitcher_hand = _dominant_pitcher_hand(pdf_df)
    metric_pairs = [
        ("Team", team), ("Throws", pitcher_hand), ("Pitches", total), ("Strike%", strike),
        ("Zone%", zone),
        ("CSW%", csw), ("Whiff%", whiff_pct), ("Stuff+", pdf_df["Stuff+"].mean() if "Stuff+" in pdf_df.columns else np.nan),
        ("Loc+", pdf_df["Loc+"].mean() if "Loc+" in pdf_df.columns else np.nan),
        ("BA", allowed["BA"]), ("OBP", allowed["OBP"]), ("SLG", allowed["SLG"]),
        ("BABIP", pa_rates.get("BABIP", np.nan)),
        ("BB%", pa_rates["BB%"]), ("K%", pa_rates["K%"]),
        ("GB%", pa_rates.get("GB%", np.nan)),
        ("Avg EV Allowed", bip["EV"].mean() if not bip.empty else np.nan),
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

    # ── PAGE 1: Cover stats (left) + Movement chart (right) ──────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#100D0C")
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_axis_off()

    title_text_color = readable_text_color(primary)

    # Header bar: full width, taller to contain name + subtitle
    ax_bg.add_patch(plt.Rectangle((0, 0.855), 1, 0.145, color=primary))
    ax_bg.add_patch(plt.Rectangle((0, 0.840), 1, 0.015, color=accent))
    ax_bg.text(0.02, 0.987, "FORDHAM BASEBALL SCOUTING ZONE",
               color=title_text_color, fontsize=13, fontweight="bold", va="top")
    name_size = 22 if len(str(pitcher)) <= 26 else 17
    ax_bg.text(0.02, 0.945, pitcher,
               color=title_text_color, fontsize=name_size, fontweight="bold", va="top")
    ax_bg.text(0.02, 0.875, f"Pitcher scouting report  ·  {pitcher_hand}",
               color=_muted_text_on(primary), fontsize=9.5, fontweight="bold", va="top")
    if team_code:
        _add_scout_logo(ax_bg, team_code, primary, accent, bounds=(0.87, 0.862, 0.095, 0.115))

    # Left block: metric boxes, dynamically spaced from just below header to footer
    cols3 = 3
    n_mp  = len(metric_pairs)
    n_rows3 = (n_mp + cols3 - 1) // cols3
    avail_h = 0.800     # from y=0.04 to y=0.840 (below accent stripe)
    row_slot = avail_h / max(n_rows3, 1)
    box_h3   = min(0.110, row_slot * 0.84)
    box_w3   = 0.148
    col_step = 0.163
    start_x3 = 0.018
    top_y    = 0.836    # first box top sits just below accent stripe

    for i, (label, value) in enumerate(metric_pairs):
        col_i = i % cols3
        row_i = i // cols3
        x3 = start_x3 + col_i * col_step
        # box bottom-left y: stack downward from top_y
        y3 = top_y - (row_i + 1) * row_slot + (row_slot - box_h3) / 2
        box_fc = _pct_box_color(str(label), value, True)
        ax_bg.add_patch(plt.Rectangle((x3, y3), box_w3, box_h3,
                        facecolor=box_fc, edgecolor="none", alpha=0.90))
        display_value = _fmt_pdf_value(value)
        vs = 11 if len(display_value) > 10 else 13
        # Dynamic text color — prevents white-on-white when box is near-white
        _bx = box_fc.lstrip("#")
        _lm = (0.299*int(_bx[0:2],16) + 0.587*int(_bx[2:4],16) + 0.114*int(_bx[4:6],16)) / 255
        _lc = "#333333" if _lm > 0.50 else "#CDBFAF"
        _vc = "#111111" if _lm > 0.50 else "#FFF7E8"
        ax_bg.text(x3 + 0.008, y3 + box_h3*0.70, str(label),
                   color=_lc, fontsize=7.0, fontweight="bold")
        ax_bg.text(x3 + 0.008, y3 + box_h3*0.22, display_value,
                   color=_vc, fontsize=vs, fontweight="bold")

    ax_bg.text(0.02, 0.020,
        "Generated from TrackMan data. Stat boxes colored by D1 percentile (blue=poor · red=elite).",
        color="#6A5C52", fontsize=7.5)

    # Right block: movement chart (53% to 99%)
    break_img = _fig_to_image(build_movement_figure(pdf_df))
    ax_mv = fig.add_axes([0.52, 0.04, 0.46, 0.79])
    ax_mv.imshow(break_img); ax_mv.axis("off")
    ax_mv.text(0.5, 1.01, "Pitch Movement  —  HB × IVB, averages labeled",
               transform=ax_mv.transAxes, color="#CDBFAF", fontsize=8.5,
               ha="center", va="bottom")

    out_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── PAGE 2: Arsenal / Notes / Splits + compact percentile tiles ───────────
    fig2 = plt.figure(figsize=(11, 8.5))
    fig2.patch.set_facecolor("#100D0C")

    # Header stripe
    ax_h2 = fig2.add_axes([0, 0.94, 1, 0.06])
    ax_h2.set_facecolor(primary); ax_h2.axis("off")
    ax_h2.text(0.02, 0.55, pitcher, color=title_text_color,
               fontsize=14, fontweight="bold", va="center")
    ax_h2.text(0.98, 0.55, "Arsenal · Splits · Percentile Rankings",
               color=_muted_text_on(primary), fontsize=9, fontweight="bold", ha="right", va="center")

    # Three-panel gridspec: arsenal / notes / splits — leaves bottom 18% for pct tiles
    gs2 = fig2.add_gridspec(3, 1, left=0.04, right=0.97,
                             top=0.92, bottom=0.20,
                             hspace=0.32, height_ratios=[1.2, 0.85, 1.0])
    _add_report_table(fig2.add_subplot(gs2[0]), _rename_compact_report_cols(arsenal.sort_values("N", ascending=False)),
                      "Pitch Arsenal", max_rows=9, font_size=6.2, context="pitching")
    _add_notes_panel(fig2.add_subplot(gs2[1]), "Quick Read", quick_notes,
                     footer="", max_notes=4, wrap_width=110,
                     title_size=12, note_size=8.0, number_size=9, footer_size=7)
    _add_report_table(fig2.add_subplot(gs2[2]), splits.sort_values(["Side", "N"], ascending=[True, False]),
                      "Batter-Side Splits", max_rows=8, font_size=6.5, context="pitching")

    # Compact percentile tiles at bottom
    try:
        pct_stats = _compute_pitcher_pct_stats(pdf_df)
        pct_rows_p = [
            ("Stuff+", "Stuff+",    "{:.0f}",     pct_stats.get("Stuff+")),
            ("Loc+",   "Loc+",      "{:.0f}",     pct_stats.get("Loc+")),
            ("Velo",   "FB Velo",   "{:.1f}",     pct_stats.get("Velo")),
            ("Whiff%", "Whiff%",    "{:.1f}%",    pct_stats.get("Whiff%")),
            ("CSW%",   "CSW%",      "{:.1f}%",    pct_stats.get("CSW%")),
            ("Zone%",  "Zone%",     "{:.1f}%",    pct_stats.get("Zone%")),
            ("K%",     "K%",        "{:.1f}%",    pct_stats.get("K%")),
            ("BB%",    "BB%",       "{:.1f}%",    pct_stats.get("BB%")),
            ("GB%",    "GB%",       "{:.1f}%",    pct_stats.get("GB%")),
            ("Avg EV", "Avg EV", "{:.1f}",     pct_stats.get("Avg EV")),
        ]
        _draw_compact_pct_tiles(fig2, [0.04, 0.03, 0.93, 0.15], pct_rows_p, _pitcher_pct_rank)
        fig2.text(0.04, 0.185, "D1 Percentile Rankings  ·  blue = poor  ·  red = elite",
                  color="#CDBFAF", fontsize=7, va="bottom")
    except Exception:
        pass

    out_pdf.savefig(fig2, bbox_inches="tight")
    plt.close(fig2)


def build_pitcher_scouting_pdf(pdf_df: pd.DataFrame, pitcher: str, team: str, team_code=None) -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as out_pdf:
        _append_pitcher_scouting_pages(out_pdf, pdf_df, pitcher, team, team_code=team_code)

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


def build_team_scouting_pdf(df: pd.DataFrame, team: str, include_individual_reports=False, packet_scope="All Players", max_players=None, team_label=None) -> bytes:
    df = df.copy()
    report_team = team_label or team_display_name(team)
    primary, accent = team_colors(team)
    hitters_df = df[df.get("BatterTeam", pd.Series("", index=df.index)).astype(str).eq(str(team))].copy()
    pitchers_df = df[df.get("PitcherTeam", pd.Series("", index=df.index)).astype(str).eq(str(team))].copy()

    hitting = team_hitting_metrics(hitters_df)
    pitching = team_pitching_metrics(pitchers_df)

    hitter_summary = summarize_contact_quality(hitters_df, "Batter").sort_values("PA", ascending=False) if not hitters_df.empty else pd.DataFrame()
    pitcher_summary = summarize_pitching_staff(pitchers_df).sort_values("Pitches", ascending=False) if not pitchers_df.empty else pd.DataFrame()
    tendency_summary = team_hitter_tendencies(hitters_df)

    hitter_cols = ["Batter", "PA", "BA", "OBP", "SLG", "OPS", "wOBA", "Bat+", "BB%", "K%", "AvgEV", "HardHit%", "Whiff%", "Chase%"]
    pitcher_cols = ["Pitcher", "Pitches", "BF", "BA", "OBP", "SLG", "OPS", "Stuff+", "Loc+", "Strike%", "Zone%", "CSW%", "Whiff%", "BB%", "K%", "AvgEV", "HH%"]
    tendency_cols = ["Hitter", "Side", "BIP", "Pull%", "Middle%", "Oppo%", "GB%", "Pull GB%", "Middle GB%", "Oppo FB%", "HH%", "AvgEV", "Tendency"]
    hitter_summary = _table_columns(hitter_summary, hitter_cols)
    pitcher_summary = _table_columns(pitcher_summary, pitcher_cols)
    tendency_summary = _table_columns(tendency_summary, tendency_cols)
    if not hitter_summary.empty:
        hitter_summary = hitter_summary.rename(columns={"HardHit%": "HH%"})
    if not tendency_summary.empty:
        tendency_summary = tendency_summary.rename(columns={
            "Middle%": "Mid%",
            "Pull GB%": "PullGB%",
            "Middle GB%": "MidGB%",
            "Oppo FB%": "OppoFB%",
        })
        if "Tendency" in tendency_summary.columns:
            tendency_summary["Tendency"] = tendency_summary["Tendency"].map(
                lambda value: textwrap.shorten(str(value), width=32, placeholder="...")
            )
    hitter_preview_cols = ["Batter", "PA", "BA", "OBP", "SLG", "OPS", "Bat+", "AvgEV", "HH%"]
    pitcher_preview_cols = ["Pitcher", "Pitches", "BF", "BA", "SLG", "Stuff+", "Loc+", "K%", "BB%", "Whiff%"]
    hitter_preview = _table_columns(hitter_summary, hitter_preview_cols)
    pitcher_preview = _table_columns(pitcher_summary, pitcher_preview_cols)

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
        ("Team", report_team), ("Hitter PA", hitting.get("PA")), ("Team BA", hitting.get("BA")), ("Team OBP", hitting.get("OBP")),
        ("Team SLG", hitting.get("SLG")), ("Team OPS", hitting.get("OPS")), ("Team wOBA", hitting.get("wOBA")), ("Team Bat+", hitting.get("Bat+")),
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
        fig = _scouting_cover_fig(report_team, "Team scouting report", metric_pairs,
                                  primary, accent, team_code=team, pitcher_context=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#100D0C")
        gs = fig.add_gridspec(3, 2, left=0.05, right=0.95, top=0.92, bottom=0.07, hspace=0.36, wspace=0.18, height_ratios=[0.9, 1.0, 1.0])
        _add_notes_panel(
            fig.add_subplot(gs[0, :]),
            "Team Snapshot",
            notes,
            footer="Selected-team batting and pitching rows from the scouting database.",
            max_notes=4,
            wrap_width=92,
            title_size=14,
            note_size=8.4,
            number_size=10,
            footer_size=7.0
        )
        _add_report_table(
            fig.add_subplot(gs[1, :]),
            hitter_preview.sort_values("PA", ascending=False).head(8) if not hitter_preview.empty else hitter_preview,
            "Top Hitters",
            max_rows=8,
            font_size=6.8,
            context="hitting"
        )
        _add_report_table(
            fig.add_subplot(gs[2, :]),
            pitcher_preview.sort_values("Pitches", ascending=False).head(8) if not pitcher_preview.empty else pitcher_preview,
            "Top Pitchers",
            max_rows=8,
            font_size=6.8,
            context="pitching"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _save_paginated_report_table(
            pdf,
            tendency_summary.sort_values(["PullGB%", "Pull%"], ascending=False) if not tendency_summary.empty and "PullGB%" in tendency_summary.columns else tendency_summary,
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
                _append_section_divider(pdf, "Individual Hitter Reports", f"{len(hitters)} hitters from {report_team}")
                for hitter in hitters:
                    player_df = hitters_df[hitters_df["Batter"].astype(str).eq(hitter)].copy()
                    if not player_df.empty:
                        _append_hitter_scouting_pages(pdf, player_df, hitter, report_team, team_code=team)

            if pitchers:
                _append_section_divider(pdf, "Individual Pitcher Reports", f"{len(pitchers)} pitchers from {report_team}")
                for pitcher in pitchers:
                    player_df = pitchers_df[pitchers_df["Pitcher"].astype(str).eq(pitcher)].copy()
                    if not player_df.empty:
                        _append_pitcher_scouting_pages(pdf, player_df, pitcher, report_team, team_code=team)

    buf.seek(0)
    return buf.getvalue()


def _scouting_pitcher_summary(df: pd.DataFrame, group_col="Pitcher") -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for name, g in df.groupby(group_col, observed=True):
        if not str(name).strip() or str(name).lower() == "nan":
            continue
        pitch_call = g.get("PitchCall", pd.Series("", index=g.index)).astype(str)
        strike_calls = [
            "StrikeCalled", "StrikeSwinging", "FoulBall", "FoulBallNotFieldable",
            "FoulBallFieldable", "InPlay", "InPlayNoOut", "InPlayOut"
        ]
        swings = float(g.get("is_swing", pd.Series(0, index=g.index)).sum())
        whiffs = float(g.get("is_whiff", pd.Series(0, index=g.index)).sum())
        slash = pitcher_allowed_slash(g)
        pa_rates = pitcher_pa_rates(g)
        primary = ""
        if "pitch_abbr" in g.columns and not g["pitch_abbr"].dropna().empty:
            primary = str(g["pitch_abbr"].value_counts().index[0])
        rows.append({
            group_col: name,
            "Pitches": len(g),
            "Batters": g["Batter"].nunique() if "Batter" in g.columns else np.nan,
            "Primary": primary,
            "Velo": pd.to_numeric(g.get("Velo", pd.Series(dtype=float)), errors="coerce").mean(),
            "Stuff+": pd.to_numeric(g.get("Stuff+", pd.Series(dtype=float)), errors="coerce").mean(),
            "Loc+": pd.to_numeric(g.get("Loc+", pd.Series(dtype=float)), errors="coerce").mean(),
            "Strike%": pitch_call.isin(strike_calls).mean() * 100 if len(g) else np.nan,
            "Zone%": pd.to_numeric(g.get("in_zone", pd.Series(dtype=float)), errors="coerce").mean() * 100,
            "Whiff%": whiffs / swings * 100 if swings else np.nan,
            "K%": pa_rates.get("K%"),
            "BB%": pa_rates.get("BB%"),
            "BA": slash.get("BA"),
            "OBP": slash.get("OBP"),
            "SLG": slash.get("SLG"),
            "OPS": slash.get("OPS"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Stuff+", "Pitches"], ascending=[False, False], na_position="last")
    return out


def _scouting_matchup_table(df: pd.DataFrame, offense_team: str, min_pa=1) -> pd.DataFrame:
    if df.empty or not {"Batter", "Pitcher", "BatterTeam"}.issubset(df.columns):
        return pd.DataFrame()
    sub = df[df["BatterTeam"].astype(str).eq(str(offense_team))].copy()
    if sub.empty:
        return pd.DataFrame()
    slash = add_ba_slg_by_group(sub, ["Batter", "Pitcher"])
    if slash.empty:
        return pd.DataFrame()
    pitches = sub.groupby(["Batter", "Pitcher"], observed=True).agg(
        Pitches=("Pitcher", "count"),
        AvgEV=("EV", "mean") if "EV" in sub.columns else ("Pitcher", "count"),
        Swings=("is_swing", "sum") if "is_swing" in sub.columns else ("Pitcher", "count"),
        Whiffs=("is_whiff", "sum") if "is_whiff" in sub.columns else ("Pitcher", "count"),
        HardHit=("hard_hit", "mean") if "hard_hit" in sub.columns else ("Pitcher", "count"),
    ).reset_index()
    out = slash.merge(pitches, on=["Batter", "Pitcher"], how="left")
    out["PA"] = out["AB"].fillna(0) + out["BB"].fillna(0) + out["HBP"].fillna(0) + out["SF"].fillna(0)
    out = out[out["PA"] >= min_pa].copy()
    out["Whiff%"] = np.where(out["Swings"] > 0, out["Whiffs"] / out["Swings"] * 100, np.nan)
    out["HardHit%"] = out["HardHit"] * 100
    for col in ["AvgEV", "Whiff%", "HardHit%"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
    return out.sort_values(["OPS", "AvgEV", "Pitches"], ascending=[False, False, False], na_position="last")


def _scouting_pitcher_vs_batter_table(df: pd.DataFrame, pitching_team: str, min_pa=1) -> pd.DataFrame:
    if df.empty or not {"Pitcher", "Batter", "PitcherTeam"}.issubset(df.columns):
        return pd.DataFrame()
    sub = df[df["PitcherTeam"].astype(str).eq(str(pitching_team))].copy()
    if sub.empty:
        return pd.DataFrame()
    slash = add_ba_slg_by_group(sub, ["Pitcher", "Batter"])
    if slash.empty:
        return pd.DataFrame()
    pitch_detail = sub.groupby(["Pitcher", "Batter"], observed=True).agg(
        Pitches=("Batter", "count"),
        AvgEV=("EV", "mean") if "EV" in sub.columns else ("Batter", "count"),
        Swings=("is_swing", "sum") if "is_swing" in sub.columns else ("Batter", "count"),
        Whiffs=("is_whiff", "sum") if "is_whiff" in sub.columns else ("Batter", "count"),
        HardHit=("hard_hit", "mean") if "hard_hit" in sub.columns else ("Batter", "count"),
    ).reset_index()
    out = slash.merge(pitch_detail, on=["Pitcher", "Batter"], how="left")
    out["PA"] = out["AB"].fillna(0) + out["BB"].fillna(0) + out["HBP"].fillna(0) + out["SF"].fillna(0)
    out = out[out["PA"] >= min_pa].copy()
    out["Whiff%"] = np.where(out["Swings"] > 0, out["Whiffs"] / out["Swings"] * 100, np.nan)
    out["HardHit%"] = out["HardHit"] * 100
    for col in ["AvgEV", "Whiff%", "HardHit%"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
    return out.sort_values(["OPS", "AvgEV", "Pitches"], ascending=[True, True, False], na_position="last")


def advanced_scouting_section(df: pd.DataFrame, team: str, team_label: str):
    st.markdown("### Advanced Scouting")
    st.caption("Player-to-player detail for the selected team. Use this to compare teammates, identify matchup edges, and build a more specific game plan.")

    team_hitters = df[df["BatterTeam"].astype(str).eq(str(team))].copy() if "BatterTeam" in df.columns else pd.DataFrame()
    team_pitchers = df[df["PitcherTeam"].astype(str).eq(str(team))].copy() if "PitcherTeam" in df.columns else pd.DataFrame()

    h_tab, p_tab, m_tab = st.tabs(["Hitter Detail", "Pitcher Detail", "Player Matchups"])

    with h_tab:
        if team_hitters.empty or "Batter" not in team_hitters.columns:
            st.info(f"No hitter rows found for {team_label}.")
        else:
            hitter_summary = summarize_contact_quality(team_hitters, "Batter")
            if hitter_summary.empty:
                st.info("No PA-ending hitter summary available for this team.")
            else:
                hitter_summary = hitter_summary.sort_values(["Bat+", "OPS", "AvgEV"], ascending=[False, False, False], na_position="last")
                show = ["Batter", "PA", "AB", "H", "HR", "xHB", "BA", "OBP", "SLG", "OPS", "wOBA", "Bat+", "BB%", "K%", "AvgEV", "HardHit%", "Barrel%", "Whiff%", "Chase%"]
                st.dataframe(style_scouting_dataframe(_table_columns(hitter_summary, show), context="hitting"), use_container_width=True, hide_index=True)

            hitters = sorted(team_hitters["Batter"].dropna().astype(str).unique())
            if hitters:
                selected_hitter = st.selectbox("Hitter Detail", hitters, key="adv_scout_hitter")
                hdf = team_hitters[team_hitters["Batter"].astype(str).eq(selected_hitter)].copy()
                c1, c2 = st.columns([1.05, 1])
                with c1:
                    st.markdown("#### Pitch-Type Plan")
                    pt = hitter_pitchtype_effectiveness(hdf)
                    show_pt = ["Pitch", "N", "BA", "OBP", "SLG", "OPS", "wOBA", "Swing%", "Whiff%", "Chase%", "AvgEV", "HardHit%"]
                    if pt.empty:
                        st.info("No pitch-type detail available for this hitter.")
                    else:
                        st.dataframe(style_scouting_dataframe(_table_columns(pt, show_pt), context="hitting"), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("#### Opposing Pitchers Faced")
                    match = _scouting_matchup_table(hdf, team, min_pa=1)
                    show_match = ["Pitcher", "PA", "Pitches", "BA", "OBP", "SLG", "OPS", "AvgEV", "HardHit%", "Whiff%"]
                    if match.empty:
                        st.info("No pitcher matchup rows available.")
                    else:
                        st.dataframe(style_scouting_dataframe(_table_columns(match, show_match), context="hitting"), use_container_width=True, hide_index=True)

    with p_tab:
        if team_pitchers.empty or "Pitcher" not in team_pitchers.columns:
            st.info(f"No pitcher rows found for {team_label}.")
        else:
            pitcher_summary = _scouting_pitcher_summary(team_pitchers, "Pitcher")
            show = ["Pitcher", "Pitches", "Batters", "Primary", "Velo", "Stuff+", "Loc+", "Strike%", "Zone%", "Whiff%", "K%", "BB%", "BA", "OBP", "SLG", "OPS"]
            if pitcher_summary.empty:
                st.info("No pitcher summary available for this team.")
            else:
                st.dataframe(style_scouting_dataframe(_table_columns(pitcher_summary, show), context="pitching"), use_container_width=True, hide_index=True)

            pitchers = sorted(team_pitchers["Pitcher"].dropna().astype(str).unique())
            if pitchers:
                selected_pitcher = st.selectbox("Pitcher Detail", pitchers, key="adv_scout_pitcher")
                pdf = team_pitchers[team_pitchers["Pitcher"].astype(str).eq(selected_pitcher)].copy()
                c1, c2 = st.columns([1.05, 1])
                with c1:
                    st.markdown("#### Arsenal Detail")
                    if "pitch_abbr" in pdf.columns:
                        arsenal = pdf.groupby("pitch_abbr", observed=True).agg(
                            N=("pitch_abbr", "count"),
                            Velo=("Velo", "mean") if "Velo" in pdf.columns else ("pitch_abbr", "count"),
                            IVB=("IVB", "mean") if "IVB" in pdf.columns else ("pitch_abbr", "count"),
                            HB=("HB", "mean") if "HB" in pdf.columns else ("pitch_abbr", "count"),
                            Stuff_plus=("Stuff+", "mean") if "Stuff+" in pdf.columns else ("pitch_abbr", "count"),
                            Loc_plus=("Loc+", "mean") if "Loc+" in pdf.columns else ("pitch_abbr", "count"),
                            Zone=("in_zone", "mean") if "in_zone" in pdf.columns else ("pitch_abbr", "count"),
                            Swings=("is_swing", "sum") if "is_swing" in pdf.columns else ("pitch_abbr", "count"),
                            Whiffs=("is_whiff", "sum") if "is_whiff" in pdf.columns else ("pitch_abbr", "count"),
                        ).reset_index().rename(columns={"pitch_abbr": "Pitch", "Stuff_plus": "Stuff+", "Loc_plus": "Loc+"})
                        arsenal["Usage%"] = arsenal["N"] / arsenal["N"].sum() * 100
                        arsenal["Zone%"] = arsenal["Zone"] * 100
                        arsenal["Whiff%"] = np.where(arsenal["Swings"] > 0, arsenal["Whiffs"] / arsenal["Swings"] * 100, np.nan)
                        show_ars = ["Pitch", "N", "Usage%", "Velo", "IVB", "HB", "Stuff+", "Loc+", "Zone%", "Whiff%"]
                        st.dataframe(style_scouting_dataframe(_table_columns(arsenal.round(1), show_ars), context="pitching"), use_container_width=True, hide_index=True)
                    else:
                        st.info("No pitch-type column available.")
                with c2:
                    st.markdown("#### Batter Matchups")
                    match = _scouting_pitcher_vs_batter_table(pdf, team, min_pa=1)
                    show_match = ["Batter", "PA", "Pitches", "BA", "OBP", "SLG", "OPS", "AvgEV", "HardHit%", "Whiff%"]
                    if match.empty:
                        st.info("No batter matchup rows available.")
                    else:
                        st.dataframe(style_scouting_dataframe(_table_columns(match, show_match), context="pitching"), use_container_width=True, hide_index=True)

    with m_tab:
        min_pa = st.slider("Minimum PA per matchup", min_value=1, max_value=20, value=2, step=1, key="adv_scout_min_pa")
        st.markdown("#### Team Hitters vs Pitchers Faced")
        hitter_matchups = _scouting_matchup_table(df, team, min_pa=min_pa)
        show_hm = ["Batter", "Pitcher", "PA", "Pitches", "BA", "OBP", "SLG", "OPS", "AvgEV", "HardHit%", "Whiff%"]
        if hitter_matchups.empty:
            st.info("No hitter matchup rows meet the selected threshold.")
        else:
            st.dataframe(style_scouting_dataframe(_table_columns(hitter_matchups, show_hm), context="hitting"), use_container_width=True, hide_index=True)

        st.markdown("#### Team Pitchers vs Batters Faced")
        pitcher_matchups = _scouting_pitcher_vs_batter_table(df, team, min_pa=min_pa)
        show_pm = ["Pitcher", "Batter", "PA", "Pitches", "BA", "OBP", "SLG", "OPS", "AvgEV", "HardHit%", "Whiff%"]
        if pitcher_matchups.empty:
            st.info("No pitcher matchup rows meet the selected threshold.")
        else:
            st.dataframe(style_scouting_dataframe(_table_columns(pitcher_matchups, show_pm), context="pitching"), use_container_width=True, hide_index=True)


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
                    if imported:
                        _SCOUTING_INDEX_FILE.unlink(missing_ok=True)
                        build_scouting_team_index.clear()
                        prepare_scouting_data.clear()
                        _scouting_parquet_index.clear()
                        _scouting_parquet_for_team.clear()
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

    if st.button("Refresh Scouting File Index", use_container_width=True):
        _SCOUTING_INDEX_FILE.unlink(missing_ok=True)
        build_scouting_team_index.clear()
        prepare_scouting_data.clear()
        _scouting_parquet_index.clear()
        _scouting_parquet_for_team.clear()
        st.rerun()

    source_sig = scouting_data_source_signature()
    csv_count = get_scouting_csv_count()
    src = _scouting_source()
    if src == "none":
        st.error("No scouting data available. Run the FTP import or ensure scouting_data.parquet is present.")
        return
    with st.spinner("Building team index…"):
        index_rows, teams = build_scouting_team_index(source_sig)
    if src == "csv":
        st.caption(f"Data source: local CSVs ({csv_count:,} files, {len(teams):,} teams indexed)")
    else:
        st.caption(f"Data source: scouting_data.parquet ({len(teams):,} teams · cloud mode)")

    if not teams:
        st.warning("No team values found in BatterTeam or PitcherTeam.")
        return
    teams = sorted(teams, key=lambda code: team_display_name(code).lower())

    d1_teams    = [c for c in teams if is_ncaa_d1_baseball_team(c)]
    other_teams = [c for c in teams if not is_ncaa_d1_baseball_team(c)]

    team_group = st.radio(
        "Team Group",
        ["NCAA D1", "Other Teams"],
        horizontal=True,
        key="scouting_team_group",
        help="'Other Teams' shows every team in the scouting data that is not mapped to a D1 conference — JUCO, D2/D3, international, travel ball, and unrecognized TrackMan codes.",
    )

    if team_group == "NCAA D1":
        teams = d1_teams
        if not teams:
            st.warning("No D1 teams found in this scouting data.")
            return
        league_options = ["All Leagues"] + sorted({team_league_name(c) for c in teams})
        filter_cols = st.columns([1.0, 2.2])
        with filter_cols[0]:
            league_filter = st.selectbox("League Filter", league_options,
                                         help="Narrow by conference.", key="scout_lg")
        if league_filter != "All Leagues":
            teams = [c for c in teams if team_league_name(c) == league_filter]
            if not teams:
                st.warning(f"No teams mapped to {league_filter} in this data.")
                return
            with filter_cols[1]:
                st.caption(f"{len(teams):,} teams in {league_filter}.")
        else:
            with filter_cols[1]:
                st.caption(f"{len(teams):,} D1 teams available.")
    else:
        teams = other_teams
        if not teams:
            st.info("No non-D1 / unrecognized team codes found in this scouting data.")
            return
        st.caption(f"{len(teams):,} other teams with tracked data (non-D1, JUCO, D2/D3, unrecognized codes).")

    mode = st.radio("Scouting View", ["PDF Reports", "2026 Leaderboards", "Advanced Scouting"], horizontal=True)

    c1, c2, c3 = st.columns([1.1, 1.0, 1.4])
    with c1:
        default_idx = teams.index("FOR_RAM") if "FOR_RAM" in teams else 0
        team = st.selectbox(
            "Team",
            teams,
            index=default_idx,
            format_func=lambda code: team_display_name(code, include_code=True)
        )
        team_label = team_display_name(team)
        st.caption(f"TrackMan code: `{team}` | League: {team_league_name(team)}")
        render_team_badge(team)

    if src == "csv":
        selected_files = _scouting_files_for_team(team, source_sig)
        st.caption(f"Selected team file set: {len(selected_files):,} CSVs involving {team_label}.")
        if not selected_files:
            st.info("No scouting CSVs found for that team.")
            return
    with st.spinner(f"Loading {team_label} scouting data..."):
        scouting_df = prepare_scouting_data(team, source_sig)

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
            sub = df[df["BatterTeam"].astype(str).isin([team, team + "1"])].copy()
            summary = summarize_contact_quality(sub, "Batter").sort_values("Bat+", ascending=False)
            table_context = "hitting"
        else:
            sub = df[df["PitcherTeam"].astype(str).isin([team, team + "1"])].copy()
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

    if mode == "Advanced Scouting":
        advanced_scouting_section(df, team, team_label)
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
            "Team": team_label,
            "Hitter PA": hitting.get("PA"),
            "BA": hitting.get("BA"),
            "OBP": hitting.get("OBP"),
            "SLG": hitting.get("SLG"),
            "wOBA": hitting.get("wOBA"),
            "Bat+": hitting.get("Bat+"),
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
                packet_scope=packet_scope,
                team_label=team_label
            )
        scope_slug = "" if not include_individual_reports else f"{_safe_pdf_name(packet_scope).lower()}_"
        file_name = f"{_safe_pdf_name(team_label)}_{scope_slug}team_scouting_report.pdf"
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
            "Team": team_label,
            "Side": card.get("Side"),
            "PA": card.get("PA"),
            "wOBA": card.get("wOBA"),
            "Bat+": card.get("Bat+"),
            "AvgEV": card.get("AvgEV"),
            "HardHit%": card.get("HardHit%"),
        }])
        st.dataframe(style_scouting_dataframe(preview, context="hitting"), hide_index=True, use_container_width=True)
        pdf_bytes = build_hitter_scouting_pdf(player_df, player, team_label, team_code=team)
        file_name = f"{_safe_pdf_name(player)}_{_safe_pdf_name(team_label)}_hitter_scout.pdf"
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
            "Team": team_label,
            "Pitches": len(player_df),
            "Strike%": round(player_df["is_strike"].mean() * 100, 1) if "is_strike" in player_df.columns and len(player_df) else np.nan,
            "Zone%": round(player_df["in_zone"].mean() * 100, 1) if "in_zone" in player_df.columns and len(player_df) else np.nan,
            "Whiff%": round(player_df["is_whiff"].sum() / swings * 100, 1) if swings else np.nan,
        }])
        st.dataframe(style_scouting_dataframe(preview, context="pitching"), hide_index=True, use_container_width=True)
        pdf_bytes = build_pitcher_scouting_pdf(player_df, player, team_label, team_code=team)
        file_name = f"{_safe_pdf_name(player)}_{_safe_pdf_name(team_label)}_pitcher_scout.pdf"

    st.download_button(
        "Download Scouting PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
        use_container_width=True
    )


# ============================================================
# HOME RUN DISTANCE LEADERBOARD
# ============================================================

def _build_hr_figure(board: pd.DataFrame) -> plt.Figure:
    """Matplotlib figure: field overhead + ranked distance chart."""
    import matplotlib.patches as mpatches

    BACKGROUND = "#100D0C"
    fig = plt.figure(figsize=(18, 9), facecolor=BACKGROUND)
    fig.suptitle("FORDHAM RAMS — Home Run Tracker 2026",
                 color=FORDHAM_GOLD, fontsize=20, fontweight="bold", y=0.97)

    ax_field = fig.add_axes([0.03, 0.08, 0.43, 0.84])
    ax_chart = fig.add_axes([0.49, 0.08, 0.50, 0.84])

    # ── Field view ────────────────────────────────────────────────────────────
    ax_field.set_facecolor("#1A1008")
    ax_field.set_aspect("equal")
    ax_field.axis("off")

    # Foul lines
    for angle in [-45, 45]:
        rad = np.radians(angle)
        ax_field.plot([0, 370*np.sin(rad)], [0, 370*np.cos(rad)],
                      color="#555", linewidth=1.5)

    # Distance rings with labels only at the edge (no center label to avoid overlap)
    for d, label_ang in [(200, 48), (300, 48), (400, 48)]:
        theta = np.linspace(np.radians(-50), np.radians(50), 120)
        ax_field.plot(d*np.sin(theta), d*np.cos(theta),
                      color="#2a2a2a", linewidth=0.9, linestyle="--")
        # Label at the right edge of the arc only
        lx = d * np.sin(np.radians(label_ang))
        ly = d * np.cos(np.radians(label_ang))
        ax_field.text(lx + 5, ly, f"{d}'", color="#666", fontsize=7,
                      ha="left", va="center")

    # Outfield arc (LF 338, CF 395, RF 320)
    arc_ang = np.linspace(np.radians(-45), np.radians(45), 120)
    arc_d   = np.interp(arc_ang,
                        [np.radians(-45), 0, np.radians(45)],
                        [338, 395, 320])
    ax_field.plot(arc_d*np.sin(arc_ang), arc_d*np.cos(arc_ang),
                  color="#555", linewidth=2.0)

    # Infield grass and dirt
    theta_c = np.linspace(0, 2*np.pi, 120)
    ax_field.fill(95*np.sin(theta_c), 95*np.cos(theta_c), color="#1E2A10", zorder=2)
    ax_field.fill(60*np.sin(theta_c), 60*np.cos(theta_c), color="#2C1F0F", zorder=3)

    # Home plate
    ax_field.scatter([0], [0], s=70, color="white", zorder=6, marker="s")

    # HR dots — one color per unique batter, no overlapping text labels
    batters = board["Batter"].unique()
    try:
        cmap = plt.colormaps.get_cmap("tab10")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10")
    batter_colors = {b: cmap(i % 10) for i, b in enumerate(batters)}

    if "Direction" in board.columns and "Dist (ft)" in board.columns:
        for _, row in board.iterrows():
            d   = row["Dist (ft)"]
            ang = float(row.get("Direction") or 0)
            if pd.isna(d) or d < 100:
                continue
            x = d * np.sin(np.radians(ang))
            y = d * np.cos(np.radians(ang))
            col = batter_colors.get(row["Batter"], "red")
            ax_field.scatter(x, y, s=140, color=col, edgecolor="white",
                             linewidth=1.2, zorder=7)

    ax_field.set_xlim(-390, 390)
    ax_field.set_ylim(-30, 440)
    ax_field.set_title("Landing Spots", color="#FFF7E8", fontsize=12,
                        fontweight="bold", pad=4)

    # Legend outside the field area, no overlap
    legend_patches = [mpatches.Patch(color=batter_colors[b],
                                     label=b.split(",")[0].strip())
                      for b in batters]
    ax_field.legend(handles=legend_patches, loc="lower center",
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=min(len(batters), 4),
                    facecolor="#1a1a1a", edgecolor="#444", labelcolor="white",
                    fontsize=8.5, framealpha=0.9)

    # ── Ranked bar chart ──────────────────────────────────────────────────────
    ax_chart.set_facecolor(BACKGROUND)
    view = board.head(15).iloc[::-1]   # top 15, bottom-to-top
    y_pos = list(range(len(view)))

    colors_bar = [batter_colors.get(b, FORDHAM_MAROON) for b in view["Batter"]]
    max_dist = view["Dist (ft)"].max() if not view.empty else 400
    bars = ax_chart.barh(y_pos, view["Dist (ft)"].values,
                         color=colors_bar, edgecolor="white", linewidth=0.5, height=0.68)

    # Label inside the bar to avoid overflow
    for bar, (_, row) in zip(bars, view.iterrows()):
        w = bar.get_width()
        dist_label = f"{w:.0f} ft"
        ev_label   = (f"  EV {row['EV (mph)']:.0f}  ·  LA {row['LA (°)']:.0f}°"
                      if pd.notna(row.get("EV (mph)")) else "")
        # Distance inside bar (right-aligned)
        ax_chart.text(w - 4, bar.get_y() + bar.get_height()/2,
                      dist_label, color="white", fontsize=9.5,
                      va="center", ha="right", fontweight="bold")
        # EV/LA outside bar only if there's room
        if w < max_dist * 0.82 and ev_label:
            ax_chart.text(w + 3, bar.get_y() + bar.get_height()/2,
                          ev_label, color="#CDBFAF", fontsize=8,
                          va="center", ha="left")

    # Y-axis labels — just first name vs opponent
    def _safe_label(r):
        name     = r["Batter"].split(",")[0].strip() if "," in r["Batter"] else r["Batter"]
        opp_raw  = str(r.get("Opponent", "—"))
        opponent = opp_raw.split()[0] if opp_raw not in ("—", "") else "—"
        return f"{name}  vs {opponent}"

    ax_chart.set_yticks(y_pos)
    ax_chart.set_yticklabels([_safe_label(r) for _, r in view.iterrows()],
                              color="white", fontsize=9, fontweight="bold")
    ax_chart.set_xlabel("Distance (ft)", color="#CDBFAF", fontsize=10, fontweight="bold")
    ax_chart.set_xlim(0, max_dist * 1.06)   # tight limit so labels stay inside
    ax_chart.tick_params(colors="#CDBFAF", labelsize=9)
    ax_chart.spines[:].set_color("#333")
    ax_chart.grid(axis="x", color="#222", linewidth=0.7, alpha=0.8)
    ax_chart.set_title("Ranked by Distance", color="#FFF7E8",
                        fontsize=12, fontweight="bold", pad=4)

    fig.text(0.5, 0.01, "Fordham Baseball  ·  2026 TrackMan",
             ha="center", color="#555", fontsize=8.5)
    return fig


def hr_distance_leaderboard_page(all_pitches_df: pd.DataFrame):
    st.markdown("## Home Run Distance Leaderboard")
    st.caption("Fordham hitters only — all tracked home runs from 2026 game data.")

    df = all_pitches_df.copy()
    if "BatterTeam" in df.columns:
        df = df[df["BatterTeam"].astype(str).str.upper().isin(["FOR_RAM", "FOR_RAM1"])]

    hr = df[df.get("PlayResult", pd.Series("", index=df.index)).astype(str).eq("HomeRun")].copy()
    for col in ["Distance", "ExitSpeed", "Angle", "Direction"]:
        if col in hr.columns:
            hr[col] = pd.to_numeric(hr[col], errors="coerce")
    hr = hr[hr.get("Distance", pd.Series(0, index=hr.index)).gt(200)]

    if hr.empty:
        st.warning("No Fordham home run data found.")
        return

    rows = []
    for _, row in hr.iterrows():
        rows.append({
            "Batter":    str(row.get("Batter",      "—")),
            "Date":      str(row.get("Date",         "—")),
            "Opponent":  team_display_name(str(row.get("PitcherTeam", ""))),
            "Pitcher":   str(row.get("Pitcher",      "—")),
            "Dist (ft)": round(float(row["Distance"]), 1),
            "EV (mph)":  round(float(row["ExitSpeed"]), 1) if pd.notna(row.get("ExitSpeed")) else float("nan"),
            "LA (°)":    round(float(row["Angle"]),     1) if pd.notna(row.get("Angle"))     else float("nan"),
            "Direction": float(row["Direction"])            if pd.notna(row.get("Direction")) else 0.0,
        })

    board = (pd.DataFrame(rows)
               .sort_values("Dist (ft)", ascending=False)
               .reset_index(drop=True))
    board.index = board.index + 1

    # Top callout metrics
    top_n = min(4, len(board))
    mcols = st.columns(top_n)
    for i, col in enumerate(mcols):
        r = board.iloc[i]
        col.metric(
            f"#{i+1}  {r['Batter'].split(',')[0].strip()}",
            f"{r['Dist (ft)']:.0f} ft",
            f"EV {r['EV (mph)']:.0f}  ·  LA {r['LA (°)']:.0f}°" if pd.notna(r["EV (mph)"]) else "",
        )

    # Graphic
    fig = _build_hr_figure(board)
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    st.download_button("Download HR Chart", buf,
                       file_name="fordham_hr_tracker_2026.png",
                       mime="image/png", use_container_width=True)

    st.markdown("---")
    display_cols = [c for c in ["Batter","Date","Opponent","Pitcher","Dist (ft)","EV (mph)","LA (°)"]
                    if c in board.columns]
    st.dataframe(
        style_scouting_dataframe(board[display_cols], context="hitting"),
        use_container_width=True,
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

    df = add_contact_quality_local(df)

    raw_teams = sorted(set(
        df.get("BatterTeam", pd.Series(dtype=str)).dropna().unique().tolist() +
        df.get("PitcherTeam", pd.Series(dtype=str)).dropna().unique().tolist()
    ))

    # Group raw codes by display name — deduplicate school entries
    name_to_codes: dict = {}
    for code in raw_teams:
        name = TEAM_CODE_NAME_OVERRIDES.get(code, code)
        name_to_codes.setdefault(name, []).append(code)

    team_names = sorted(name_to_codes.keys())
    if not team_names:
        st.warning("No team info found.")
        return

    default_name = TEAM_CODE_NAME_OVERRIDES.get("FOR_RAM", "FOR_RAM")
    default_idx  = team_names.index(default_name) if default_name in team_names else 0
    team_name    = st.selectbox("Select Team", team_names, index=default_idx)
    team_variants = name_to_codes[team_name]

    mode = st.radio("View:", ["Hitters", "Pitchers"], horizontal=True)

    if mode == "Hitters":
        sub = df[df["BatterTeam"].astype(str).isin(team_variants)]
        summary = summarize_contact_quality(sub, "Batter")
        summary = summary.sort_values("Bat+", ascending=False)
        st.dataframe(style_scouting_dataframe(summary, context="hitting"), use_container_width=True)

    else:
        sub = df[df["PitcherTeam"].astype(str).isin(team_variants)]
        summary = summarize_contact_quality(sub, "Pitcher")
        if "Stuff+" in sub.columns and not summary.empty:
            stuff_summary = sub.groupby("Pitcher").agg(Stuff_plus=("Stuff+", "mean")).reset_index()
            summary = summary.merge(stuff_summary, on="Pitcher", how="left")
            summary = summary.rename(columns={"Stuff_plus": "Stuff+"})
        summary = summary.sort_values("HardHit%", ascending=True)
        pitcher_cols = [c for c in [
            "Pitcher", "PA", "AB", "BIP", "HR", "BA", "OBP", "SLG", "BABIP",
            "wOBA", "Bat+", "AvgEV", "HardHit%", "Barrel%", "K%", "BB%",
            "Whiff%", "Chase%", "Stuff+",
        ] if c in summary.columns]
        st.dataframe(
            style_scouting_dataframe(_table_columns(summary, pitcher_cols), context="pitching"),
            use_container_width=True,
        )


def _legacy_hitter_development_page_table_only(all_pitches_df: pd.DataFrame):

    st.title("Hitter Development & Approach")

    df = normalize_hitter_columns(all_pitches_df)
    df = add_contact_quality_local(df)

    # Only FOR_RAM hitters (no pitchers)
    if "BatterTeam" in df.columns:
        df = df[df["BatterTeam"].astype(str).str.upper().str.startswith("FOR_RAM")]

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
        st.metric("xHB", card.get("xHB", card["2B"] + card["3B"] + card["HR"]))

    with c2:
        st.metric("BB%", f"{card['BB%']}%")
        st.metric("K%", f"{card['K%']}%")
        st.metric("Swing%", f"{card['Swing%']}%")
        st.metric("Chase%", f"{card['Chase%']}%")

    with c3:
        st.metric("wOBA", f"{card['wOBA']:.3f}")
        st.metric("Bat+", f"{card.get('Bat+', card.get('wRC+', '-'))}")
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

    df = apply_date_range_filter(df, "pitcher_development")
    if df.empty:
        st.warning("No FOR_RAM pitcher data found in the selected date range.")
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

    # SECTION 1 - ARSENAL OVERVIEW
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

    # ── Arsenal Visuals: Movement + Usage/Velo bars ──────────────────────────
    _DEV_PC = {
        "FB": "#1f77b4", "SI": "#17becf", "FC": "#ff9f1c",
        "SL": "#e63946", "CU": "#7b2cbf", "CH": "#2a9d8f", "SW": "#b56576",
    }

    move_c1, move_c2 = st.columns([3, 2])
    with move_c1:
        # Pitch break plot
        _mv = pdf[["pitch_abbr", "HB", "IVB", "Velo"]].copy()
        _mv["HB"]  = pd.to_numeric(_mv["HB"],  errors="coerce")
        _mv["IVB"] = pd.to_numeric(_mv["IVB"], errors="coerce")
        _mv["Velo"]= pd.to_numeric(_mv["Velo"],errors="coerce")
        _mv = _mv.dropna(subset=["HB", "IVB"])
        if not _mv.empty:
            _throws = str(pdf["PitcherThrows"].dropna().iloc[0]).upper() if "PitcherThrows" in pdf.columns and pdf["PitcherThrows"].notna().any() else "R"
            _xlim = max(26, float(np.nanmax(np.abs(_mv["HB"]))) + 4)
            _ylo  = min(-20, float(_mv["IVB"].min()) - 3)
            _yhi  = max(22, float(_mv["IVB"].max()) + 3)
            fig_mv, ax_mv = plt.subplots(figsize=(7, 6))
            fig_mv.patch.set_facecolor("#100D0C")
            ax_mv.set_facecolor("#181412")
            if _throws.startswith("R"):
                ax_mv.axvspan(0, _xlim,  facecolor="#13365F", alpha=0.18)
                ax_mv.axvspan(-_xlim, 0, facecolor="#5E1814", alpha=0.18)
                ax_mv.text(_xlim*0.55,  _yhi-2.5, "ARM SIDE",   color="#9FC7FF", fontsize=9, fontweight="bold", ha="center")
                ax_mv.text(-_xlim*0.55, _yhi-2.5, "GLOVE SIDE", color="#FFB1A8", fontsize=9, fontweight="bold", ha="center")
            else:
                ax_mv.axvspan(-_xlim, 0, facecolor="#13365F", alpha=0.18)
                ax_mv.axvspan(0, _xlim,  facecolor="#5E1814", alpha=0.18)
                ax_mv.text(-_xlim*0.55, _yhi-2.5, "ARM SIDE",   color="#9FC7FF", fontsize=9, fontweight="bold", ha="center")
                ax_mv.text(_xlim*0.55,  _yhi-2.5, "GLOVE SIDE", color="#FFB1A8", fontsize=9, fontweight="bold", ha="center")
            for _pt, _sub in _mv.groupby("pitch_abbr", sort=False):
                _col = _DEV_PC.get(_pt, "#CDBFAF")
                ax_mv.scatter(_sub["HB"], _sub["IVB"], s=30, alpha=0.22, color=_col, edgecolor="none", zorder=2)
            _cen = _mv.groupby("pitch_abbr").agg(N=("pitch_abbr","count"), HB=("HB","mean"), IVB=("IVB","mean"), Velo=("Velo","mean")).reset_index()
            for _, _r in _cen.iterrows():
                _col = _DEV_PC.get(_r["pitch_abbr"], "#CDBFAF")
                ax_mv.scatter(_r["HB"], _r["IVB"], s=500, color=_col, edgecolor="#FFF7E8", linewidth=1.6, zorder=5)
                ax_mv.text(_r["HB"], _r["IVB"], _r["pitch_abbr"], color="#FFFFFF", fontsize=12, fontweight="bold", ha="center", va="center", zorder=6)
            ax_mv.axhline(0, color="#FFF7E8", linewidth=1.2, alpha=0.65)
            ax_mv.axvline(0, color="#FFF7E8", linewidth=1.2, alpha=0.65)
            ax_mv.grid(True, color="#C7A45D", alpha=0.12, linewidth=0.7)
            ax_mv.set_xlim(-_xlim, _xlim)
            ax_mv.set_ylim(_ylo, _yhi)
            ax_mv.set_aspect("equal", adjustable="box")
            ax_mv.set_title("Pitch Movement Profile", color="#FFF7E8", fontsize=16, fontweight="bold", pad=10)
            ax_mv.set_xlabel("Horizontal Break (in.)", color="#CDBFAF", fontsize=11, fontweight="bold")
            ax_mv.set_ylabel("Induced Vert Break (in.)", color="#CDBFAF", fontsize=11, fontweight="bold")
            ax_mv.tick_params(colors="#CDBFAF", labelsize=9)
            for sp in ax_mv.spines.values():
                sp.set_color("#4E4036"); sp.set_linewidth(0.8)
            st.pyplot(fig_mv, use_container_width=True)
            plt.close(fig_mv)

    with move_c2:
        # Usage % and avg velo bar charts stacked
        if not arsenal.empty:
            _pitches_sorted = arsenal.sort_values("Usage%", ascending=True)
            fig_bars, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(6, 5))
            fig_bars.patch.set_facecolor("#100D0C")
            for _axi in (ax_u, ax_v):
                _axi.set_facecolor("#181412")
                for sp in _axi.spines.values():
                    sp.set_color("#4E4036"); sp.set_linewidth(0.7)
                _axi.tick_params(colors="#CDBFAF", labelsize=9)

            _pts = list(_pitches_sorted.index)
            _cols = [_DEV_PC.get(p, "#CDBFAF") for p in _pts]
            _usages = list(_pitches_sorted["Usage%"])
            _velos  = list(_pitches_sorted["Velo"])

            ax_u.barh(_pts, _usages, color=_cols, edgecolor="#4E4036", linewidth=0.6, height=0.65)
            for i, (v, p) in enumerate(zip(_usages, _pts)):
                ax_u.text(v + 0.5, i, f"{v:.0f}%", va="center", color="#FFF7E8", fontsize=9, fontweight="bold")
            ax_u.set_xlim(0, max(_usages) * 1.22)
            ax_u.set_xlabel("Usage %", color="#CDBFAF", fontsize=10, fontweight="bold")
            ax_u.set_title("Usage", color="#FFF7E8", fontsize=13, fontweight="bold", pad=8)
            ax_u.yaxis.label.set_color("#CDBFAF")
            ax_u.set_facecolor("#181412")

            ax_v.barh(_pts, _velos, color=_cols, edgecolor="#4E4036", linewidth=0.6, height=0.65)
            _vmin_disp = max(0, min(_velos) - 3) if _velos else 0
            for i, (v, p) in enumerate(zip(_velos, _pts)):
                if not np.isnan(v):
                    ax_v.text(v + 0.3, i, f"{v:.1f}", va="center", color="#FFF7E8", fontsize=9, fontweight="bold")
            ax_v.set_xlim(_vmin_disp, max(_velos) * 1.06 if _velos else 100)
            ax_v.set_xlabel("Avg Velo (mph)", color="#CDBFAF", fontsize=10, fontweight="bold")
            ax_v.set_title("Avg Velocity", color="#FFF7E8", fontsize=13, fontweight="bold", pad=8)
            ax_v.set_yticks([])
            fig_bars.tight_layout(pad=1.2)
            st.pyplot(fig_bars, use_container_width=True)
            plt.close(fig_bars)

    # SECTION 2 - COUNT-BASED EFFECTIVENESS
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

    # SECTION 3 - STRIKE ZONE 9-BOX
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

    # SECTION 4 - RELEASE CONSISTENCY
    st.markdown("### Release Consistency")

    rel = pdf.groupby("pitch_abbr").agg(
        RelH_std=("RelH", "std"),
        RelS_std=("RelS", "std"),
        Ext_std=("Ext", "std")
    ).round(3)

    st.dataframe(style_scouting_dataframe(rel, context="pitching"), use_container_width=True)

    # Release point scatter (visual consistency check)
    _rel_raw = pdf[["pitch_abbr", "RelS", "RelH"]].copy()
    _rel_raw["RelS"] = pd.to_numeric(_rel_raw["RelS"], errors="coerce")
    _rel_raw["RelH"] = pd.to_numeric(_rel_raw["RelH"], errors="coerce")
    _rel_raw = _rel_raw.dropna(subset=["RelS", "RelH"])
    if not _rel_raw.empty:
        fig_rel, ax_rel = plt.subplots(figsize=(6.5, 5.5))
        fig_rel.patch.set_facecolor("#100D0C")
        ax_rel.set_facecolor("#181412")
        _rx_pad = max(0.4, float(_rel_raw["RelS"].std()) * 3 + 0.3)
        _ry_pad = max(0.4, float(_rel_raw["RelH"].std()) * 3 + 0.3)
        _rx_c   = float(_rel_raw["RelS"].mean())
        _ry_c   = float(_rel_raw["RelH"].mean())
        for _pt2, _sg in _rel_raw.groupby("pitch_abbr", sort=False):
            _col2 = _DEV_PC.get(_pt2, "#CDBFAF")
            ax_rel.scatter(_sg["RelS"], _sg["RelH"], s=28, alpha=0.45, color=_col2, edgecolor="none", label=_pt2, zorder=3)
        for _pt2, _sg in _rel_raw.groupby("pitch_abbr", sort=False):
            _col2 = _DEV_PC.get(_pt2, "#CDBFAF")
            ax_rel.scatter(_sg["RelS"].mean(), _sg["RelH"].mean(), s=260, color=_col2, edgecolor="#FFF7E8", linewidth=1.4, zorder=5)
            ax_rel.text(_sg["RelS"].mean(), _sg["RelH"].mean(), _pt2, color="#FFFFFF", fontsize=10, fontweight="bold", ha="center", va="center", zorder=6)
        ax_rel.set_xlim(_rx_c - _rx_pad, _rx_c + _rx_pad)
        ax_rel.set_ylim(_ry_c - _ry_pad, _ry_c + _ry_pad)
        ax_rel.grid(True, color="#C7A45D", alpha=0.10, linewidth=0.7)
        ax_rel.set_title("Release Point by Pitch Type", color="#FFF7E8", fontsize=15, fontweight="bold", pad=10)
        ax_rel.set_xlabel("Horizontal Release Side (ft.)", color="#CDBFAF", fontsize=10, fontweight="bold")
        ax_rel.set_ylabel("Release Height (ft.)", color="#CDBFAF", fontsize=10, fontweight="bold")
        ax_rel.tick_params(colors="#CDBFAF", labelsize=9)
        for sp in ax_rel.spines.values():
            sp.set_color("#4E4036"); sp.set_linewidth(0.8)
        _rel_handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=_DEV_PC.get(p,"#CDBFAF"), markersize=9, label=p) for p in _rel_raw["pitch_abbr"].unique()]
        ax_rel.legend(handles=_rel_handles, loc="best", facecolor="#211C1A", edgecolor="#C7A45D", labelcolor="#FFF7E8", fontsize=9)
        fig_rel.tight_layout()
        st.pyplot(fig_rel, use_container_width=True)
        plt.close(fig_rel)

    # SECTION 5 - PITCH-TO-PITCH SEQUENCING
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

    # Sequencing transition matrix heatmap (Whiff%)
    _seq_pitches = sorted(set(list(seq_stats["PrevPitch"].unique()) + list(seq_stats["pitch_abbr"].unique())))
    if len(_seq_pitches) >= 2 and not seq_stats.empty:
        _mat_w  = pd.DataFrame(np.nan, index=_seq_pitches, columns=_seq_pitches)
        _mat_n  = pd.DataFrame(0,      index=_seq_pitches, columns=_seq_pitches)
        for _, _sr in seq_stats.iterrows():
            _mat_w.at[_sr["PrevPitch"], _sr["pitch_abbr"]]  = _sr["Whiff%"]
            _mat_n.at[_sr["PrevPitch"], _sr["pitch_abbr"]]  = int(_sr["N"])

        _seq_cols2, _ = st.columns([3, 1])
        with _seq_cols2:
            _n  = len(_seq_pitches)
            _fs = max(4, int(10 - _n * 0.5))
            fig_seq, ax_seq = plt.subplots(figsize=(min(9, _n * 1.6 + 1.5), min(8, _n * 1.4 + 1.2)))
            fig_seq.patch.set_facecolor("#100D0C")
            ax_seq.set_facecolor("#181412")
            import matplotlib.colors as mcolors
            _cmap_seq = mcolors.LinearSegmentedColormap.from_list("seq", [
                (0.00, "#0a2e6e"), (0.35, "#5ea3d0"), (0.50, "#787878"),
                (0.65, "#f5a17a"), (1.00, "#8b0000"),
            ])
            _arr = _mat_w.values.astype(float)
            _im  = ax_seq.imshow(_arr, cmap=_cmap_seq, vmin=0, vmax=100, aspect="auto")
            for _ri in range(_n):
                for _ci in range(_n):
                    _val = _arr[_ri, _ci]
                    _cnt = int(_mat_n.iloc[_ri, _ci])
                    if not np.isnan(_val):
                        _bg_norm = _val / 100
                        _tc = "#FFF7E8" if _bg_norm > 0.45 else "#130F0D"
                        ax_seq.text(_ci, _ri, f"{_val:.0f}%\nn={_cnt}", ha="center", va="center",
                                    color=_tc, fontsize=_fs + 1, fontweight="bold", linespacing=1.2)
                    else:
                        ax_seq.text(_ci, _ri, "–", ha="center", va="center", color="#4E4036", fontsize=_fs)
            ax_seq.set_xticks(range(_n)); ax_seq.set_xticklabels(_seq_pitches, fontsize=_fs+2, fontweight="bold", color="#FFF7E8")
            ax_seq.set_yticks(range(_n)); ax_seq.set_yticklabels(_seq_pitches, fontsize=_fs+2, fontweight="bold", color="#FFF7E8")
            ax_seq.set_xlabel("Next Pitch →", color="#CDBFAF", fontsize=11, fontweight="bold", labelpad=8)
            ax_seq.set_ylabel("← Previous Pitch", color="#CDBFAF", fontsize=11, fontweight="bold", labelpad=8)
            ax_seq.set_title("Sequencing Matrix  (Whiff% after each combo)", color="#FFF7E8", fontsize=14, fontweight="bold", pad=12)
            ax_seq.tick_params(length=0)
            for sp in ax_seq.spines.values():
                sp.set_color("#4E4036")
            _cb = fig_seq.colorbar(_im, ax=ax_seq, fraction=0.035, pad=0.03)
            _cb.set_label("Whiff%", color="#CDBFAF", fontsize=9, fontweight="bold")
            _cb.ax.tick_params(colors="#CDBFAF", labelsize=8)
            _cb.outline.set_edgecolor("#4E4036")
            fig_seq.tight_layout()
            st.pyplot(fig_seq, use_container_width=True)
            plt.close(fig_seq)

    st.dataframe(
        style_scouting_dataframe(seq_stats[["PrevPitch", "pitch_abbr", "N", "Whiff%", "HardHit%"]], context="pitching"),
        use_container_width=True
    )

    # SECTION 6 - LHH vs RHH SPLITS
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

    # SECTION 7 - SMART DEVELOPMENT RECOMMENDATIONS
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
                f"Increase **{pitch}** usage: elite Whiff% ({whiff}) with low damage ({hardhit} HardHit%)."
            )

        if hardhit >= 40 and whiff <= 20:
            recs.append(
                f"Reduce **{pitch}** usage: high HardHit% ({hardhit}) with limited swing/miss ({whiff} Whiff%)."
            )

    seq_good = seq_stats[seq_stats["N"] >= 10].sort_values("Whiff%", ascending=False)
    if not seq_good.empty:
        best = seq_good.iloc[0]
        recs.append(
            f"Best sequencing pair: **{best['PrevPitch']} -> {best['pitch_abbr']}** "
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
                    f"Improve release consistency on **{pitch}**: large variance in release height, side, or extension."
                )

    if not recs:
        st.success("No major issues detected. Arsenal is well optimized.")
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
        df = df[df["BatterTeam"].astype(str).str.upper().str.startswith("FOR_RAM")]

    if df.empty:
        st.error("No FOR_RAM hitters found.")
        return

    df = apply_date_range_filter(df, "hitter_development")
    if df.empty:
        st.warning("No FOR_RAM hitter data found in the selected date range.")
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

    # ── Visual hitter summary card ────────────────────────────────────────────
    _hitter_side_h = str(hdf["BatterSide"].dropna().mode().iloc[0]) if "BatterSide" in hdf.columns and hdf["BatterSide"].notna().any() else ""

    _hcard_metrics = [
        # (label, value_str, good_is_high, lo, hi)
        ("PA",       f"{card['PA']}",         None,  0,   1),
        ("H",        f"{card['H']}",           None,  0,   1),
        ("HR",       f"{card['HR']}",          None,  0,   1),
        ("BB%",      f"{card['BB%']}%",        True,  5,  16),
        ("K%",       f"{card['K%']}%",         False, 12,  28),
        ("Swing%",   f"{card['Swing%']}%",     None,  0,   1),
        ("Chase%",   f"{card['Chase%']}%",     False, 20,  38),
        ("Whiff%",   f"{card['Whiff%']}%",     False, 18,  35),
        ("wOBA",     f"{card['wOBA']:.3f}",    True, .250,.430),
        ("Bat+",     f"{card.get('Bat+', card.get('wRC+', '-'))}",        True,  70, 140),
        ("HardHit%", f"{card['HardHit%']}%",   True,  25,  55),
        ("Barrel%",  f"{card['Barrel%']}%",    True,   3,  15),
        ("Avg EV",   f"{card['AvgEV']}",       True,  82,  95),
        ("Max EV",   f"{card['MaxEV']}",       True,  95, 115),
    ]

    def _hcard_color_fn(good_is_high, lo, hi, val_str):
        if good_is_high is None:
            return "#4E4036"
        try:
            v = float(str(val_str).replace("%",""))
        except Exception:
            return "#4E4036"
        norm = float(np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1))
        if not good_is_high:
            norm = 1 - norm
        # Baseball Savant stops
        if norm >= 0.85:  return "#8b0000"
        if norm >= 0.65:  return "#d13c28"
        if norm >= 0.50:  return "#787878"
        if norm >= 0.35:  return "#5ea3d0"
        if norm >= 0.15:  return "#1956a0"
        return "#0a2e6e"

    _n_hc = len(_hcard_metrics)
    _hc_cols = 7
    _hc_rows = (_n_hc + _hc_cols - 1) // _hc_cols
    fig_hcard, ax_hcard = plt.subplots(figsize=(13.5, _hc_rows * 1.55))
    fig_hcard.patch.set_facecolor("#100D0C")
    ax_hcard.set_facecolor("#100D0C")
    ax_hcard.axis("off")
    _cell_w = 1.0 / _hc_cols
    _cell_h = 1.0 / _hc_rows
    for _idx, (_lbl, _vstr, _gih, _lo, _hi) in enumerate(_hcard_metrics):
        _col_i = _idx % _hc_cols
        _row_i = _idx // _hc_cols
        _xb = _col_i * _cell_w + 0.006
        _yb = 1.0 - (_row_i + 1) * _cell_h + 0.008
        _bcolor = _hcard_color_fn(_gih, _lo, _hi, _vstr)
        # Parse lum for dynamic text
        _hx = _bcolor.lstrip("#")
        _hr, _hg, _hb = (int(_hx[i:i+2],16) for i in (0,2,4))
        _lum = (0.299*_hr + 0.587*_hg + 0.114*_hb) / 255
        _val_txt = "#111111" if _lum > 0.50 else "#FFF7E8"
        _lbl_txt = "#444444" if _lum > 0.50 else "#CDBFAF"
        import matplotlib.patches as _mp
        ax_hcard.add_patch(_mp.FancyBboxPatch(
            (_xb, _yb), _cell_w - 0.012, _cell_h - 0.016,
            boxstyle="round,pad=0.01", transform=ax_hcard.transAxes,
            facecolor=_bcolor, edgecolor="none"
        ))
        ax_hcard.text(_xb + (_cell_w - 0.012)/2, _yb + (_cell_h - 0.016)*0.72, _vstr,
                      transform=ax_hcard.transAxes, ha="center", va="center",
                      fontsize=16, fontweight="bold", color=_val_txt)
        ax_hcard.text(_xb + (_cell_w - 0.012)/2, _yb + (_cell_h - 0.016)*0.22, _lbl,
                      transform=ax_hcard.transAxes, ha="center", va="center",
                      fontsize=9, fontweight="bold", color=_lbl_txt)
    fig_hcard.tight_layout(pad=0.3)
    st.pyplot(fig_hcard, use_container_width=True)
    plt.close(fig_hcard)

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

        # Visual: grouped bar chart for Whiff%, Chase%, HardHit% by pitch type
        _pt_plot = pitchtype_df[pitchtype_df["N"] >= 5].copy()
        if not _pt_plot.empty:
            _DEV_PC_H = {
                "FB": "#1f77b4", "SI": "#17becf", "FC": "#ff9f1c",
                "SL": "#e63946", "CU": "#7b2cbf", "CH": "#2a9d8f",
                "SW": "#b56576", "BR": "#b56576",
            }
            _pt_lbls = list(_pt_plot["Pitch"])
            _n_pt = len(_pt_lbls)
            _metrics_bar = ["Whiff%", "Chase%", "HardHit%"]
            _bar_colors  = ["#e63946", "#ff9f1c", "#2a9d8f"]
            _bar_width = 0.25
            _x = np.arange(_n_pt)

            fig_pt, ax_pt = plt.subplots(figsize=(max(7, _n_pt * 1.4), 5.5))
            fig_pt.patch.set_facecolor("#100D0C")
            ax_pt.set_facecolor("#181412")
            for _mi, (_met, _bcol) in enumerate(zip(_metrics_bar, _bar_colors)):
                _vals = [float(_pt_plot.loc[_pt_plot["Pitch"] == p, _met].values[0]) if p in _pt_plot["Pitch"].values else 0 for p in _pt_lbls]
                _bars = ax_pt.bar(_x + (_mi - 1) * _bar_width, _vals, _bar_width - 0.03,
                                  label=_met, color=_bcol, edgecolor="#4E4036", linewidth=0.5, alpha=0.88)
                for _b, _v in zip(_bars, _vals):
                    if _v > 2:
                        ax_pt.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 0.8,
                                   f"{_v:.0f}", ha="center", va="bottom", color="#FFF7E8", fontsize=8.5, fontweight="bold")
            ax_pt.set_xticks(_x)
            ax_pt.set_xticklabels(_pt_lbls, color="#FFF7E8", fontsize=12, fontweight="bold")
            ax_pt.set_ylabel("Rate (%)", color="#CDBFAF", fontsize=10, fontweight="bold")
            ax_pt.set_title("Pitch Type Effectiveness", color="#FFF7E8", fontsize=15, fontweight="bold", pad=10)
            ax_pt.set_ylim(0, 105)
            ax_pt.tick_params(colors="#CDBFAF", labelsize=9)
            ax_pt.grid(axis="y", color="#C7A45D", alpha=0.10, linewidth=0.7)
            ax_pt.legend(facecolor="#211C1A", edgecolor="#C7A45D", labelcolor="#FFF7E8", fontsize=10)
            for sp in ax_pt.spines.values():
                sp.set_color("#4E4036"); sp.set_linewidth(0.7)
            fig_pt.tight_layout()
            st.pyplot(fig_pt, use_container_width=True)
            plt.close(fig_pt)

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


def _practice_metric_value(df: pd.DataFrame, col: str, pct=False):
    if df is None or df.empty or col not in df.columns:
        return np.nan
    values = pd.to_numeric(df[col], errors="coerce")
    value = values.mean()
    return value * 100 if pct else value


def _practice_arsenal_table(pdf: pd.DataFrame) -> pd.DataFrame:
    if pdf is None or pdf.empty or "pitch_abbr" not in pdf.columns:
        return pd.DataFrame()

    work = pdf.copy()
    for col in ["Velo", "PerceivedVelo", "IVB", "HB", "Spin", "Ext", "RelH", "RelS", "Stuff+", "Loc+"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("pitch_abbr", dropna=False).agg(
        N=("pitch_abbr", "count"),
        Velo=("Velo", "mean") if "Velo" in work.columns else ("pitch_abbr", "count"),
        PerVelo=("PerceivedVelo", "mean") if "PerceivedVelo" in work.columns else ("pitch_abbr", "count"),
        IVB=("IVB", "mean") if "IVB" in work.columns else ("pitch_abbr", "count"),
        HB=("HB", "mean") if "HB" in work.columns else ("pitch_abbr", "count"),
        Spin=("Spin", "mean") if "Spin" in work.columns else ("pitch_abbr", "count"),
        Ext=("Ext", "mean") if "Ext" in work.columns else ("pitch_abbr", "count"),
        RelHt=("RelH", "mean") if "RelH" in work.columns else ("pitch_abbr", "count"),
        StuffPlus=("Stuff+", "mean") if "Stuff+" in work.columns else ("pitch_abbr", "count"),
        LocPlus=("Loc+", "mean") if "Loc+" in work.columns else ("pitch_abbr", "count"),
        Zone=("in_zone", "mean") if "in_zone" in work.columns else ("pitch_abbr", "count"),
        Strike=("is_strike", "mean") if "is_strike" in work.columns else ("pitch_abbr", "count"),
        CSW=("is_csw", "mean") if "is_csw" in work.columns else ("pitch_abbr", "count"),
        Swings=("is_swing", "sum") if "is_swing" in work.columns else ("pitch_abbr", "count"),
        Whiffs=("is_whiff", "sum") if "is_whiff" in work.columns else ("pitch_abbr", "count"),
    ).reset_index().rename(columns={"pitch_abbr": "Pitch", "StuffPlus": "Stuff+", "LocPlus": "Loc+"})

    total = max(len(work), 1)
    grouped["Usage%"] = grouped["N"] / total * 100
    grouped["Zone%"] = grouped["Zone"] * 100
    grouped["Strike%"] = grouped["Strike"] * 100
    grouped["CSW%"] = grouped["CSW"] * 100
    grouped["Whiff%"] = np.where(grouped["Swings"] > 0, grouped["Whiffs"] / grouped["Swings"] * 100, np.nan)
    view_cols = [
        "Pitch", "N", "Usage%", "Velo", "PerVelo", "IVB", "HB", "Spin",
        "Ext", "RelHt", "Stuff+", "Loc+", "Zone%", "Strike%", "CSW%", "Whiff%",
    ]
    return grouped[[c for c in view_cols if c in grouped.columns]].round(1).sort_values("N", ascending=False)


def practice_review_page(page_title="Bullpen Review", allowed_session_types=None, live_only=True):
    st.title(page_title)
    if live_only:
        st.caption("Review uses PitchSession = Live rows only, so warmups are ignored while every tracked bullpen pitch is kept.")
    else:
        st.caption("Upload practice TrackMan CSVs, keep them separate from game files, and review coach-facing pitch data.")

    with st.expander("Upload bullpen CSVs", expanded=True):
        upload_cols = st.columns([1.05, 1, 1.4])
        with upload_cols[0]:
            session_type = st.radio("Session Type", ["Bullpen"], horizontal=True)
        with upload_cols[1]:
            session_label = st.text_input("Session Label", placeholder="Optional: May 1 bullpen")
        with upload_cols[2]:
            uploaded = st.file_uploader(
                "Bullpen TrackMan CSV files",
                type=["csv"],
                accept_multiple_files=True,
                help="These files save locally in /practice_data and do not change the game data in /data.",
            )
        if st.button("Save Uploaded Bullpen Data", use_container_width=True):
            if not uploaded:
                st.warning("Choose one or more TrackMan CSVs first.")
            else:
                saved = save_practice_uploads(uploaded, session_type, session_label)
                st.success(f"Saved {len(saved)} file(s) to {PRACTICE_DATA_DIR}.")

    summary = summarize_practice_files()
    if summary.empty:
        st.info("No bullpen CSVs have been uploaded yet.")
        return

    st.subheader("Bullpen Data Library")
    if allowed_session_types:
        summary_view = summary[summary["Type"].isin(allowed_session_types)].copy()
    else:
        summary_view = summary.copy()
    if summary_view.empty:
        st.info(f"No {page_title.lower()} files have been uploaded yet.")
        return
    st.dataframe(summary_view, use_container_width=True, hide_index=True)

    files = get_practice_csv_files()
    type_options = sorted(summary_view["Type"].dropna().unique())
    selected_types = st.multiselect("Session Type Filter", type_options, default=type_options)
    visible_files = [
        path for path in files
        if _practice_session_type_from_name(path) in selected_types
        and (not allowed_session_types or _practice_session_type_from_name(path) in allowed_session_types)
    ]
    selected_files = st.multiselect(
        "Sessions",
        visible_files,
        default=visible_files,
        format_func=lambda path: f"{_practice_session_type_from_name(path)} - {_practice_file_label(path)}",
    )

    if not selected_files:
        st.warning("Select at least one bullpen session.")
        return

    df = prepare_practice_data(selected_files)
    if df.empty:
        st.error("No valid TrackMan pitch-by-pitch data found in the selected bullpen files.")
        return

    if live_only:
        before_live = len(df)
        df = filter_live_practice_pitches(df)
        if df.empty:
            st.error("No live bullpen pitches found in the selected files. Warmups were ignored.")
            return
        if "PitchSession" in df.columns:
            st.caption(f"PitchSession filter kept {len(df):,} Live pitch rows from {before_live:,} tracked rows. Warmup rows were ignored.")
        else:
            st.caption(f"Live-pitch filter kept {len(df):,} of {before_live:,} tracked pitch rows.")

    df = apply_date_range_filter(df, "practice_review")
    if df.empty:
        st.warning("No bullpen pitches found in the selected date range.")
        return

    pitchers = get_pitcher_list(df)
    pitcher = st.selectbox("Pitcher", ["All Pitchers"] + pitchers)
    pdf = df.copy() if pitcher == "All Pitchers" else df[df["Pitcher"] == pitcher].copy()

    if pdf.empty:
        st.warning("No pitches match the selected filters.")
        return

    st.subheader("Session Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pitches", f"{len(pdf):,}")
    m2.metric("Avg Velo", _fmt_pdf_value(_practice_metric_value(pdf, "Velo"), "Velo"))
    m3.metric("PerVelo", _fmt_pdf_value(_practice_metric_value(pdf, "PerceivedVelo"), "PerceivedVelo"))
    m4.metric("Stuff+", _fmt_pdf_value(_practice_metric_value(pdf, "Stuff+"), "Stuff+"))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Loc+", _fmt_pdf_value(_practice_metric_value(pdf, "Loc+"), "Loc+"))
    m6.metric("Strike%", f"{_fmt_pdf_value(_practice_metric_value(pdf, 'is_strike', pct=True), 'Strike%')}%")
    m7.metric("Zone%", f"{_fmt_pdf_value(_practice_metric_value(pdf, 'in_zone', pct=True), 'Zone%')}%")
    m8.metric("CSW%", f"{_fmt_pdf_value(_practice_metric_value(pdf, 'is_csw', pct=True), 'CSW%')}%")

    st.subheader("Pitch Mix And Bullpen Quality")
    arsenal = _practice_arsenal_table(pdf)
    if arsenal.empty:
        st.info("No pitch-type summary available.")
    else:
        st.dataframe(style_scouting_dataframe(arsenal, context="pitching"), use_container_width=True, hide_index=True)

    visual_a, visual_b = st.columns(2)
    with visual_a:
        st.markdown("### Pitch Break")
        fig = build_movement_figure(pdf)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Strike Zone 9-Box")
        zone_pitch_options = ["All"] + sorted(pdf["pitch_abbr"].dropna().astype(str).unique()) if "pitch_abbr" in pdf.columns else ["All"]
        zone_pitch = st.selectbox("Zone Pitch Type", zone_pitch_options, key="practice_zone_pitch")
        zone_df = pdf if zone_pitch == "All" else pdf[pdf["pitch_abbr"].astype(str) == zone_pitch]
        zone_fig = make_savant_zone_heatmap(zone_df, "Zone%", "Bullpen Zone% By Zone", "Zone rate by pitch type")
        if zone_fig:
            st.pyplot(zone_fig)
            plt.close(zone_fig)
        else:
            st.info("No zone data available for this selection.")

    with visual_b:
        st.markdown("### Command Heatmap")
        heat_fig = make_zone_heatmap(pdf, "Zone%", "Bullpen Zone% Heatmap")
        if heat_fig:
            st.pyplot(heat_fig)
            plt.close(heat_fig)
        else:
            st.info("No plate-location data available.")

        st.markdown("### Release Consistency")
        rel_cols = [c for c in ["RelH", "RelS", "Ext"] if c in pdf.columns]
        if rel_cols and "pitch_abbr" in pdf.columns:
            rel = pdf.groupby("pitch_abbr")[rel_cols].agg(["mean", "std"]).round(2)
            st.dataframe(rel, use_container_width=True)
        else:
            st.info("Release data unavailable for these files.")

    st.subheader("Pitch-Level Review")
    raw_cols = [
        "PracticeSession", "SessionType", "Pitcher", "PitcherTeam", "Batter", "pitch_abbr",
        "Velo", "PerceivedVelo", "IVB", "HB", "Spin", "Ext", "PlateLocSide",
        "PlateLocHeight", "PitchCall", "PlayResult", "Stuff+", "Loc+",
    ]
    raw_view = _table_columns(pdf, raw_cols).copy()
    st.dataframe(style_scouting_dataframe(raw_view.head(500), context="pitching"), use_container_width=True, hide_index=True)
    if len(raw_view) > 500:
        st.caption("Showing the first 500 matching pitches to keep the page responsive.")


def bullpen_review_page():
    practice_review_page(
        page_title="Bullpen Review",
        allowed_session_types=["Bullpen"],
        live_only=True,
    )


def batting_practice_page():
    st.title("Batting Practice Review")
    st.caption("Upload BP TrackMan CSVs and review hitter contact quality. Warmup/setup rows without hitter contact are ignored.")

    with st.expander("Upload batting practice CSVs", expanded=True):
        upload_cols = st.columns([1, 1.8])
        with upload_cols[0]:
            session_label = st.text_input("Session Label", placeholder="Optional: May 1 BP", key="bp_upload_label")
        with upload_cols[1]:
            uploaded = st.file_uploader(
                "Batting Practice TrackMan CSV files",
                type=["csv"],
                accept_multiple_files=True,
                key="bp_upload_files",
            )
        if st.button("Save Batting Practice Data", use_container_width=True):
            if not uploaded:
                st.warning("Choose one or more batting practice TrackMan CSVs first.")
            else:
                saved = save_practice_uploads(uploaded, "Batting Practice", session_label)
                st.success(f"Saved {len(saved)} batting practice file(s) to {PRACTICE_DATA_DIR}.")

    summary = summarize_practice_files()
    if summary.empty or "Batting Practice" not in set(summary.get("Type", [])):
        st.info("Upload batting practice TrackMan CSVs above to build the review.")
        return

    bp_files = [
        path for path in get_practice_csv_files()
        if _practice_session_type_from_name(path) == "Batting Practice"
    ]
    selected_files = st.multiselect(
        "Batting Practice Sessions",
        bp_files,
        default=bp_files,
        format_func=lambda path: _practice_file_label(path),
    )
    if not selected_files:
        st.warning("Select at least one batting practice session.")
        return

    df = prepare_practice_data(selected_files)
    tracked_rows = len(df)
    df = filter_batting_practice_rows(df)
    if df.empty:
        st.error("No batting-practice contact rows found. Make sure the CSV has Batter plus EV/LA/direction or InPlay rows.")
        return
    if "PitchCall" in df.columns:
        pitch_call = df["PitchCall"].astype(str).str.strip()
        contact_cols = [c for c in ["EV", "ExitSpeed", "LA", "Angle", "Distance", "Direction", "Bearing"] if c in df.columns]
        if contact_cols:
            contact_mask = pd.Series(False, index=df.index)
            for col in contact_cols:
                contact_mask = contact_mask | pd.to_numeric(df[col], errors="coerce").notna()
            blank_call = pitch_call.eq("") | pitch_call.str.lower().isin(["nan", "none", "null", "undefined"])
            df.loc[contact_mask & blank_call, "PitchCall"] = "InPlay"
    if "PitchSession" in df.columns:
        st.caption(f"BP PitchSession filter kept {len(df):,} Live/contact rows from {tracked_rows:,} tracked rows. Warmup rows were ignored.")
    else:
        st.caption(f"BP contact filter kept {len(df):,} hitter-contact rows from {tracked_rows:,} tracked rows.")

    df = apply_date_range_filter(df, "batting_practice")
    if df.empty:
        st.warning("No batting practice contact found in the selected date range.")
        return

    df = normalize_hitter_columns(df)
    df = add_contact_quality(df)

    st.subheader("Contact Quality Leaderboard")
    min_bip = st.slider("Minimum BIP", min_value=1, max_value=100, value=5, step=1)
    board = _practice_hitter_contact_leaderboard(df, "Batter")
    basic_board = _practice_hitter_basic_stats(df)
    if not board.empty and not basic_board.empty:
        board = board.merge(basic_board, on="Batter", how="left")
    if not board.empty:
        board = board[board["BIP"] >= min_bip].sort_values(["AvgEV", "HardHit%"], ascending=False)
    cols = [
        "Batter", "Pitches", "BIP", "PA", "AB", "H", "K", "BB", "K%", "BB%",
        "BA", "OBP", "SLG", "OPS", "AvgEV", "MaxEV", "HardHit%", "Barrel%",
        "SweetSpot%", "AvgLA", "AvgDist", "MaxDist", "Most Seen",
    ]
    if board.empty:
        st.info("No hitters meet the selected BIP threshold.")
    else:
        st.dataframe(style_scouting_dataframe(_table_columns(board, cols), context="hitting"), use_container_width=True, hide_index=True)

    hitters = sorted(df["Batter"].dropna().astype(str).unique()) if "Batter" in df.columns else []
    if not hitters:
        return
    hitter = st.selectbox("Hitter", hitters, key="bp_hitter")
    hdf = df[df["Batter"].astype(str) == hitter].copy()

    c1, c2, c3, c4 = st.columns(4)
    bip = get_true_bip_with_ev(hdf) if {"EV", "PitchCall"}.issubset(hdf.columns) else hdf.dropna(subset=["EV"]) if "EV" in hdf.columns else pd.DataFrame()
    hitter_basic_df = _practice_hitter_basic_stats(hdf)
    hitter_basic = hitter_basic_df.iloc[0].to_dict() if not hitter_basic_df.empty else {}
    c1.metric("Tracked Contact", f"{len(bip):,}")
    c2.metric("Avg EV", _fmt_pdf_value(pd.to_numeric(bip.get("EV", pd.Series(dtype=float)), errors="coerce").mean(), "AvgEV"))
    c3.metric("Max EV", _fmt_pdf_value(pd.to_numeric(bip.get("EV", pd.Series(dtype=float)), errors="coerce").max(), "MaxEV"))
    c4.metric("HardHit%", f"{_fmt_pdf_value((pd.to_numeric(bip.get('EV', pd.Series(dtype=float)), errors='coerce') >= 95).mean() * 100, 'HardHit%')}%")

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("BA", _fmt_pdf_value(hitter_basic.get("BA"), "BA"))
    h2.metric("OBP", _fmt_pdf_value(hitter_basic.get("OBP"), "OBP"))
    h3.metric("SLG", _fmt_pdf_value(hitter_basic.get("SLG"), "SLG"))
    h4.metric("OPS", _fmt_pdf_value(hitter_basic.get("OPS"), "OPS"))

    h5, h6, h7, h8 = st.columns(4)
    h5.metric("K", _fmt_pdf_value(hitter_basic.get("K"), "K"))
    h6.metric("BB", _fmt_pdf_value(hitter_basic.get("BB"), "BB"))
    h7.metric("K%", f"{_fmt_pdf_value(hitter_basic.get('K%'), 'K%')}%")
    h8.metric("BB%", f"{_fmt_pdf_value(hitter_basic.get('BB%'), 'BB%')}%")

    st.subheader("Spray Chart")
    st.pyplot(build_hitter_spray_chart(hdf, hitter))

    spray = hitter_spray_profile(hdf)
    if spray.empty:
        st.info("Not enough directional data for spray profile.")
    else:
        st.dataframe(style_scouting_dataframe(spray, context="hitting"), use_container_width=True, hide_index=True)

    st.subheader("Pitch-Level Contact")
    raw_cols = [
        "PracticeSession", "Batter", "Pitcher", "pitch_abbr", "EV", "LA", "Distance",
        "Direction", "Bearing", "PitchCall", "PlayResult", "Velo", "PlateLocSide", "PlateLocHeight",
    ]
    st.dataframe(style_scouting_dataframe(_table_columns(hdf, raw_cols).head(500), context="hitting"), use_container_width=True, hide_index=True)


def intersquad_leaderboard_page():
    st.title("Intersquad Live Review")
    st.caption("Use uploaded intersquad CSVs to review every PitchSession = Live row. Warmups are ignored; outcome stats appear when the file includes official result columns.")

    with st.expander("Upload intersquad CSVs", expanded=True):
        upload_cols = st.columns([1, 1.8])
        with upload_cols[0]:
            session_label = st.text_input("Session Label", placeholder="Optional: Friday scrimmage", key="intersquad_upload_label")
        with upload_cols[1]:
            uploaded = st.file_uploader(
                "Intersquad TrackMan CSV files",
                type=["csv"],
                accept_multiple_files=True,
                key="intersquad_upload_files",
            )
        if st.button("Save Intersquad Data", use_container_width=True):
            if not uploaded:
                st.warning("Choose one or more intersquad TrackMan CSVs first.")
            else:
                saved = save_practice_uploads(uploaded, "Intersquad", session_label)
                st.success(f"Saved {len(saved)} intersquad file(s) to {PRACTICE_DATA_DIR}.")

    summary = summarize_practice_files()
    if summary.empty or "Intersquad" not in set(summary.get("Type", [])):
        st.info("Upload intersquad TrackMan CSVs above to build the leaderboard.")
        return

    intersquad_files = [
        path for path in get_practice_csv_files()
        if _practice_session_type_from_name(path) == "Intersquad"
    ]
    selected_files = st.multiselect(
        "Intersquad Sessions",
        intersquad_files,
        default=intersquad_files,
        format_func=lambda path: _practice_file_label(path),
    )
    if not selected_files:
        st.warning("Select at least one intersquad session.")
        return

    df = prepare_practice_data(selected_files)
    tracked_rows = len(df)
    df = filter_live_practice_pitches(df)
    if df.empty:
        st.error("No live intersquad pitches found. Make sure PitchSession has Live rows.")
        return
    if "PitchSession" in df.columns:
        st.caption(f"Intersquad PitchSession filter kept {len(df):,} Live pitch rows from {tracked_rows:,} tracked rows. Warmup rows were ignored.")
    else:
        st.caption(f"Live intersquad filter kept {len(df):,} live pitch rows from {tracked_rows:,} tracked rows.")

    official_hitter_outcomes = (
        ("KorBB" in df.columns and df["KorBB"].isin(["Walk", "Strikeout"]).any())
        or ("PitchCall" in df.columns and df["PitchCall"].isin(["HitByPitch"]).any())
        or ("PlayResult" in df.columns and df["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice", "Error", "Sacrifice"]).any())
    )

    if "PitchCall" in df.columns:
        pitch_call = df["PitchCall"].astype(str).str.strip()
        contact_cols = [c for c in ["EV", "ExitSpeed", "LA", "Angle", "Distance", "Direction", "Bearing"] if c in df.columns]
        if contact_cols:
            contact_mask = pd.Series(False, index=df.index)
            for col in contact_cols:
                contact_mask = contact_mask | pd.to_numeric(df[col], errors="coerce").notna()
            blank_call = pitch_call.eq("") | pitch_call.str.lower().isin(["nan", "none", "null", "undefined"])
            df.loc[contact_mask & blank_call, "PitchCall"] = "InPlay"

    df = apply_date_range_filter(df, "intersquad_leaderboard")
    if df.empty:
        st.warning("No intersquad data found in the selected date range.")
        return

    df = normalize_hitter_columns(df)
    df = add_contact_quality(df)

    min_bip = st.slider("Minimum BIP", min_value=1, max_value=25, value=1, step=1)
    if official_hitter_outcomes:
        hitter_board = summarize_contact_quality(df, "Batter")
        if not hitter_board.empty:
            hitter_board = hitter_board[hitter_board["PA"] >= min_bip].sort_values(["OPS", "AvgEV"], ascending=False)
        hitter_cols = ["Batter", "PA", "AB", "H", "BA", "OBP", "SLG", "OPS", "wOBA", "Bat+", "BB%", "K%", "AvgEV", "HardHit%", "Barrel%", "Whiff%", "Chase%"]
        threshold_label = "PA"
    else:
        hitter_board = _practice_hitter_contact_leaderboard(df, "Batter")
        basic_board = _practice_hitter_basic_stats(df)
        if not hitter_board.empty and not basic_board.empty:
            hitter_board = hitter_board.merge(basic_board, on="Batter", how="left")
        if not hitter_board.empty:
            hitter_board = hitter_board[hitter_board["BIP"] >= min_bip].sort_values(["AvgEV", "HardHit%"], ascending=False)
        hitter_cols = [
            "Batter", "Pitches", "BIP", "PA", "AB", "H", "K", "BB", "K%", "BB%",
            "BA", "OBP", "SLG", "OPS", "AvgEV", "MaxEV", "HardHit%", "Barrel%",
            "SweetSpot%", "AvgLA", "AvgDist", "MaxDist", "Most Seen",
        ]
        threshold_label = "BIP"

    st.subheader("Hitter Leaderboard")
    if hitter_board.empty:
        st.info(f"No hitters meet the selected {threshold_label} threshold.")
    else:
        st.dataframe(style_scouting_dataframe(_table_columns(hitter_board, hitter_cols), context="hitting"), use_container_width=True, hide_index=True)

    st.subheader("Hitter Intersquad Data Card")
    hitters = sorted(df["Batter"].dropna().astype(str).unique()) if "Batter" in df.columns else []
    if hitters:
        player = st.selectbox("Select Hitter", hitters, key="intersquad_player_card")
        pdf = df[df["Batter"].astype(str) == player].copy()
        contact_board = _practice_hitter_contact_leaderboard(pdf, "Batter")
        card = contact_board.iloc[0].to_dict() if not contact_board.empty else {}
        basic_df = _practice_hitter_basic_stats(pdf)
        basic_card = basic_df.iloc[0].to_dict() if not basic_df.empty else {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pitches Seen", _fmt_pdf_value(card.get("Pitches"), "Pitches"))
        c2.metric("BIP", _fmt_pdf_value(card.get("BIP"), "BIP"))
        c3.metric("Avg EV", _fmt_pdf_value(card.get("AvgEV"), "AvgEV"))
        c4.metric("Max EV", _fmt_pdf_value(card.get("MaxEV"), "MaxEV"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("HardHit%", f"{_fmt_pdf_value(card.get('HardHit%'), 'HardHit%')}%")
        c6.metric("Barrel%", f"{_fmt_pdf_value(card.get('Barrel%'), 'Barrel%')}%")
        c7.metric("SweetSpot%", f"{_fmt_pdf_value(card.get('SweetSpot%'), 'SweetSpot%')}%")
        c8.metric("Most Seen", str(card.get("Most Seen", "")))

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("BA", _fmt_pdf_value(basic_card.get("BA"), "BA"))
        s2.metric("OBP", _fmt_pdf_value(basic_card.get("OBP"), "OBP"))
        s3.metric("SLG", _fmt_pdf_value(basic_card.get("SLG"), "SLG"))
        s4.metric("OPS", _fmt_pdf_value(basic_card.get("OPS"), "OPS"))

        s5, s6, s7, s8 = st.columns(4)
        s5.metric("K", _fmt_pdf_value(basic_card.get("K"), "K"))
        s6.metric("BB", _fmt_pdf_value(basic_card.get("BB"), "BB"))
        s7.metric("K%", f"{_fmt_pdf_value(basic_card.get('K%'), 'K%')}%")
        s8.metric("BB%", f"{_fmt_pdf_value(basic_card.get('BB%'), 'BB%')}%")

        card_cols = st.columns([1.15, 1])
        with card_cols[0]:
            st.markdown("### Spray / Contact")
            st.pyplot(build_hitter_spray_chart(pdf, player))
        with card_cols[1]:
            st.markdown("### Pitch-Type Results")
            pt = _practice_hitter_contact_leaderboard(pdf, "pitch_abbr") if "pitch_abbr" in pdf.columns else pd.DataFrame()
            if pt.empty:
                st.info("No pitch-type contact detail for this hitter.")
            else:
                st.dataframe(
                    style_scouting_dataframe(
                        _table_columns(pt.rename(columns={"pitch_abbr": "Pitch"}), ["Pitch", "Pitches", "BIP", "AvgEV", "MaxEV", "HardHit%", "Barrel%", "AvgLA", "AvgDist"]),
                        context="hitting",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    st.subheader("Pitcher Leaderboard")
    min_pitches = st.slider("Minimum Pitches", min_value=1, max_value=100, value=5, step=1)
    pitcher_board = _practice_pitcher_tracking_leaderboard(df, min_pitches=min_pitches)
    pitcher_basic = _practice_pitcher_basic_stats(df)
    if not pitcher_board.empty and not pitcher_basic.empty:
        pitcher_board = pitcher_board.merge(pitcher_basic, on="Pitcher", how="left")
    pitcher_cols = [
        "Rank", "Pitcher", "Pitches", "Batters", "BF", "IP", "ERA", "Primary Pitch",
        "Velo", "MaxVelo", "Zone%", "K", "BB", "K%", "BB%", "BA", "OBP", "SLG", "OPS",
        "IVB", "HB", "Ext", "Stuff+", "Loc+",
    ]
    if pitcher_board.empty:
        st.info("No pitchers meet the selected pitch threshold.")
    else:
        st.dataframe(style_scouting_dataframe(_table_columns(pitcher_board, pitcher_cols), context="pitching"), use_container_width=True, hide_index=True)

    st.subheader("Pitcher Intersquad Data Card")
    pitchers = sorted(df["Pitcher"].dropna().astype(str).unique()) if "Pitcher" in df.columns else []
    if pitchers:
        pitcher = st.selectbox("Select Pitcher", pitchers, key="intersquad_pitcher_card")
        ppdf = df[df["Pitcher"].astype(str) == pitcher].copy()
        pitcher_card = _practice_pitcher_tracking_leaderboard(ppdf, min_pitches=1)
        pcard = pitcher_card.iloc[0].to_dict() if not pitcher_card.empty else {}
        basic_card_df = _practice_pitcher_basic_stats(ppdf)
        basic = basic_card_df.iloc[0].to_dict() if not basic_card_df.empty else {}

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Pitches", _fmt_pdf_value(pcard.get("Pitches"), "Pitches"))
        p2.metric("Batters", _fmt_pdf_value(pcard.get("Batters"), "Batters"))
        p3.metric("Avg Velo", _fmt_pdf_value(pcard.get("Velo"), "Velo"))
        p4.metric("Max Velo", _fmt_pdf_value(pcard.get("MaxVelo"), "Velo"))

        p5, p6, p7, p8 = st.columns(4)
        p5.metric("Zone%", f"{_fmt_pdf_value(pcard.get('Zone%'), 'Zone%')}%")
        p6.metric("IVB", _fmt_pdf_value(pcard.get("IVB"), "IVB"))
        p7.metric("HB", _fmt_pdf_value(pcard.get("HB"), "HB"))
        p8.metric("Ext", _fmt_pdf_value(pcard.get("Ext"), "Ext"))

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("ERA", _fmt_pdf_value(basic.get("ERA"), "ERA"))
        b2.metric("K", _fmt_pdf_value(basic.get("K"), "K"))
        b3.metric("BB", _fmt_pdf_value(basic.get("BB"), "BB"))
        b4.metric("K% / BB%", f"{_fmt_pdf_value(basic.get('K%'), 'K%')}% / {_fmt_pdf_value(basic.get('BB%'), 'BB%')}%")

        b5, b6, b7, b8 = st.columns(4)
        b5.metric("BAA", _fmt_pdf_value(basic.get("BA"), "BA"))
        b6.metric("OBP", _fmt_pdf_value(basic.get("OBP"), "OBP"))
        b7.metric("SLG", _fmt_pdf_value(basic.get("SLG"), "SLG"))
        b8.metric("OPS", _fmt_pdf_value(basic.get("OPS"), "OPS"))

        pc_a, pc_b = st.columns([1.15, 1])
        with pc_a:
            st.markdown("### Pitch Break")
            fig = build_movement_figure(ppdf)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Zone% By Pitch Type")
            pitch_options = ["All"] + sorted(ppdf["pitch_abbr"].dropna().astype(str).unique()) if "pitch_abbr" in ppdf.columns else ["All"]
            selected_pitch = st.selectbox("Pitch Type", pitch_options, key=f"intersquad_pitcher_zone_{pitcher}")
            zone_df = ppdf if selected_pitch == "All" else ppdf[ppdf["pitch_abbr"].astype(str) == selected_pitch]
            zone_fig = make_savant_zone_heatmap(zone_df, "Zone%", "Intersquad Zone%", "Live pitches only")
            if zone_fig:
                st.pyplot(zone_fig)
                plt.close(zone_fig)
            else:
                st.info("No zone data available for this pitcher.")

        with pc_b:
            st.markdown("### Arsenal")
            p_arsenal = _practice_arsenal_table(ppdf)
            if p_arsenal.empty:
                st.info("No arsenal detail available for this pitcher.")
            else:
                st.dataframe(
                    style_scouting_dataframe(
                        _table_columns(p_arsenal, ["Pitch", "N", "Usage%", "Velo", "IVB", "HB", "Ext", "Stuff+", "Loc+", "Zone%"]),
                        context="pitching",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Contact Allowed")
            allowed = _practice_hitter_contact_leaderboard(ppdf, "Pitcher")
            if allowed.empty:
                st.info("No contact data allowed for this pitcher.")
            else:
                st.dataframe(
                    style_scouting_dataframe(
                        _table_columns(allowed, ["Pitcher", "Pitches", "BIP", "AvgEV", "HardHit%", "Barrel%", "AvgLA", "AvgDist"]),
                        context="pitching",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    st.subheader("Pitch-Type Leaderboard")
    pitch_mix = _practice_arsenal_table(df)
    if pitch_mix.empty:
        st.info("No pitch-type data available.")
    else:
        st.dataframe(style_scouting_dataframe(pitch_mix, context="pitching"), use_container_width=True, hide_index=True)


def glossary_page():
    st.title("Advanced Stats Glossary")

    st.markdown("### Outing Grading System")
    grade_terms = pd.DataFrame([
        {"Grade": "Outing Grade",
         "What it measures": "Overall pitcher outing quality — run-independent, contact-first.",
         "Components & weights": "Contact Quality 40% (Avg EV 16%, HH% 16%, Barrel% 8%) · Swing & Miss 25% (CSW% 13%, Whiff% 12%) · Command 20% (BB% 10%, FPS% 10%) · Pitch Models 15% (Stuff+ 8%, Loc+ 7%)",
         "Scale": "A+ elite → F poor. Missing metrics are dropped and remaining weights rescaled. Design is ERA-independent — a pitcher who allows hard-hit balls that find gaps still grades poorly; soft contact that becomes hits grades well."},
        {"Grade": "Pure Stuff Grade",
         "What it measures": "Raw pitch quality in isolation — no location or results context.",
         "Components & weights": "Stuff+ only (100-centered, 100 = D1 average).",
         "Scale": "A+ ≥118 · A ≥113 · A- ≥108 · B+ ≥105 · B ≥102 · B- ≥99 · C+ ≥96 · C ≥92 · C- ≥87 · D ≥80 · F <80"},
        {"Grade": "Pitch Efficiency Grade",
         "What it measures": "How many pitches it takes to record outs — lower is better.",
         "Components & weights": "P/IP = total pitches ÷ true innings (outs ÷ 3). Baseball IP notation (e.g. 1.2) is never used as a decimal; always converted via outs.",
         "Scale": "A ≤14.5 · A- ≤16.5 · B+ ≤18.5 · B ≤20.5 · B- ≤22.5 · C+ ≤24.5 · C ≤26.5 · C- ≤28.5 · D ≤30.0 · F >30.0"},
    ])
    st.dataframe(grade_terms, hide_index=True, use_container_width=True)

    st.markdown("### Outing Grade — D1 Normalization Anchors")
    st.caption("Each metric is converted to a 100-centered score before weighting. These are the D1 averages used as the 'par' value.")
    anchor_terms = pd.DataFrame([
        {"Metric": "Avg EV allowed",  "D1 Average": "86.0 mph",  "Direction": "Lower = better (inverted)", "Scale": "3.0 pts per mph — a pitcher allowing 83 mph avg EV scores ~109"},
        {"Metric": "HH% allowed",     "D1 Average": "34%",       "Direction": "Lower = better (inverted)", "Scale": "2.0 pts per % — allowing 28% HH scores ~112"},
        {"Metric": "Barrel% allowed", "D1 Average": "15%",       "Direction": "Lower = better (inverted)", "Scale": "3.0 pts per % (using 92 mph / 16-36° threshold)"},
        {"Metric": "CSW%",            "D1 Average": "27%",       "Direction": "Higher = better",           "Scale": "2.5 pts per % — 31% CSW scores ~110"},
        {"Metric": "Whiff%",          "D1 Average": "22%",       "Direction": "Higher = better",           "Scale": "2.5 pts per % — 28% whiff scores ~115"},
        {"Metric": "BB%",             "D1 Average": "12%",       "Direction": "Lower = better (inverted)", "Scale": "3.0 pts per % — 8% BB scores ~112"},
        {"Metric": "FPS%",            "D1 Average": "58%",       "Direction": "Higher = better",           "Scale": "2.0 pts per % — 65% FPS scores ~114"},
        {"Metric": "Stuff+",          "D1 Average": "100",       "Direction": "Higher = better",           "Scale": "Already 100-centered by model"},
        {"Metric": "Loc+",            "D1 Average": "100",       "Direction": "Higher = better",           "Scale": "Already 100-centered by model"},
    ])
    st.dataframe(anchor_terms, hide_index=True, use_container_width=True)

    st.markdown("### Pitching Metrics")
    pitching_terms = pd.DataFrame([
        {"Stat": "Stuff+",    "What it means": "Pitch quality based on raw stuff — velocity, movement, spin, and release point.", "App logic": "LightGBM model trained on college TrackMan data. 100 = average D1 pitcher. Above 100 is better."},
        {"Stat": "Loc+",      "What it means": "Command quality — how well the pitcher locates given the count and situation.", "App logic": "LightGBM model using plate location, count, and zone context. 100 = average. Above 100 is better."},
        {"Stat": "FPS%",      "What it means": "First-pitch strike percentage — how often the pitcher gets ahead 0-1.", "App logic": "Pitches thrown with Balls=0, Strikes=0 that result in a strike (called, swinging, foul, or in-play) ÷ total first pitches. D1 avg ≈ 58%. Research shows 69% of all strikeouts and 70% of all walks start with the first pitch."},
        {"Stat": "P/IP",      "What it means": "Pitches per true inning — efficiency of getting outs.", "App logic": "Total pitches ÷ (total outs ÷ 3). Always computed from raw outs, never from baseball IP notation. D1 average varies; A grade ≤14.5, B grade ≤20.5."},
        {"Stat": "Avg EV allowed", "What it means": "Average exit velocity on true balls in play — contact quality allowed.", "App logic": "Only InPlay pitch calls with valid EV. D1 average ≈ 86 mph (historic high 2025). Lower is better for pitchers."},
        {"Stat": "HH% allowed",    "What it means": "Hard-hit rate allowed — share of BIP at 95 mph or harder.", "App logic": "Hard-hit BIP ÷ total true BIP. D1 average ≈ 34%. Lower is better."},
        {"Stat": "Barrel% allowed","What it means": "Barrel rate allowed — optimal contact given up.", "App logic": f"EV ≥ {BARREL_EV_MIN} mph and LA {BARREL_LA_MIN}–{BARREL_LA_MAX}°. D1 average ≈ 15% using this threshold (lower than MLB barrel def of 98 mph). Lower is better."},
        {"Stat": "PerVelo",   "What it means": "Perceived fastball velocity, accounting for extension and pitch shape.", "App logic": f"Fastballs only: Velo × ((60.5 − {PERCEIVED_VELO_EXT_BASELINE:.1f}) ÷ (60.5 − Extension)) plus a small IVB/spin shape adjustment capped at ±{PERCEIVED_VELO_SHAPE_CAP:.1f} mph."},
        {"Stat": "Usage%",    "What it means": "Share of pitches thrown as that pitch type.", "App logic": "Pitch type count divided by total pitches in the sample."},
        {"Stat": "Zone%",     "What it means": "How often pitches land in the strike zone.", "App logic": "PlateLocSide −0.83 to 0.83 ft and PlateLocHeight 1.5 to 3.5 ft."},
        {"Stat": "Strike%",   "What it means": "Percent of pitches resulting in a strike of any kind.", "App logic": "Called strikes, swinging strikes, fouls, and balls put in play all count as strikes."},
        {"Stat": "CSW%",      "What it means": "Called strikes plus whiffs — the strongest single-pitch predictor of ERA (r²=.568 vs SIERA).", "App logic": "(Called strikes + swinging strikes) ÷ total pitches. D1 avg ≈ 27%. Elite = 30%+."},
        {"Stat": "Whiff%",    "What it means": "Misses per swing.", "App logic": "Swinging strikes ÷ total swings. D1 avg ≈ 22%."},
        {"Stat": "Chase%",    "What it means": "Opponent swings on pitches outside the strike zone.", "App logic": "Out-of-zone swings ÷ total swings."},
        {"Stat": "K%",        "What it means": "Strikeout rate per PA.", "App logic": "Strikeouts ÷ PA-ending events."},
        {"Stat": "BB%",       "What it means": "Walk rate per PA — single best proxy for command.", "App logic": "Walks ÷ PA-ending events. D1 avg ≈ 12% (higher than MLB 8.2%)."},
        {"Stat": "GB%",       "What it means": "Ground ball percentage of balls in play.", "App logic": "Batted balls tagged as GroundBall ÷ all batted balls. More grounders generally means fewer HR allowed."},
        {"Stat": "BABIP",     "What it means": "Batting average on balls in play — measures defense and luck on contact.", "App logic": "(Hits − HR) ÷ (AB − K − HR). Pitchers with low BABIP may be over-performing or getting good defense."},
        {"Stat": "BA / OBP / SLG allowed", "What it means": "Slash line from the opposing hitter's perspective.", "App logic": "Computed from PA-ending rows where the pitcher is Fordham. Lower is better for pitchers."},
        {"Stat": "IVB",       "What it means": "Induced vertical break — how much the pitch rises vs. a spinless ball.", "App logic": "TrackMan InducedVertBreak. Positive = rising action (fastball ride). Negative = downward break (curveball)."},
        {"Stat": "HB",        "What it means": "Horizontal break — arm-side (+) or glove-side (−) movement.", "App logic": "TrackMan HorzBreak."},
        {"Stat": "Spin",      "What it means": "Pitch spin rate in rpm.", "App logic": "TrackMan SpinRate, averaged by pitch type."},
        {"Stat": "Ext / RelExt", "What it means": "Release extension — how far in front of the rubber the pitcher releases.", "App logic": "TrackMan Extension. Higher extension = closer to the plate = more reaction time taken from hitter."},
        {"Stat": "RelHt",     "What it means": "Release height in feet.", "App logic": "TrackMan RelHeight, averaged by pitch type."},
        {"Stat": "Pitch Break Plot", "What it means": "Movement chart by pitch type.", "App logic": "Every pitch plotted by HB and IVB. Large labeled markers show pitch-type centroids."},
    ])
    st.dataframe(pitching_terms, hide_index=True, use_container_width=True)

    st.markdown("### Hitting / Contact Metrics")
    hitting_terms = pd.DataFrame([
        {"Stat": "BA",        "What it means": "Batting average.", "App logic": "Hits ÷ at-bats. Walks, HBP, and sacrifice plays are excluded from AB."},
        {"Stat": "OBP",       "What it means": "On-base percentage.", "App logic": "(H + BB + HBP) ÷ (AB + BB + HBP + SF)."},
        {"Stat": "SLG",       "What it means": "Slugging percentage.", "App logic": "Total bases ÷ at-bats."},
        {"Stat": "OPS",       "What it means": "OBP plus SLG.", "App logic": "OBP + SLG from PA-ending pitch rows."},
        {"Stat": "HR",        "What it means": "Home runs.", "App logic": "PA-ending rows with PlayResult == HomeRun."},
        {"Stat": "xHB",       "What it means": "Extra base hits — doubles, triples, and home runs combined.", "App logic": "2B + 3B + HR. A quick gauge of a hitter's power without needing SLG context."},
        {"Stat": "BABIP",     "What it means": "Batting average on balls in play.", "App logic": "(H − HR) ÷ (AB − K − HR). Hitters who hit hard and in gaps sustain higher BABIP."},
        {"Stat": "wOBA",      "What it means": "Weighted on-base average — values each outcome by its run impact.", "App logic": "BB .69 · HBP .72 · 1B .88 · 2B 1.247 · 3B 1.578 · HR 2.031."},
        {"Stat": "Bat+",      "What it means": "Run creation relative to the 2026 college average.", "App logic": f"Player wOBA ÷ college average wOBA {COLLEGE_AVG_WOBA:.3f}, scaled to 100. 110 = 10% above average."},
        {"Stat": "Avg EV",    "What it means": "Average exit velocity on true in-play contact.", "App logic": "True BIP only (InPlay PitchCall, EV > 45 mph). D1 avg ≈ 86 mph."},
        {"Stat": "Max EV",    "What it means": "Single hardest batted ball in the sample.", "App logic": "Max EV on true BIP. Requires ≥ 5 BIP to display."},
        {"Stat": "EV 90th%",  "What it means": "90th percentile exit velocity — measures ceiling, not just average.", "App logic": "bip_ev.quantile(0.90). Requires ≥ 10 BIP. D1 avg ≈ 100.8 mph."},
        {"Stat": "HH%",       "What it means": "Hard-hit rate — share of BIP at 95 mph or harder.", "App logic": "Hard-hit BIP ÷ total true BIP. D1 avg ≈ 34%."},
        {"Stat": "Barrel%",   "What it means": "Optimal contact — high EV at a productive launch angle.", "App logic": f"EV ≥ {BARREL_EV_MIN} mph and LA {BARREL_LA_MIN}–{BARREL_LA_MAX}°. D1 avg ≈ 15% (our threshold is lower than MLB's 98 mph def)."},
        {"Stat": "SweetSpot%","What it means": "Line-drive and productive fly-ball launch angle window.", "App logic": "Launch angle 8–32°."},
        {"Stat": "Swing%",    "What it means": "Overall swing rate — swings on all pitches.", "App logic": "Swings ÷ total pitches. D1 avg ≈ 42.7%. No clear good/bad direction — neutral stat."},
        {"Stat": "Z-Swing%",  "What it means": "In-zone swing rate — aggression on hittable pitches.", "App logic": "In-zone swings ÷ total in-zone pitches. D1 avg ≈ 66.7%. Higher = more aggressive on strikes."},
        {"Stat": "O-Swing%",  "What it means": "Out-of-zone swing rate — chase rate (plate discipline).", "App logic": "Out-zone swings ÷ total out-zone pitches. D1 avg ≈ 24.4%. Lower = better discipline for hitters. Higher = more chases induced for pitchers."},
        {"Stat": "Contact%",  "What it means": "Contact rate on all swings — how often the bat hits the ball.", "App logic": "(Fouls + in-play) ÷ total swings. D1 avg ≈ 76.7% (MLB ≈ 82%)."},
        {"Stat": "Z-Contact%","What it means": "Contact rate on in-zone swings.", "App logic": "In-zone contact ÷ in-zone swings. D1 avg ≈ 85.6% (MLB ≈ 87%)."},
        {"Stat": "O-Contact%","What it means": "Contact rate on out-of-zone swings (chase contact).", "App logic": "Out-zone contact ÷ out-zone swings. D1 avg ≈ 59.5% (MLB ≈ 66%). For hitters: higher = tougher out when chasing. For pitchers: lower = batters whiff when chasing."},
        {"Stat": "Whiff%",    "What it means": "Miss rate per swing.", "App logic": "Swinging strikes ÷ total swings. D1 avg ≈ 22.5%. Preferred over SwStr% for per-swing measurement."},
        {"Stat": "Chase%",    "What it means": "Chases per swing (alternate chase measure).", "App logic": "Out-of-zone swings ÷ total swings. Different denominator from O-Swing%. Lower is better discipline."},
        {"Stat": "K%",        "What it means": "Strikeout rate per PA.", "App logic": "Strikeouts ÷ PA-ending events."},
        {"Stat": "BB%",       "What it means": "Walk rate per PA.", "App logic": "Walks ÷ PA-ending events."},
        {"Stat": "Bat+",      "What it means": "Run creation vs. D1 average. 100 = average, 110 = 10% above average.", "App logic": "(Player wOBA ÷ league wOBA) × 100 using D1 collegiate weights. Formerly wRC+."},
        {"Stat": "Spray",     "What it means": "Pull / middle / opposite field batted-ball tendency.", "App logic": "TrackMan Direction/Bearing converted into hitter-relative spray buckets using handedness."},
        {"Stat": "Shift Read","What it means": "Defensive alignment cue against a hitter.", "App logic": "Combines spray bucket, GB rate, HH rate, oppo air contact, and bunt frequency."},
    ])
    st.dataframe(hitting_terms, hide_index=True, use_container_width=True)

    st.markdown("### Color Scale Logic")
    color_terms = pd.DataFrame([
        {"Area": "Color direction",     "App logic": "Blue = weaker/worse. Red = stronger/better. Context-aware — the same stat colors differently for pitchers vs. hitters."},
        {"Area": "Pitching context",    "App logic": "Red is good for: Stuff+, Loc+, K%, Zone%, CSW%, Whiff%, Strike%, GB%, Ext, Swing% (induced), O-Swing% (induced). Red is bad for: BB%, BA/OBP/SLG allowed, Avg EV, HH%, Barrel%, Z-Contact%, O-Contact% allowed."},
        {"Area": "Hitting context",     "App logic": "Red is good for: BA, OBP, SLG, HR, xHB, wOBA, Bat+, Avg EV, Max EV, EV90, HH%, Barrel%, BB%, Z-Swing%, Z-Contact%, O-Contact%. Red is bad for: K%, Whiff%, Chase%, O-Swing%."},
        {"Area": "Swing% (neutral)",    "App logic": "Swing% has no color — there is no universally good/bad swing rate. It is displayed as an informational stat only."},
        {"Area": "Benchmarking",        "App logic": "Color scales use the 10th–90th percentile range of the selected table's own data so grades are relative to the sample shown, not a fixed league average."},
        {"Area": "Neutral / uncolored", "App logic": "Count columns (N, Pitches, PA, AB, H, HR, xHB, BIP) and label columns (Pitcher, Pitch, Side) are not colored."},
    ])
    st.dataframe(color_terms, hide_index=True, use_container_width=True)

    st.markdown("### Pitch Tag Cleaning")
    tag_terms = pd.DataFrame([
        {"Area": "Undefined / Other removed", "App logic": "TaggedPitchType values of Undefined, Other, and TwoSeam map to UN, OT, or TW — all are dropped before any graphic or table is built."},
        {"Area": "Pitcher-specific overrides", "App logic": "Some Fordham pitchers have pitch-type remappings applied automatically (e.g. Stewart FB→SI, Hanawalt SL splits by IVB threshold, Murray CU→SL). These run in basic_clean() before any metric is computed."},
        {"Area": "Conditional overrides",      "App logic": "Overrides can depend on a pitch metric. For example, Hanawalt sliders with IVB ≥ −6 become cutters; IVB < −6 become curveballs. Berg sliders with IVB < −5 become curveballs."},
        {"Area": "Scouting deduplication",     "App logic": "The same game CSV can be re-imported on multiple daily SFTP runs. The app keeps only the first copy of each GameID so no game is double-counted."},
    ])
    st.dataframe(tag_terms, hide_index=True, use_container_width=True)

    st.markdown("### Zone And Positioning Logic")
    zone_terms = pd.DataFrame([
        {"Area": "Strike Zone Heatmaps",  "App logic": "Plate width −0.83 to 0.83 ft, zone height 1.5 to 3.5 ft."},
        {"Area": "9-Box Breakdown",       "App logic": "Strike zone divided into equal 3×3 boxes, Baseball Savant style."},
        {"Area": "Pitch Type Filter",     "App logic": "Filters the 9-box sample to all pitches or one pitch type before computing zone metrics."},
        {"Area": "Spray Chart",           "App logic": "Batted-ball depth from TrackMan Distance, Bearing for angle. Field scaled to LF 338, CF 395, RF 320."},
        {"Area": "Ground-Ball Lines",     "App logic": "Ground balls keep their true landing point plus an extended direction guide toward the 5-6, middle, or 3-4 lane."},
        {"Area": "Infield Alignment",     "App logic": "Recommends standard, pull-side shift, middle pinch, guard lines, corners-in, or 3B bunt alert based on spray/GB/air/bunt rates."},
        {"Area": "Outfield Alignment",    "App logic": "Shades toward primary spray bucket and moves deeper when air contact + HH rate are elevated."},
    ])
    st.dataframe(zone_terms, hide_index=True, use_container_width=True)

    st.markdown("### Report Formatting")
    report_terms = pd.DataFrame([
        {"Area": "Postgame / Season graphics",  "App logic": "Pitch movement, LHH/RHH location maps, pitch usage by batter side, and arsenal table (Pitch, N, Usage%, Velo, PerVelo, IVB, HB, Spin, Stuff+, Loc+, CSW%, Whiff%, Strike%, Zone%)."},
        {"Area": "Hitter scouting PDF",         "App logic": "Two pages: (1) header stats + pitch-type table + count tendencies + splits + quick reads. (2) spray chart + zone heatmaps + damage/miss tables."},
        {"Area": "Pitcher scouting PDF",        "App logic": "Cover page with Stuff+, Loc+, BA/OBP/SLG allowed, BABIP, GB%, K%, BB%, plus movement, location, and arsenal pages."},
        {"Area": "Decimal display",             "App logic": "Slash-line stats, wOBA, and BABIP show three decimals (.xxx). Percentages, velocity, movement, Stuff+, and Loc+ show one decimal. Counts (HR, xHB, PA, K, BB) show whole numbers."},
        {"Area": "Pitch mix bar",               "App logic": "Color-coded bar in stat cards and PDF footers showing each pitch type's usage share."},
    ])
    st.dataframe(report_terms, hide_index=True, use_container_width=True)

    st.markdown("### Practice / Intersquad Logic")
    practice_terms = pd.DataFrame([
        {"Area": "Bullpen Review",        "App logic": "Keeps PitchSession = Live rows only. Warmups excluded. All live tracked bullpen pitches included even without contact outcomes."},
        {"Area": "Batting Practice",      "App logic": "Keeps live/contact rows with a hitter plus EV, launch angle, direction, distance, or an in-play pitch call."},
        {"Area": "Intersquad",            "App logic": "Keeps every PitchSession = Live row, then builds hitter and pitcher cards. Slash-line stats stay blank if no PA-ending outcomes are found."},
        {"Area": "Practice persistence",  "App logic": "Uploaded CSVs save locally and also push to GitHub via API (GITHUB_TOKEN secret required). Files survive app restarts on Streamlit Cloud."},
        {"Area": "Fordham practice tags", "App logic": "FOR_RAM1 is normalized to FOR_RAM so practice files match game-data player logic."},
    ])
    st.dataframe(practice_terms, hide_index=True, use_container_width=True)

    st.markdown("### Data Sources And Auto-Update")
    data_terms = pd.DataFrame([
        {"Area": "Game data (data/)",              "App logic": "Fordham game CSVs from TrackMan FTP. Automatically downloaded and pushed to GitHub every 24 hours by the macOS LaunchAgent. Streamlit Cloud redeploys on each push."},
        {"Area": "Scouting data (scouting_2026_trackman/)", "App logic": "All D1 opponent games from the same TrackMan FTP feed. Stored locally only (gitignored). Updated on the same 24-hour cycle as game data."},
        {"Area": "Practice data (practice_data/)", "App logic": "Manually uploaded CSVs (bullpen, BP, intersquad). Persisted to GitHub via the GitHub API on upload so they survive redeploys."},
        {"Area": "Duplicate filtering",            "App logic": "The same game file is often re-imported on multiple daily runs. Only the first import of each GameID is used so pitch counts are not inflated."},
        {"Area": "Pitch tag overrides",            "App logic": "Applied in basic_clean() before any analysis. See Pitch Tag Cleaning section above."},
    ])
    st.dataframe(data_terms, hide_index=True, use_container_width=True)

    st.markdown("### Team Tags And Names")
    team_terms = pd.DataFrame([
        {"Area": "Readable team names",  "App logic": "TrackMan team codes checked against a 300+ entry override dict (FOR_RAM → Fordham Rams, etc.) before falling back to code-splitting logic."},
        {"Area": "Team colors",          "App logic": "Known programs use official primary/accent colors. Unknown codes get stable auto-generated colors from the tag string."},
        {"Area": "NCAA D1 gate",         "App logic": "Scouting Zone only shows teams mapped to D1 baseball conferences. D2, D3, NAIA, JUCO, and unmapped codes are hidden from the team dropdown."},
        {"Area": "Conference index",     "App logic": "D1 teams are further organized by conference (ACC, SEC, Big Ten, Big 12, A-10, MAAC, Patriot, Ivy, CAA, etc.) for filtered browsing."},
        {"Area": "Team colors (reports)","App logic": "Postgame, season, scouting, and PDF headers all use the same team color lookup so branding is consistent across every output."},
    ])
    st.dataframe(team_terms, hide_index=True, use_container_width=True)


# ============================================================
# SEASON GRADE TRENDS
# ============================================================
def season_grade_trends_page(all_pitches_df: pd.DataFrame):
    st.title("Season Grade Trends")
    st.caption("Outing Grade, Pure Stuff Grade, and Pitch Efficiency Grade plotted game-by-game.")

    df = all_pitches_df.copy() if all_pitches_df is not None else pd.DataFrame()
    df = filter_fordham_only(df) if not df.empty else df
    if df.empty:
        st.error("No Fordham pitcher data loaded.")
        return

    pitchers = get_pitcher_list(df)
    pitcher  = st.selectbox("Select pitcher", pitchers, key="sgt_pitcher")
    pdf      = df[df["Pitcher"] == pitcher].copy()

    if "Date" not in pdf.columns:
        st.warning("No date column in data.")
        return

    pdf["Date"] = pd.to_datetime(pdf["Date"], errors="coerce")
    pdf = pdf.dropna(subset=["Date"])

    # Per-game stats
    game_col = "GameID" if "GameID" in pdf.columns else "Date"
    rows = []
    for gid, g in pdf.groupby(game_col):
        date_val = g["Date"].min()
        stats    = _compute_pitcher_pct_stats(g)
        bip_g    = get_true_bip_with_ev(g) if {"EV","PitchCall"}.issubset(g.columns) else pd.DataFrame()
        avg_ev_g = float(bip_g["EV"].mean()) if not bip_g.empty else float("nan")
        hh_g     = float((bip_g["EV"]>=95).mean()*100) if not bip_g.empty else float("nan")
        brl_g    = float(bip_g["barrel"].mean()*100) if not bip_g.empty and "barrel" in bip_g.columns else float("nan")
        csw_g    = float(g["is_csw"].mean()*100) if "is_csw" in g.columns else float("nan")
        sw_g     = g["is_swing"].sum() if "is_swing" in g.columns else 0
        wh_g     = float(g["is_whiff"].sum()/sw_g*100) if sw_g else float("nan")
        bb_g     = stats.get("BB%") or float("nan")
        fps_g    = stats.get("FPS%") or float("nan")
        stuff_v  = stats.get("Stuff+") or float("nan")
        _, _, _, outing_score = outing_grade(stuff_v, stats.get("Loc+") or float("nan"),
                                             fps_g, csw_g, wh_g, bb_g, avg_ev_g, hh_g, brl_g)
        p_per_ip, _ = compute_pitch_efficiency(g)
        # Map efficiency P/IP to a 100-centered score for consistent scale
        eff_score = max(0, min(200, 100 + (18.5 - p_per_ip) * 3.5)) if not pd.isna(p_per_ip) else float("nan")
        rows.append({
            "Date":       date_val,
            "Pitches":    len(g),
            "Outing":     round(outing_score, 1) if outing_score else float("nan"),
            "Stuff":      round(stuff_v, 1) if not pd.isna(stuff_v) else float("nan"),
            "Efficiency": round(eff_score, 1) if not pd.isna(eff_score) else float("nan"),
            "P/IP":       round(p_per_ip, 1) if not pd.isna(p_per_ip) else float("nan"),
        })

    if not rows:
        st.info("Not enough game data for this pitcher.")
        return

    gdf = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    gdf["Game"] = gdf["Date"].dt.strftime("%-m/%-d")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("#100D0C")
    BG, TXT, GRID = "#181412", "#FFF7E8", "#C7A45D"

    configs = [
        ("Outing",     "#C7A45D", "Outing Grade Score"),
        ("Stuff",      "#1f77b4", "Stuff+ (Pure Pitch Quality)"),
        ("Efficiency", "#2ca02c", "Pitch Efficiency Score"),
    ]
    for ax, (col, color, title) in zip(axes, configs):
        ax.set_facecolor(BG)
        vals = gdf[col].dropna()
        if not vals.empty:
            ax.plot(gdf.index[gdf[col].notna()], vals,
                    color=color, linewidth=2.2, marker="o", markersize=6)
            ax.axhline(100, color="#6B7A93", linewidth=1, linestyle="--", alpha=0.6)
            ax.fill_between(gdf.index[gdf[col].notna()], vals, 100,
                            alpha=0.12, color=color)
        ax.set_ylabel(title, color=TXT, fontsize=9)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.set_ylim(60, 140)
        ax.grid(axis="y", color=GRID, alpha=0.15)
        for sp in ax.spines.values(): sp.set_color("#2E3D55")

    axes[-1].set_xticks(gdf.index)
    axes[-1].set_xticklabels(gdf["Game"], rotation=45, ha="right", color=TXT, fontsize=8)
    axes[0].set_title(f"{pitcher} — Season Grade Trends  ·  100 = D1 Average",
                      color=TXT, fontsize=13, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(
        gdf[["Game","Pitches","Outing","Stuff","Efficiency","P/IP"]].set_index("Game"),
        use_container_width=True)


# ============================================================
# GAME REVIEW PAGE
# ============================================================
def game_review_page(all_pitches_df: pd.DataFrame):
    st.title("Game Review")
    st.caption("Full advanced breakdown for any game — pitching, hitting, and umpire scorecard in one view.")

    if all_pitches_df is None or all_pitches_df.empty:
        st.warning("No game data loaded.")
        return

    df_full = all_pitches_df.copy()
    for col in ["Date", "PitcherTeam", "BatterTeam"]:
        if col not in df_full.columns:
            df_full[col] = ""

    # Identify opponent for each pitch (the non-Fordham team)
    df_full["_opp"] = np.where(
        df_full["PitcherTeam"].astype(str).str.upper() == "FOR_RAM",
        df_full["BatterTeam"],
        df_full["PitcherTeam"]
    )
    df_full["_opp"] = df_full["_opp"].astype(str).str.strip()

    game_id_cols = [c for c in ["GameID", "GameUID", "GameForeignID"] if c in df_full.columns]
    group_cols = ["Date", "_opp"] + game_id_cols

    games_df = (
        df_full.groupby(group_cols, observed=True, dropna=False)
        .size().reset_index(name="_n")
        .sort_values("Date", ascending=False)
    )

    def _glabel(row):
        d = pd.to_datetime(row["Date"], errors="coerce")
        dstr = d.strftime("%b %d, %Y") if pd.notna(d) else str(row["Date"])
        opp = team_display_name(str(row["_opp"]))
        return f"{dstr}  ·  vs  {opp}  ({int(row['_n'])} pitches)"

    games_df["_label"] = games_df.apply(_glabel, axis=1)

    if games_df.empty:
        st.warning("No games found in the loaded data.")
        return

    selected_label = st.selectbox("Select Game", games_df["_label"].tolist(), key="game_review_sel")
    sel = games_df[games_df["_label"] == selected_label].iloc[0]

    # Filter all data to this game
    mask = df_full["Date"].astype(str) == str(sel["Date"])
    mask &= df_full["_opp"].astype(str) == str(sel["_opp"])
    for gc in game_id_cols:
        sv = sel.get(gc)
        if sv and str(sv) not in ("", "nan"):
            mask &= df_full[gc].astype(str) == str(sv)
            break

    game_df = df_full[mask].copy()
    if game_df.empty:
        st.warning("No pitches found for the selected game.")
        return

    d_fmt = pd.to_datetime(sel["Date"], errors="coerce")
    dstr_full = d_fmt.strftime("%B %d, %Y") if pd.notna(d_fmt) else str(sel["Date"])
    opp_name = team_display_name(str(sel["_opp"]))

    st.markdown(f"## Fordham  vs  {opp_name}  ·  {dstr_full}")

    for_pitches = (game_df["PitcherTeam"].astype(str).str.upper() == "FOR_RAM").sum()
    opp_pitches = len(game_df) - for_pitches
    for_pa = (game_df["BatterTeam"].astype(str).str.upper().str.startswith("FOR_RAM")).sum()

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Pitches", len(game_df))
    mc2.metric("Fordham Pitches Thrown", int(for_pitches))
    mc3.metric("Pitches Fordham Faced", int(opp_pitches))
    mc4.metric("Fordham PA Rows", int(for_pa))

    # ── PITCHING ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Pitching")

    pitch_df = game_df[game_df["PitcherTeam"].astype(str).str.upper() == "FOR_RAM"].copy()
    pitch_df = add_perceived_velocity(pitch_df)
    if "is_swing" not in pitch_df.columns:
        pitch_df = add_flags(pitch_df) if not pitch_df.empty else pitch_df

    if pitch_df.empty:
        st.info("No Fordham pitching data found for this game.")
    else:
        pitchers_g = [p for p in pitch_df.groupby("Pitcher", sort=False)["Pitcher"].first().tolist()
                      if pd.notna(p)]
        pitchers_g = sorted(set(pitchers_g), key=lambda p: pitch_df[pitch_df["Pitcher"]==p].index[0])

        # Summary table
        p_rows = []
        for p in pitchers_g:
            pdf = pitch_df[pitch_df["Pitcher"] == p].copy()
            stats  = _compute_pitcher_pct_stats(pdf)
            pa_r   = pitcher_pa_rates(pdf)
            bip    = get_true_bip_with_ev(pdf) if {"EV","PitchCall"}.issubset(pdf.columns) else pd.DataFrame()
            allow  = pitcher_allowed_slash(pdf)
            swings = float(pdf["is_swing"].sum()) if "is_swing" in pdf.columns else 0
            whiffs = float(pdf["is_whiff"].sum()) if "is_whiff" in pdf.columns else 0
            p_rows.append({
                "Pitcher":  p,
                "Pitches":  len(pdf),
                "Stuff+":   round(stats.get("Stuff+") or np.nan, 1),
                "Loc+":     round(stats.get("Loc+") or np.nan, 1),
                "Velo":     round(stats.get("Velo") or np.nan, 1),
                "CSW%":     round(float(pdf["is_csw"].mean()*100) if "is_csw" in pdf.columns else np.nan, 1),
                "Whiff%":   round(whiffs/swings*100 if swings else np.nan, 1),
                "Zone%":    round(float(pdf["in_zone"].mean()*100) if "in_zone" in pdf.columns else np.nan, 1),
                "K%":       round(pa_r.get("K%", np.nan), 1),
                "BB%":      round(pa_r.get("BB%", np.nan), 1),
                "BA vs":    allow.get("BA"),
                "SLG vs":   allow.get("SLG"),
                "Avg EV":round(bip["EV"].mean() if not bip.empty else np.nan, 1),
                "HH% vs":   round((bip["EV"]>=95).mean()*100 if not bip.empty else np.nan, 1),
            })
        p_tbl = pd.DataFrame(p_rows).set_index("Pitcher")
        st.dataframe(style_scouting_dataframe(p_tbl, context="pitching"), use_container_width=True)

        # ── Outing grades per pitcher ──────────────────────────────────────────
        st.markdown("#### Outing Grades")
        grade_cols = st.columns(len(pitchers_g)) if len(pitchers_g) <= 5 else st.columns(5)
        for gi, p in enumerate(pitchers_g):
            pdf_g  = pitch_df[pitch_df["Pitcher"] == p].copy()
            stats_g = _compute_pitcher_pct_stats(pdf_g)
            bip_g   = get_true_bip_with_ev(pdf_g) if {"EV","PitchCall"}.issubset(pdf_g.columns) else pd.DataFrame()
            avg_ev_g = float(bip_g["EV"].mean()) if not bip_g.empty else float("nan")
            hh_g     = float((bip_g["EV"]>=95).mean()*100) if not bip_g.empty else float("nan")
            brl_g    = float(bip_g["barrel"].mean()*100) if not bip_g.empty and "barrel" in bip_g.columns else float("nan")
            csw_g    = float(pdf_g["is_csw"].mean()*100) if "is_csw" in pdf_g.columns else float("nan")
            sw_g     = pdf_g["is_swing"].sum() if "is_swing" in pdf_g.columns else 0
            wh_g     = float(pdf_g["is_whiff"].sum()/sw_g*100) if sw_g else float("nan")
            bb_g     = stats_g.get("BB%", float("nan")) or float("nan")
            fps_g    = stats_g.get("FPS%", float("nan")) or float("nan")
            letter_o, color_o, _, _ = outing_grade(
                stats_g.get("Stuff+"), stats_g.get("Loc+"),
                fps_g, csw_g, wh_g, bb_g, avg_ev_g, hh_g, brl_g)
            p_ip, p_outs = compute_pitch_efficiency(pdf_g)
            letter_e, color_e, _ = pitch_efficiency_grade(p_ip)
            letter_s, color_s, _ = pure_stuff_grade(stats_g.get("Stuff+") or float("nan"))
            tc_o = "#0f172a" if color_o in ("#bef264","#fde047","#86efac","#4ade80") else "#ffffff"
            tc_e = "#0f172a" if color_e in ("#bef264","#fde047","#86efac","#4ade80") else "#ffffff"
            tc_s = "#0f172a" if color_s in ("#bef264","#fde047","#86efac","#4ade80") else "#ffffff"
            short = p.split(",")[0] if "," in p else p.split()[-1]
            with grade_cols[gi % 5]:
                st.markdown(
                    f'<div style="background:#1a2535;border:1px solid #2E3D55;border-radius:10px;'
                    f'padding:10px 8px;text-align:center;margin:4px 0">'
                    f'<div style="color:#9BAABF;font-size:.72rem;font-weight:700;margin-bottom:6px">{short}</div>'
                    f'<div style="display:flex;justify-content:center;gap:6px">'
                    f'<div style="background:{color_o};color:{tc_o};border-radius:6px;padding:4px 8px;font-weight:900;font-size:1rem" title="Outing Grade">{letter_o}</div>'
                    f'<div style="background:{color_s};color:{tc_s};border-radius:6px;padding:4px 8px;font-weight:900;font-size:1rem" title="Stuff Grade">{letter_s}</div>'
                    f'<div style="background:{color_e};color:{tc_e};border-radius:6px;padding:4px 8px;font-weight:900;font-size:1rem" title="Efficiency Grade">{letter_e}</div>'
                    f'</div>'
                    f'<div style="color:#6B7A93;font-size:.62rem;margin-top:4px">Outing · Stuff · Eff</div>'
                    f'</div>', unsafe_allow_html=True)

        st.markdown("#### Pitcher Detail")
        for p in pitchers_g:
            pdf = pitch_df[pitch_df["Pitcher"] == p].copy()
            hand = _dominant_pitcher_hand(pdf)
            with st.expander(f"{p}  ·  {hand}  ·  {len(pdf)} pitches", expanded=(len(pitchers_g)==1)):
                col_mv, col_arsen = st.columns([3, 2])
                with col_mv:
                    st.pyplot(build_movement_figure(pdf), use_container_width=True)
                with col_arsen:
                    arsen = pdf.groupby("pitch_abbr", sort=False).agg(
                        N=("pitch_abbr","count"),
                        Velo=("Velo","mean"),
                        IVB=("IVB","mean"),
                        HB=("HB","mean"),
                    ).reset_index().rename(columns={"pitch_abbr":"Pitch"})
                    arsen["Usage%"] = (arsen["N"]/arsen["N"].sum()*100).round(1)
                    for extra_col, src in [("Stuff+","Stuff+"),("Loc+","Loc+")]:
                        if src in pdf.columns:
                            arsen = arsen.merge(
                                pdf.groupby("pitch_abbr")[src].mean().reset_index().rename(columns={"pitch_abbr":"Pitch",src:extra_col}),
                                on="Pitch", how="left")
                    w_df = pdf.groupby("pitch_abbr").agg(Sw=("is_swing","sum"), Wh=("is_whiff","sum")).reset_index().rename(columns={"pitch_abbr":"Pitch"})
                    arsen = arsen.merge(w_df, on="Pitch", how="left")
                    arsen["Whiff%"] = np.where(arsen["Sw"]>0, arsen["Wh"]/arsen["Sw"]*100, np.nan).round(1)
                    show_cols = [c for c in ["Pitch","N","Usage%","Velo","IVB","HB","Stuff+","Loc+","Whiff%"] if c in arsen.columns]
                    st.dataframe(style_scouting_dataframe(arsen[show_cols].set_index("Pitch").round(1), context="pitching"), use_container_width=True)

                z1, z2, z3, z4 = st.columns(4)
                for ax_col, metric, title in zip(
                    [z1,z2,z3,z4],
                    ["CSW%","Whiff%","AvgEV","Usage%"],
                    ["CSW% by Zone","Whiff% by Zone","Avg EV","Location%"]
                ):
                    fig_z = make_savant_zone_heatmap(pdf, metric, title, "")
                    if fig_z:
                        ax_col.pyplot(fig_z, use_container_width=True)

    # ── HITTING ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hitting")

    hit_df_raw = game_df[game_df["BatterTeam"].astype(str).str.upper().str.startswith("FOR_RAM")].copy()

    if hit_df_raw.empty:
        st.info("No Fordham hitting data found for this game.")
    else:
        hit_df = normalize_hitter_columns(hit_df_raw)
        hit_df = add_contact_quality_local(hit_df)
        if "is_swing" not in hit_df.columns:
            hit_df = add_flags(hit_df)
        lgwoba = compute_league_woba(game_df)

        hitters_g = hit_df["Batter"].dropna().unique().tolist()

        h_rows = []
        h_cards = {}
        for h in hitters_g:
            hdf = hit_df[hit_df["Batter"] == h].copy()
            card = compute_hitter_card(hdf, lgwoba)
            h_cards[h] = card
            slash_df = add_ba_slg_by_group(hdf.assign(_P=h), ["_P"])
            sl = {} if slash_df.empty else {c: slash_df[c].iloc[0] for c in ["BA","OBP","SLG","OPS"] if c in slash_df.columns}
            # Traditional box score + advanced
            h_rows.append({
                "Batter":  h,
                "PA":      card["PA"],
                "AB":      card["AB"],
                "H":       card["H"],
                "2B":      card.get("2B", 0),
                "3B":      card.get("3B", 0),
                "HR":      card["HR"],
                "BB":      card.get("BB", 0),
                "K":       card.get("K", 0),
                "wOBA":    f"{card['wOBA']:.3f}",
                "Bat+":    card["Bat+"],
                "Avg EV":  card["AvgEV"],
                "Max EV":  card["MaxEV"],
                "HH%":     card["HardHit%"],
                "Whiff%":  card["Whiff%"],
                "Chase%":  card["Chase%"],
            })
        h_tbl = pd.DataFrame(h_rows).set_index("Batter")
        st.dataframe(style_scouting_dataframe(h_tbl, context="hitting"), use_container_width=True)

        st.markdown("#### Hitter Detail")
        for h in hitters_g:
            hdf = hit_df[hit_df["Batter"] == h].copy()
            card = h_cards[h]
            if card["PA"] == 0:
                continue
            slash_df2 = add_ba_slg_by_group(hdf.assign(_P=h), ["_P"])
            sl2 = {} if slash_df2.empty else {c: slash_df2[c].iloc[0] for c in ["BA","OBP","SLG"] if c in slash_df2.columns}
            _ba  = f"{sl2['BA']:.3f}"  if sl2.get("BA")  else "—"
            _obp = f"{sl2['OBP']:.3f}" if sl2.get("OBP") else "—"
            _slg = f"{sl2['SLG']:.3f}" if sl2.get("SLG") else "—"
            _hr  = card["HR"]; _bb = card.get("BB",0); _k = card.get("K",0)
            _ev  = f"{card['AvgEV']:.1f}" if card["AvgEV"] else "—"
            _mev = f"{card['MaxEV']:.1f}" if card["MaxEV"] else "—"
            _exp_label = (
                f"**{h}**  ·  {card['H']}/{card['AB']}-{_hr}HR-{_bb}BB-{_k}K  "
                f"·  {_ba}/{_obp}/{_slg}  ·  wOBA {card['wOBA']:.3f}  "
                f"·  EV {_ev} / Max {_mev}"
            )
            with st.expander(_exp_label, expanded=False):
                hc1, hc2 = st.columns([1.4, 1])
                with hc1:
                    st.pyplot(build_hitter_spray_chart(hdf, h, annotate_ev=True), use_container_width=True)
                with hc2:
                    zza, zzb = st.columns(2)
                    for zcol, metric, title in [
                        (zza,"Swing%","In-Zone Swing%"), (zzb,"Whiff%","In-Zone Whiff%"),
                        (zza,"HardHit%","HardHit%"),    (zzb,"AvgEV","Avg EV"),
                    ]:
                        fig_z = make_savant_zone_heatmap(hdf, metric, title, "")
                        if fig_z:
                            zcol.pyplot(fig_z, use_container_width=True)

                pt_df = hitter_pitchtype_effectiveness(hdf)
                if not pt_df.empty:
                    show = [c for c in ["Pitch","N","BA","SLG","Swing%","Whiff%","Chase%","AvgEV","HardHit%"] if c in pt_df.columns]
                    st.dataframe(style_scouting_dataframe(pt_df[show].set_index("Pitch"), context="hitting"), use_container_width=True)

    # ── UMPIRE SCORECARD ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Umpire Scorecard")

    try:
        scorecard = _build_umpire_from_df(game_df)
        m = scorecard["metrics"]
        um1, um2, um3, um4, um5 = st.columns(5)
        um1.metric("Called Pitches",    int(m["called_pitches"]))
        um2.metric("Overall Accuracy",  f"{m['overall_accuracy']:.1f}%")
        um3.metric("Missed Calls",      int(m["missed_calls"]))
        um4.metric("Fordham Favored",   int(m["fordham_favor"]))
        um5.metric("Net Fordham",       f"{m['fordham_net']:+d}")

        if st.button("Show Full Scorecard", key="game_review_umpire_btn"):
            _, sc_fig = generate_umpire_scorecard(scorecard)
            st.pyplot(sc_fig, use_container_width=True)

        if not scorecard["missed"].empty:
            with st.expander(f"Missed Calls ({len(scorecard['missed'])})", expanded=False):
                st.dataframe(scorecard["missed"].reset_index(drop=True), use_container_width=True, hide_index=True)
    except Exception as _ump_err:
        st.info(f"Umpire scorecard unavailable for this game: {_ump_err}")


# ------------------------------------------------------------
# PRIVATE REPORTS
# ------------------------------------------------------------
PRIVATE_REPORTS_PASSWORD = "Rams1"
REPORTS_DIR = ROOT / "personal_reports"


def _prettify(s: str) -> str:
    return " ".join(w.capitalize() for w in s.replace("_", " ").replace("-", " ").split())


def _group_reports(reports_dir: Path) -> dict:
    groups: dict = {}
    for f in sorted(reports_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".pdf", ".png", ".jpg", ".jpeg"):
            continue
        parts = f.stem.split("_")
        subject = _prettify("_".join(parts[:2])) if len(parts) >= 2 else _prettify(f.stem)
        groups.setdefault(subject, []).append(f)
    # PDFs first within each group
    for key in groups:
        groups[key].sort(key=lambda f: (0 if f.suffix.lower() == ".pdf" else 1, f.name))
    return groups


def _rendered_pages(pdf_path: Path) -> list:
    pages_dir = pdf_path.parent / pdf_path.stem
    if pages_dir.is_dir():
        return sorted([p for p in pages_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    return []


def private_reports_page():
    st.subheader("Private Reports")

    if not st.session_state.get("private_reports_authed"):
        st.markdown("This section is restricted to coaching staff.")
        _, center, _ = st.columns([1.15, 1, 1.15])
        with center:
            pw = st.text_input("Reports password", type="password",
                               placeholder="Enter password", key="priv_pw_input")
            if st.button("Unlock Reports", use_container_width=True, key="priv_pw_btn"):
                if pw == PRIVATE_REPORTS_PASSWORD:
                    st.session_state["private_reports_authed"] = True
                    rerun_app()
                else:
                    st.error("Incorrect password.")
        if pw == PRIVATE_REPORTS_PASSWORD:
            st.session_state["private_reports_authed"] = True
            rerun_app()
        return

    if not REPORTS_DIR.exists():
        st.info("No reports found.")
        return

    groups = _group_reports(REPORTS_DIR)
    if not groups:
        st.info("No reports found.")
        return

    selected_group = st.session_state.get("selected_report_group")

    if selected_group is None:
        st.markdown("### Reports")
        for group_name in groups:
            if st.button(group_name, key=f"rg_{group_name}", use_container_width=True):
                st.session_state["selected_report_group"] = group_name
                rerun_app()
    else:
        if st.button("Back to Reports", key="reports_back_btn"):
            st.session_state["selected_report_group"] = None
            rerun_app()
        st.markdown(f"### {selected_group}")
        for f in groups[selected_group]:
            section = _prettify(f.stem)
            st.markdown(f"#### {section}")
            suffix = f.suffix.lower()
            if suffix in (".png", ".jpg", ".jpeg"):
                st.image(str(f), use_container_width=True)
            elif suffix == ".pdf":
                pages = _rendered_pages(f)
                if pages:
                    for pg in pages:
                        st.image(str(pg), use_container_width=True)
                else:
                    st.info("PDF preview unavailable — use the download button below.")
                st.download_button("Download PDF", data=f.read_bytes(),
                                   file_name=f.name, mime="application/pdf",
                                   key=f"dl_{f.name}")


# ─────────────────────────────────────────────────────────────────────────────
# AI ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────

def _clean_gemini_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end   = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text  = "\n".join(lines[start:end])
    return text.strip()


def _fuzzy_team_match(opp: str) -> list:
    if not opp:
        return []
    opp_l = opp.lower().strip()
    hits = []
    for code, name in TEAM_CODE_NAME_OVERRIDES.items():
        if opp_l in name.lower() or opp_l in code.lower():
            hits.append(code)
    if not hits:
        words = [w for w in opp_l.split() if len(w) > 2]
        for code, name in TEAM_CODE_NAME_OVERRIDES.items():
            if any(w in name.lower() for w in words):
                hits.append(code)
    return hits


_PT_ALIASES = {
    "Fastball":        ["Fastball", "FourSeamFastBall"],
    "FourSeamFastBall":["Fastball", "FourSeamFastBall"],
    "Sinker":          ["Sinker",   "TwoSeamFastBall"],
    "TwoSeamFastBall": ["Sinker",   "TwoSeamFastBall"],
}


def _compute_ai_metric(filtered: pd.DataFrame, metric: str):
    """Return (answer_str, optional_DataFrame)."""
    n = len(filtered)
    if n == 0:
        return "No pitches found for that query — double-check the player name or filters.", None

    bip = filtered[filtered["ExitSpeed"].notna() & (filtered["ExitSpeed"] > 0)].copy() \
          if "ExitSpeed" in filtered.columns else pd.DataFrame()

    swing_calls  = {"StrikeSwinging", "FoulBall", "FoulTip", "InPlay"}
    whiff_calls  = {"StrikeSwinging"}
    strike_calls = {"StrikeCalled", "StrikeSwinging", "FoulBall", "FoulTip", "InPlay"}

    pc = filtered["PitchCall"] if "PitchCall" in filtered.columns else pd.Series(dtype=str)
    n_swings = pc.isin(swing_calls).sum()
    n_whiffs  = pc.isin(whiff_calls).sum()
    n_csw     = pc.isin({"StrikeCalled"} | whiff_calls).sum()

    pa_cols = {"Pitcher", "Batter", "Inning", "PAofInning"}
    tbf = filtered.groupby(list(pa_cols)).ngroups if pa_cols.issubset(filtered.columns) else max(1, n // 4)

    if metric == "avg_ev":
        if bip.empty: return "No balls in play found.", None
        v = bip["ExitSpeed"].mean()
        return f"**{v:.1f} mph** avg EV ({len(bip)} BIP)", None

    if metric == "max_ev":
        if bip.empty: return "No balls in play found.", None
        v   = bip["ExitSpeed"].max()
        idx = bip["ExitSpeed"].idxmax()
        who = bip.loc[idx, "Batter"] if "Batter" in bip.columns else "unknown"
        return f"**{v:.1f} mph** max EV (by {who})", None

    if metric == "hard_hit_pct":
        if bip.empty: return "No balls in play found.", None
        hh  = (bip["ExitSpeed"] >= 92).sum()
        pct = hh / len(bip) * 100
        return f"**{pct:.1f}%** hard-hit rate ({hh}/{len(bip)} BIP ≥ 92 mph)", None

    if metric == "barrel_pct":
        if bip.empty or "Angle" not in bip.columns:
            return "No BIP or launch angle data found.", None
        barrels = bip[(bip["ExitSpeed"] >= 92) & bip["Angle"].between(16, 36)]
        pct = len(barrels) / len(bip) * 100
        return f"**{pct:.1f}%** barrel rate ({len(barrels)}/{len(bip)} BIP)", None

    if metric == "whiff_pct":
        if n_swings == 0: return "No swings found.", None
        return f"**{n_whiffs/n_swings*100:.1f}%** whiff rate ({n_whiffs} whiffs / {n_swings} swings)", None

    if metric == "csw_pct":
        return f"**{n_csw/n*100:.1f}%** CSW ({n_csw}/{n} pitches)", None

    if metric == "k_pct":
        ks  = (filtered["KorBB"] == "Strikeout").sum() if "KorBB" in filtered.columns else 0
        return f"**{ks/tbf*100:.1f}%** K rate ({ks} K / {tbf} TBF)", None

    if metric == "bb_pct":
        bbs = (filtered["KorBB"] == "Walk").sum() if "KorBB" in filtered.columns else 0
        return f"**{bbs/tbf*100:.1f}%** BB rate ({bbs} BB / {tbf} TBF)", None

    if metric == "avg_velo":
        if "RelSpeed" not in filtered.columns: return "No velocity data.", None
        return f"**{filtered['RelSpeed'].dropna().mean():.1f} mph** avg velocity ({n} pitches)", None

    if metric == "max_velo":
        if "RelSpeed" not in filtered.columns: return "No velocity data.", None
        return f"**{filtered['RelSpeed'].dropna().max():.1f} mph** max velocity", None

    if metric == "avg_stuff":
        if "Stuff+" not in filtered.columns: return "No Stuff+ data.", None
        return f"**{filtered['Stuff+'].dropna().mean():.0f}** avg Stuff+ ({n} pitches)", None

    if metric == "avg_loc":
        if "Loc+" not in filtered.columns: return "No Loc+ data.", None
        return f"**{filtered['Loc+'].dropna().mean():.0f}** avg Loc+ ({n} pitches)", None

    if metric == "pitch_count":
        tbl = None
        if "TaggedPitchType" in filtered.columns:
            vc  = filtered["TaggedPitchType"].value_counts().reset_index()
            vc.columns = ["Pitch Type", "Count"]
            tbl = vc
        return f"**{n}** total pitches", tbl

    if metric == "usage_pct":
        if "TaggedPitchType" not in filtered.columns: return "No pitch type data.", None
        vc  = filtered["TaggedPitchType"].value_counts()
        tbl = (vc / vc.sum() * 100).round(1).reset_index()
        tbl.columns = ["Pitch Type", "Usage %"]
        return f"Pitch usage breakdown ({n} pitches):", tbl

    if metric == "fps_pct":
        if "Balls" not in filtered.columns: return "No count data.", None
        fp  = filtered[(filtered["Balls"] == 0) & (filtered["Strikes"] == 0)]
        if fp.empty: return "No first-pitch data.", None
        fps = fp[pc.reindex(fp.index).isin(strike_calls)].shape[0] if not pc.empty else 0
        return f"**{fps/len(fp)*100:.1f}%** first-pitch strike rate ({fps}/{len(fp)} first pitches)", None

    if metric == "avg_spin":
        if "SpinRate" not in filtered.columns: return "No spin data.", None
        return f"**{filtered['SpinRate'].dropna().mean():.0f} rpm** avg spin ({n} pitches)", None

    # all_stats — full summary
    rows = []
    if "RelSpeed" in filtered.columns:
        rows.append({"Stat": "Avg Velo", "Value": f"{filtered['RelSpeed'].dropna().mean():.1f} mph"})
        rows.append({"Stat": "Max Velo", "Value": f"{filtered['RelSpeed'].dropna().max():.1f} mph"})
    if not bip.empty:
        rows.append({"Stat": "Avg EV",     "Value": f"{bip['ExitSpeed'].mean():.1f} mph"})
        rows.append({"Stat": "Hard Hit%",  "Value": f"{(bip['ExitSpeed']>=92).mean()*100:.1f}%"})
        if "Angle" in bip.columns:
            bl = bip[(bip["ExitSpeed"] >= 92) & bip["Angle"].between(16, 36)]
            rows.append({"Stat": "Barrel%", "Value": f"{len(bl)/len(bip)*100:.1f}%"})
    if n_swings > 0:
        rows.append({"Stat": "Whiff%", "Value": f"{n_whiffs/n_swings*100:.1f}%"})
    rows.append({"Stat": "CSW%", "Value": f"{n_csw/n*100:.1f}%"})
    if "KorBB" in filtered.columns:
        ks  = (filtered["KorBB"] == "Strikeout").sum()
        bbs = (filtered["KorBB"] == "Walk").sum()
        rows.append({"Stat": "K%",  "Value": f"{ks/tbf*100:.1f}%"})
        rows.append({"Stat": "BB%", "Value": f"{bbs/tbf*100:.1f}%"})
    if "Stuff+" in filtered.columns:
        rows.append({"Stat": "Stuff+", "Value": f"{filtered['Stuff+'].dropna().mean():.0f}"})
    if "Loc+"   in filtered.columns:
        rows.append({"Stat": "Loc+",   "Value": f"{filtered['Loc+'].dropna().mean():.0f}"})
    rows.append({"Stat": "Pitches", "Value": str(n)})
    return f"Full breakdown ({n} pitches):", pd.DataFrame(rows) if rows else None


def _run_ai_query(df: pd.DataFrame, question: str, api_key: str):
    import json
    from groq import Groq

    client = Groq(api_key=api_key)

    pitchers = sorted(df["Pitcher"].dropna().unique().tolist()) if "Pitcher" in df.columns else []
    batters  = sorted(df["Batter"].dropna().unique().tolist())  if "Batter"  in df.columns else []
    fordham_pitchers = [p for p in pitchers if any(
        df[(df["Pitcher"] == p) & df["PitcherTeam"].str.contains("FOR", na=False)].shape[0] > 0
        for _ in [1])] if "PitcherTeam" in df.columns else pitchers
    fordham_batters = [b for b in batters if any(
        df[(df["Batter"] == b) & df["BatterTeam"].str.contains("FOR", na=False)].shape[0] > 0
        for _ in [1])] if "BatterTeam" in df.columns else batters

    pitcher_ctx = ", ".join(sorted(fordham_pitchers)[:60])
    batter_ctx  = ", ".join(sorted(fordham_batters)[:60])

    system_prompt = f"""You are a baseball data assistant for Fordham Baseball.
Parse the user's question about TrackMan pitch-by-pitch data and return ONLY valid JSON.

Available Fordham PITCHERS (exact "Last, First" format): {pitcher_ctx}
Available Fordham HITTERS (exact "Last, First" format): {batter_ctx}

Columns available:
- Pitcher / Batter: "Last, First" format
- PitcherTeam / BatterTeam: team codes (e.g. FOR_RAM, VCU_RAM)
- TaggedPitchType: Fastball, Sinker, TwoSeamFastBall, FourSeamFastBall, Slider, Cutter, Curveball, ChangeUp, Splitter
- RelSpeed: pitch velo | ExitSpeed: EV on BIP | Angle: launch angle
- KorBB: "Strikeout" or "Walk" | PitchCall: StrikeCalled, StrikeSwinging, Ball, InPlay, FoulBall
- PitcherThrows / BatterSide: Right or Left | Balls / Strikes: count
- Stuff+ / Loc+: pitcher model scores (100 = average) | SpinRate

Pitch type mapping (accept plurals, abbreviations):
  fastball/four-seam → Fastball or FourSeamFastBall
  sinker/two-seam → Sinker or TwoSeamFastBall
  slider → Slider | cutter → Cutter | curve/hook → Curveball
  change/changeup → ChangeUp | splitter/split → Splitter

Valid metrics: avg_ev, max_ev, hard_hit_pct, barrel_pct, whiff_pct, csw_pct,
  k_pct, bb_pct, avg_velo, max_velo, avg_stuff, avg_loc, pitch_count,
  usage_pct, fps_pct, avg_spin, all_stats

Return ONLY this JSON (no markdown):
{{
  "player_type": "pitcher" or "hitter",
  "pitcher": "Last, First" or null,
  "batter": "Last, First" or null,
  "opponent": "team name fragment" or null,
  "pitch_types": ["Cutter"] or null,
  "metric": "<one of the valid metrics above>",
  "pitcher_throws": "Right" or "Left" or null,
  "batter_side": "Right" or "Left" or null,
  "context": "one-sentence plain-English description of what was asked"
}}

Rules:
- If the name matches a hitter, set player_type="hitter" and put name in "batter" field.
- If the name matches a pitcher, set player_type="pitcher" and put name in "pitcher" field.
- For hitters: avg_ev/hard_hit_pct/barrel_pct/k_pct/bb_pct describe what the HITTER does at the plate.
- If no specific metric is implied, use all_stats."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        query = json.loads(response.choices[0].message.content)
    except Exception as e:
        return f"Could not parse your question ({e}). Try rephrasing.", None

    # Apply filters
    filt = df.copy()
    player_type  = query.get("player_type", "pitcher")
    player_name  = None

    if player_type == "hitter" and query.get("batter"):
        last = query["batter"].split(",")[0].strip()
        player_name = last
        filt = filt[filt["Batter"].str.contains(last, case=False, na=False)]
        # For hitters: opponent is PitcherTeam, pitch_type filter still applies
        if query.get("opponent"):
            codes = _fuzzy_team_match(query["opponent"])
            if codes:
                filt = filt[filt["PitcherTeam"].isin(codes)]
        if query.get("pitch_types"):
            expanded = []
            for pt in query["pitch_types"]:
                expanded.extend(_PT_ALIASES.get(pt, [pt]))
            filt = filt[filt["TaggedPitchType"].isin(expanded)]
        if query.get("pitcher_throws"):
            filt = filt[filt["PitcherThrows"] == query["pitcher_throws"]]
    else:
        if query.get("pitcher"):
            last = query["pitcher"].split(",")[0].strip()
            player_name = last
            filt = filt[filt["Pitcher"].str.contains(last, case=False, na=False)]
        if query.get("batter"):
            last = query["batter"].split(",")[0].strip()
            filt = filt[filt["Batter"].str.contains(last, case=False, na=False)]
        if query.get("opponent"):
            codes = _fuzzy_team_match(query["opponent"])
            if codes:
                filt = filt[filt["BatterTeam"].isin(codes)]
        if query.get("pitch_types"):
            expanded = []
            for pt in query["pitch_types"]:
                expanded.extend(_PT_ALIASES.get(pt, [pt]))
            filt = filt[filt["TaggedPitchType"].isin(expanded)]
        if query.get("batter_side"):
            filt = filt[filt["BatterSide"] == query["batter_side"]]

    # Player not found — show helpful roster
    if filt.empty and player_name:
        if player_type == "hitter":
            available = sorted(df["Batter"].dropna().unique().tolist()) if "Batter" in df.columns else []
            label = "hitters"
        else:
            available = sorted(df["Pitcher"].dropna().unique().tolist()) if "Pitcher" in df.columns else []
            label = "pitchers"
        roster = "  \n".join(f"• {p}" for p in available)
        return (f"**'{player_name}' not found in the dataset.**\n\n"
                f"**Available {label}:**  \n{roster}"), None

    answer, table = _compute_ai_metric(filt, query.get("metric", "all_stats"))

    ctx = query.get("context", "")
    return (f"_{ctx}_\n\n{answer}" if ctx else answer), table


def ask_ai_page(all_df: pd.DataFrame):
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.warning(
            "**GROQ_API_KEY not configured.**  \n"
            "Add a free key to `.streamlit/secrets.toml`:  \n"
            "```\nGROQ_API_KEY = \"your-key-here\"\n```  \n"
            "Get one free at **console.groq.com** (takes ~2 min, no credit card needed)."
        )
        return

    if all_df.empty:
        st.info("No pitch data loaded yet.")
        return

    st.markdown("### Ask anything about your TrackMan data")

    EXAMPLES = [
        "Tell me about McAndrews",
        "What's McAndrews avg EV this season?",
        "McAndrews barrel rate vs righties",
        "What's Hanawalt's avg EV against on cutters?",
        "Hanawalt whiff rate on sliders",
        "Give me Hanawalt's full stat breakdown",
    ]
    st.markdown("**Quick examples** — click to load:")
    ecols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if ecols[i % 3].button(ex, key=f"ai_ex_{i}", use_container_width=True):
            st.session_state["_ai_q"] = ex

    prefill = st.session_state.pop("_ai_q", "")
    question = st.text_input(
        "Your question:",
        value=prefill,
        placeholder="e.g. What's Hanawalt's avg EV against on cutters?",
        key="ai_q_input",
    )

    if not question:
        return

    with st.spinner("Thinking..."):
        try:
            answer, table = _run_ai_query(all_df, question, api_key)
        except Exception as e:
            st.error(f"Error: {e}")
            return

    st.markdown("---")
    st.markdown(answer)
    if table is not None and not table.empty:
        st.dataframe(table, use_container_width=True, hide_index=True)

    # Session history
    hist = st.session_state.setdefault("ai_history", [])
    if not hist or hist[-1]["q"] != question:
        hist.append({"q": question, "a": answer})

    prior = hist[:-1][-5:]
    if prior:
        with st.expander("Previous questions this session"):
            for item in reversed(prior):
                st.markdown(f"**Q:** {item['q']}")
                st.markdown(f"**A:** {item['a']}")
                st.markdown("---")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    inject_fordham_theme(show_logo=False)
    logo_b64 = get_logo_b64()
    logo_tag = (f'<img src="data:image/png;base64,{logo_b64}" class="fordham-hero-logo">'
                if logo_b64 else "")
    st.markdown(
        f"""
        <div class="fordham-hero" style="display:flex;align-items:center;gap:20px">
            {logo_tag}
            <div>
                <h1 style="margin:0">Fordham Baseball Advanced Analytics</h1>
                <p style="margin:0.3rem 0 0 0">Pitching plans, hitter development, TrackMan contact quality, and game-report visuals in one staff dashboard.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load all processed pitch-by-pitch data ONCE
    all_pitches_df = prepare_data()

    page_options = {
        "Game Review": ["Game Review"],
        "Reports": ["Postgame Summary", "Season Summary", "Pitcher Profile",
                    "Percentile Cards", "Hitter Percentile Cards"],
        "Leaderboards": ["Stuff+", "Location+", "Pitch Efficiency", "Pitch-Type Leaderboards", "Contact Quality", "HR Distance"],
        "Development": ["Pitcher Advanced Info", "Season Grade Trends", "Hitter Advanced Info"],
        "Practice": ["Bullpen Review", "Batting Practice", "Intersquad Leaderboard"],
        "Scouting Zone": ["Player Reports"],
        "Glossary": ["Advanced Stats Glossary"],
        "Private Reports": ["Private Reports"],
        "AI Assistant": ["Ask AI"],
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

    if page == "Game Review":
        game_review_page(all_pitches_df)
    elif page == "Postgame Summary":
        postgame_page()
    elif page == "Season Summary":
        season_page()
    elif page == "Pitcher Profile":
        pitcher_profile_page()
    elif page == "Percentile Cards":
        percentile_card_page()
    elif page == "Hitter Percentile Cards":
        hitter_percentile_card_page(all_pitches_df)
    elif page == "Stuff+":
        stuff_leaderboard_page()
    elif page == "Location+":
        location_leaderboard_page()
    elif page == "Pitch Efficiency":
        pitch_efficiency_leaderboard_page()
    elif page == "Pitch-Type Leaderboards":
        pitchtype_grids_page()
    elif page == "Contact Quality":
        contact_quality_leaderboard_page(all_pitches_df)
    elif page == "HR Distance":
        hr_distance_leaderboard_page(all_pitches_df)
    elif page == "Pitcher Advanced Info":
        sequencing_page(all_pitches_df)
    elif page == "Season Grade Trends":
        season_grade_trends_page(all_pitches_df)
    elif page == "Hitter Advanced Info":
        hitter_development_page(all_pitches_df)
    elif page == "Bullpen Review":
        bullpen_review_page()
    elif page == "Batting Practice":
        batting_practice_page()
    elif page == "Intersquad Leaderboard":
        intersquad_leaderboard_page()
    elif page == "Player Reports":
        scouting_zone_page(all_pitches_df)
    elif page == "Private Reports":
        private_reports_page()
    elif page == "Ask AI":
        ask_ai_page(all_pitches_df)
    else:
        glossary_page()


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if check_password():
    main()
else:
    st.stop()
