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
    "GEO_COL": "Georgia College Bobcats",
    "GEO_PAT": "George Mason Patriots",
    "GEO_COL1": "Georgia College Bobcats",
    "GEO_COL2": "Georgia College Bobcats",
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
}

TEAM_COLORS = {
    "FOR_RAM": ("#8C1515", "#C7A45D"),
    "FOR_RAM1": ("#8C1515", "#C7A45D"),
    "FLA__GAT": ("#0021A5", "#FA4616"),
    "FLO_SEM": ("#782F40", "#CEB888"),
    "FLA_GAT": ("#0021A5", "#FA4616"),
    "TEN_VOL": ("#FF8200", "#58595B"),
    "VIR_CAV": ("#232D4B", "#F84C1E"),
    "UCLA": ("#2774AE", "#FFD100"),
    "VAN_COM": ("#000000", "#B3A369"),
    "AKR_ZIP": ("#041E42", "#A89968"),
    "ALA_CRI": ("#9E1B32", "#FFFFFF"),
    "BOC_EAG": ("#003263", "#BC9B6A"),
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
    "GEO_COL": ("#003057", "#C0C0C0"),
    "GEO_PAT": ("#006633", "#FFCC33"),
    "GEO_COL1": ("#003057", "#C0C0C0"),
    "GEO_COL2": ("#003057", "#C0C0C0"),
    "GEO_EAG": ("#011E41", "#A99260"),
    "GEO_FOX": ("#002F6C", "#C8102E"),
    "GEO_GWI": ("#033C5A", "#AA9868"),
    "GEO_HOY": ("#041E42", "#8D817B"),
    "GEO_PAN": ("#0039A6", "#C60C30"),
    "GEO_SOU": ("#011E41", "#A99260"),
    "SAC_PIO": ("#CE1141", "#FFFFFF"),
    "ION_GAL": ("#6F2C91", "#FFB81C"),
    "ION_GAE": ("#6F2C91", "#FFB81C"),
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
    "ARK_LIO": ("#003A70", "#C8102E"),
    "SOU_LIO": ("#006747", "#F1B82D"),
    "SAC_DON": ("#862633", "#FFFFFF"),
    "LOY_UNI": ("#8D0034", "#FFC72C"),
    "MAR_HAW": ("#7A0019", "#000000"),
    "MAR_HAR": ("#003A70", "#C8102E"),
    "MAR_BAL": ("#0033A0", "#FFFFFF"),
    "MAR_SAI": ("#002F6C", "#C8102E"),
    "MAR_TER": ("#B31B1B", "#FFFFFF"),
    "MAR_THU": ("#00B140", "#000000"),
    "MAR_UNI3": ("#003A70", "#C8102E"),
    "SBU_SEA": ("#990000", "#1F1F1F"),
    "STJ_RED": ("#BA0C2F", "#FFFFFF"),
    "DUK_BLU": ("#012169", "#FFFFFF"),
    "UMASS_RIV": ("#003DA5", "#C0C0C0"),
    "OHIO_BOB": ("#00694E", "#FFFFFF"),
    "PUR_BOI": ("#CEB888", "#000000"),
    "RIC_OWL": ("#00205B", "#FFFFFF"),
    "SET_PIR": ("#003366", "#FFFFFF"),
    "PIT_PAN": ("#003594", "#FFB81C"),
    "BYU_COU": ("#002255", "#FFFFFF"),
    "CIN_BEA": ("#E00122", "#000000"),
    "CLE_TIG": ("#F66733", "#522D80"),
    "MIC_SPA": ("#18453B", "#FFFFFF"),
    "MIC_WOL": ("#00274C", "#FFCB05"),
    "CEN_MIC": ("#6A0032", "#FFCB05"),
    "LOU_CAJ": ("#CE181E", "#000000"),
    "LOU_CAR": ("#AD0000", "#000000"),
    "FDU_KNI": ("#0033A0", "#C8102E"),
    "FRE_BUL": ("#003A70", "#C41230"),
    "LSU_TIG": ("#461D7C", "#FDD023"),
    "MIA_HUR": ("#F47321", "#005030"),
    "MIL_UNI": ("#000000", "#FFC72C"),
    "MSU_BDG": ("#660000", "#FFFFFF"),
    "NEW_HAV": ("#0033A0", "#FFCD00"),
    "NIU_HUS": ("#BA0C2F", "#000000"),
    "ARI_SUN": ("#8C1D40", "#FFC627"),
    "AIR_FOR": ("#003087", "#8A8D8F"),
    "ARM_BLA": ("#000000", "#D4AF37"),
    "NAV_MID": ("#00205B", "#C5B783"),
    "MEX_LOB": ("#BA0C2F", "#000000"),
    "XAV_MUS": ("#0C2340", "#9EA2A2"),
    "AUB_TIG": ("#0C2340", "#E87722"),
    "NEV_WOL": ("#003366", "#A7A9AC"),
    "UCF_KNI": ("#000000", "#BA9B37"),
    "HOU_COG": ("#C8102E", "#FFFFFF"),
    "YAL_BUL": ("#00356B", "#FFFFFF"),
    "NOR_TAR": ("#7BAFD4", "#13294B"),
    "NOR_WOL": ("#CC0000", "#000000"),
    "ORE_DUC": ("#154733", "#FEE123"),
    "CCU_BLD": ("#0033A0", "#A7A9AC"),
    "UCO_HUS": ("#000E2F", "#FFFFFF"),
    "CON_HUS": ("#000E2F", "#FFFFFF"),
    "BOS_COL": ("#003263", "#BC9B6A"),
    "DEL_BLU": ("#00539B", "#FFD200"),
    "DEL_STA": ("#EE3124", "#00539B"),
    "HOF_PRI": ("#003591", "#FFB81C"),
    "DRE_DRA": ("#07294D", "#FFC600"),
    "NOR_HUS": ("#CC0000", "#000000"),
    "WIL_SEA": ("#006666", "#CBA052"),
    "ELON_PHO": ("#73000A", "#B59A57"),
    "CAM_CAM": ("#F47920", "#000000"),
    "CHS_COU": ("#73000A", "#000000"),
    "BRY_BUL": ("#000000", "#C8102E"),
    "LIU_SHA": ("#69BE28", "#002F6C"),
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
    "ION_GAE1": ("#6F2C91", "#FFB81C"),
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
    "UMA_AMH": ("#971B2F", "#FFFFFF"),
    "UMA_BOS": ("#0033A0", "#FFFFFF"),
    "UMBC_RET": ("#FFCC00", "#000000"),
    "UNC_SPA": ("#003366", "#FFC72C"),
    "UNL_REB": ("#BA0C2F", "#000000"),
    "USF_BUL": ("#006747", "#CFC493"),
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


def _build_usage_bars(ax: plt.Axes, arsenal: pd.DataFrame, total_pitches: int) -> None:
    """Draw horizontal pitch usage % bars onto a given axes."""
    arsenal_sorted = arsenal.sort_values("N", ascending=False).reset_index(drop=True)
    n = len(arsenal_sorted)
    if n == 0:
        return

    bar_height = 0.55
    for i, row in arsenal_sorted.iterrows():
        y = n - 1 - i
        pct = row["N"] / total_pitches if total_pitches > 0 else 0
        color = PITCH_COLORS.get(row["Pitch"], "#94A3B8")

        # Background track
        ax.barh(y, 1.0, height=bar_height, color="#1e293b", left=0, zorder=1)
        # Filled portion
        ax.barh(y, pct, height=bar_height, color=color, left=0, alpha=0.90, zorder=2)

        # Pitch label on left
        ax.text(-0.03, y, row["Pitch"], va="center", ha="right",
                color=color, fontsize=9, fontweight="bold")
        # Percentage on right
        ax.text(1.03, y, f"{pct * 100:.1f}%", va="center", ha="left",
                color="#cbd5e1", fontsize=8.5)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axis("off")


def build_player_card_png(df: pd.DataFrame, pitcher: str, team_code: str) -> bytes:
    card = pitcher_stat_card(df)
    primary, accent = team_colors(team_code)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#090B10")
    ax.set_facecolor("#090B10")
    ax.axis("off")

    # Header bar
    ax.add_patch(plt.Rectangle((0, .78), 1, .22, transform=ax.transAxes, color=primary))
    ax.text(.04, .91, pitcher, transform=ax.transAxes, color="white",
            fontsize=28, fontweight="bold", va="center")
    ax.text(.04, .835, safe_team_name(team_code), transform=ax.transAxes,
            color=accent, fontsize=15, fontweight="bold")

    # Team logo
    logo = logo_path_for_team(team_code)
    if logo:
        try:
            img = Image.open(logo)
            logo_ax = fig.add_axes([.80, .80, .13, .13])
            logo_ax.imshow(img)
            logo_ax.axis("off")
        except Exception:
            pass

    # Stat boxes
    items = ["Pitches", "Games", "IP", "K", "BB", "K%", "BB%", "BAA", "SLG",
             "Velo", "MaxVelo", "Stuff+", "Loc+", "Whiff%", "Zone%", "CSW%"]
    for i, key in enumerate(items):
        x = .045 + (i % 8) * .116
        y = .58 - (i // 8) * .25
        ax.add_patch(plt.Rectangle((x, y), .095, .15, transform=ax.transAxes,
                                    facecolor="#111827", edgecolor="#334155", linewidth=1))
        ax.text(x + .0475, y + .09, fmt(card.get(key), key), transform=ax.transAxes,
                color="#fff7ed", ha="center", fontsize=16, fontweight="bold")
        ax.text(x + .0475, y + .035, key, transform=ax.transAxes,
                color="#cbd5e1", ha="center", fontsize=8.5, fontweight="bold")

    ax.text(.04, .08, "CBBReports | College Baseball Pitching Plus | 2026 TrackMan",
            transform=ax.transAxes, color="#94a3b8", fontsize=10)

    out = BytesIO()
    fig.savefig(out, format="png", dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out.read()


def build_pitcher_summary_png(
    df: pd.DataFrame,
    pitcher: str,
    team_code: str,
    game_id: str | None = None,
    summary_type: str = "Postgame",
) -> bytes:
    if summary_type == "Season" or not game_id:
        game_df = df.copy()
        label = "Season Summary"
        date = "2026 Season"
    else:
        game_df = df[df["GameID"].astype(str).eq(str(game_id))].copy() if "GameID" in df.columns else df.copy()
        if game_df.empty:
            game_df = df.copy()
        label = "Postgame Summary"
        date = (
            game_df["Date"].dropna().astype(str).iloc[0]
            if "Date" in game_df.columns and not game_df["Date"].dropna().empty
            else "2026"
        )

    card = pitcher_stat_card(game_df)
    primary, accent = team_colors(team_code)
    total_pitches = len(game_df)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#090B10")

    # Grid: 2 rows, 3 cols — bottom row is movement | zone | usage bars
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[.75, 1.35],
        width_ratios=[1.1, 1.1, 0.8],
        hspace=.32,
        wspace=.30,
    )

    # ── Title row ────────────────────────────────────────────────────────────
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.add_patch(plt.Rectangle((0, .05), 1, .9, transform=title_ax.transAxes, color="#111827"))
    title_ax.add_patch(plt.Rectangle((0, .05), .015, .9, transform=title_ax.transAxes, color=accent))
    title_ax.text(.035, .62, pitcher, color="#fff7ed", fontsize=28,
                  fontweight="bold", transform=title_ax.transAxes)
    title_ax.text(
        .035, .30,
        f"{safe_team_name(team_code)} | {label} | {date}"
        + (f" | {game_id}" if game_id and summary_type != "Season" else ""),
        color="#cbd5e1", fontsize=13, transform=title_ax.transAxes,
    )
    for i, key in enumerate(["Pitches", "IP", "K", "BB", "Stuff+", "Loc+", "Whiff%", "Zone%", "CSW%"]):
        x = .46 + i * .055
        title_ax.text(x, .62, fmt(card.get(key), key), color="#fff7ed",
                      fontsize=16, fontweight="bold", ha="center", transform=title_ax.transAxes)
        title_ax.text(x, .34, key, color="#94a3b8", fontsize=8,
                      fontweight="bold", ha="center", transform=title_ax.transAxes)

    # ── Movement plot — FIXED ±30" axes, square aspect ────────────────────
    ax_move = fig.add_subplot(gs[1, 0])
    ax_move.set_facecolor("#111827")
    ax_move.set_xlim(-30, 30)
    ax_move.set_ylim(-30, 30)
    ax_move.set_aspect("equal", adjustable="box")
    ax_move.axhline(0, color="#475569", linestyle="--", linewidth=0.8, zorder=1)
    ax_move.axvline(0, color="#475569", linestyle="--", linewidth=0.8, zorder=1)
    if "HB" in game_df.columns and "IVB" in game_df.columns:
        for pitch, g in game_df.groupby("Pitch"):
            ax_move.scatter(
                g["HB"], g["IVB"],
                s=48, color=PITCH_COLORS.get(pitch, "#94a3b8"),
                edgecolor="white", linewidth=0.35,
                label=pitch, alpha=0.90, zorder=3,
            )
    ax_move.set_title("Movement Profile", color="#fff7ed", fontweight="bold", fontsize=11, pad=8)
    ax_move.set_xlabel("Horizontal Break (in)", color="#94a3b8", fontsize=8)
    ax_move.set_ylabel("Induced Vert Break (in)", color="#94a3b8", fontsize=8)
    ax_move.tick_params(colors="#64748b", labelsize=7)
    ax_move.set_xticks([-20, -10, 0, 10, 20])
    ax_move.set_yticks([-20, -10, 0, 10, 20])
    ax_move.grid(color="#1e293b", linewidth=0.6, zorder=0)
    ax_move.spines[:].set_edgecolor("#334155")
    ax_move.legend(
        facecolor="#0f172a", edgecolor="#334155", labelcolor="#fff7ed",
        fontsize=7.5, framealpha=0.9, loc="upper right",
    )

    # ── Zone / location plot ──────────────────────────────────────────────
    ax_zone = fig.add_subplot(gs[1, 1])
    ax_zone.set_facecolor("#111827")
    ax_zone.plot(
        [-0.83, .83, .83, -.83, -.83],
        [1.5, 1.5, 3.5, 3.5, 1.5],
        color="#fff7ed", linewidth=2,
    )
    if "PlateLocSide" in game_df.columns and "PlateLocHeight" in game_df.columns:
        for pitch, g in game_df.groupby("Pitch"):
            ax_zone.scatter(
                g["PlateLocSide"], g["PlateLocHeight"],
                s=45, color=PITCH_COLORS.get(pitch, "#94a3b8"),
                edgecolor="white", linewidth=0.35, alpha=0.88,
            )
    ax_zone.set_xlim(-2.2, 2.2)
    ax_zone.set_ylim(.5, 4.5)
    ax_zone.set_title("Pitch Locations", color="#fff7ed", fontweight="bold", fontsize=11, pad=8)
    ax_zone.tick_params(colors="#64748b", labelsize=7)
    ax_zone.grid(color="#1e293b", linewidth=0.5, alpha=0.5)
    ax_zone.spines[:].set_edgecolor("#334155")

    # ── Usage bars + arsenal table ────────────────────────────────────────
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
    arsenal = arsenal.sort_values("N", ascending=False).reset_index(drop=True)

    # Top half of right column: usage bars
    ax_bars = fig.add_subplot(gs[1, 2])
    ax_bars.set_facecolor("#111827")
    ax_bars.spines[:].set_visible(False)
    _build_usage_bars(ax_bars, arsenal, total_pitches)
    ax_bars.set_title("Usage %", color="#fff7ed", fontweight="bold", fontsize=11, pad=8)

    # Arsenal stats as a small text block below usage bars (inset axes)
    inset = ax_bars.inset_axes([0.0, -0.52, 1.0, 0.48])
    inset.set_facecolor("#111827")
    inset.axis("off")

    col_headers = ["Pitch", "Velo", "IVB", "HB", "Stf+", "Loc+", "CSW%"]
    col_x = [0.0, 0.20, 0.36, 0.52, 0.66, 0.80, 0.93]
    header_y = 1.0
    for hdr, cx in zip(col_headers, col_x):
        inset.text(cx, header_y, hdr, color="#94a3b8", fontsize=7.5,
                   fontweight="bold", ha="left", transform=inset.transAxes)

    row_h = 0.18
    for i, row in arsenal.iterrows():
        y = header_y - (i + 1) * row_h
        if y < -0.05:
            break
        color = PITCH_COLORS.get(row["Pitch"], "#94A3B8")
        vals = [
            row["Pitch"],
            fmt(row.get("Velo"), "Velo"),
            fmt(row.get("IVB"), "IVB"),
            fmt(row.get("HB"), "HB"),
            fmt(row.get("Stuff+"), "Stuff+"),
            fmt(row.get("Loc+"), "Loc+"),
            fmt(row.get("CSW%"), "CSW%"),
        ]
        for j, (val, cx) in enumerate(zip(vals, col_x)):
            txt_color = color if j == 0 else "#fff7ed"
            inset.text(cx, y, val, color=txt_color, fontsize=7.5,
                       fontweight="bold", ha="left", transform=inset.transAxes)

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
        team_code = st.selectbox("Team", teams["TeamCode"].tolist(),
                                  format_func=lambda c: f"{safe_team_name(c)} ({c})")
    team_pitchers = index[index["TeamCode"].eq(team_code)].sort_values(
        ["Pitches", "Pitcher"], ascending=[False, True]
    )
    with c2:
        pitcher = st.selectbox("Pitcher", team_pitchers["Pitcher"].tolist())
    with c3:
        view = st.radio("Graphic", ["Player Stat Card", "Postgame Summary", "Season Summary"],
                        horizontal=False)

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
        st.download_button(
            "Download Player Stat Card PNG", png,
            file_name=f"{pitcher.replace(',', '').replace(' ', '_')}_stat_card.png",
            mime="image/png", use_container_width=True,
        )
    elif view == "Postgame Summary":
        if "GameID" in df.columns:
            games = (
                df.groupby("GameID")
                .agg(Date=("Date", "first"), Pitches=("Pitch", "count"))
                .reset_index()
                .sort_values("Date")
            )
            game_id = st.selectbox(
                "Game", games["GameID"].astype(str).tolist(),
                format_func=lambda g: f"{games.loc[games['GameID'].astype(str).eq(str(g)), 'Date'].iloc[0]} | {g}",
            )
        else:
            game_id = "Season"
        png = build_pitcher_summary_png(df, pitcher, team_code, game_id, summary_type="Postgame")
        st.image(png, use_container_width=True)
        st.download_button(
            "Download Postgame Summary PNG", png,
            file_name=f"{pitcher.replace(',', '').replace(' ', '_')}_{game_id}_postgame.png",
            mime="image/png", use_container_width=True,
        )
    else:
        png = build_pitcher_summary_png(df, pitcher, team_code, summary_type="Season")
        st.image(png, use_container_width=True)
        st.download_button(
            "Download Season Summary PNG", png,
            file_name=f"{pitcher.replace(',', '').replace(' ', '_')}_season_summary.png",
            mime="image/png", use_container_width=True,
        )

    with st.expander("Team logos"):
        st.write("Add licensed PNG logos to `national_pitchingplus_app/team_logos/TEAM_CODE.png`. Missing logos use a branded color fallback.")
        logo = logo_path_for_team(team_code)
        if logo:
            st.image(str(logo), width=120)
        else:
            primary, accent = team_colors(team_code)
            st.markdown(
                f"<div style='background:{primary};color:{accent};padding:16px;border-radius:8px;"
                f"font-weight:800;width:260px'>{safe_team_name(team_code)}</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
