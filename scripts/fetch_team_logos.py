#!/usr/bin/env python3
"""
Download team logo PNGs from ESPN for every team in the CBB Plus app.

Run from the repo root:
    python3 scripts/fetch_team_logos.py

Logos are saved to national_pitchingplus_app/team_logos/<TEAM_CODE>.png
Already-downloaded logos are skipped unless --force is passed.
"""

import re
import sys
import time
import argparse
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGO_DIR  = REPO_ROOT / "national_pitchingplus_app" / "team_logos"
LOGO_DIR.mkdir(exist_ok=True)

# ── Full team name dict (same as CBB Plus app) ────────────────────────────────
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
    "SBU_BON":"St. Bonaventure Bonnies","JOE_HAW":"Saint Joseph's Hawks",
    "DUQ_DUK":"Duquesne Dukes","LOY_RAM":"Loyola Chicago Ramblers",
    "UMASS":"Massachusetts Minutemen","COL_LION":"Columbia Lions",
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
    "ARI_WIL":"Arizona Wildcats",
    "AIR_FOR":"Air Force Falcons","ARM_BLA":"Army Black Knights",
    "NAV_MID":"Navy Midshipmen","XAV_MUS":"Xavier Musketeers",
    "AUB_TIG":"Auburn Tigers","NEV_WOL":"Nevada Wolf Pack",
    "UCF_KNI":"UCF Knights","HOU_COG":"Houston Cougars","HOU_COU":"Houston Cougars",
    "YAL_BUL":"Yale Bulldogs","NOR_TAR":"North Carolina Tar Heels",
    "NOR_WOL":"NC State Wolfpack","ORE_DUC":"Oregon Ducks",
    "UCO_HUS":"UConn Huskies","CON_HUS":"UConn Huskies",
    "BOS_COL":"Boston College Eagles","DEL_BLU":"Delaware Blue Hens",
    "HOF_PRI":"Hofstra Pride","DRE_DRA":"Drexel Dragons",
    "NOR_HUS":"Northeastern Huskies","ELON_PHO":"Elon Phoenix",
    "CAM_CAM":"Campbell Camels","CHS_COU":"Charleston Cougars",
    "BRY_BUL":"Bryant Bulldogs","LIU_SHA":"LIU Sharks",
    "MER_WAR":"Merrimack Warriors","MAI_BLA":"Maine Black Bears",
    "ALB_GRE":"UAlbany Great Danes","UML_RIV":"UMass Lowell River Hawks",
    "LEH_MOU":"Lehigh Mountain Hawks","LAF_LEO":"Lafayette Leopards",
    "BUC_BIS":"Bucknell Bison","HOL_CRO":"Holy Cross Crusaders",
    "COL_GAT":"Colgate Raiders","RID_BRO":"Rider Broncs",
    "OLD_DOM":"Old Dominion Monarchs","CHA_49E":"Charlotte 49ers",
    "LIB_FLA":"Liberty Flames","APP_MOU":"App State Mountaineers",
    "WVU_MOU":"West Virginia Mountaineers","GEO_STA":"Georgia State Panthers",
    "COA_CHA":"Coastal Carolina Chanticleers","UMBC_RET":"UMBC Retrievers",
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
    "BUT_BUL":"Butler Bulldogs","MIL_UNI":"Milwaukee Panthers",
    "WIN_BUL":"Winthrop Eagles","BIN_BEA":"Binghamton Bearcats",
    "OKL_SOO":"Oklahoma Sooners","NEB":"Nebraska Cornhuskers",
    "IU":"Indiana Hoosiers","RUT_SCA":"Rutgers Scarlet Knights",
    "MIN_GOL":"Minnesota Golden Gophers","MIC_WOL":"Michigan Wolverines",
    "MIC_SPA":"Michigan State Spartans","OSU_BUC":"Ohio State Buckeyes",
    "PUR_BOI":"Purdue Boilermakers","ILL_ILL":"Illinois Fighting Illini",
}

# Manual overrides for tricky name matches (team code → ESPN display name)
MANUAL_OVERRIDES = {
    "UCLA":     "UCLA Bruins",
    "NEB":      "Nebraska Cornhuskers",
    "IU":       "Indiana Hoosiers",
    "FGCU":     "Florida Gulf Coast Eagles",
    "UNCW":     "UNC Wilmington Seahawks",
    "UNC_SEA":  "UNC Wilmington Seahawks",
    "VCU_RAM":  "VCU Rams",
    "UAB_BLA":  "UAB Blazers",
    "MTSU_BLU": "Middle Tennessee Blue Raiders",
    "ULM_WAR":  "Louisiana Monroe Warhawks",
    "ECU_PIR":  "East Carolina Pirates",
    "HOU_COG":  "Houston Cougars",
    "HOU_COU":  "Houston Cougars",
    "GIT_YEL":  "Georgia Tech Yellow Jackets",
    "OLE_REB":  "Ole Miss Rebels",
    "SOU_MIS":  "Southern Miss Golden Eagles",
    "ETS_BUC":  "ETSU Buccaneers",
    "COA_CHA":  "Coastal Carolina Chanticleers",
    "GEO_GWI":  "George Washington Revolutionaries",
    "RIC_OWL":  "Rice Owls",
    "GEO_PAT":  "George Mason Patriots",
    "LIB_FLA":  "Liberty Flames",
    "APP_MOU":  "Appalachian State Mountaineers",
    "WVU_MOU":  "West Virginia Mountaineers",
    "NIU_HUS":  "Northern Illinois Huskies",
    "CHA_49E":  "Charlotte 49ers",
    "BOC_EAG":  "Boston College Eagles",
    "BOS_COL":  "Boston College Eagles",
    "NOR_WOL":  "NC State Wolfpack",
    "NOR_TAR":  "North Carolina Tar Heels",
    "NOR_HUS":  "Northeastern Huskies",
    "NOR_AGG":  "North Carolina A&T Aggies",
    "UCO_HUS":  "Connecticut Huskies",
    "CON_HUS":  "Connecticut Huskies",
    "UMASS_RIV":"UMass Lowell River Hawks",
    "UML_RIV":  "UMass Lowell River Hawks",
    "LOW_RIV":  "UMass Lowell River Hawks",
    "ALB_GRE":  "Albany Great Danes",
    "ALB_DAN":  "Albany Great Danes",
    "STA_CAR":  "Stanford Cardinal",
    "TCU_HFG":  "TCU Horned Frogs",
    "TEX_BOB":  "Texas State Bobcats",
    "TEX_LON":  "Texas Longhorns",
    "TEX_RAI":  "Texas Tech Red Raiders",
    "PEN_NIT":  "Penn State Nittany Lions",
    "PEN_QUA":  "Penn Quakers",
    "CSU_BAK":  "Cal State Bakersfield Roadrunners",
    "CAL_FUL":  "Cal State Fullerton Titans",
    "CAL_MUS":  "Cal Poly Mustangs",
    "CAL_ANT":  "UC Irvine Anteaters",
    "CAL_AGO":  "UC Davis Aggies",
    "CAL_BEA":  "California Golden Bears",
    "SAN_GAU":  "UC Santa Barbara Gauchos",
    "GRA_CAN":  "Grand Canyon Lopes",
    "OKL_COW":  "Oklahoma State Cowboys",
    "OKL_SOO":  "Oklahoma Sooners",
    "MIZ_TIG":  "Missouri Tigers",
    "MSU_BDG":  "Mississippi State Bulldogs",
    "ARI_SUN":  "Arizona State Sun Devils",
    "ARI_WIL":  "Arizona Wildcats",
    "FLO_SEM":  "Florida State Seminoles",
    "LOU_CAJ":  "Louisiana Ragin' Cajuns",
    "LOU_CAR":  "Louisville Cardinals",
    "PIT_PAN":  "Pittsburgh Panthers",
    "UNC_SPA":  "UNC Greensboro Spartans",
    "HIG_PAN":  "High Point Panthers",
    "GEO_EAG":  "Georgia Southern Eagles",
    "GEO_SOU":  "Georgia Southern Eagles",
    "GEO_HOY":  "Georgetown Hoyas",
    "GEO_PAN":  "Georgia State Panthers",
    "GEO_STA":  "Georgia State Panthers",
    "GEO_BUL":  "Georgia Bulldogs",
    "SAC_PIO":  "Sacred Heart Pioneers",
    "SAC_HOR":  "Sacramento State Hornets",
    "ION_GAL":  "Iona Gaels",
    "ION_GAE":  "Iona Gaels",
    "STM_GAE":  "Saint Mary's (CA) Gaels",
    "MAR_RED":  "Marist Red Foxes",
    "LOY_LIO":  "Loyola Marymount Lions",
    "LOY_RAM":  "Loyola Chicago Ramblers",
    "MAN_JAS":  "Manhattan Jaspers",
    "SIE_SAI":  "Siena Saints",
    "MON_HAW":  "Monmouth Hawks",
    "QUI_BOB":  "Quinnipiac Bobcats",
    "RID_BRO":  "Rider Broncs",
    "SPU_PEA":  "Saint Peter's Peacocks",
    "FAI_STA":  "Fairfield Stags",
    "WAG_SEA":  "Wagner Seahawks",
    "NIA_EAG":  "Niagara Purple Eagles",
    "CAN_GRI":  "Canisius Golden Griffins",
    "SAI_BIL":  "Saint Louis Billikens",
    "SAI_JOE":  "Saint Joseph's Hawks",
    "JOE_HAW":  "Saint Joseph's Hawks",
    "STJ_HAW":  "Saint Joseph's Hawks",
    "STJ_RED":  "St. John's Red Storm",
    "SBU_BON":  "St. Bonaventure Bonnies",
    "SBU_SEA":  "Stony Brook Seawolves",
    "DUQ_DUK":  "Duquesne Dukes",
    "JMU_DUK":  "James Madison Dukes",
    "LAS_EXP":  "La Salle Explorers",
    "RHO_RAM":  "Rhode Island Rams",
    "RIC_SPI":  "Richmond Spiders",
    "DAY_FLY":  "Dayton Flyers",
    "VCU_RAM":  "VCU Rams",
    "DAV_WIL":  "Davidson Wildcats",
    "BIN_BEA":  "Binghamton Bearcats",
    "SET_PIR":  "Seton Hall Pirates",
    "FDU_KNI":  "Fairleigh Dickinson Knights",
    "LIU_SHA":  "LIU Sharks",
    "MER_WAR":  "Merrimack Warriors",
    "BRY_BUL":  "Bryant Bulldogs",
    "COR_BRE":  "Cornell Big Red",
    "COL_LION": "Columbia Lions",
    "PRI_TIG":  "Princeton Tigers",
    "YAL_BUL":  "Yale Bulldogs",
    "PEN_QUA":  "Penn Quakers",
    "HOL_CRO":  "Holy Cross Crusaders",
    "LAF_LEO":  "Lafayette Leopards",
    "LEH_MOU":  "Lehigh Mountain Hawks",
    "BUC_BIS":  "Bucknell Bison",
    "COL_GAT":  "Colgate Raiders",
    "ARM_BLA":  "Army West Point Black Knights",
    "NAV_MID":  "Navy Midshipmen",
    "AIR_FOR":  "Air Force Falcons",
    "DEL_BLU":  "Delaware Blue Hens",
    "HOF_PRI":  "Hofstra Pride",
    "DRE_DRA":  "Drexel Dragons",
    "ELON_PHO": "Elon Phoenix",
    "CAM_CAM":  "Campbell Camels",
    "CHS_COU":  "Charleston Cougars",
    "TOW_TIG":  "Towson Tigers",
    "IND_SYC":  "Indiana State Sycamores",
    "MUR_RAC":  "Murray State Racers",
    "SIU_SAL":  "Southern Illinois Salukis",
    "WRI_RAI":  "Wright State Raiders",
    "WIN_BUL":  "Winthrop Eagles",
    "CEN_MIC":  "Central Michigan Chippewas",
    "OHIO_BOB": "Ohio Bobcats",
    "AKR_ZIP":  "Akron Zips",
    "NEV_WOL":  "Nevada Wolf Pack",
    "FRE_BUL":  "Fresno State Bulldogs",
    "SAN_AZT":  "San Diego State Aztecs",
    "SAN_TOR":  "San Diego Toreros",
    "SAN_DON":  "San Francisco Dons",
    "POR_PIL":  "Portland Pilots",
    "GON_BUL":  "Gonzaga Bulldogs",
    "PEP_WAV":  "Pepperdine Waves",
    "BUT_BUL":  "Butler Bulldogs",
    "CRE_BLU":  "Creighton Bluejays",
    "XAV_MUS":  "Xavier Musketeers",
    "GEO_HOY":  "Georgetown Hoyas",
    "MIL_UNI":  "Milwaukee Panthers",
    "UIC_FLA":  "Illinois-Chicago Flames",
    "TRO_T":    "Troy Trojans",
    "SAL_JAG":  "South Alabama Jaguars",
    "MAI_BLA":  "Maine Black Bears",
    "UMBC_RET": "UMBC Retrievers",
    "CSU_BAK":  "Cal State Bakersfield Roadrunners",
    "ETS_BUC":  "East Tennessee State Buccaneers",
    "CHA_49E":  "Charlotte 49ers",
    "LIP_BIS":  "Lipscomb Bisons",
    "USF_BUL":  "South Florida Bulls",
    "HAW_WAR":  "Hawaii Warriors",
    "STE_HAT":  "Stetson Hatters",
    "FGCU":     "Florida Gulf Coast Eagles",
    "HIG_PAN":  "High Point Panthers",
}


def load_app_team_names() -> dict[str, str]:
    """Use the live CBB Plus team dictionary so this script does not drift."""
    app_path = REPO_ROOT / "national_pitchingplus_app" / "app.py"
    try:
        text = app_path.read_text()
        end = text.index("BG   =")
        ns = {"__file__": str(app_path)}
        exec(text[:end], ns)
        names = ns.get("TEAM_NAMES", {})
        conferences = ns.get("TEAM_CONFERENCES", {})
        return {
            code: name
            for code, name in names.items()
            if code in conferences and isinstance(name, str) and name.strip()
        }
    except Exception as exc:
        print(f"Warning: could not read CBB Plus team dictionary ({exc}); using script fallback.")
        return TEAM_NAMES


def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\b(university|college|state|the|of|at|a&m|a&t|&)\b", "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return " ".join(name.split())


def word_overlap_score(a: str, b: str) -> float:
    wa, wb = set(normalize(a).split()), set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def fetch_espn_teams() -> list[dict]:
    url = ("http://site.api.espn.com/apis/site/v2/sports/baseball/"
           "college-baseball/teams?limit=1000")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    sports  = data.get("sports", [])
    leagues = sports[0].get("leagues", []) if sports else []
    raw     = leagues[0].get("teams", []) if leagues else []
    teams   = []
    for entry in raw:
        t     = entry.get("team", {})
        logos = t.get("logos", [])
        if logos:
            teams.append({
                "id":          t.get("id"),
                "displayName": t.get("displayName", ""),
                "logo":        logos[0]["href"],
            })
    return teams


def find_best_match(target_name: str, espn_teams: list[dict]) -> tuple[dict | None, float]:
    best, best_score = None, 0.0
    for t in espn_teams:
        score = word_overlap_score(target_name, t["displayName"])
        if score > best_score:
            best_score, best = score, t
    return best, best_score


def download_logo(url: str, path: Path) -> bool:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return True
        return False
    except Exception as e:
        print(f"    Error downloading: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-download logos that already exist")
    parser.add_argument("--min-score", type=float, default=0.35,
                        help="Minimum name match score (0-1, default 0.35)")
    args = parser.parse_args()

    print("Fetching ESPN college baseball team list...")
    espn_teams = fetch_espn_teams()
    print(f"Found {len(espn_teams)} ESPN teams\n")

    # Deduplicate team codes (some have multiple TrackMan codes for same school)
    seen_names: dict[str, str] = {}   # display_name → code (first wins for download)
    unique_codes: dict[str, str] = {} # code → display_name to look up

    for code, name in load_app_team_names().items():
        lookup_name = MANUAL_OVERRIDES.get(code, name)
        unique_codes[code] = lookup_name

    downloaded = skipped = failed = 0

    for code, lookup_name in sorted(unique_codes.items()):
        out_path = LOGO_DIR / f"{code}.png"

        # Skip if already downloaded (unless --force)
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        # If a sibling code already downloaded this same school, just copy it
        if lookup_name in seen_names and not args.force:
            src = LOGO_DIR / f"{seen_names[lookup_name]}.png"
            if src.exists():
                out_path.write_bytes(src.read_bytes())
                print(f"  [COPY ] {code} ← {seen_names[lookup_name]}")
                downloaded += 1
                continue

        match, score = find_best_match(lookup_name, espn_teams)

        if score < args.min_score or not match:
            print(f"  [SKIP ] {code:12s}  {lookup_name!r:<40s}  no match (best {score:.2f})")
            failed += 1
            continue

        print(f"  [GET  ] {code:12s}  {lookup_name!r:<40s}  → {match['displayName']!r} ({score:.2f})")
        if download_logo(match["logo"], out_path):
            seen_names[lookup_name] = code
            downloaded += 1
        else:
            print(f"           Download failed")
            failed += 1

        time.sleep(0.15)   # be polite to ESPN's CDN

    print(f"\nDone: {downloaded} downloaded/copied, {skipped} already existed, {failed} failed/skipped")
    print(f"Logos saved to: {LOGO_DIR}")


if __name__ == "__main__":
    main()
