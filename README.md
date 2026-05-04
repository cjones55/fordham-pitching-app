# ⚾ Fordham Baseball – Pitching & Hitting Analytics Platform

https://fordhampitchingplus.streamlit.app/

A full‑stack, data‑driven analytics system built for **Fordham Baseball**.  
This platform ingests TrackMan pitch‑by‑pitch data, computes advanced metrics (Stuff+, Location+, Contact Quality), and generates MLB‑grade visuals, scouting reports, and development insights — all inside a streamlined Streamlit interface.

Designed for:
- Player development  
- Game preparation  
- Opponent scouting  
- Postgame review  
- Practice tracking  
- Staff collaboration  

---

## 🔐 Password
Baseball_1

---

# 🗂 Repository Layout

| Folder | Purpose |
|--------|---------|
| **app.py** | Main Streamlit application; builds all pages and UI |
| **utils/** | Core analytics engine (cleaning, flags, models, helpers, calculators) |
| **models/** | Stuff+ and Location+ models + league averages |
| **data/** | Raw TrackMan game files (auto‑updated every 24 hours) |
| **practice_data/** | Manually uploaded bullpens, BP, and scrimmage sessions |
| **teamstat/** | Season‑long pitching stats for profiles & summaries |
| **assets/** | Logos, icons, and report graphics |
| **scouting_2026_trackman/** | SFTP‑imported scouting data for D1 opponents |
| **machinelearning_submission/** | ML coursework, scrapers, model builders, evaluators, and final paper |
| **scripts/** | Automated TrackMan updater + SFTP import scripts |
| **static/** | Legacy static files |
| **.streamlit/** | Theme + layout configuration |
| **.devcontainer/** | Reproducible development environment |
| **requirements.txt** | Python dependencies |

---

# 🚀 Platform Overview

The application is organized into six major modules, accessible from the top navigation bar:

**Reports | Leaderboards | Development | Practice | Scouting Zone | Glossary**

Each module is described below.

---

# 📊 1. Reports

## Postgame Summary
A full MLB‑style game report including:
- Pitch movement charts  
- Release point maps  
- Stuff+ / Location+  
- LHH/RHH splits  
- Batted‑ball outcomes  
- Count‑based performance  
- High‑resolution PNG export  

## Season Summary
Aggregated season‑long analytics:
- Stuff+ and Location+ trends  
- Movement & release consistency  
- Pitch‑type summaries  
- Contact quality over time  

## Pitcher Profile
A complete pitcher breakdown:
- Movement clusters  
- Release consistency  
- Arsenal overview  
- Count‑based performance  
- Season stats  
- LHH/RHH splits  
- Contact quality  

---

# 🏆 2. Leaderboards

## Stuff+ Leaderboard
Ranks all FOR_RAM pitchers by average Stuff+.

## Location+ Leaderboard
Ranks pitchers by command quality.

## Pitch‑Type Grids
2×3 grid of Stuff+ and Location+ leaderboards by pitch type.

## Contact Quality Leaderboard
Hard‑hit %, average EV, and batted‑ball performance.

---

# 🛠 3. Development Engine

A full pitcher development and sequencing module including:
- Arsenal overview  
- Count‑based effectiveness (Whiff%, Chase%, Zone%, CSW%, K%, HardHit%, AvgEV)  
- Pitch‑to‑pitch sequencing  
- LHH/RHH splits  
- Release consistency  
- Automated development recommendations  

Built for bullpen work, pitch design, and in‑season adjustments.

---

# 🟡 4. Practice Review

A dedicated workflow for **bullpens, BP, and scrimmages**.

### Manual Uploads
Coaches can upload any practice‑day TrackMan CSV:
- Bullpens  
- Live BP  
- Intrasquad scrimmages  

Uploaded data is processed through the **same engine** as NCAA game files:
- Stuff+  
- Location+  
- Movement  
- Release metrics  
- Contact quality  
- Pitch‑type summaries  

### Automation
- **NCAA game data → auto‑updated every 24 hours**  
- **Practice data → manual upload only**  

This separation keeps practice sessions flexible and game data official.

---

# 🟣 5. Scouting Zone

A full scouting‑report engine for **any team or any player in Division I baseball**.

### Capabilities
- Search any D1 team or player  
- Pull TrackMan, SFTP‑imported, or manually uploaded scouting data  
- Auto‑compute pitch metrics, tendencies, and contact quality  
- Generate **clean, printable PDF scouting reports**  
- Include visuals:
  - Heatmaps  
  - Pitch usage charts  
  - Movement plots  
  - Release maps  
  - Attack‑plan summaries  

### Use Cases
- Opponent scouting  
- Weekend series preparation  
- Individual hitter/pitcher breakdowns  
- Staff meeting packets  
- Player development tracking  

### Report Contents
- Player bio + handedness  
- Arsenal overview  
- Stuff+ / Location+  
- Movement + release visuals  
- Count tendencies  
- LHH/RHH splits  
- Contact quality  
- Attack plan recommendations  

---

# 📡 6. Automated Data Pipelines

### Daily TrackMan Updater
A scheduled script that:
- Pulls new NCAA game files every 24 hours  
- Cleans and standardizes data  
- Updates season summaries and leaderboards  

### SFTP Scouting Import
A secure pipeline for importing opponent scouting data:
- Connects to remote SFTP  
- Downloads scouting TrackMan files  
- Auto‑processes into the Scouting Zone module  

---

# 🎨 Visual System

The platform uses MLB‑grade visuals:
- Pitcher‑view heatmaps  
- Strike zone outlines  
- Movement scatter plots  
- Release point clusters  
- Clean maroon/gold Fordham branding  
- High‑resolution PNG exports  

---

# 🧪 Machine Learning Folder

The `machinelearning_submission/` directory includes:
- TrackMan scrapers  
- Model builders (Stuff+, Location+)  
- Model evaluators  
- Coursework notebooks  
- Final written PDF  
- Example datasets  

This folder is preserved for academic reference and reproducibility.

---

# 🛠 Installation

Clone the repository: 
git clone https://github.com/cjones55/fordham-pitching-app.git
cd fordham-pitching-app

Install dependencies: 

pip install -r requirements.txt


Run the app:

streamlit run app.py

---

# 👥 Contributors
- **Chris Jones** — Developer, Data Scientist, Fordham Baseball Analytics  
- Fordham Baseball Coaching Staff  

---

# 📄 License
Internal use for Fordham Baseball.





