# fordham-pitching-app
Fordham Pitching Analyzer

# ⚾ Fordham Baseball – Pitching Analytics Platform

A full-stack, data-driven pitching analytics platform built for Fordham Baseball.  
This application processes TrackMan pitch-by-pitch data, computes advanced metrics (Stuff+, Location+, Contact Quality), and generates interactive dashboards, leaderboards, and automated reports — all inside a streamlined Streamlit interface.

The platform is designed for player development, game preparation, and postgame review, with MLB-style visualizations and automated insights.

---
PASSWORD: Baseball_1
---

# 📐 Repository Layout

| Folder | Purpose |
|--------|---------|
| **app.py** | Main Streamlit app; builds all pages and UI |
| **utils/** | Core analytics engine (cleaning, flags, models, helpers) |
| **models/** | Stuff+ and Location+ models + league averages |
| **data/** | Raw TrackMan CSVs for Fordham |
| **teamstat/** | General season pitching stats used in profiles & summaries |
| **assets/** | Logos and images used in reports |
| **static/** | Legacy static files (safe to keep) |
| **machinelearning_submission/** | ML coursework + trackman scrapers + model makers + model evaluators + final writting PDF + example data |
| **.streamlit/** | Theme + layout configuration |
| **.devcontainer/** | Reproducible development environment |

---

# 🚀 Features

## **📊 Summaries**

### **Postgame Summary**
- Full game report with:
  - Movement plots  
  - Release maps  
  - Stuff+ / Location+  
  - LHH/RHH splits  
  - Batted-ball outcomes  
- High-resolution PNG export

### **Season Summary**
- Aggregated Stuff+ and Location+ across all games  
- Season-long movement and release visuals  

### **Pitcher Profile**
- Movement clusters  
- Release consistency  
- Pitch metrics  
- Season stats  
- Count-based performance  

---

## **🏆 Leaderboards**
### **Stuff+ Leaderboard**
Ranks all FOR_RAM pitchers by average Stuff+.

### **Location+ Leaderboard**
Ranks pitchers by command quality.

### **Pitch-Type Grids**
2×3 grid of Stuff+ and Location+ leaderboards by pitch type.

### **Contact Quality Leaderboard**
Hard-hit %, average EV, and batted-ball performance.

---

## **🛠 Tools**
### **Pitcher Development & Sequencing**
A full development engine including:

- Arsenal overview  
- Count-based effectiveness (Whiff%, Chase%, Zone%, CSW%, K%, HardHit%, AvgEV)  
- Pitch-to-pitch sequencing  
- LHH/RHH splits  
- Release consistency  
- Automated development recommendations  

### **Umpire Scorecard Generator**
Produces an MLB-style scorecard with:

- Overall accuracy  
- Strike accuracy  
- Ball accuracy  
- Team bias summary  
- Touch-zone visualization  
- Missed call table  
- High-resolution PNG export  

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/cjones55/fordham-pitching-app.git
cd fordham-pitching-app



