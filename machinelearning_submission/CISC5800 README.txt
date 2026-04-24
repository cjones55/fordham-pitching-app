Machine Learning Submission — README
This folder contains all of the core materials for my CISC 5800 Machine Learning Final Project, which focuses on building Division‑I–calibrated Stuff+ and Location+ models using more than ten million TrackMan‑tracked pitches from NCAA baseball. Everything in this directory represents the full machine learning workflow: raw data samples, ingestion scripts, model‑building code, evaluation scripts, and the final IEEE‑style project report.

📁 Folder Purpose
The machinelearning_submission directory is a self‑contained snapshot of the machine learning portion of the project. It includes:

Raw TrackMan CSV samples used for demonstration and testing

Python scripts for model training, evaluation, and data ingestion

Model‑specific code for Stuff+ and Location+

The final written report submitted for the course

Documentation describing the project and workflow

This folder is intentionally separated from the main Streamlit app so the ML submission stands alone for academic review.

📄 Files Overview
1. Raw Data Samples (CSV files)
These CSVs are small, representative samples of the much larger private TrackMan dataset used in the project.
Examples include:

20250201-BobSteinStadium-1.csv

20250201-LeeUniversity-2.csv

20250201-LimestoneUniversity-1.csv

20250201-UofMontevallo-1.csv

etc.

These files show the schema, feature availability, and raw pitch‑level structure used during ingestion and model training.
The full dataset (10M+ pitches) cannot be shared due to NCAA and program‑level privacy restrictions.

2. trackman_data_scrapper.py
This script handles FTP ingestion from Fordham’s TrackMan FileZilla server.
It performs:

secure connection

recursive directory scanning

CSV downloading

file integrity checks

local organization by season/game

This is the first step of the full pipeline.

3. cjstuff+_model_designer.py
This script builds the Stuff+ model, which predicts the probability of generating a called or swinging strike (CSW) using only ball‑flight characteristics.

Key components:

feature engineering (velo, IVB, HB, spin, release metrics, VAA/HAA)

LightGBM classifier training

hyperparameter tuning

scaling logic to convert raw probabilities → Stuff+ (centered at 100)

4. location+ model maker.py
This script builds the Location+ model, which predicts the expected run value of a pitch based on its location and count.

Includes:

zone‑based features

count leverage

pitch type encoding

LightGBM regression

scaling to Location+ (centered at 100)

5. stuff+ eval.py
Evaluation script for the Stuff+ model.

Outputs:

accuracy

precision

recall

F1

ROC–AUC

log loss

calibration behavior

This script validates that the model meaningfully distinguishes high‑quality pitches.

6. loc+ eval.py
Evaluation script for the Location+ model.

Outputs:

RMSE

MAE

R²

baseline comparisons

distribution checks

This confirms that the model captures meaningful patterns in pitch location and run value.

7. Chris Jones CISC5800 Final Project IEEE Writing.pdf
This is the final written report for the course, formatted in IEEE conference style.
It includes:

introduction

related work

data description

methodology

model evaluation

results

system design

discussion

conclusion

This PDF is the formal academic deliverable for the project.

8. CISC5800 README.txt
A short text file summarizing the project for the course submission.
The Markdown README (this file) is the expanded, more complete version.

🧠 Summary of the ML Workflow
Ingest TrackMan CSVs from FileZilla

Engineer features for Stuff+ and Location+

Train LightGBM models

Scale predictions to Division‑I–calibrated “plus” metrics

Evaluate models using standard ML metrics

Export results for use in the Streamlit analytics app

This folder contains all the code and sample data needed to reproduce the machine learning portion of the project.

📌 Notes on Data Privacy
The full TrackMan dataset cannot be included due to NCAA and program‑level restrictions.
Only representative samples are included here.
All modeling code is fully reproducible with any TrackMan‑formatted dataset.

✔️ What This Folder Demonstrates
Real‑world machine learning on a massive sports dataset

End‑to‑end pipeline design

Feature engineering for ball‑flight physics

Model training and evaluation

Reproducible academic documentation

Integration with a deployed analytics application

This folder is the complete ML submission for the course.
