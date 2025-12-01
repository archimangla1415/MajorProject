MajorProject

Authors:
Archi Mangla
Nehal Pruthi

This repository contains all the code and files for our Major Project, including the Streamlit dashboard, phase-wise scripts, and the cleaned dataset. The structure is straightforward and easy to follow.

Project Overview

This project includes:

A Streamlit-based dashboard for visualizing and exploring the dataset

Python scripts divided into phases for:

Data cleaning

Model training

Evaluation

Final output preparation

You can run the scripts independently or use the dashboard to interact with the results.

Project Structure
MajorProject/
│
├─ dashboardstreamlitcode.py         # Main Streamlit dashboard
├─ phase1code.py                     # Data cleaning and EDA
├─ phase2code.py                     # Model building
├─ phase3code.py                     # Model evaluation and tuning
├─ phase4code.py                     # Final outputs and processing
│
├─ combined_year_dataset_cleaned.xlsx   # Cleaned dataset used in the project
├─ Educational Intelligence Platform_Sonia_2.pdf
│
└─ README.md

Requirements

Make sure Python is installed.
Typical libraries used:

streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
openpyxl
joblib


Install everything with:

pip install -r requirements.txt


(You can create a requirements.txt using this list.)

How to Run the Project
Run the Streamlit Dashboard

From the project folder:

streamlit run dashboardstreamlitcode.py

Run Phase-wise Scripts

Each script can be run individually:

python phase1code.py
python phase2code.py
python phase3code.py
python phase4code.py

Dataset

The repository includes the cleaned dataset file combined_year_dataset_cleaned.xlsx, which is used across all phases and the dashboard.

Notes

Make sure the dataset file is in the same directory as the scripts.

If any paths inside the scripts need updating, adjust them accordingly.
