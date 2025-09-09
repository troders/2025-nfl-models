# fetch_and_train.py
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import datetime, timezone

# -------------------------------
# Paths
DATA_FILE = os.path.join("datasets", "nfl_gamelogs_vegas_2015-2025_FINAL.csv")
MODEL_FILE = "nfl_model.pkl"

# -------------------------------
# Step 1: Load Existing Dataset
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    raise FileNotFoundError(f"Dataset {DATA_FILE} not found!")

# Ensure necessary columns exist
required_cols = ["Season", "Week", "Date", "Team", "Opponent",
                 "Spread", "Total", "Home", "Win"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Column {c} is missing in dataset. Check format!")

# -------------------------------
# Step 2: Check if we need to append new week results
# (Here we just trust the dataset gets updated manually by weekly data collection,
# but if you want real pull, you could integrate NFLVerse API here.)

today = datetime.now(timezone.utc)
season_start = datetime(2025, 9, 4, tzinfo=timezone.utc)  # adjust NFL kickoff
days_since_start = (today - season_start).days
current_week = max(1, days_since_start // 7 + 1)
last_completed_week = current_week -
