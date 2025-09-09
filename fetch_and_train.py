# fetch_and_train.py
import nflfastpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load most recent NFL season data (adjust year as needed)
season = 2025
schedule = nflfastpy.load_schedule([season])

# Step 2: Make sure spreads are available
df = schedule[["week", "home_team", "away_team", "spread_line", "total_line", "result"]].copy()

# Small cleanup: spread_line can be NaN for future games
df = df.dropna(subset=["spread_line", "result"])

# Step 3: Very simple model: predict win/loss using spread only
df["home_win"] = (df["result"] > 0).astype(int)

X = df[["spread_line"]]  # features (could add more later)
y = df["home_win"]       # target

# Step 4: Train model
model = RandomForestClassifier()
model.fit(X, y)

# Step 5: Save it so Streamlit app can load it
joblib.dump(model, "nfl_model.pkl")

print("✅ Model updated and saved as nfl_model.pkl")
