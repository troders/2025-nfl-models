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
# Load Existing Dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset {DATA_FILE} not found!")

df = pd.read_csv(DATA_FILE)

# Just check for the columns we actually need
required_cols = ["Season", "Week", "Date", "Team", "Opponent", 
                 "Spread", "Total", "Home", "Win"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Column {c} missing in dataset!")

# -------------------------------
# Get current week / last completed week
today = datetime.now(timezone.utc)
season_start = datetime(2025, 9, 4, tzinfo=timezone.utc)  # NFL 2025 kickoff
days_since_start = (today - season_start).days
current_week = max(1, days_since_start // 7 + 1)
last_completed_week = current_week - 1

print(f"📅 Today={today}, Current={current_week}, LastCompleted={last_completed_week}")

# -------------------------------
# Train Model on cleaned data
df_clean = df.dropna(subset=["Spread", "Total", "Home", "Win"])
X = df_clean[["Spread", "Total", "Home"]]
y = df_clean["Win"]

model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X, y)

# Save model file
joblib.dump(model, MODEL_FILE)
print(f"✅ Model retrained and saved to {MODEL_FILE}")

# -------------------------------
# Evaluate last completed week
if last_completed_week > 0:
    df_last = df_clean[df_clean["Week"] == last_completed_week]
    if len(df_last) > 0:
        X_last = df_last[["Spread", "Total", "Home"]]
        y_last = df_last["Win"]

        preds = model.predict(X_last)
        correct = (preds == y_last).sum()
        total = len(y_last)
        record = f"{correct}-{total - correct}"

        print(f"📊 Last Week (Week {last_completed_week}) record: {record}")
    else:
        print(f"ℹ️ No rows found for Week {last_completed_week}.")
