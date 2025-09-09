# fetch_and_train.py
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import datetime, timezone
import nflfastpy

DATA_FILE = os.path.join("datasets", "nfl_gamelogs_vegas_2015-2025_FINAL.csv")
MODEL_FILE = "nfl_model.pkl"

# -------------------------------
# 1. Load existing CSV
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"CSV missing: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required_cols = ["Season","Week","Date","Team","Opp","Home","Win","Spread","Total"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing {c} in CSV. Found {df.columns.tolist()}")

print(f"📂 Loaded {DATA_FILE}, shape={df.shape}")

# -------------------------------
# 2. Pull schedule/results from NFLVerse
this_season = 2025
schedule = nflfastpy.load_schedule([this_season])

rows = []
for _, g in schedule.iterrows():
    # Only finished games
    if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
        continue

    spread = g.get("spread_line", None)
    total = g.get("total_line", None)

    # Home row
    rows.append({
        "Season": this_season,
        "Week": g["week"],
        "Date": g["gameday"],
        "Team": g["home_team"],
        "Opp": g["away_team"],
        "Home": 1,
        "Win": 1 if g["home_score"] > g["away_score"] else 0,
        "Spread": spread,
        "Total": total
    })
    # Away row
    rows.append({
        "Season": this_season,
        "Week": g["week"],
        "Date": g["gameday"],
        "Team": g["away_team"],
        "Opp": g["home_team"],
        "Home": 0,
        "Win": 1 if g["away_score"] > g["home_score"] else 0,
        "Spread": -spread if spread is not None else None,
        "Total": total
    })

df_new = pd.DataFrame(rows)

# -------------------------------
# 3. Merge into existing CSV, avoid duplicates
before = len(df)
df = pd.concat([df, df_new], ignore_index=True)
df = df.drop_duplicates(subset=["Season","Week","Team"], keep="last")
after = len(df)

added = after - before
print(f"🆕 Added {added} new rows, total {after}")

df.to_csv(DATA_FILE, index=False)
print(f"💾 Updated CSV written to {DATA_FILE}")

# -------------------------------
# 4. Train Logistic Regression
df_train = df.dropna(subset=["Spread","Total","Home","Win"])
X = df_train[["Spread","Total","Home"]]
y = df_train["Win"]

model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X,y)

joblib.dump(model, MODEL_FILE)
print(f"✅ Trained + saved model at {MODEL_FILE}")

# -------------------------------
# 5. Show last completed week record
today = datetime.now(timezone.utc)
season_start = datetime(2025,9,4,tzinfo=timezone.utc)
days_since = (today - season_start).days
current_week = max(1, days_since//7 + 1)
last_week = current_week - 1

df_last = df_train[df_train["Week"] == last_week]
if not df_last.empty:
    preds = model.predict(df_last[["Spread","Total","Home"]])
    correct = (preds == df_last["Win"]).sum()
    total = len(df_last)
    print(f"📊 Last Week {last_week} record: {correct}-{total-correct}")
else:
    print(f"ℹ️ No completed games for Week {last_week} yet")
