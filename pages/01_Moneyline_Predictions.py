# pages/01_Moneyline_Predictions.py
import streamlit as st
import requests
import pandas as pd
import os
import joblib
from datetime import datetime, timezone

st.set_page_config(page_title="🏈 NFL Predictions", page_icon="🏈")
st.title("🏈 NFL Moneyline Predictions")

# -------------------------------
# CONFIG
API_KEY = os.environ.get("ODDS_API_KEY")  # in Streamlit secrets
SPORT = "americanfootball_nfl"
BOOKMAKERS = ["fanduel", "draftkings"]

# Load weekly updated trained model
MODEL_PATH = "nfl_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("No model file found yet. Please wait for GitHub Action to generate.")
    st.stop()

features = ["Spread", "Total", "Home"]

# -------------------------------
# DETERMINE CURRENT WEEK
today = datetime.now(timezone.utc)
season_start = datetime(2025, 9, 4, tzinfo=timezone.utc)  # kickoff date
days_since_start = (today - season_start).days
current_week = max(1, days_since_start // 7 + 1)

# -------------------------------
# SECTION 1: THIS WEEK'S PREDICTIONS
st.header(f"📅 Week {current_week} Predictions")

url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": "spreads,totals",
    "oddsFormat": "american",
    "bookmakers": ",".join(BOOKMAKERS),
}
resp = requests.get(url, params=params)

if resp.status_code == 200:
    games = resp.json()
else:
    st.warning("⚠️ Could not fetch odds API. Showing historical only.")
    games = []

rows = []
for g in games:
    home_team = g["home_team"]
    away_team = g["away_team"]

    # filter only this week
    game_time = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
    game_week = (game_time - season_start).days // 7 + 1
    if game_week != current_week:
        continue

    spread, total = None, None
    for bm in g["bookmakers"]:
        if bm["key"] in BOOKMAKERS:
            for market in bm["markets"]:
                if market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home_team:
                            spread = outcome.get("point")
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Over":
                            total = outcome.get("point")
    if spread is None or total is None:
        continue

    rows.append({
        "Matchup": f"{away_team} @ {home_team}",
        "Spread": spread,
        "Total": total,
        "Home": 1,
        "Home_Team": home_team,
        "Away_Team": away_team,
    })

df_games = pd.DataFrame(rows)

if not df_games.empty:
    probs = model.predict_proba(df_games[features])[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = []
    for i, row in df_games.iterrows():
        winner = row["Home_Team"] if preds[i] == 1 else row["Away_Team"]
        results.append({
            "Matchup": row["Matchup"],
            "Home Win Probability": probs[i],
            "Predicted Winner": winner,
        })

    st.dataframe(
        pd.DataFrame(results).style.format({"Home Win Probability": "{:.2%}"})
    )
else:
    st.info("No upcoming games found for this week.")

# -------------------------------
# SECTION 2: HISTORICAL PERFORMANCE
st.header("📊 Historical Predictions & Records")

csv_path = os.path.join("datasets", "nfl_gamelogs_vegas_2015-2025_FINAL.csv")
if os.path.exists(csv_path):
    df_hist = pd.read_csv(csv_path)
    df_hist = df_hist.dropna(subset=["Spread","Total","Home","Win"])
    
    # You can add Week column if not present
    if "Week" not in df_hist.columns:
        df_hist["Date"] = pd.to_datetime(df_hist["Date"])
        df_hist["Week"] = ((df_hist["Date"] - season_start).dt.days // 7) + 1
    
    week_choice = st.selectbox(
        "Select a past week", 
        sorted(df_hist["Week"].unique())
    )
    
    df_week = df_hist[df_hist["Week"] == week_choice].copy()
    X = df_week[features]
    y_true = df_week["Win"]

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    df_week["Predicted_Home_Win_Prob"] = y_prob
    df_week["Predicted_Home_Win"] = y_pred
    df_week["Predicted_Winner"] = df_week.apply(
        lambda r: r["Team"] if r["Predicted_Home_Win"] == 1 else r["Opponent"], axis=1
    )
    
    # Evaluate record
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    record = f"{correct}-{total - correct}"

    st.subheader(f"Week {week_choice} Record: {record}")
    st.dataframe(
        df_week[["Team","Opponent","Spread","Total","Win",
                 "Predicted_Home_Win_Prob","Predicted_Winner"]]
        .style.format({"Predicted_Home_Win_Prob":"{:.2%}"})
    )
else:
    st.warning("Historical dataset not found.")
