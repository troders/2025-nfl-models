# pages/01_Moneyline_Predictions.py
import streamlit as st
import requests
import pandas as pd
import os
import joblib
from datetime import datetime, timezone

st.set_page_config(page_title="🏈 NFL Predictions", page_icon="🏈")
st.title("🏈 NFL Moneyline Predictions")

# ------------------------------------------------------
# BOOKMAKER + API CONFIG
API_KEY = os.environ.get("ODDS_API_KEY")  # stored in Streamlit secrets
SPORT = "americanfootball_nfl"
BOOKMAKERS = ["fanduel", "draftkings"]

# Model + feature setup
MODEL_PATH = "nfl_model.pkl"
features = ["Spread", "Total", "Home"]

# dictionary mapping abbreviations -> full team names
TEAM_MAP = {
    "CRD": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GNB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KAN": "Kansas City Chiefs",
    "LVR": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "RAM": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NWE": "New England Patriots",
    "NOR": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SFO": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TAM": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    # legacy teams if needed
    "STL": "St. Louis Rams"
}

# ------------------------------------------------------
# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("❌ No model file found yet. Run GitHub Action to generate nfl_model.pkl.")
    st.stop()

# ------------------------------------------------------
# Determine current week number (by date)
today = datetime.now(timezone.utc)
season_start = datetime(2025, 9, 4, tzinfo=timezone.utc)  # NFL 2025 kickoff Thursday
days_since_start = (today - season_start).days
current_week = max(1, days_since_start // 7 + 1)

# ------------------------------------------------------
# SECTION 1: Current Week Predictions
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

games = []
if resp.status_code == 200:
    games = resp.json()
else:
    st.warning("⚠️ Could not fetch live odds. Check your API key.")

rows = []
for g in games:
    home_team = g["home_team"]
    away_team = g["away_team"]

    # filter: only games in current week
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

if rows:
    df_games = pd.DataFrame(rows)
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

    out = pd.DataFrame(results)
    st.dataframe(out.style.format({"Home Win Probability": "{:.2%}"}))
else:
    st.info("No upcoming games found for this week.")

# ------------------------------------------------------
# SECTION 2: Historical Predictions & Records
st.header("📊 Historical Predictions & Records")

csv_path = os.path.join("datasets", "nfl_gamelogs_vegas_2015-2025_FINAL.csv")
if os.path.exists(csv_path):
    df_hist = pd.read_csv(csv_path)
    df_hist = df_hist.dropna(subset=["Spread", "Total", "Home", "Win", "Opp"])

    # Map team abbreviations to full names
    df_hist["Team"] = df_hist["Team"].map(TEAM_MAP).fillna(df_hist["Team"])
    df_hist["Opp"] = df_hist["Opp"].map(TEAM_MAP).fillna(df_hist["Opp"])

    # Compute Week if not present
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
        lambda r: r["Team"] if r["Predicted_Home_Win"] == 1 else r["Opp"], axis=1
    )

    # Compute accuracy record
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    record = f"{correct}-{total - correct}"

    st.subheader(f"Week {week_choice} Record: {record}")

    st.dataframe(
        df_week[["Team", "Opp", "Spread", "Total", "Win",
                 "Predicted_Home_Win_Prob", "Predicted_Winner"]]
        .style.format({"Predicted_Home_Win_Prob": "{:.2%}"})
    )
else:
    st.warning("❌ Historical dataset not found.")
