# pages/01_Moneyline_Predictions.py
import streamlit as st
import requests
import pandas as pd
import os
import joblib

st.set_page_config(page_title="🏈 NFL Predictions", page_icon="🏈")
st.title("🏈 Automated NFL Moneyline Predictions")

# === CONFIG ===
API_KEY = os.environ.get("ODDS_API_KEY")  # stored securely in Streamlit secrets
SPORT = "americanfootball_nfl"
BOOKMAKERS = ["fanduel", "draftkings"]

# Load latest trained model (auto updated weekly by GitHub Action)
model = joblib.load("nfl_model.pkl")

# Keep feature names consistent with how the model was trained
features_avg = ["Spread", "Total", "Home"]

# === Fetch upcoming games with odds ===
url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": "spreads,totals",
    "oddsFormat": "american",
    "bookmakers": ",".join(BOOKMAKERS),
}
resp = requests.get(url, params=params)

if resp.status_code != 200:
    st.error(f"Failed to fetch odds data: {resp.text}")
    st.stop()

games = resp.json()

rows = []
for g in games:
    home_team = g["home_team"]
    away_team = g["away_team"]

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

    rows.append(
        {
            "Matchup": f"{away_team} @ {home_team}",
            "Spread": spread,
            "Total": total,
            "Home": 1,
            "Home_Team": home_team,
            "Away_Team": away_team,
        }
    )

df_games = pd.DataFrame(rows)

if df_games.empty:
    st.warning("No games available yet from the Odds API.")
    st.stop()

# === Make Predictions ===
probs = model.predict_proba(df_games[features_avg])[:, 1]
preds = (probs >= 0.5).astype(int)

results = []
for i, row in df_games.iterrows():
    winner = row["Home_Team"] if preds[i] == 1 else row["Away_Team"]
    results.append(
        {
            "Matchup": row["Matchup"],
            "Home Win Probability": probs[i],
            "Predicted Winner": winner,
        }
    )

out = pd.DataFrame(results)
st.subheader("This Week's Predictions")
st.dataframe(out.style.format({"Home Win Probability": "{:.2%}"}))
