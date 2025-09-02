#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.title("Moneyline Model – Logistic Regression")

# --- Controls
debug = st.checkbox("Debug mode (print intermediate variables)")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2024_NEW.csv")

# --- Load dataset
df = pd.read_csv(csv_path)

if debug:
    st.subheader("Head()")
    st.dataframe(df.head())

# Binary target
df['Win_Binary'] = df['Win']

# Feature engineering
df['Tm_3DConv_Rate'] = df['Tm_3DConv'] / df['Tm_3DAtt'].replace(0, 1)
df['Opp_3DConv_Rate'] = df['Opp_3DConv'] / df['Opp_3DAtt'].replace(0, 1)
df['Turnover_Diff'] = df['Opp_TO'] - df['Tm_TO']

stat_cols = [
    'Tm_pY/A', 'Tm_rY/A', 'Tm_Y/P',
    'Opp_pY/A', 'Opp_rY/A', 'Opp_Y/P',
    'Tm_TO', 'Opp_TO', 'Tm_PenYds', 'Opp_PenYds',
    'Tm_3DConv_Rate', 'Opp_3DConv_Rate',
    'Turnover_Diff'
]

# Leakage-free rolling averages
for col in stat_cols:
    df[f'{col}_avg'] = (
        df.groupby(['Season', 'Team'])[col]
          .apply(lambda x: x.shift().expanding().mean())
          .reset_index(level=[0,1], drop=True)
    )

# Fill NaN (Week 1) with league average
for col in stat_cols:
    league_avg = df[col].mean()
    df[f'{col}_avg'].fillna(league_avg, inplace=True)

# Features
features_avg = ['Spread', 'Total', 'Home'] + [f'{col}_avg' for col in stat_cols]
df_clean = df.dropna(subset=features_avg + ['Win_Binary'])

if debug:
    st.write("Dataset shape:", df_clean.shape)
    st.write("Missing values after dropna():")
    st.write(df_clean.isna().sum())

X = df_clean[features_avg]
y = df_clean['Win_Binary']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Evaluation
st.write(f"**Train Accuracy:** {model.score(X_train, y_train):.2%}")
st.write(f"**Test Accuracy:** {model.score(X_test, y_test):.2%}")

if debug:
    y_pred = model.predict(X_test)
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

# --- Week 1 Predictions
st.markdown("---")
st.subheader("Week 1 Predictions")

# Team matchups in the same order as spreads
week1_teams = [
    {"Home": "Eagles", "Away": "Cowboys"},
    {"Home": "Chargers", "Away": "Chiefs"},
    {"Home": "Commanders", "Away": "Giants"},
    {"Home": "Jaguars", "Away": "Panthers"},
    {"Home": "Jets", "Away": "Steelers"},
    {"Home": "Patriots", "Away": "Raiders"},
    {"Home": "Saints", "Away": "Cardinals"},
    {"Home": "Browns", "Away": "Bengals"},
    {"Home": "Colts", "Away": "Dolphins"},
    {"Home": "Falcons", "Away": "Buccaneers"},
    {"Home": "Broncos", "Away": "Titans"},
    {"Home": "Seahawks", "Away": "49ers"},
    {"Home": "Packers", "Away": "Lions"},
    {"Home": "Rams", "Away": "Texans"},
    {"Home": "Bills", "Away": "Ravens"},
    {"Home": "Bears", "Away": "Vikings"}
]

week1_games = [
    {'Spread': -8.5, 'Total': 47.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  3.0, 'Total': 45.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -6.5, 'Total': 45.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -3.5, 'Total': 46.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  2.5, 'Total': 38.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -2.5, 'Total': 43.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  6.5, 'Total': 42.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  5.5, 'Total': 47.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -1.5, 'Total': 46.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  2.5, 'Total': 47.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -7.5, 'Total': 42.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  2.5, 'Total': 43.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -2.5, 'Total': 47.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -3.0, 'Total': 44.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -1.5, 'Total': 50.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  1.5, 'Total': 44.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
]
week1_df = pd.DataFrame(week1_games)

if st.button("Run Week 1 Predictions"):
    probs = LogisticRegression(max_iter=1000, solver='liblinear').fit(X_train, y_train).predict_proba(week1_df[features_avg])[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        home = week1_teams[i]["Home"]
        away = week1_teams[i]["Away"]
        winner = home if pred == 1 else away
        results.append({
            "Matchup": f"{away} @ {home}",
            "Home Win Probability": prob,
            "Predicted Winner": winner
        })

    out = pd.DataFrame(results)
    st.dataframe(out.style.format({"Home Win Probability": "{:.2%}"}))
