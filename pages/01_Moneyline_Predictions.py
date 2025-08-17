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

# --- Load dataset (exact filename, relative to repo)
df = pd.read_csv(csv_path)

# Display first few rows (matches notebook)
if debug:
    st.subheader("Head()")
    st.dataframe(df.head())

# Binary target: 1 if Win, 0 if Loss
df['Win_Binary'] = df['Win']

# 3rd down conversion rates
df['Tm_3DConv_Rate'] = df['Tm_3DConv'] / df['Tm_3DAtt'].replace(0, 1)
df['Opp_3DConv_Rate'] = df['Opp_3DConv'] / df['Opp_3DAtt'].replace(0, 1)

# Per-game turnover differential
df['Turnover_Diff'] = df['Opp_TO'] - df['Tm_TO']

# Columns for leakage-free rolling avgs
stat_cols = [
    'Tm_pY/A', 'Tm_rY/A', 'Tm_Y/P',
    'Opp_pY/A', 'Opp_rY/A', 'Opp_Y/P',
    'Tm_TO', 'Opp_TO', 'Tm_PenYds', 'Opp_PenYds',
    'Tm_3DConv_Rate', 'Opp_3DConv_Rate',
    'Turnover_Diff'
]

# Leakage-free rolling means (shift then expanding)
for col in stat_cols:
    df[f'{col}_avg'] = (
        df.groupby(['Season', 'Team'])[col]
          .apply(lambda x: x.shift().expanding().mean())
          .reset_index(level=[0,1], drop=True)
    )

# Fill NaN (Week 1) with league average of the raw column (matches code)
for col in stat_cols:
    league_avg = df[col].mean()
    df[f'{col}_avg'].fillna(league_avg, inplace=True)

# Features used
features_avg = ['Spread', 'Total', 'Home'] + [f'{col}_avg' for col in stat_cols]
df_clean = df.dropna(subset=features_avg + ['Win_Binary'])

if debug:
    st.write("Dataset shape:", df_clean.shape)
    st.write("Missing values after dropna():")
    st.write(df_clean.isna().sum())

X = df_clean[features_avg]
y = df_clean['Win_Binary']

# Train/Test split (deterministic)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Evaluation (exact prints -> shown via st.write)
st.write(f"**Train Accuracy:** {model.score(X_train, y_train):.2%}")
st.write(f"**Test Accuracy:** {model.score(X_test, y_test):.2%}")

# Optional detailed report in Debug
if debug:
    y_pred = model.predict(X_test)
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

st.markdown("---")
st.subheader("Try a Custom Matchup")

col1, col2, col3 = st.columns(3)
with col1:
    spread = st.number_input("Vegas Spread (Home team)", value=-6.5, format="%.1f")
with col2:
    total = st.number_input("Vegas Total", value=46.5, format="%.1f")
with col3:
    team_choice = st.selectbox("Which team to predict?", ["Home", "Away"])

# Convert dropdown to binary flag for model
home_flag = 1 if team_choice == "Home" else 0

# Create one-row DataFrame
new_game = pd.DataFrame([{
    'Spread': spread,
    'Total': total,
    'Home': home_flag,
    **{f'{col}_avg': df[col].mean() for col in stat_cols}
}])

if st.button("Predict Outcome"):
    prob = model.predict_proba(new_game)[:, 1][0]
    st.write(f"**Win Probability for {team_choice} Team: {prob:.2%}**")
    st.write("**Prediction:** WIN" if prob >= 0.5 else "**Prediction:** LOSS")


st.markdown("---")
st.subheader("Week 1 Predictions")

week1_games = [
    {'Spread': -6.5, 'Total': 46.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  3.0, 'Total': 44.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -6.5, 'Total': 45.0, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -6.5, 'Total': 45.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  3.0, 'Total': 38.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -2.5, 'Total': 42.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  5.5, 'Total': 41.0, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  5.5, 'Total': 45.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -1.5, 'Total': 46.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  1.5, 'Total': 48.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -7.5, 'Total': 41.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  2.5, 'Total': 45.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -1.5, 'Total': 49.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -2.5, 'Total': 44.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread': -1.5, 'Total': 51.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
    {'Spread':  1.5, 'Total': 43.5, 'Home': 1, **{f'{c}_avg': df[c].mean() for c in stat_cols}},
]
week1_df = pd.DataFrame(week1_games)

if st.button("Run Week 1 Predictions"):
    probs = LogisticRegression(max_iter=1000, solver='liblinear').fit(X_train, y_train).predict_proba(week1_df[features_avg])[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = pd.DataFrame({
        "Game#": range(1, len(probs)+1),
        "Home_Win_Prob": probs,
        "Prediction": ["WIN" if p else "LOSS" for p in preds]
    })
    st.dataframe(out.style.format({"Home_Win_Prob": "{:.2%}"}))

