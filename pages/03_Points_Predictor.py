import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Points Predictor – Linear Regression")

# --- Controls
debug = st.checkbox("Debug mode (print intermediate variables)")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_FINAL.csv")

# 2. Load and Prepare the Data
df = pd.read_csv(csv_path)
df["Total_Points_Scored"] = df["Tm_Pts"] + df["Opp_Pts"]

# Sort to calculate rolling averages correctly
df = df.sort_values(by=["Team", "Season", "Week"])

# 3. Select Features
features_early = ["Season", "Week", "Home", "Team", "Opp", "Spread", "Total"]
target_early = "Tm_Pts"

X_early = df[features_early]
y_early = df[target_early]

# 4. Preprocessing: One-hot encode 'Team' and 'Opp'
categorical_early = ["Team", "Opp"]
numerical_early = ["Season", "Week", "Home", "Spread", "Total"]

preprocessor_early = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_early)],
    remainder="passthrough"
)

X_processed_early = preprocessor_early.fit_transform(X_early)

# 5. Split the Data
X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
    X_processed_early, y_early, test_size=0.2, random_state=42
)

# 6. Train Model
model_early = LinearRegression()
model_early.fit(X_train_early, y_train_early)

# 7. Evaluate
y_pred_early = model_early.predict(X_test_early)
mae_early = mean_absolute_error(y_test_early, y_pred_early)
rmse_early = np.sqrt(mean_squared_error(y_test_early, y_pred_early))
r2_early = r2_score(y_test_early, y_pred_early)

st.write(f"[Early Model] **R²:** {r2_early:.2f}")
st.write(f"[Early Model] **MAE:** {mae_early:.2f}")
st.write(f"[Early Model] **RMSE:** {rmse_early:.2f}")

# 9. Prediction Function
def predict_week_points_early(games):
    input_df = pd.DataFrame(games)
    input_processed = preprocessor_early.transform(input_df)
    predictions = model_early.predict(input_processed)
    input_df["Predicted_Points"] = predictions
    input_df = input_df.rename(columns={"Opp": "Opponent"})
    # Format spread/total
    input_df["Spread"] = input_df["Spread"].map(lambda x: f"{x:.1f}")
    input_df["Total"] = input_df["Total"].map(lambda x: f"{x:.1f}")
    return input_df

# Week 1 Games (FanDuel)
week1_games_fd = [
    {"Season": 2025, "Week": 1, "Home": 1, "Team": "PHI", "Opp": "DAL", "Spread": -8.5, "Total": 48.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "DAL", "Opp": "PHI", "Spread":  8.5, "Total": 48.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "LAC", "Opp": "KAN", "Spread":  3.0, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "KAN", "Opp": "LAC", "Spread": -3.0, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "WAS", "Opp": "NYG", "Spread": -6.5, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "NYG", "Opp": "WAS", "Spread":  6.5, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "JAX", "Opp": "CAR", "Spread": -4.5, "Total": 46.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CAR", "Opp": "JAX", "Spread":  4.5, "Total": 46.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NYJ", "Opp": "PIT", "Spread":  3.0, "Total": 37.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "PIT", "Opp": "NYJ", "Spread": -3.0, "Total": 37.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NWE", "Opp": "RAI", "Spread": -2.5, "Total": 43.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "RAI", "Opp": "NWE", "Spread":  2.5, "Total": 43.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NOR", "Opp": "CRD", "Spread":  6.5, "Total": 44.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CRD", "Opp": "NOR", "Spread": -6.5, "Total": 44.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CLE", "Opp": "CIN", "Spread":  4.5, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CIN", "Opp": "CLE", "Spread": -4.5, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CLT", "Opp": "MIA", "Spread": -1.5, "Total": 46.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "MIA", "Opp": "CLT", "Spread":  1.5, "Total": 46.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "ATL", "Opp": "TAM", "Spread":  1.5, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "TAM", "Opp": "ATL", "Spread": -1.5, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "DEN", "Opp": "OTI", "Spread": -8.5, "Total": 42.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "OTI", "Opp": "DEN", "Spread":  8.5, "Total": 42.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "SEA", "Opp": "SFO", "Spread":  1.5, "Total": 43.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "SFO", "Opp": "SEA", "Spread": -1.5, "Total": 43.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "GNB", "Opp": "DET", "Spread": -1.5, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "DET", "Opp": "GNB", "Spread":  1.5, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "RAM", "Opp": "HTX", "Spread": -3.0, "Total": 43.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "HTX", "Opp": "RAM", "Spread":  3.0, "Total": 43.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "BUF", "Opp": "RAV", "Spread": -1.5, "Total": 50.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "RAV", "Opp": "BUF", "Spread":  1.5, "Total": 50.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CHI", "Opp": "MIN", "Spread":  1.5, "Total": 43.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "MIN", "Opp": "CHI", "Spread": -1.5, "Total": 43.5},
]

# Week 1 Games (DraftKings)
week1_games_dk = [
    {"Season": 2025, "Week": 1, "Home": 1, "Team": "PHI", "Opp": "DAL", "Spread": -8.5, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "DAL", "Opp": "PHI", "Spread":  8.5, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "LAC", "Opp": "KAN", "Spread":  3.0, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "KAN", "Opp": "LAC", "Spread": -3.0, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "WAS", "Opp": "NYG", "Spread": -6.0, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "NYG", "Opp": "WAS", "Spread":  6.0, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "JAX", "Opp": "CAR", "Spread": -4.5, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CAR", "Opp": "JAX", "Spread":  4.5, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NYJ", "Opp": "PIT", "Spread":  3.0, "Total": 37.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "PIT", "Opp": "NYJ", "Spread": -3.0, "Total": 37.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NWE", "Opp": "RAI", "Spread": -2.5, "Total": 44.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "RAI", "Opp": "NWE", "Spread":  2.5, "Total": 44.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NOR", "Opp": "CRD", "Spread":  6.0, "Total": 44.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CRD", "Opp": "NOR", "Spread": -6.0, "Total": 44.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CLE", "Opp": "CIN", "Spread":  5.5, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CIN", "Opp": "CLE", "Spread": -5.5, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CLT", "Opp": "MIA", "Spread": -1.5, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "MIA", "Opp": "CLT", "Spread":  1.5, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "ATL", "Opp": "TAM", "Spread":  1.5, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "TAM", "Opp": "ATL", "Spread": -1.5, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "DEN", "Opp": "OTI", "Spread": -8.5, "Total": 42.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "OTI", "Opp": "DEN", "Spread":  8.5, "Total": 42.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "SEA", "Opp": "SFO", "Spread":  1.5, "Total": 43.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "SFO", "Opp": "SEA", "Spread": -1.5, "Total": 43.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "GNB", "Opp": "DET", "Spread": -1.5, "Total": 47.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "DET", "Opp": "GNB", "Spread":  1.5, "Total": 47.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "RAM", "Opp": "HTX", "Spread": -3.0, "Total": 43.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "HTX", "Opp": "RAM", "Spread":  3.0, "Total": 43.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "BUF", "Opp": "RAV", "Spread": -1.5, "Total": 50.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "RAV", "Opp": "BUF", "Spread":  1.5, "Total": 50.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CHI", "Opp": "MIN", "Spread":  1.5, "Total": 43.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "MIN", "Opp": "CHI", "Spread": -1.5, "Total": 43.5},
]

# --- FanDuel Predictions
st.markdown("---")
st.subheader("Week 1 Predictions – FanDuel Lines")
if st.button("Run Week 1 Predictions – FanDuel"):
    week1_predictions_fd = predict_week_points_early(week1_games_fd)

    # Add Matchup column
    week1_predictions_fd["Matchup"] = week1_predictions_fd.apply(
        lambda row: f"{row['Team']} vs {row['Opponent']}" if row["Home"] == 1 else None,
        axis=1
    )
    week1_predictions_fd["Matchup"].ffill(inplace=True)

    # Calculate predicted total per game
    totals_fd = week1_predictions_fd.groupby("Matchup").agg(
        Home_Team=("Team", lambda x: x.iloc[0]),
        Away_Team=("Opponent", lambda x: x.iloc[0]),
        Home_Predicted=("Predicted_Points", lambda x: x.iloc[0]),
        Away_Predicted=("Predicted_Points", lambda x: x.iloc[1]),
        Predicted_Total=("Predicted_Points", "sum")
    ).reset_index(drop=True)

    st.dataframe(week1_predictions_fd.style.format({"Predicted_Points": "{:.2f}"}))
    st.write("**Predicted Totals (FanDuel):**")
    st.dataframe(totals_fd.style.format({
        "Home_Predicted": "{:.2f}",
        "Away_Predicted": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))

# --- DraftKings Predictions
st.markdown("---")
st.subheader("Week 1 Predictions – DraftKings Lines")
if st.button("Run Week 1 Predictions – DraftKings"):
    week1_predictions_dk = predict_week_points_early(week1_games_dk)

    # Add Matchup column
    week1_predictions_dk["Matchup"] = week1_predictions_dk.apply(
        lambda row: f"{row['Team']} vs {row['Opponent']}" if row["Home"] == 1 else None,
        axis=1
    )
    week1_predictions_dk["Matchup"].ffill(inplace=True)

    # Calculate predicted total per game
    totals_dk = week1_predictions_dk.groupby("Matchup").agg(
        Home_Team=("Team", lambda x: x.iloc[0]),
        Away_Team=("Opponent", lambda x: x.iloc[0]),
        Home_Predicted=("Predicted_Points", lambda x: x.iloc[0]),
        Away_Predicted=("Predicted_Points", lambda x: x.iloc[1]),
        Predicted_Total=("Predicted_Points", "sum")
    ).reset_index(drop=True)

    st.dataframe(week1_predictions_dk.style.format({"Predicted_Points": "{:.2f}"}))
    st.write("**Predicted Totals (DraftKings):**")
    st.dataframe(totals_dk.style.format({
        "Home_Predicted": "{:.2f}",
        "Away_Predicted": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))

