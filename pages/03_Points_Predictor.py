#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2024_NEW.csv")

# 2. Load and Prepare the Data
df = pd.read_csv(csv_path)
df["Total_Points_Scored"] = df["Tm_Pts"] + df["Opp_Pts"]

# Sort to calculate rolling averages correctly (code sorts for early model anyway)
df = df.sort_values(by=["Team", "Season", "Week"])

# 3. Select Features for Early-Season Model (No rolling stats)
features_early = ["Season", "Week", "Home", "Team", "Opp_x", "Spread", "Total"]
target_early = "Tm_Pts"

X_early = df[features_early]
y_early = df[target_early]

# 4. Preprocessing: One-hot encode 'Team' and 'Opp_x'
categorical_early = ["Team", "Opp_x"]
numerical_early = ["Season", "Week", "Home", "Spread", "Total"]

preprocessor_early = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_early)
    ],
    remainder="passthrough"
)

X_processed_early = preprocessor_early.fit_transform(X_early)

# 5. Split the Data (deterministic)
X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
    X_processed_early, y_early, test_size=0.2, random_state=42
)

# 6. Train the Early-Season Model
model_early = LinearRegression()
model_early.fit(X_train_early, y_train_early)

# 7. Evaluate Model Performance
y_pred_early = model_early.predict(X_test_early)
mae_early = mean_absolute_error(y_test_early, y_pred_early)
rmse_early = np.sqrt(mean_squared_error(y_test_early, y_pred_early))
r2_early = r2_score(y_test_early, y_pred_early)

st.write(f"[Early Model] **R²:** {r2_early:.2f}")
st.write(f"[Early Model] **MAE:** {mae_early:.2f}")
st.write(f"[Early Model] **RMSE:** {rmse_early:.2f}")

# 8. View Top Features
encoded_columns_early = preprocessor_early.named_transformers_["cat"].get_feature_names_out(categorical_early)
all_columns_early = np.concatenate([encoded_columns_early, numerical_early])

if debug:
    coef_df_early = pd.DataFrame({
        "Feature": all_columns_early,
        "Coefficient": model_early.coef_
    }).sort_values(by="Coefficient", ascending=False)

    st.subheader("Top Features (Early Model)")
    st.dataframe(coef_df_early.head(10))

# 9. Early-Season Prediction Function (unchanged core logic)
def predict_week_points_early(games):
    """
    games: list of dicts, each with Season, Week, Home, Team, Opp_x, Spread, Total
    """
    input_df = pd.DataFrame(games)
    input_processed = preprocessor_early.transform(input_df)
    predictions = model_early.predict(input_processed)
    input_df["Predicted_Points"] = predictions
    if debug:
        # print line-by-line like your notebook
        for i, row in input_df.iterrows():
            st.text(f"{row['Team']} vs {row['Opp_x']} -> {row['Predicted_Points']:.2f} points")
    return input_df

# 10. Week 1 preset
week1_games = [
    {"Season": 2025, "Week": 1, "Home": 1, "Team": "PHI", "Opp_x": "DAL", "Spread": -6.5, "Total": 46.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "DAL", "Opp_x": "PHI", "Spread":  6.5, "Total": 46.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "LAC", "Opp_x": "KAN", "Spread":  3.0, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "KAN", "Opp_x": "LAC", "Spread": -3.0, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "WAS", "Opp_x": "NYG", "Spread": -6.5, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "NYG", "Opp_x": "WAS", "Spread":  6.5, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "JAX", "Opp_x": "CAR", "Spread": -2.5, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CAR", "Opp_x": "JAX", "Spread":  2.5, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NYJ", "Opp_x": "PIT", "Spread":  3.0, "Total": 38.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "PIT", "Opp_x": "NYJ", "Spread": -3.0, "Total": 38.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NWE", "Opp_x": "RAI", "Spread": -2.5, "Total": 42.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "RAI", "Opp_x": "NWE", "Spread":  2.5, "Total": 42.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "NOR", "Opp_x": "CRD", "Spread":  5.5, "Total": 41.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CRD", "Opp_x": "NOR", "Spread": -5.5, "Total": 41.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CLE", "Opp_x": "CIN", "Spread":  5.5, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "CIN", "Opp_x": "CLE", "Spread": -5.5, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CLT", "Opp_x": "MIA", "Spread": -1.5, "Total": 46.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "MIA", "Opp_x": "CLT", "Spread":  1.5, "Total": 46.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "ATL", "Opp_x": "TAM", "Spread":  1.5, "Total": 48.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "TAM", "Opp_x": "ATL", "Spread": -1.5, "Total": 48.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "DEN", "Opp_x": "OTI", "Spread": -7.5, "Total": 41.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "OTI", "Opp_x": "DEN", "Spread":  7.5, "Total": 41.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "SEA", "Opp_x": "SFO", "Spread":  2.5, "Total": 45.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "SFO", "Opp_x": "SEA", "Spread": -2.5, "Total": 45.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "GNB", "Opp_x": "DET", "Spread": -1.5, "Total": 49.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "DET", "Opp_x": "GNB", "Spread":  1.5, "Total": 49.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "RAM", "Opp_x": "HTX", "Spread": -2.5, "Total": 44.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "HTX", "Opp_x": "RAM", "Spread":  2.5, "Total": 44.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "BUF", "Opp_x": "RAV", "Spread": -1.5, "Total": 51.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "RAV", "Opp_x": "BUF", "Spread":  1.5, "Total": 51.5},

    {"Season": 2025, "Week": 1, "Home": 1, "Team": "CHI", "Opp_x": "MIN", "Spread":  1.5, "Total": 43.5},
    {"Season": 2025, "Week": 1, "Home": 0, "Team": "MIN", "Opp_x": "CHI", "Spread": -1.5, "Total": 43.5},
]

st.markdown("**Week 1 predictions**")
if st.button("Run Week 1 Predictions"):
    week1_predictions = predict_week_points_early(week1_games)
    st.dataframe(week1_predictions.style.format({"Predicted_Points": "{:.2f}"}))

