# fetch_and_train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Paths
DATA_PATH = os.path.join("datasets", "nfl_gamelogs_vegas_2015-2025_FINAL.csv")
MODEL_PATH = "nfl_model.pkl"

print("📂 Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Keep only rows with needed features + target
required_cols = ["Spread", "Total", "Home", "Win"]
df = df.dropna(subset=required_cols)

# Features & target
X = df[["Spread", "Total", "Home"]]
y = df["Win"]   # already binary: 1=win, 0=lose

print(f"✅ Training dataset shape: X={X.shape}, y={y.shape}")

# Train simple Logistic Regression
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X, y)

# Save model to repo root
joblib.dump(model, MODEL_PATH)
print(f"✅ Model trained & saved to {MODEL_PATH}")
