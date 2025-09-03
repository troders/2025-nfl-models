#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os

st.title("DraftKings NFL Over/Under Predictions")

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Instead of scanning all *_predictions.csv, filter for DK files
prediction_files = [f for f in os.listdir(BASE_DIR) if f.endswith("_predictions_dk.csv")]

# Extract week numbers
weeks_available = sorted([
    int(f.split("_")[0].replace("week", "")) 
    for f in prediction_files
])

selected_week = st.selectbox("Select Week:", weeks_available)

# Load DK predictions file
pred_file = f"week{selected_week}_2025_predictions_dk.csv"
preds = pd.read_csv(os.path.join(BASE_DIR, pred_file))

for i, row in preds.iterrows():
    st.markdown(
        f"**{row['Game']}** | Spread: {row['Spread']:.1f} | Total: {row['Total']:.1f} "
        f"| **Prediction:** {row['Prediction']}"
    )
    st.write(
        f"Confidence %: {row['ConfidencePercent']*100:.1f}% "
        f"| Avg Distance: {row['AvgDistance']} "
        f"| Score: {row['ConfidenceScore']:.3f}"
    )

    # Neighbor file pattern for DK
    neighbors_file = f"dk_neighbors_{i+1}.csv"
    neighbors_path = os.path.join(BASE_DIR, neighbors_file)

    if os.path.exists(neighbors_path):
        neighbors = pd.read_csv(neighbors_path)
        st.dataframe(neighbors.round(3))
    else:
        st.warning(f"Neighbors file not found: {neighbors_file}")
