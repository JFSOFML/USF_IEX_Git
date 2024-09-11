"""Prediction tool page for streamlit app"""

import pickle
import streamlit as st
import pandas as pd

st.header("Predictions")

with open("Scaler.pkl", "rb") as f:
    sc = pickle.load(f)
with open("forest.pkl", "rb") as f:
    model = pickle.load(f)


overall_qual = st.select_slider(
    "Overall Quality (1-10):", options=range(1, 11), value=5
)
overall_cond = st.select_slider(
    "Overall Condition (1-10):", options=range(1, 11), value=5
)
gr_liv_area = st.slider("Above Grade Living Area (sq. ft):", 0, 5000, 1500)
Central_Air = st.selectbox("Central Air", ["Yes", "No"])
Central_Air_Binary = {"Yes": 1, "No": 0}
total_bsmt_sf = st.slider("Total Basement SF (sq. ft.):", 0, 3000, 1500)

Input = pd.DataFrame(
    {
        "Overall Qual": [overall_qual],
        "Overall Cond": [overall_cond],
        "Total Bsmt SF": [total_bsmt_sf],
        "Central Air": Central_Air_Binary[Central_Air],
        "Gr Liv Area": [gr_liv_area],
    }
)

if st.button("Push me!"):
    user_input_scaled = sc.transform(Input)
    prediction = model.predict(user_input_scaled)[0]

    st.subheader("Predicted Sale Price")
    st.write(f"${prediction:,.2f}")