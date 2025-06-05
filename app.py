import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load trained models
model_bcva = joblib.load("models/model_bcva.pkl")
model_k1 = joblib.load("models/model_k1.pkl")
model_k2 = joblib.load("models/model_k2.pkl")
model_thickness = joblib.load("models/model_thickness.pkl")

st.set_page_config(page_title="LASIK Surgery Outcome Predictor", layout="centered")

st.title("🔬 LASIK Surgery Outcome Predictor")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=70, value=30)
sph = st.number_input("Spherical (D)", value=0.0)
cyl = st.number_input("Cylinder (D)", value=0.0)
k1 = st.number_input("K1 (D)", value=43.0)
k2 = st.number_input("K2 (D)", value=44.0)
pachy = st.number_input("Corneal Thickness (µm)", value=520)
bcva = st.number_input("Pre-op BCVA (decimal)", value=0.8)

# Derived Features
se = sph + (cyl / 2)
k_avg = (k1 + k2) / 2

input_features = pd.DataFrame([[sph, cyl, k1, k2, pachy, bcva, age, se, k_avg]],
    columns=["Spherical", "Cylinder", "K1", "K2", "Pachymetry", "Pre-op BCVA", "Age", "SE", "Kavg"])

# Predict Post-op Values
post_bcva = model_bcva.predict(input_features)[0]
post_k1 = model_k1.predict(input_features)[0]
post_k2 = model_k2.predict(input_features)[0]
post_pachy = model_thickness.predict(input_features)[0]
post_kavg = (post_k1 + post_k2) / 2
ablation = pachy - post_pachy

# Surgery Recommendation Logic (hardcoded rules)
def recommend_surgery():
    if (
        pachy >= 500 and
        post_pachy > 405 and
        ablation <= 140 and
        36 <= post_kavg <= 49
    ):
        return "LASIK"
    elif (
        se < 0 and
        460 <= pachy <= 550 and
        post_pachy >= 395 and
        ablation <= 100 and
        36 <= post_kavg <= 49
    ):
        return "PRK"
    elif age < 40:
        return "Phakic IOL"
    else:
        return "Pseudophakic IOL"

recommended = recommend_surgery()

# Display Results
st.markdown("### 📈 Predicted Post-Op Results")
st.write(f"**BCVA:** {round(post_bcva, 2)}")
st.write(f"**K1:** {round(post_k1, 2)} D")
st.write(f"**K2:** {round(post_k2, 2)} D")
st.write(f"**Corneal Thickness:** {round(post_pachy)} µm")
st.write(f"**Kavg:** {round(post_kavg, 2)} D")
st.write(f"**Ablation:** {round(ablation)} µm")

st.markdown("### 💡 Recommended Surgery Type")
st.success(recommended)
