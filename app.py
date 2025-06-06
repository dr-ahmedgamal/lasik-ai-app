import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the single model file containing all predictors
model_dict = joblib.load("model.pkl")
model_bcva = model_dict["bcva"]
model_k1 = model_dict["k1"]
model_k2 = model_dict["k2"]
model_thickness = model_dict["thickness"]

st.set_page_config(page_title="LASIK Predictor", layout="centered")
st.title("🔬 LASIK Predictor")

# Inputs with correct naming and step formatting
age = st.number_input("Age", min_value=18, max_value=70, step=1, format="%d")
spherical = st.number_input("Spherical (D)", format="%.2f", step=0.25, value=None)
cylinder = st.number_input("Cylinder (D)", format="%.2f", step=0.25, value=None)
k1 = st.number_input("K1 (D)", format="%.2f", step=0.1, value=40.00)
k2 = st.number_input("K2 (D)", format="%.2f", step=0.1, value=40.00)
pachymetry = st.number_input("Corneal Thickness (µm)", format="%d", step=1, value=500)
preop_bcva = st.number_input("Pre-op BCVA (decimal)", min_value=0.0, max_value=1.0, format="%.1f", step=0.1)

if st.button("Predict"):
    # Derived features
    se = spherical + (cylinder / 2)
    k_avg = (k1 + k2) / 2

    # Ensure input feature names match model training
    input_features = pd.DataFrame([[
        age, spherical, cylinder, k1, k2, pachymetry, preop_bcva, se, k_avg
    ]], columns=[
        "Age", "Spherical", "Cylinder", "K1", "K2", "Pachymetry", "Pre-op BCVA", "SE", "Kavg"
    ])

    # Predictions
    post_bcva = model_bcva.predict(input_features)[0]
    post_k1 = model_k1.predict(input_features)[0]
    post_k2 = model_k2.predict(input_features)[0]
    post_thickness = model_thickness.predict(input_features)[0]

    post_kavg = (post_k1 + post_k2) / 2
    ablation = pachymetry - post_thickness

    # Surgery recommendation logic
    def recommend_surgery():
        if (
            pachymetry >= 500 and
            post_thickness > 405 and
            ablation <= 140 and
            36 <= post_kavg <= 49
        ):
            return "LASIK"
        elif (
            se < 0 and
            460 <= pachymetry <= 550 and
            post_thickness >= 395 and
            ablation <= 100 and
            36 <= post_kavg <= 49
        ):
            return "PRK"
        elif age < 40:
            return "Phakic IOL"
        else:
            return "Pseudophakic IOL"

    surgery_type = recommend_surgery()

    # Display results
    st.markdown("### 📈 Predicted Post-Op Results")
    st.write(f"**BCVA:** {round(post_bcva, 2)}")
    st.write(f"**K1:** {round(post_k1, 2)} D")
    st.write(f"**K2:** {round(post_k2, 2)} D")
    st.write(f"**Corneal Thickness:** {round(post_thickness)} µm")
    st.write(f"**Kavg:** {round(post_kavg, 2)} D")
    st.write(f"**Ablation Depth:** {round(ablation)} µm")

    st.markdown("### 💡 Recommended Surgery Type")
    st.success(surgery_type)
