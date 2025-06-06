import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load unified model dictionary from model.pkl
model_dict = joblib.load("model.pkl")
model_bcva = model_dict["bcva"]
model_k1 = model_dict["k1"]
model_k2 = model_dict["k2"]
model_thickness = model_dict["thickness"]

st.set_page_config(page_title="LASIK Surgery Outcome Predictor", layout="centered")
st.title("🔬 LASIK Surgery Outcome Predictor")

# User Inputs (with default values and proper increments)
age = st.number_input("Age", min_value=18, max_value=70, format="%d", step=1)
sph = st.number_input("Spherical (D)", format="%.2f", step=0.25)
cyl = st.number_input("Cylinder (D)", format="%.2f", step=0.25)
k1 = st.number_input("K1 (D)", format="%.2f", step=0.01, value=40.00)
k2 = st.number_input("K2 (D)", format="%.2f", step=0.01, value=40.00)
pachy = st.number_input("Corneal Thickness (µm)", format="%d", step=1, value=500)
bcva = st.number_input("Pre-op BCVA (decimal)", min_value=0.0, max_value=1.0, format="%.2f", step=0.1)

if st.button("Predict"):
    # Derived Features
    se = sph + (cyl / 2)
    k_avg = (k1 + k2) / 2

    input_features = pd.DataFrame([[age, sph, cyl, k1, k2, pachy, bcva, se, k_avg]],
                                  columns=["Age", "Sph", "Cyl", "K1", "K2", "Pachy", "BCVA", "SE", "Kavg"])

    # Make Predictions
    post_bcva = model_bcva.predict(input_features)[0]
    post_k1 = model_k1.predict(input_features)[0]
    post_k2 = model_k2.predict(input_features)[0]
    post_pachy = model_thickness.predict(input_features)[0]
    post_kavg = (post_k1 + post_k2) / 2
    ablation = pachy - post_pachy

    # Surgery Decision Logic
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

    # Show Results
    st.markdown("### 📈 Predicted Post-Op Results")
    st.write(f"**BCVA:** {round(post_bcva, 2)}")
    st.write(f"**K1:** {round(post_k1, 2)} D")
    st.write(f"**K2:** {round(post_k2, 2)} D")
    st.write(f"**Corneal Thickness:** {round(post_pachy)} µm")
    st.write(f"**Kavg:** {round(post_kavg, 2)} D")
    st.write(f"**Ablation:** {round(ablation)} µm")

    st.markdown("### 💡 Recommended Surgery Type")
    st.success(recommended)
