import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="LASIK / PRK / Phakic IOL Predictor", layout="centered")
st.title("Refractive Surgery Outcome & Recommendation App")

# Load models
@st.cache_resource
def load_models():
    k1_model = joblib.load("models/k1_model.pkl")
    k2_model = joblib.load("models/k2_model.pkl")
    thickness_model = joblib.load("models/thickness_model.pkl")
    return k1_model, k2_model, thickness_model

k1_model, k2_model, thickness_model = load_models()

# Input fields
st.subheader("Enter Patient Data")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    se = st.number_input("Spherical Equivalent (SE) in D", value=-3.0)
    preop_thickness = st.number_input("Pre-op Corneal Thickness (in µm)", value=520)

with col2:
    k1 = st.number_input("Pre-op K1", value=43.0)
    k2 = st.number_input("Pre-op K2", value=44.0)

if st.button("Predict Outcomes and Recommend Surgery"):
    # Predict post-op outcomes
    features = np.array([[se, preop_thickness, k1, k2]])
    post_k1 = round(k1_model.predict(features)[0], 2)
    post_k2 = round(k2_model.predict(features)[0], 2)
    post_thickness = round(thickness_model.predict(features)[0], 2)
    k_avg = round((post_k1 + post_k2) / 2, 2)
    ablation = preop_thickness - post_thickness

    # Recommendation logic
    if (
        preop_thickness >= 500
        and post_thickness > 405
        and k_avg >= 36 and k_avg <= 49
        and ablation <= 140
    ):
        recommended = "LASIK"
    elif (
        se < 0
        and 460 <= preop_thickness <= 550
        and post_thickness >= 395
        and k_avg >= 36 and k_avg <= 49
        and ablation <= 100
    ):
        recommended = "PRK"
    elif age < 40:
        recommended = "Phakic IOL"
    else:
        recommended = "Pseudophakic IOL"

    # Show results
    st.success("Prediction Complete")
    st.write(f"**Predicted Post-op K1:** {post_k1} D")
    st.write(f"**Predicted Post-op K2:** {post_k2} D")
    st.write(f"**Predicted Post-op Corneal Thickness:** {post_thickness} µm")
    st.write(f"**Ablation Amount:** {ablation} µm")
    st.write(f"**Average K (Kavg):** {k_avg} D")
    st.markdown(f"### 🔹 Recommended Surgery: **{recommended}**")

    # Summary table
    st.subheader("Summary")
    summary = pd.DataFrame({
        "Metric": ["Post-op K1", "Post-op K2", "Post-op Thickness", "Ablation", "Kavg", "Recommended Surgery"],
        "Value": [post_k1, post_k2, post_thickness, ablation, k_avg, recommended]
    })
    st.dataframe(summary, use_container_width=True)

# Option to upload and analyze batch patient CSV
st.divider()
st.subheader("Batch Prediction from patients.csv")

@st.cache_data
def load_csv():
    url = "https://raw.githubusercontent.com/dr-ahmedgamal/lasik-ai-app/main/patients.csv"
    return pd.read_csv(url)

if st.checkbox("Show Predictions from GitHub CSV"):
    df = load_csv()
    st.write(df)
