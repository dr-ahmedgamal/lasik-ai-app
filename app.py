import streamlit as st
import pandas as pd
import joblib

# Load models
reg_model = joblib.load("model.pkl")
class_model = joblib.load("surgery_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🔬 LASIK Surgery Outcome Predictor")

sph = st.number_input("Spherical (D)", -10.0, 10.0, step=0.25, value=None, format="%.2f")
cyl = st.number_input("Cylinder (D)", -6.0, 6.0, step=0.25, value=None, format="%.2f")
k1 = st.number_input("K1 (D)", 35.0, 50.0, value=None, format="%.2f")
k2 = st.number_input("K2 (D)", 35.0, 50.0, value=None, format="%.2f")
pachy = st.number_input("Corneal Thickness (µm)", 400, 700, value=None)
bcva = st.number_input("Pre-op BCVA (decimal)", 0.0, 1.2, value=None, format="%.2f")
age = st.number_input("Age", 10, 70, value=None)

if st.button("Predict"):
    if None in [sph, cyl, k1, k2, pachy, bcva, age]:
        st.warning("Please fill in all fields before predicting.")
    else:
        input_df = pd.DataFrame([[sph, cyl, k1, k2, pachy, bcva, age]],
                                columns=["Spherical", "Cylinder", "K1", "K2", "Pachymetry", "Pre-op BCVA", "Age"])

        reg_pred = reg_model.predict(input_df)[0]
        surgery = le.inverse_transform(class_model.predict(input_df))[0]

        st.subheader("📈 Predicted Post-Op Results")
        st.write(f"**BCVA:** {reg_pred[0]:.2f}")
        st.write(f"**K1:** {reg_pred[1]:.2f}")
        st.write(f"**K2:** {reg_pred[2]:.2f}")
        st.write(f"**Corneal Thickness:** {int(reg_pred[3])} µm")

        st.subheader("💡 Recommended Surgery Type")
        st.success(surgery)
