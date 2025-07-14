import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
t_encoder     = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.pkl'))
scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
model         = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))

region_list = [
    "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Meghalaya", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Chandigarh"
]

st.set_page_config(page_title="Unemployment Predictor", layout="centered")
st.title("📊 Unemployment Rate Predictor")
st.markdown("Predict the **Estimated Unemployment Rate (%)** based on economic indicators and region.")

# --- Inputs ---
area = st.selectbox("Select Area", label_encoder.classes_)
region = st.selectbox("Select Region", region_list)
year = st.selectbox("Select Year", [2019, 2020])
month = st.number_input("Enter Month", min_value=1, max_value=12, value=6)
labour_rate = st.number_input("Estimated Labour Participation Rate (%)", value=40.0)
employed = st.number_input("Estimated Employed", value=35.0)

# --- Predict Button ---
if st.button("🔮 Predict Unemployment Rate"):
    try:
        # Encoding
        region_df = pd.DataFrame({'Region': [region]})
        region_encoded = t_encoder.transform(region_df)['Region'].values[0]
        area_encoded = label_encoder.transform([area])[0]

        # Create input DataFrame
        input_df = pd.DataFrame([[region_encoded, employed, labour_rate, area_encoded, month, year]],
                                columns=["Region", "Estimated Employed", "Estimated Labour Participation Rate (%)", "Area", "month", "year"])

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]

        st.success(f"📈 Predicted Unemployment Rate: **{pred:.2f}%**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
