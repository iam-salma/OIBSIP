import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load model and scaler
model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

# UI
st.set_page_config(page_title="📈 Sales Predictor", layout="centered")
st.title("📊 Advertising Sales Predictor")
st.markdown("Enter advertisement spend across media to predict **sales** (in thousands of units).")

# Inputs
tv = st.number_input("TV Advertising Budget ($)", min_value=0.0, value=100.0)
radio = st.number_input("Radio Advertising Budget ($)", min_value=0.0, value=25.0)
newspaper = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, value=20.0)

if st.button("🔮 Predict Sales"):
    try:
        # Create DataFrame
        input_df = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"📈 Predicted Sales: **{prediction:.2f}k units**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
