import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

fuel_le = joblib.load(os.path.join(MODELS_DIR, 'fuel_label_encoder.pkl'))
sell_le = joblib.load(os.path.join(MODELS_DIR, 'sell_label_encoder.pkl'))
trans_le = joblib.load(os.path.join(MODELS_DIR, 'trans_label_encoder.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))

st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor")
st.markdown("Estimate a car's **selling price** based on its features.")

# User inputs
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, max_value=100.0, value=6.5)
year = st.selectbox("Year of Manufacture", list(range(2005, 2018))[::-1])
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000)
fuel_type = st.selectbox("Fuel Type", fuel_le.classes_)
transmission = st.selectbox("Transmission", trans_le.classes_)
selling_type = st.selectbox("Seller Type", sell_le.classes_)
owner_map = {"Yes": 0, "No": 1}
owner_display = st.selectbox("Are you the owner of the car?", list(owner_map.keys()))
owner = owner_map[owner_display]

if st.button("🔮 Predict Price"):
    try:
        # Encode categorical variables
        fuel_encoded = fuel_le.transform([fuel_type])[0]
        trans_encoded = trans_le.transform([transmission])[0]
        sell_encoded = sell_le.transform([selling_type])[0]

        # Prepare input
        input_data = pd.DataFrame([[year, present_price, kms_driven, fuel_encoded, sell_encoded, trans_encoded, owner]],
                                    columns=['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner'])

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]

        st.success(f"💰 Predicted Selling Price: ₹{predicted_price:.2f} Lakhs")
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")