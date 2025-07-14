import streamlit as st
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
model = joblib.load(os.path.join(MODELS_DIR, 'iris_model.pkl'))

from sklearn.datasets import load_iris
iris = load_iris()

# Streamlit UI
st.title("🌸 Iris Species Predictor")
st.write("Input flower measurements and get the predicted species.")

# User inputs
sl = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sw = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
pl = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
pw = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

# Create input DataFrame
input_df = pd.DataFrame([[sl, sw, pl, pw]], columns=iris.feature_names)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0]

# Output
st.success(f"🌼 Predicted Species: **{prediction.capitalize()}**")
