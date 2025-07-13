import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

iris = load_iris()

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

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
