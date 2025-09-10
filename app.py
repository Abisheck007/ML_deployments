import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("co2_model.pkl")

st.title("🚗 CO₂ Emissions Prediction")

# User inputs
eng_size = st.number_input("Engine Size (Liters)", min_value=0.0)
cylinders = st.number_input("Number of Cylinders", min_value=0, max_value=16, step=1)
fuel_comb = st.number_input("Fuel Consumption (L/100km - Combined)", min_value=0.0)

if st.button("Predict CO₂ Emissions"):
    features = np.array([[eng_size, cylinders, fuel_comb]])
    prediction = model.predict(features)[0]
    st.success(f"🌍 Estimated CO₂ Emissions: {prediction:.2f} g/km")
