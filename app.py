import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the trained model
model = joblib.load("train_delay_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Streamlit UI
st.title("🚆 Train Delay Prediction System")
st.write("Enter the details to predict train delay.")

# User inputs
distance = st.number_input("Distance Between Stations (km)", min_value=1, max_value=1000, step=1)
weather = st.selectbox("Weather Conditions", label_encoders['Weather Conditions'].classes_)
day_of_week = st.selectbox("Day of the Week", label_encoders['Day of the Week'].classes_)
time_of_day = st.selectbox("Time of Day", label_encoders['Time of Day'].classes_)
train_type = st.selectbox("Train Type", label_encoders['Train Type'].classes_)
route_congestion = st.selectbox("Route Congestion", label_encoders['Route Congestion'].classes_)

# Encode user inputs
input_data = np.array([
    distance,
    label_encoders['Weather Conditions'].transform([weather])[0],
    label_encoders['Day of the Week'].transform([day_of_week])[0],
    label_encoders['Time of Day'].transform([time_of_day])[0],
    label_encoders['Train Type'].transform([train_type])[0],
    label_encoders['Route Congestion'].transform([route_congestion])[0]
]).reshape(1, -1)

# Prediction
if st.button("Predict Delay"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Train Delay: {prediction:.2f} minutes")

st.write("Powered by Machine Learning 🚀")
