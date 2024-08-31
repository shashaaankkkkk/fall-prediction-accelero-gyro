import streamlit as st

import numpy as np
import joblib

# Load the pre-trained model, scaler, and label encoder classes
model = joblib.load('fall_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_classes = joblib.load('label_encoder_classes.npy')

# Function to make a prediction
def predict_fall(features):
    # Standardize the input features
    features_scaled = scaler.transform([features])
    # Make the prediction
    prediction = model.predict(features_scaled)
    return label_encoder_classes[prediction][0]

# Streamlit app setup
st.title("Fall Detection Predictor")

st.write("This application predicts if a fall is detected based on sensor input data.")

# Input form for sensor data
with st.form(key='prediction_form'):
    st.header("Input Sensor Data")

    x_acc = st.number_input("X Acceleration (g)", format="%.2f")
    y_acc = st.number_input("Y Acceleration (g)", format="%.2f")
    z_acc = st.number_input("Z Acceleration (g)", format="%.2f")
    x_gyro = st.number_input("X Gyroscope (deg/s)", format="%.2f")
    y_gyro = st.number_input("Y Gyroscope (deg/s)", format="%.2f")
    z_gyro = st.number_input("Z Gyroscope (deg/s)", format="%.2f")

    submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Collect input features
        features = [x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro]
        # Make prediction
        prediction = predict_fall(features)
        st.write(f"Prediction: {prediction}")



