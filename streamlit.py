import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model and scaler
model = joblib.load('fall_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_classes = joblib.load('label_encoder_classes.npy')

# Create a function to predict fall
def predict_fall(features):
    # Standardize the features
    features_scaled = scaler.transform([features])
    # Make prediction
    prediction = model.predict(features_scaled)
    return label_encoder_classes[prediction][0]

# Streamlit app
st.title("Fall Detection Predictor")

# Input form
with st.form(key='prediction_form'):
    st.header("Input Sensor Data")

    x_acc = st.number_input("X Acceleration", format="%.2f")
    y_acc = st.number_input("Y Acceleration", format="%.2f")
    z_acc = st.number_input("Z Acceleration", format="%.2f")
    x_gyro = st.number_input("X Gyroscope", format="%.2f")
    y_gyro = st.number_input("Y Gyroscope", format="%.2f")
    z_gyro = st.number_input("Z Gyroscope", format="%.2f")

    submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Collect input features
        features = [x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro]
        # Predict
        prediction = predict_fall(features)
        st.write(f"Prediction: {prediction}")


