import streamlit as st
import numpy as np
import joblib

def load_model():
    """Load the pre-trained model, scaler, and label encoder classes."""
    model = joblib.load('fall_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder_classes = joblib.load('label_encoder_classes.npy')
    return model, scaler, label_encoder_classes

def predict_fall(model, scaler, label_encoder_classes, features):
    """Predict if a fall is detected based on input features."""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return label_encoder_classes[prediction][0]

def main():
    """Main function to run the Streamlit app."""
    st.title("Fall Detection Predictor")
    st.write("This application predicts if a fall is detected based on sensor input data.")

    # Load model and preprocessing objects
    model, scaler, label_encoder_classes = load_model()

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
            prediction = predict_fall(model, scaler, label_encoder_classes, features)
            st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()

