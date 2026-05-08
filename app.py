import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("leaf_folder_legacy.h5", compile=False)

# Load scaler
scaler = joblib.load("scaler.pkl")

# App title
st.title("Rice Leaf Folder Forecasting System")

st.write("Enter weekly environmental parameters for prediction")

# User Inputs
temp = st.number_input("Weekly Mean Temperature")
rh = st.number_input("Weekly Mean Relative Humidity")
rf = st.number_input("Weekly Total Rainfall")
lf = st.number_input("Previous Week Mean Leaf Folder damaged leaves per Hill")

# Prediction
if st.button("Predict"):

    # Input array
    data = np.array([[temp, rh, rf, lf]])

    # Scale
    data_scaled = scaler.transform(data)

    # Reshape for LSTM
    data_scaled = data_scaled.reshape((1,1,4))

    # Predict
    prediction = model.predict(data_scaled)

    pred_value = float(prediction[0][0])

    # Risk category
    if pred_value < 5:
        risk = "Low"
        advice = "Regular monitoring recommended."
    elif pred_value < 10:
        risk = "Moderate"
        advice = "Field scouting and preventive measures advised."
    else:
        risk = "Severe"
        advice = "Immediate leaf folder management intervention recommended."

    # Output
    st.success(f"Predicted Leaf Folder Population: {pred_value:.2f}")
    st.warning(f"Risk Level: {risk}")
    st.info(f"Recommendation: {advice}")
