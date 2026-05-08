import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved files
model = load_model("leaf_folder_model.h5",, compile=False)
scaler = joblib.load("scaler.pkl")

st.title("Rice Leaf Folder Forecasting System")

temp = st.number_input("Weekly Mean Temperature")
rh = st.number_input("Weekly Mean Relative Humidity")
rf = st.number_input("Weekly Total Rainfall")
lf = st.number_input("Previous Week Mean Leaf Folder damaged leaves per Hill")

if st.button("Predict"):
    data = np.array([[temp, rh, rf, lf]])

    # Scale
    data_scaled = scaler.transform(data)

    # Reshape for LSTM
    data_scaled = data_scaled.reshape((1, 1, 4))

    prediction = model.predict(data_scaled)

    st.success(f"Predicted Leaf Folder: {prediction[0][0]:.2f}")
