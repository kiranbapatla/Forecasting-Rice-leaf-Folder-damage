import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, Input

# Rebuild model architecture
def build_model():
    model = Sequential()
    model.add(Input(shape=(1,4)))

    model.add(LSTM(100, return_sequences=True))
    model.add(LeakyReLU(alpha=0.5))

    model.add(LSTM(100, return_sequences=True))
    model.add(LeakyReLU(alpha=0.5))

    model.add(Dropout(0.3))

    model.add(LSTM(50, return_sequences=False))

    model.add(Dropout(0.3))

    model.add(Dense(1, activation='linear'))

    return model

# Load model
model = build_model()
model.load_weights("leaf_folder.weights.h5")

# Load scaler
scaler = joblib.load("scaler.pkl")

# UI
st.title("Rice Leaf Folder Forecasting")

temp = st.number_input("Weekly Mean Temperature")
rh = st.number_input("Weekly Mean Relative Humidity")
rf = st.number_input("Weekly Total Rainfall")
lf = st.number_input("Previous Week Mean Leaf Folder damaged leaves per Hill")

if st.button("Predict"):
    data = np.array([[temp, rh, rf, lf]])

    data_scaled = scaler.transform(data)

    data_scaled = data_scaled.reshape((1,1,4))

    prediction = model.predict(data_scaled)

    st.success(f"Predicted Leaf Folder Population: {prediction[0][0]:.2f}")
