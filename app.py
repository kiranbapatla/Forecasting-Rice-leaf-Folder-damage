import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU

# Build exact model architecture
def build_model():
    model = Sequential()

    model.add(LSTM(100, return_sequences=True, input_shape=(1,4)))
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

# Build model explicitly
model.build(input_shape=(None,1,4))

# Load weights
model.load_weights("leaf_folder.weights.h5")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Rice Leaf Folder Forecasting System")

temp = st.number_input("Weekly Mean Temperature")
rh = st.number_input("Weekly Mean Relative Humidity")
rf = st.number_input("Weekly Mean Rainfall")
lf = st.number_input("Previous Week Mean Leaf Folder damaged leaves per Hill")

if st.button("Predict"):

    data = np.array([[temp, rh, rf, lf]])

    # Scale
    data_scaled = scaler.transform(data)

    # Reshape for LSTM
    data_scaled = data_scaled.reshape((1,1,4))

    # Predict
    prediction = model.predict(data_scaled)

    pred_value = prediction[0][0]

    # Risk classification
    if pred_value < 5:
        risk = "Low"
    elif pred_value < 10:
        risk = "Moderate"
    else:
        risk = "Severe"

    st.success(f"Predicted Leaf Folder Population: {pred_value:.2f}")
    st.warning(f"Risk Level: {risk}")
