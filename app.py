import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import resample_sequence

# Load model and normalization
model = load_model("model/airdraw_model.h5")
mean = np.load("model/mean.npy")
std = np.load("model/std.npy")


def preprocess_file(df):
    # Rename columns
    df = df.rename(columns={
    # Accelerometer (handle both cases)
    "gFx": "ax", "gFy": "ay", "gFz": "az",
    "aFx": "ax", "aFy": "ay", "aFz": "az",

    # Gyroscope
    "wx": "gx", "wy": "gy", "wz": "gz"
})

    df["time"] = df["time"] - df["time"].iloc[0]

    seq = df[["time", "ax", "ay", "az", "gx", "gy", "gz"]].values
    seq = seq[:, 1:]

    # Split
    ax, ay, az = seq[:, 0], seq[:, 1], seq[:, 2]
    gx, gy, gz = seq[:, 3], seq[:, 4], seq[:, 5]

    # Magnitude
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Velocity
    velocity = np.diff(seq, axis=0)
    velocity = np.vstack((velocity[0], velocity))

    # Combine
    seq = np.column_stack((seq, acc_mag, gyro_mag, velocity))

    # Resample
    seq = resample_sequence(seq)

    # Normalize
    seq = (seq - mean) / (std + 1e-8)

    return np.squeeze(seq)[np.newaxis, :, :]


# UI
st.title("✍️ AirDraw Digit Recognition")

st.write("Upload a CSV file of air-drawn digit")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of data:")
    st.dataframe(df.head())

    if st.button("Predict"):
        processed = preprocess_file(df)

        prediction = model.predict(processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f}")