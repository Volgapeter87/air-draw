import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from preprocess import resample_sequence

# Load model + normalization
model = load_model("model/airdraw_model.h5")
mean = np.load("model/mean.npy")
std = np.load("model/std.npy")


def preprocess_single_file(file_path):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Rename columns
    df = df.rename(columns={
    
    # Accelerometer (handle both cases)
    "gFx": "ax", "gFy": "ay", "gFz": "az",
    "aFx": "ax", "aFy": "ay", "aFz": "az",

    # Gyroscope
    "wx": "gx", "wy": "gy", "wz": "gz"
})

    # Fix time
    df["time"] = df["time"] - df["time"].iloc[0]

    # Select columns
    seq = df[["time", "ax", "ay", "az", "gx", "gy", "gz"]].values

    # Remove time
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

    # Combine → 14 features
    seq = np.column_stack((seq, acc_mag, gyro_mag, velocity))

    # Resample
    seq = resample_sequence(seq)

    # Normalize
    seq = (seq - mean) / (std + 1e-8)
    


    if len(seq.shape) == 2:
       seq = seq[np.newaxis, :, :]
    elif len(seq.shape) == 3:
       seq = seq  # already correct

    return seq


def predict(file_path):
    processed = preprocess_single_file(file_path)

    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    return predicted_class, confidence


if __name__ == "__main__":
    file_path = "sample.csv" 

    digit, conf = predict(file_path)

    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {conf:.2f}")