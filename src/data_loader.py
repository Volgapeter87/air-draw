import os
import pandas as pd
import numpy as np
import re

DATA_PATH = "data/raw"

def extract_label(filename):
    match = re.search(r'digit(\d)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def load_data():
    all_samples = []
    all_labels = []

    # Loop through folders (0–9)
    for folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder)

        if os.path.isdir(folder_path):

            for file in os.listdir(folder_path):
                if file.endswith(".csv"):

                    file_path = os.path.join(folder_path, file)

                    # Extract label from filename
                    label = extract_label(file)
                    if label is None:
                        continue

                    df = pd.read_csv(file_path)

                    # Rename columns- data standardization
                    df = df.rename(columns={
                        "gFx": "ax",
                        "gFy": "ay",
                        "gFz": "az",
                        "wx": "gx",
                        "wy": "gy",
                        "wz": "gz"
                    })

                    # Fix time- time column is normalized
                    df["time"] = df["time"] - df["time"].iloc[0]

                    # Select columns
                    df = df[["time", "ax", "ay", "az", "gx", "gy", "gz"]]

                    all_samples.append(df.values)
                    all_labels.append(label)

    return all_samples, np.array(all_labels)


if __name__ == "__main__":
    X, y = load_data()
    print("Total samples:", len(X))
    print("Labels shape:", y.shape)
    print("Sample labels:", y[:10])