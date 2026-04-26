# ✍️ AirDraw Digit Recognition

## 📌 Overview

AirDraw is a deep learning project that recognizes digits (0–9) drawn in air using smartphone motion sensor data (accelerometer and gyroscope).
The system processes time-series sensor data and predicts the drawn digit using a CNN + LSTM model.

---

## 🚀 Features

* Time-series data processing from sensor recordings
* Feature engineering (magnitude + velocity)
* Deep learning model (CNN + LSTM)
* Model evaluation with accuracy, confusion matrix, and F1-score
* Streamlit web app for real-time prediction

---

## 🗂️ Project Structure

```
Airdraw/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── model/
│   ├── airdraw_model.h5
│   ├── mean.npy
│   └── std.npy
│
├── notebooks/
│   └── evaluation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* Collected using Physics Toolbox app
* 10 classes (digits 0–9)
* 300 samples per digit
* Total: 3000 samples

Each sample is converted into a tensor of shape:
```
(200, 14)
```

---

## ⚙️ Data Processing

* Resampling sequences to fixed length (200 timesteps)
* Removing time column
* Feature engineering:

  * Accelerometer + Gyroscope (6 features)
  * Magnitude features (2)
  * Velocity features (6)
* Normalization using mean and standard deviation

---

## 🧠 Model Architecture

```
Conv1D → Conv1D → LSTM → LSTM → Dense → Softmax
```

* CNN layers extract local motion patterns
* LSTM layers capture temporal dependencies

---

## 🏋️ Training

* Loss: Categorical Crossentropy
* Optimizer: Adam
* Early stopping applied
* Train/Validation/Test split: 64% / 16% / 20%

---

## 📈 Results

* Test Accuracy: **~81%**
* Evaluated using:

  * Confusion Matrix
  * Precision, Recall, F1-score

---

## 💻 Streamlit App

A simple web interface to test predictions.

### Run the app:

```bash
streamlit run app/app.py
```

### Usage:

1. Upload a CSV file
2. Click "Predict"
3. View predicted digit and confidence

---

## ⚡ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/airdraw.git
cd airdraw
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

* Increase dataset size for better generalization
* Experiment with Transformer-based models
* Improve accuracy beyond 85%
* Real-time prediction using mobile sensors
* Deploy as a mobile or web application

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Streamlit
* Matplotlib, Seaborn

---

## 📌 Conclusion

This project demonstrates how time-series sensor data can be used with deep learning to recognize air-drawn digits.
Feature engineering and sequence modeling significantly improved performance.

---


```
