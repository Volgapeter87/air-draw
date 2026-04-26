from data_loader import load_data
from preprocess import preprocess_data, normalize_data

# Load data
X, y = load_data()

# Preprocess
X_processed = preprocess_data(X)

# Normalize
X_normalized, mean, std = normalize_data(X_processed)

print("Final shape:", X_normalized.shape)
print("Labels shape:", y.shape)