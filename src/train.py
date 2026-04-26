from data_loader import load_data
from preprocess import preprocess_data, normalize_data
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load data
X, y = load_data()

# Preprocess
X = preprocess_data(X)
X, mean, std = normalize_data(X)

# One-hot encode labels
y = to_categorical(y, num_classes=10)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Add noise to training data ONLY
#noise = np.random.normal(0, 0.02, X_train.shape)
#X_train = X_train + noise


print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Build model
from tensorflow.keras.layers import MaxPooling1D

model = Sequential([
    Conv1D(128, 3, activation='relu', input_shape=(200, 14)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(256, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),

    LSTM(128, return_sequences=True),
    LSTM(128),

    Dropout(0.4),

    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Save model
model.save("model/airdraw_model.h5")

# Save normalization parameters
np.save("model/mean.npy", mean)
np.save("model/std.npy", std)

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes))


