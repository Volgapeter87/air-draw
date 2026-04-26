import numpy as np
from scipy.interpolate import interp1d


TARGET_LENGTH = 200


def resample_sequence(sequence, target_length=TARGET_LENGTH):
    """
    Resample sequence to fixed length
    """
    original_length = sequence.shape[0]

    # Original time steps
    original_indices = np.linspace(0, 1, original_length)

    # Target time steps
    target_indices = np.linspace(0, 1, target_length)

    resampled = []

    for i in range(sequence.shape[1]):  # for each feature
        f = interp1d(original_indices, sequence[:, i], kind='linear')
        resampled_feature = f(target_indices)
        resampled.append(resampled_feature)

    return np.stack(resampled, axis=1)


def preprocess_data(X):
    processed = []

    for seq in X:
        seq = seq[:, 1:]  # remove time


        # Split accel & gyro
        ax, ay, az = seq[:, 0], seq[:, 1], seq[:, 2]
        gx, gy, gz = seq[:, 3], seq[:, 4], seq[:, 5]

        # Magnitude features
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # Add new features 
        # Magnitude features
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # Velocity (difference between steps)
        velocity = np.diff(seq, axis=0)
        velocity = np.vstack((velocity[0], velocity))  # keep same length

        # Combine all features
        seq = np.column_stack((seq, acc_mag, gyro_mag, velocity))

        # Resample
        resampled = resample_sequence(seq)
        processed.append(resampled)

    return np.array(processed)


def normalize_data(X):
    """
    Normalize data using mean and std
    """
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)

    X_norm = (X - mean) / (std + 1e-8)

    return X_norm, mean, std


def trim_sequence(seq, threshold=0.02):
    """
    Remove idle parts where movement is very low
    """
    magnitude = np.linalg.norm(seq, axis=1)

    active = magnitude > threshold

    if np.any(active):
        start = np.argmax(active)
        end = len(active) - np.argmax(active[::-1])
        return seq[start:end]
    
    return seq