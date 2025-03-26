import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Load data
data = pd.read_csv('data/ai4i2020.csv')

# Separate normal (non-failure) and anomaly (failure) data
normal_data = data[data['Machine failure'] == 0]  # Train only on normal data
anomaly_data = data[data['Machine failure'] == 1]  # Used for testing
print(len(normal_data))
print(len(anomaly_data))
# Select relevant features (sensor data)
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X_normal = normal_data[features].values
X_anomaly = anomaly_data[features].values

# Normalize data (fit only on normal data)
scaler = MinMaxScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_anomaly_scaled = scaler.transform(X_anomaly)

# Split normal data into train/test
X_train, X_test_normal = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)

# Combine test data (normal + anomalies for evaluation)
X_test = np.concatenate([X_test_normal, X_anomaly_scaled])
y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_anomaly_scaled))])  # 0=normal, 1=anomaly

# Build One-Class Autoencoder
input_dim = X_train.shape[1]
encoding_dim = 2  # Smaller latent space for compression

input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation='relu')(input_layer)
encoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(64, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train on normal data only
history = autoencoder.fit(
    X_train, X_train,  # Input and target are the same (reconstruction)
    epochs=50,
    batch_size=32,
    validation_data=(X_test_normal, X_test_normal),  # Validate on normal test data
    verbose=1
)

# Calculate reconstruction error on test data
reconstructions = autoencoder.predict(X_anomaly_scaled)
mse = np.mean(np.power(X_anomaly_scaled - reconstructions, 2), axis=1)

# Set threshold based on normal training data
train_reconstructions = autoencoder.predict(X_train)
train_mse = np.mean(np.power(X_train - train_reconstructions, 2), axis=1)
threshold = np.mean(train_mse) + 3 * np.std(train_mse)  # 3-sigma rule (adjust as needed)

# Classify anomalies
y_pred_anomaly = (mse > threshold).astype(int)  # 1 if above threshold (anomaly), else 0

# Evaluate performance only on anomaly data
y_true_anomaly = np.ones(len(X_anomaly_scaled))  # Ground truth: all are anomalies

print(f"Precision (Anomalies Only): {precision_score(y_true_anomaly, y_pred_anomaly):.3f}")
print(f"Recall (Anomalies Only): {recall_score(y_true_anomaly, y_pred_anomaly):.3f}")
print(f"F1-Score (Anomalies Only): {f1_score(y_true_anomaly, y_pred_anomaly):.3f}")
print(f"AUC-ROC (Anomalies Only): {roc_auc_score(y_true_anomaly, mse):.3f}")

# Plot reconstruction error distribution
plt.figure(figsize=(10, 6))
plt.hist(mse, bins=50, alpha=0.5, label='Anomaly', color='red')
plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.3f}')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Anomaly Data Reconstruction Error Distribution')
plt.show()

# Calculate reconstruction error on test data
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Set threshold based on normal training data
train_reconstructions = autoencoder.predict(X_train)
train_mse = np.mean(np.power(X_train - train_reconstructions, 2), axis=1)
threshold = np.mean(train_mse) + 3 * np.std(train_mse)  # 3-sigma rule (adjust as needed)

# Classify anomalies
y_pred = (mse > threshold).astype(int)

# Evaluate performance
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, mse):.3f}")  # AUC using reconstruction error scores

# Plot reconstruction error distribution
plt.figure(figsize=(10, 6))
plt.hist(mse[y_test == 0], bins=50, alpha=0.5, label='Normal', color='green')
plt.hist(mse[y_test == 1], bins=50, alpha=0.5, label='Anomaly', color='red')
plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.3f}')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Frequency')
plt.legend()
plt.title('One-Class Autoencoder: Anomaly Detection')
plt.show()