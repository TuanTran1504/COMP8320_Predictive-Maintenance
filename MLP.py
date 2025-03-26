import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# 1. Read Data from 5 CSV Files
#    Replace these filenames with your actual CSV filenames for unbalance levels 0..4
df0 = pd.read_csv("data/0E/0E.csv")
df1 = pd.read_csv("data/1E/1E.csv")
df2 = pd.read_csv("data/2E/2E.csv")
df3 = pd.read_csv("data/3E/3E.csv")
df4 = pd.read_csv("data/4E/4E.csv")
df0["unbalance"] = 0
df1["unbalance"] = 1
df2["unbalance"] = 2
df3["unbalance"] = 3
df4["unbalance"] = 4

# 3. Combine All Data
df_all = pd.concat([df0, df1, df2, df3, df4], ignore_index=True)

# 4. Optional: Basic Data Cleaning
#    - Example: remove negative or zero RPM if it makes no sense physically.
#    - Adjust this step to your domain knowledge.
df_all = df_all[df_all["Measured_RPM"] > 0].copy()

# 5. Define Features (X) and Target (y)
#    The columns will vary depending on your data structure.
#    Common columns might be V_in, Measured_RPM, Vibration_1, Vibration_2, Vibration_3.
feature_cols = ["V_in", "Measured_RPM", "Vibration_1", "Vibration_2", "Vibration_3"]
X = df_all[feature_cols].values  # as NumPy array
y = df_all["unbalance"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3,       # 30% goes to temp
    random_state=42, 
    stratify=y           # preserve class proportions
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5,       # half of temp to val, half to test
    random_state=42,
    stratify=y_temp
)

print("Train distribution:\n", pd.Series(y_train).value_counts())
print("Val distribution:\n", pd.Series(y_val).value_counts())
print("Test distribution:\n", pd.Series(y_test).value_counts())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# 7. Random Forest Classifier
num_features = X_train_scaled.shape[1]
num_classes = len(np.unique(y))  # should be 5 for unbalance 0..4

model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(526, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")  #   for multi-class classification
])

model.compile(
    loss="sparse_categorical_crossentropy",  # y is integer-labeled
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()
batch_size = 1024  # Larger batch sizes can speed up training for big data, if you have enough GPU memory
epochs = 30      # Increase if needed

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import classification_report

y_pred_probs = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_test, y_pred_classes))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot training & validation loss
plt.figure(figsize=(12, 5))

# --- Loss Plot ---
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# --- Accuracy Plot ---
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()