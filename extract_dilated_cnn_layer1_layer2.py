import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D,
    AveragePooling1D,
    Dropout,
    BatchNormalization,
    Flatten,
    Dense
)
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Load dataset
# -----------------------------
data = np.load("music_mfcc_dataset.npz")

X = data["X"]
y = data["y"]

print("Original MFCC shape:", X.shape)

# Convert:
# (500, 20, 1293) -> (500, 1293, 20)
X = np.transpose(X, (0, 2, 1))

print("CNN input shape:", X.shape)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train_cat = to_categorical(y_train, num_classes=5)

# -----------------------------
# Dilated CNN
# -----------------------------
cnn = Sequential([
    Conv1D(
        filters=8,
        kernel_size=8,
        dilation_rate=4,
        activation="relu",
        input_shape=(1293, 20)
    ),

    AveragePooling1D(pool_size=32),

    Dropout(0.5),

    BatchNormalization(),

    Conv1D(
        filters=16,
        kernel_size=4,
        dilation_rate=2,
        activation="relu"
    ),

    AveragePooling1D(pool_size=8),

    Dropout(0.5),

    BatchNormalization(),

    Flatten(),

    Dense(128, activation="relu"),

    Dense(5, activation="softmax")
])

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train CNN
# -----------------------------
cnn.fit(
    X_train,
    y_train_cat,
    epochs=20,
    batch_size=16,
    validation_split=0.2
)

# -----------------------------
# Extract Layer 1
# -----------------------------
layer1_extractor = Model(
    inputs=cnn.inputs,
    outputs=cnn.layers[0].output
)

X_train_layer1 = layer1_extractor.predict(X_train)
X_test_layer1 = layer1_extractor.predict(X_test)

print("\nLayer 1 before flattening:")
print("Training:", X_train_layer1.shape)
print("Testing:", X_test_layer1.shape)

X_train_layer1 = X_train_layer1.reshape(
    X_train_layer1.shape[0], -1
)

X_test_layer1 = X_test_layer1.reshape(
    X_test_layer1.shape[0], -1
)

print("Layer 1 after flattening:")
print("Training:", X_train_layer1.shape)
print("Testing:", X_test_layer1.shape)

np.savez_compressed(
    "dilated_cnn_layer1_features.npz",
    X_train=X_train_layer1,
    X_test=X_test_layer1,
    y_train=y_train,
    y_test=y_test
)

# -----------------------------
# Extract Layer 2
# -----------------------------
layer2_extractor = Model(
    inputs=cnn.inputs,
    outputs=cnn.layers[4].output
)

X_train_layer2 = layer2_extractor.predict(X_train)
X_test_layer2 = layer2_extractor.predict(X_test)

print("\nLayer 2 before flattening:")
print("Training:", X_train_layer2.shape)
print("Testing:", X_test_layer2.shape)

X_train_layer2 = X_train_layer2.reshape(
    X_train_layer2.shape[0], -1
)

X_test_layer2 = X_test_layer2.reshape(
    X_test_layer2.shape[0], -1
)

print("Layer 2 after flattening:")
print("Training:", X_train_layer2.shape)
print("Testing:", X_test_layer2.shape)

np.savez_compressed(
    "dilated_cnn_layer2_features.npz",
    X_train=X_train_layer2,
    X_test=X_test_layer2,
    y_train=y_train,
    y_test=y_test
)

print("\nBoth Dilated CNN feature sets saved successfully!")