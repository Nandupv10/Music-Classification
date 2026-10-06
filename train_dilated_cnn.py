import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
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
# Load MFCC dataset
# -----------------------------
data = np.load("music_mfcc_dataset.npz")

X = data["X"]
y = data["y"]

print("Original MFCC shape:", X.shape)

# X = (500, 20, 1293)
# Convert to (500, 1293, 20)
# 1293 = time steps
# 20 = MFCC features
X = np.transpose(X, (0, 2, 1))

print("CNN input shape:", X.shape)

# -----------------------------
# Train-test split
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

# -----------------------------
# 2-Layer Dilated CNN
# -----------------------------
model = Sequential([

    # Dilated Convolution Layer 1
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

    # Dilated Convolution Layer 2
    Conv1D(
        filters=16,
        kernel_size=4,
        dilation_rate=2,
        activation="relu"
    ),

    AveragePooling1D(pool_size=8),

    Dropout(0.5),

    BatchNormalization(),

    # Classifier
    Flatten(),

    Dense(128, activation="relu"),

    Dense(5, activation="softmax")
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Show architecture
model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train_cat,
    epochs=20,
    batch_size=16,
    validation_split=0.2
)

# -----------------------------
# Test
# -----------------------------
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_cat
)

print("\nDilated CNN Test Accuracy:", test_accuracy)