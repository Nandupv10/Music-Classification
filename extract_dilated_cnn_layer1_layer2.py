import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model


# ============================================================
# LOAD MFCC DATASET
# ============================================================

data = np.load("music_mfcc_dataset.npz")

X = data["X"]
y = data["y"]

print("Original MFCC shape:", X.shape)


# ============================================================
# CONVERT MFCC FORMAT
# (500, 20, 1293) -> (500, 1293, 20)
# ============================================================

X = np.transpose(X, (0, 2, 1))

print("CNN input shape:", X.shape)


# ============================================================
# SAME TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# LOAD THE ALREADY TRAINED CNN
# ============================================================

print("\n========================================")
print("Loading Saved Dilated CNN")
print("========================================")

cnn = tf.keras.models.load_model(
    "dilated_cnn_model.keras"
)

print("Loaded:")
print("dilated_cnn_model.keras")


# ============================================================
# CHECK SAVED CNN
# ============================================================

print("\n========================================")
print("Checking Saved CNN")
print("========================================")

test_loss, test_accuracy = cnn.evaluate(
    X_test,
    tf.keras.utils.to_categorical(y_test, num_classes=5),
    verbose=0
)

print("Standalone Dilated CNN Test Accuracy:")
print(f"{test_accuracy * 100:.2f}%")


# ============================================================
# EXTRACT LAYER 1 FEATURES
# ============================================================

print("\n========================================")
print("Extracting Layer 1 Features")
print("========================================")

layer1_extractor = Model(
    inputs=cnn.inputs,
    outputs=cnn.layers[0].output
)

X_train_layer1 = layer1_extractor.predict(
    X_train,
    verbose=0
)

X_test_layer1 = layer1_extractor.predict(
    X_test,
    verbose=0
)

print("\nLayer 1 before flattening:")
print("Training:", X_train_layer1.shape)
print("Testing:", X_test_layer1.shape)


# Flatten
X_train_layer1 = X_train_layer1.reshape(
    X_train_layer1.shape[0],
    -1
)

X_test_layer1 = X_test_layer1.reshape(
    X_test_layer1.shape[0],
    -1
)

print("\nLayer 1 after flattening:")
print("Training:", X_train_layer1.shape)
print("Testing:", X_test_layer1.shape)


# Save Layer 1
np.savez_compressed(
    "dilated_cnn_layer1_features.npz",
    X_train=X_train_layer1,
    X_test=X_test_layer1,
    y_train=y_train,
    y_test=y_test
)

print("\nSaved:")
print("dilated_cnn_layer1_features.npz")


# ============================================================
# EXTRACT LAYER 2 FEATURES
# ============================================================

print("\n========================================")
print("Extracting Layer 2 Features")
print("========================================")

layer2_extractor = Model(
    inputs=cnn.inputs,
    outputs=cnn.layers[4].output
)

X_train_layer2 = layer2_extractor.predict(
    X_train,
    verbose=0
)

X_test_layer2 = layer2_extractor.predict(
    X_test,
    verbose=0
)

print("\nLayer 2 before flattening:")
print("Training:", X_train_layer2.shape)
print("Testing:", X_test_layer2.shape)


# Flatten
X_train_layer2 = X_train_layer2.reshape(
    X_train_layer2.shape[0],
    -1
)

X_test_layer2 = X_test_layer2.reshape(
    X_test_layer2.shape[0],
    -1
)

print("\nLayer 2 after flattening:")
print("Training:", X_train_layer2.shape)
print("Testing:", X_test_layer2.shape)


# Save Layer 2
np.savez_compressed(
    "dilated_cnn_layer2_features.npz",
    X_train=X_train_layer2,
    X_test=X_test_layer2,
    y_train=y_train,
    y_test=y_test
)

print("\nSaved:")
print("dilated_cnn_layer2_features.npz")


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("FEATURE EXTRACTION COMPLETE")
print("========================================")

print("Saved files:")
print("1. dilated_cnn_layer1_features.npz")
print("2. dilated_cnn_layer2_features.npz")

print("\nCNN was NOT retrained.")
print("Features were extracted from the saved CNN.")