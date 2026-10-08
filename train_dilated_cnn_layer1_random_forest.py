import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# LOAD LAYER 1 FEATURES
# ============================================================

data = np.load("dilated_cnn_layer1_features.npz")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# PCA
# ============================================================

pca = PCA(n_components=50, svd_solver="full")
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("After PCA:")
print("Training shape:", X_train_pca.shape)
print("Testing shape:", X_test_pca.shape)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_pca, y_train)


# ============================================================
# TEST
# ============================================================

y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)

print("\nDilated CNN Layer 1 + Random Forest")
print("Test Accuracy:", accuracy)
print("Test Accuracy (%):", accuracy * 100)


# ============================================================
# SAVE EXACT PIPELINE
# ============================================================

joblib.dump(scaler, "layer1_scaler.pkl")
joblib.dump(pca, "layer1_pca.pkl")
joblib.dump(model, "layer1_random_forest.pkl")

print("\n========================================")
print("MODELS SAVED")
print("========================================")
print("1. layer1_scaler.pkl")
print("2. layer1_pca.pkl")
print("3. layer1_random_forest.pkl")
print("========================================")