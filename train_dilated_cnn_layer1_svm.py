import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
# Load Layer 1 features
data = np.load("dilated_cnn_layer1_features.npz")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce dimensions using PCA
pca = PCA(n_components=50, svd_solver="full")
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("After PCA:")
print("Training shape:", X_train_pca.shape)
print("Testing shape:", X_test_pca.shape)

# SVM with RBF kernel
model = SVC(kernel="rbf")

model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)

print("\nDilated CNN Layer 1 + SVM")
print("Test Accuracy:", accuracy)
print("Test Accuracy (%):", accuracy * 100)