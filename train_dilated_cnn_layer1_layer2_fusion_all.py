import os
import random
import numpy as np

# ========================================
# Reproducibility
# ========================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)


# ========================================
# Imports
# ========================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ========================================
# Load Layer 1 Features
# ========================================

layer1 = np.load("dilated_cnn_layer1_features.npz")

X_train_layer1 = layer1["X_train"]
X_test_layer1 = layer1["X_test"]

y_train = layer1["y_train"]
y_test = layer1["y_test"]


# ========================================
# Load Layer 2 Features
# ========================================

layer2 = np.load("dilated_cnn_layer2_features.npz")

X_train_layer2 = layer2["X_train"]
X_test_layer2 = layer2["X_test"]


# ========================================
# Combine Layer 1 + Layer 2
# ========================================

X_train = np.concatenate(
    [X_train_layer1, X_train_layer2],
    axis=1
)

X_test = np.concatenate(
    [X_test_layer1, X_test_layer2],
    axis=1
)

print("Layer 1 Shape:", X_train_layer1.shape)
print("Layer 2 Shape:", X_train_layer2.shape)

print("\nCombined Feature Shape:")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)


# ========================================
# Standardization
# ========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ========================================
# PCA → 50 Components
# ========================================

pca = PCA(
    n_components=50,
    svd_solver="full"
)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("\nAfter PCA:", X_train.shape)


# ========================================
# 1. Logistic Regression
# ========================================

lr = LogisticRegression(
    max_iter=1000,
    random_state=SEED
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)


# ========================================
# 2. GDA
# ========================================

gda = LinearDiscriminantAnalysis()

gda.fit(X_train, y_train)

gda_pred = gda.predict(X_test)

gda_accuracy = accuracy_score(y_test, gda_pred)


# ========================================
# 3. Random Forest
# ========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)


# ========================================
# 4. SVM
# ========================================

svm = SVC(
    kernel="rbf"
)

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)


# ========================================
# Final Results
# ========================================

print("\n========================================")
print("Layer 1 + Layer 2 Feature Fusion")
print("========================================")

print(f"Logistic Regression: {lr_accuracy * 100:.2f}%")
print(f"GDA:                 {gda_accuracy * 100:.2f}%")
print(f"Random Forest:       {rf_accuracy * 100:.2f}%")
print(f"SVM:                 {svm_accuracy * 100:.2f}%")

print("========================================")