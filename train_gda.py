import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Load dataset
data = np.load("music_mfcc_dataset.npz")

X = data["X"]
y = data["y"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Flatten MFCC
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print("Before PCA:", X_train.shape)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=50, svd_solver="full")

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("After PCA:", X_train.shape)

# GDA (Linear Discriminant Analysis)
model = LinearDiscriminantAnalysis()

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("GDA Accuracy:", accuracy)