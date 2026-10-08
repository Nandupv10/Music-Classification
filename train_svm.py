import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = np.load("music_mfcc_dataset.npz")

X = data["X"]
y = data["y"]

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

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA → 50 components
pca = PCA(n_components=50, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("After PCA:", X_train.shape)

# SVM
model = SVC(kernel="rbf")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("SVM Accuracy:", accuracy)