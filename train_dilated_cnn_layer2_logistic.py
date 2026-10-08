import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Layer 2 features
data = np.load("dilated_cnn_layer2_features.npz")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nDilated CNN Layer 2 + Logistic Regression")
print("Test Accuracy:", accuracy)
print("Test Accuracy (%):", accuracy * 100)
