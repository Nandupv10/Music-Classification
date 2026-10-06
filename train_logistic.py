import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = np.load("music_mfcc_dataset.npz")

X = data["X"]
y = data["y"]

# Split into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Flatten MFCC for Logistic Regression
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Create and train Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", accuracy)