import numpy as np
from sklearn.model_selection import train_test_split

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

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)