import sys
import numpy as np
import librosa
import joblib
import tensorflow as tf


# ============================================================
# GENRE LABELS
# ============================================================

genres = {
    0: "BLUES",
    1: "CLASSICAL",
    2: "HIPHOP",
    3: "METAL",
    4: "POP"
}


# ============================================================
# CHECK INPUT
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print("py -3.12 demo_predict.py path_to_song.wav")
    sys.exit()


file_path = sys.argv[1]

print("\nLoading song:")
print(file_path)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

cnn = tf.keras.models.load_model(
    "dilated_cnn_model.keras"
)

scaler = joblib.load(
    "layer1_scaler.pkl"
)

pca = joblib.load(
    "layer1_pca.pkl"
)

rf = joblib.load(
    "layer1_random_forest.pkl"
)

print("Saved models loaded successfully.")


# ============================================================
# LOAD AUDIO
# ============================================================

audio, sample_rate = librosa.load(
    file_path,
    sr=None
)

print("Sample rate:", sample_rate)
print("Audio length:", len(audio))


# ============================================================
# EXTRACT MFCC
# ============================================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=20
)

# Make exactly 1293 time frames
mfcc = librosa.util.fix_length(
    mfcc,
    size=1293,
    axis=1
)

print("MFCC shape:", mfcc.shape)


# ============================================================
# CNN INPUT
# (20, 1293) -> (1293, 20)
# ============================================================

cnn_input = np.transpose(
    mfcc,
    (1, 0)
)

cnn_input = np.expand_dims(
    cnn_input,
    axis=0
)

print("CNN input shape:", cnn_input.shape)


# ============================================================
# EXTRACT LAYER 1 FEATURES
# ============================================================

layer1_model = tf.keras.Model(
    inputs=cnn.inputs,
    outputs=cnn.layers[0].output
)

layer1_features = layer1_model.predict(
    cnn_input,
    verbose=0
)

print("Layer 1 shape:", layer1_features.shape)


# ============================================================
# FLATTEN
# ============================================================

layer1_features = layer1_features.reshape(
    1,
    -1
)


# ============================================================
# SAME PIPELINE AS TRAINING
# ============================================================

features_scaled = scaler.transform(
    layer1_features
)

features_pca = pca.transform(
    features_scaled
)


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

prediction = rf.predict(
    features_pca
)[0]

probabilities = rf.predict_proba(
    features_pca
)[0]


# ============================================================
# RESULT
# ============================================================

predicted_genre = genres[int(prediction)]

print("\n========================================")
print("       MUSIC GENRE PREDICTION")
print("========================================")

print("Predicted Genre:", predicted_genre)

print("\nConfidence scores:")

for i, probability in enumerate(probabilities):
    print(
        f"{genres[i]:10s}: {probability * 100:.2f}%"
    )

print("========================================")