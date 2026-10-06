import os
import librosa
import numpy as np

DATASET_PATH = "genres_original"

genres = ["blues", "classical", "hiphop", "metal", "pop"]

X = []
y = []

for label, genre in enumerate(genres):

    genre_path = os.path.join(DATASET_PATH, genre)

    print(f"Processing {genre}...")

    for file in os.listdir(genre_path):

        if file.endswith(".wav"):

            file_path = os.path.join(genre_path, file)

            audio, sample_rate = librosa.load(file_path)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=20
            )
            mfcc = librosa.util.fix_length(mfcc, size=1293, axis=1)   
            X.append(mfcc)
            y.append(label)

print("\nFinished!")
print("Number of songs:", len(X))
print("Labels:", len(y))

X = np.array(X)
y = np.array(y)

np.savez_compressed("music_mfcc_dataset.npz", X=X, y=y)

print("Saved dataset!")
print("X shape:", X.shape)
print("y shape:", y.shape)