import librosa
import matplotlib.pyplot as plt
import librosa.display

file_path = "genres_original/blues/blues.00000.wav"

audio, sample_rate = librosa.load(file_path)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=20
)

print("MFCC shape:", mfcc.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    mfcc,
    x_axis="time",
    sr=sample_rate
)
plt.colorbar()
plt.title("MFCC of Blues Song")
plt.tight_layout()
plt.show()