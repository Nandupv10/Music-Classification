import librosa

file_path = "genres_original/blues/blues.00000.wav"

audio, sample_rate = librosa.load(file_path)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio length:", len(audio))