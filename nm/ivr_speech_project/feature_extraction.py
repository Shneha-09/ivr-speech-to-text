import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_features(audio_path, sr=16000):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)

    # 1. MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 2. Pitch (Fundamental Frequency using YIN algorithm)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sr
    )

    # 3. Energy (Root Mean Square Energy)
    energy = librosa.feature.rms(y=y)[0]

    return mfccs, f0, energy, sr

def plot_features(audio_path):
    mfccs, f0, energy, sr = extract_features(audio_path)

    plt.figure(figsize=(12, 10))

    # MFCC plot
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title("MFCCs")

    # Pitch plot
    plt.subplot(3, 1, 2)
    times = librosa.times_like(f0)
    plt.plot(times, f0, label='F0 (Pitch)', color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch (Fundamental Frequency)")
    plt.legend()

    # Energy plot
    plt.subplot(3, 1, 3)
    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=sr)
    plt.plot(t, energy, color='green', label='Energy')
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Energy")
    plt.title("Energy")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_path = r"C:\Users\shneh\nm\ivr_speech_project\data\ivr_clean.wav"  # Update if your file has a different name or path
    plot_features(audio_path)