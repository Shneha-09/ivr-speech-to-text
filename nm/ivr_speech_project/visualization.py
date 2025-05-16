import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot_waveforms(y1, y2, sr, title1="Raw Audio", title2="Noise Reduced Audio"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(y1, sr=sr)
    plt.title(title1)
    plt.subplot(1, 2, 2)
    librosa.display.waveshow(y2, sr=sr)
    plt.title(title2)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, sr, title="Spectrogram"):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_mfcc(y, sr, title="MFCC"):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pitch_energy(y, sr):
    # Pitch (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)].mean()

    # Energy
    energy = np.sum(librosa.feature.rms(y=y), axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(energy, label="Energy")
    plt.title("Energy over time")
    plt.xlabel("Frame")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Estimated average pitch: {pitch:.2f} Hz")

def plot_vad(y, sr, frame_length=2048, hop_length=512, threshold_ratio=1.5):
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.mean(energy) * threshold_ratio
    speech_frames = energy > threshold

    times = librosa.frames_to_time(np.arange(len(speech_frames)), sr=sr, hop_length=hop_length)
    plt.figure(figsize=(10, 3))
    plt.plot(times, energy, label='Frame Energy')
    plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
    plt.fill_between(times, 0, energy, where=speech_frames, color='green', alpha=0.5, label='Speech')
    plt.title("VAD - Energy Based")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- Load and Plot ----------

# ---------- Load and Plot ----------

base_dir = os.path.join(os.getcwd(), "ivr_speech_project", "data")
raw_audio_path = os.path.join(base_dir, "ivr_clean.wav")
denoised_audio_path = os.path.join(base_dir, "ivr_noise.wav")  # Update this to "ivr_noise.wav" if needed

try:
    y_raw, sr = librosa.load(raw_audio_path, sr=16000)
    print("✅ Raw audio loaded.")
except FileNotFoundError:
    print(f"❌ Raw audio file not found at {raw_audio_path}")
    exit(1)

try:
    y_denoised, _ = librosa.load(denoised_audio_path, sr=16000)
    print("✅ Denoised audio loaded.")
except FileNotFoundError:
    print(f"⚠️ Denoised audio file not found at {denoised_audio_path}. Using raw audio as placeholder.")
    y_denoised = y_raw  # fallback

# Run visualizations
plot_waveforms(y_raw, y_denoised, sr)
plot_spectrogram(y_raw, sr, "Raw Audio Spectrogram")
plot_spectrogram(y_denoised, sr, "Denoised Audio Spectrogram")
plot_mfcc(y_raw, sr, "MFCC - Raw Audio")
plot_pitch_energy(y_raw, sr)
plot_vad(y_raw, sr)
