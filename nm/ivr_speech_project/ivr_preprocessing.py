import librosa
import numpy as np
import soundfile as sf
import os

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def add_noise_to_audio(clean_audio_path, noise_audio_path, output_path, snr_db=10):
    clean = load_audio(clean_audio_path)
    noise = load_audio(noise_audio_path)

    # Pad noise if too short
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]

    # Adjust noise to SNR
    rms_clean = np.sqrt(np.mean(clean ** 2))
    rms_noise = np.sqrt(np.mean(noise ** 2))
    desired_rms_noise = rms_clean / (10 ** (snr_db / 20))
    noise = noise * (desired_rms_noise / rms_noise)

    mixed = clean + noise
    sf.write(output_path, mixed, 16000)
    print(f"Noisy audio saved at: {output_path}")
