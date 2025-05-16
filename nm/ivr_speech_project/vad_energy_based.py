import librosa
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt

def vad_energy_based(audio_path, output_path, frame_length=2048, hop_length=512, energy_threshold_ratio=0.5):
    # Load the audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Compute short-time energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Set energy threshold
    threshold = np.mean(rms) * energy_threshold_ratio

    # Identify speech frames
    speech_frames = rms > threshold

    # Map frames to samples
    speech_mask = np.zeros(len(y), dtype=bool)
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * hop_length
            end = min(start + frame_length, len(y))
            speech_mask[start:end] = True

    # Extract and save speech-only audio
    speech_audio = y[speech_mask]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, speech_audio, sr)
    print(f"Speech-only audio saved to: {output_path}")

    # Optional: Plot RMS and detected regions
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms, label="RMS Energy")
    plt.hlines(threshold, times[0], times[-1], color='r', linestyle='--', label="Threshold")
    plt.fill_between(times, 0, rms, where=speech_frames, color='green', alpha=0.4, label="Detected Speech")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Voice Activity Detection (Energy-Based)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_audio = r"C:\Users\shneh\nm\ivr_speech_project\data\ivr_clean.wav"
    output_audio = r"C:\Users\shneh\nm\ivr_speech_project\data\ivr_noise.wav"
    vad_energy_based(input_audio, output_audio)
