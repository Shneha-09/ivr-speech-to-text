import numpy as np
from jiwer import wer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

# ----------------- Example Ground Truth and Predictions -----------------

# Replace these with your real transcription texts
true_transcripts = [
    "Welcome to customer support. Please press 1 for billing.",
    "To speak to a representative, press 0 now.",
    "Your call is important to us. Please stay on the line.",
    "Say 'account' to access your account information.",
    "For technical support, press 2."
]

predicted_transcripts = [
    "Welcome to customer support. Please press 1 for billing.",
    "To speak to a representative, press 0 now.",
    "Your call is important to us. Please stay on the line.",
    "Say 'account' to access your account information.",
    "For technical support, press 2."
]

# For classification (e.g. Voice Activity Detection - VAD)
# Replace with your real labels and predictions (binary or multi-class)
y_true = np.array([1, 0, 1, 1, 0, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1])

# ----------------- SNR Calculation -----------------

def calculate_snr(clean_signal, noise_signal):
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean((noise_signal - clean_signal) ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Dummy audio signals (replace with your actual audio waveforms)
clean_signal = np.random.randn(16000)
noise_signal = clean_signal + 0.5 * np.random.randn(16000)
denoised_signal = clean_signal + 0.2 * np.random.randn(16000)

# ----------------- Evaluation -----------------

# 1. Word Error Rate (WER)
wers = [wer(t, p) for t, p in zip(true_transcripts, predicted_transcripts)]
avg_wer = np.mean(wers)
print(f"Average WER: {avg_wer:.3f}")

# 2. Accuracy, Precision, Recall, F1-Score for VAD or classification
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1_score:.3f}")

# 3. SNR Improvement
snr_before = calculate_snr(clean_signal, noise_signal)
snr_after = calculate_snr(clean_signal, denoised_signal)
snr_improvement = snr_after - snr_before

print(f"SNR Before: {snr_before:.2f} dB")
print(f"SNR After: {snr_after:.2f} dB")
print(f"SNR Improvement: {snr_improvement:.2f} dB")

# 4. Training Time and Inference Latency (Example with dummy model)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dummy model and data for timing demonstration
input_dim = 20
num_classes = 2

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')

X_train_dummy = np.random.randn(100, input_dim)
y_train_dummy = tf.keras.utils.to_categorical(np.random.randint(num_classes, size=100))

start_train = time.time()
model.fit(X_train_dummy, y_train_dummy, epochs=1, batch_size=16, verbose=0)
end_train = time.time()

print(f"Training Time: {end_train - start_train:.3f} seconds")

X_test_dummy = np.random.randn(10, input_dim)

start_infer = time.time()
model.predict(X_test_dummy)
end_infer = time.time()

avg_latency = (end_infer - start_infer) / len(X_test_dummy)
print(f"Average Inference Latency: {avg_latency * 1000:.3f} ms per sample")
