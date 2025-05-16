from jiwer import wer
from ivr_transcriber import transcribe_audio

ground_truth = "press one for support"
transcription = transcribe_audio("data/ivr_noisy.wav")

print("Transcription:", transcription)
print("WER (Word Error Rate):", wer(ground_truth.lower(), transcription.lower()))
