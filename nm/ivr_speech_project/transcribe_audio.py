import whisper

model = whisper.load_model("base")
result = model.transcribe("path/to/your_audio.wav")
print("✅ Transcription:", result["text"])
