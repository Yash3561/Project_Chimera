import os
from transformers import pipeline

class AudioTool:
    def __init__(self):
        print("Initializing Audio Tool (Whisper)...")
        self.pipe = pipeline("automatic-speech-recognition", model="distil-whisper/distil-medium.en")
        print("Audio Tool ready.")

    def transcribe(self, audio_path: str, output_filename: str) -> str:
        """Transcribes an audio file and saves the full text to a file."""
        full_audio_path = os.path.join("sandbox", audio_path.lstrip('/\\'))
        if not os.path.exists(full_audio_path):
            return f"Error: Audio file not found at '{audio_path}'."

        full_output_path = os.path.join("sandbox", output_filename.lstrip('/\\'))
        if not os.path.abspath(full_output_path).startswith(os.path.abspath("sandbox")):
            return f"Error: Access denied. Output path is outside the sandbox."

        try:
            transcript_result = self.pipe(full_audio_path, chunk_length_s=30, batch_size=8, return_timestamps=False)
            transcript_text = transcript_result['text']
            
            with open(full_output_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            
            return f"Successfully transcribed '{audio_path}' and saved the transcript to '{output_filename}'."
        except Exception as e:
            return f"Error processing audio: {e}"