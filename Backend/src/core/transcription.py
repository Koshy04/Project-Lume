import torch
import traceback
import config
from faster_whisper import WhisperModel

class Transcription:
    """
    A class to encapsulate the speech-to-text model and its functionality.
    It handles model loading and provides a simple interface for transcription.
    """
    def __init__(self):
        self.model: WhisperModel | None = None
        self.is_initialized = False

    def initialize_model(self) -> bool:
        """Loads the Faster Whisper model into memory. Returns True on success."""
        if self.is_initialized:
            return True

        print("Loading Faster Whisper model (this may take a minute)...")
        try:
            model_size = config.STT_MODEL
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            compute_precision = config.STT_COMPUTE_TYPE

            self.model = WhisperModel(model_size, device=device_type, compute_type=compute_precision)
            self.is_initialized = True
            
            print(f"Faster Whisper model '{model_size}' loaded on '{device_type}' with compute type '{compute_precision}'.")
            return True
        except Exception as e:
            print(f"FATAL: Could not load Faster Whisper model: {e}. Transcription will fail.")
            traceback.print_exc()
            return False

    def transcribe(self, file_path: str) -> str:
        """Transcribes an audio file and returns the text."""
        if not self.is_initialized or not self.model:
            print("Whisper model not loaded. Cannot transcribe.")
            return ""
        try:
            segments, info = self.model.transcribe(file_path, language="en", beam_size=config.STT_BEAM_SIZE)
            
            # Optional: Log detected language for debugging
            # print(f"Detected language '{info.language}' with probability {info.language_probability} in {info.duration}s")
            
            transcription = "".join(segment.text for segment in segments).strip()
            return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
            traceback.print_exc()
            return ""