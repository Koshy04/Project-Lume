import torch
import traceback
import config
from faster_whisper import WhisperModel

whisper_model = None

def initialize_transcription_model():
    """Loads the Faster Whisper model into memory."""
    global whisper_model
    if whisper_model:
        return True

    print("Loading Faster Whisper model (this may take a minute)...")
    try:
        model_size = config.STT_MODEL
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        compute_precision = config.STT_COMPUTE_TYPE

        whisper_model = WhisperModel(model_size, device=device_type, compute_type=compute_precision)
        print(f"Faster Whisper model '{model_size}' loaded on '{device_type}' with compute type '{compute_precision}'.")
        return True
    except Exception as e:
        print(f"FATAL: Could not load Faster Whisper model: {e}. Transcription will fail.")
        traceback.print_exc()
        return False

def transcribe_audio(file_path: str) -> str:
    """Transcribes an audio file and returns the text."""
    if not whisper_model:
        print("Whisper model not loaded. Cannot transcribe.")
        return ""
    try:
        segments, info = whisper_model.transcribe(file_path, language="en", beam_size=config.STT_BEAM_SIZE)
        print(f"Detected language '{info.language}' with probability {info.language_probability} in {info.duration}s")
        
        transcription = "".join(segment.text for segment in segments).strip()
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        traceback.print_exc()
        return ""