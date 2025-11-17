import os
import time
import wave
import torch
import config
import tempfile
import traceback
import numpy as np
import threading
import asyncio
from queue import Queue, Empty as QueueEmpty
from pynput import keyboard
import sounddevice as sd
from src.log.custom_logger import logger

# --- Configuration Constants for Local Audio ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BLOCKSIZE = 512
VAD_SPEECH_THRESHOLD = 0.5
VAD_SILENCE_TIMEOUT_CHUNKS = 25  # ~0.8 seconds of silence

class LocalAudioHandler:
    """
    Handles local audio input (PTT or VAD). Its sole responsibility is to capture
    raw audio data and pass it to the main async event loop via a thread-safe queue.
    It does NOT perform transcription.
    """
    def __init__(self, mode: str = "ptt", loop: asyncio.AbstractEventLoop = None, queue: asyncio.Queue = None):
        if not loop or not queue:
            raise ValueError("LocalAudioHandler requires an asyncio event loop and queue.")

        self.mode = mode
        self.loop = loop
        self.response_queue = queue
        self.raw_audio_queue = Queue()
        self._stop_event = threading.Event()

        self.ptt_key = keyboard.KeyCode.from_char(config.PTT_INPUT_KEY)
        self.is_recording_ptt = False
        self.ptt_recording_buffer = []

        self.vad_model, self.utils = None, None
        self.is_speaking_vad = False
        self.vad_recording_buffer = []
        self.vad_silence_chunks = 0

        self.ptt_thread = None
        self.stream_thread = None
        self.processing_thread = None
        logger.info(f"Local Audio Handler initialized in '{self.mode}' mode.")

    def start(self):
        """Starts all necessary threads for the selected audio mode."""
        logger.info("Starting Local Audio Handler services...")
        self._stop_event.clear()

        if self.mode == 'vad' and not self._initialize_vad():
            logger.critical("VAD model failed to initialize. Handler cannot start.")
            return

        self.processing_thread = threading.Thread(target=self._pass_audio_to_main_queue, daemon=True)
        self.processing_thread.start()

        self.stream_thread = threading.Thread(target=self._audio_stream_worker, daemon=True)
        self.stream_thread.start()

        if self.mode == 'ptt':
            self.ptt_thread = threading.Thread(target=self._ptt_listener_worker, daemon=True)
            self.ptt_thread.start()

        logger.info(f"--- Local {self.mode.upper()} Mode Activated ---")
        if self.mode == 'ptt':
            logger.info(f"Press and hold '{config.PTT_INPUT_KEY}' to talk.")
        else:
            logger.info("Listening for speech...")

    def stop(self):
        """Gracefully stops all running threads."""
        logger.info("Stopping Local Audio Handler services...")
        self._stop_event.set()
        time.sleep(0.2)
        if self.ptt_thread and self.ptt_thread.is_alive(): self.ptt_thread.join(timeout=1)
        if self.stream_thread and self.stream_thread.is_alive(): self.stream_thread.join(timeout=1)
        if self.processing_thread and self.processing_thread.is_alive(): self.processing_thread.join(timeout=1)
        logger.info("Local Audio Handler stopped.")

    def _initialize_vad(self):
        """Loads the Silero VAD model into memory."""
        try:
            self.vad_model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            logger.info("Silero VAD model loaded successfully.")
            return True
        except Exception as e:
            logger.critical(f"Error loading Silero VAD model: {e}\n{traceback.format_exc()}")
            return False

    def _ptt_listener_worker(self):
        """Listens for PTT key presses and releases."""
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _audio_stream_worker(self):
        """Opens and maintains the audio input stream from the microphone."""
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                callback=self._audio_callback, blocksize=BLOCKSIZE
            ):
                logger.info("Audio stream is now active.")
                self._stop_event.wait()
        except Exception as e:
            logger.critical(f"Could not open audio stream: {e}\n{traceback.format_exc()}")
        logger.info("Audio stream has been closed.")

    def _pass_audio_to_main_queue(self):
        """
        Worker thread that simply takes raw audio from the internal queue and
        places it onto the main Bot's asyncio queue for transcription and processing.
        """
        while not self._stop_event.is_set():
            try:
                audio_data = self.raw_audio_queue.get(timeout=1.0)
                user_id = config.user_name
                channel_id = "local_chat_history"
                item_to_queue = (user_id, audio_data, None, channel_id)
                self.loop.call_soon_threadsafe(self.response_queue.put_nowait, item_to_queue)
            except QueueEmpty:
                continue
            except Exception as e:
                logger.error(f"Error in local audio passing thread: {e}\n{traceback.format_exc()}")

    def _on_press(self, key):
        """Callback for when the PTT key is pressed."""
        if key == self.ptt_key and not self.is_recording_ptt:
            logger.info("Recording started...")
            self.is_recording_ptt = True
            self.ptt_recording_buffer.clear()

    def _on_release(self, key):
        """Callback for when the PTT key is released."""
        if key == self.ptt_key and self.is_recording_ptt:
            logger.info("Recording stopped.")
            self.is_recording_ptt = False
            if self.ptt_recording_buffer:
                audio_data = np.concatenate(self.ptt_recording_buffer, axis=0)
                self.raw_audio_queue.put(audio_data)

    def _audio_callback(self, indata, frames, time_info, status):
        """
        This function is called by the sounddevice stream for each new audio chunk.
        It sorts the audio into PTT or VAD buffers.
        """
        if status:
            logger.warning(status)

        if self._stop_event.is_set():
            raise sd.CallbackStop

        if self.mode == 'ptt':
            if self.is_recording_ptt:
                self.ptt_recording_buffer.append(indata.copy())

        elif self.mode == 'vad':
            audio_tensor = torch.from_numpy(indata.flatten().astype(np.float32) / 32768.0)
            speech_prob = self.vad_model(audio_tensor, SAMPLE_RATE).item()

            if speech_prob > VAD_SPEECH_THRESHOLD:
                if not self.is_speaking_vad:
                    logger.info("Speech detected...")
                    self.is_speaking_vad = True
                    self.vad_recording_buffer.clear()
                self.vad_silence_chunks = 0
                self.vad_recording_buffer.append(indata.copy())
            elif self.is_speaking_vad:
                self.vad_silence_chunks += 1
                self.vad_recording_buffer.append(indata.copy())
                if self.vad_silence_chunks > VAD_SILENCE_TIMEOUT_CHUNKS:
                    logger.info("Silence detected, processing speech.")
                    self.is_speaking_vad = False
                    audio_data = np.concatenate(self.vad_recording_buffer, axis=0)
                    self.raw_audio_queue.put(audio_data)
                    self.vad_recording_buffer.clear()