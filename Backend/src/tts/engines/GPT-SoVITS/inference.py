import sys
import os
import traceback
import asyncio
from . import config as engine_config
from src.log.custom_logger import logger

class TTSEngine:
    """
    Encapsulates all logic for initializing and using the GPT-SoVITS model.
    """
    def __init__(self):
        self.is_initialized = False
        self.i18n = None
        self.get_tts_wav = None
        self.change_gpt_weights = None
        self.change_sovits_weights = None
        self._load_libraries()

    def _load_libraries(self):
        """Dynamically imports the necessary functions from the GPT-SoVITS library."""
        base_path = engine_config.BASE_PATH
        
        if not os.path.exists(base_path):
            logger.warning(f"[GPT_SoVITS] Base path not found at '{base_path}'. This engine will be disabled.")
            return

        if base_path not in sys.path:
            sys.path.append(base_path)
        
        try:
            from inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
            from tools.i18n.i18n import I18nAuto

            self.get_tts_wav = get_tts_wav
            self.change_gpt_weights = change_gpt_weights
            self.change_sovits_weights = change_sovits_weights
            self.i18n = I18nAuto()
        except ImportError as e:
            logger.error(f"[GPT_SoVITS] Failed to import modules: {e}\n{traceback.format_exc()}")

    def initialize(self) -> bool:
        """Initializes the TTS models using paths from its own config. Returns True on success."""
        if self.is_initialized:
            return True
        
        if not all([self.get_tts_wav, self.change_gpt_weights, self.change_sovits_weights, self.i18n]):
            logger.critical("[GPT_SoVITS] Library functions were not loaded. Cannot initialize.")
            return False

        try:
            self.change_gpt_weights(gpt_path=engine_config.GPT_MODEL_PATH)
            self.change_sovits_weights(sovits_path=engine_config.SOVITS_MODEL_PATH)
            
            self.is_initialized = True
            logger.info("[GPT_SoVITS] Engine initialized successfully.")
            return True
        except Exception as e:
            logger.critical(f"[GPT_SoVITS] FATAL INITIALIZATION ERROR: {e}")
            return False

    def stream_generate(
        self,
        text: str,
        queue_to_async: asyncio.Queue,
        event_loop_for_queue: asyncio.AbstractEventLoop
    ):
        """
        Blocking function that runs in a separate thread to generate audio.
        """
        if not self.is_initialized:
            error_msg = "GPT-SoVITS Inference: Models not initialized. Aborting."
            logger.error(error_msg)
            asyncio.run_coroutine_threadsafe(queue_to_async.put(Exception(error_msg)), event_loop_for_queue).result()
            return

        try:
            tts_params = {
                "how_to_cut": self.i18n(engine_config.how_to_cut), "sample_steps": engine_config.sample_steps,
                "top_k": engine_config.top_k, "top_p": engine_config.top_p, 
                "temperature": engine_config.temperature, "speed": engine_config.speed,
            }

            synthesis_generator = self.get_tts_wav(
                ref_wav_path=engine_config.REF_AUDIO_PATH, prompt_text=engine_config.REF_TEXT_CONTENT,
                prompt_language=self.i18n(engine_config.REF_LANG), text=text,
                text_language=self.i18n(engine_config.TARGET_LANG), **tts_params
            )

            for sr, data_chunk in synthesis_generator:
                if data_chunk is not None and sr is not None:
                    item_to_queue = (sr, data_chunk)
                    asyncio.run_coroutine_threadsafe(queue_to_async.put(item_to_queue), event_loop_for_queue).result()

        except Exception as e:
            logger.critical(f"FATAL Error in GPT-SoVITS Producer Thread: {e}\n{traceback.format_exc()}")
            asyncio.run_coroutine_threadsafe(queue_to_async.put(e), event_loop_for_queue).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue_to_async.put(None), event_loop_for_queue).result()