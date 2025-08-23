import sys
import os
import traceback
import asyncio
import functools
from config import (
    GPT_SOVITS_BASE_PATH, GPT_MODEL_PATH, SOVITS_MODEL_PATH,
    REF_AUDIO_PATH, REF_TEXT_CONTENT, REF_LANG, TARGET_LANG,
    top_p, top_k, temperature, sample_steps, speed, how_to_cut
)

# --- Module State ---
TTS_MODELS_INITIALIZED = False
i18n_instance = None
get_tts_wav = None
change_gpt_weights = None
change_sovits_weights = None

# --- Dynamic Import of GPT-SoVITS ---
print(f"[TTS_INTERFACE] >>> Checking for GPT-SoVITS path: {GPT_SOVITS_BASE_PATH}")
if os.path.exists(GPT_SOVITS_BASE_PATH):
    print("[TTS_INTERFACE] >>> Path exists. Appending to sys.path.")
    sys.path.append(GPT_SOVITS_BASE_PATH)
    try:
        print("[TTS_INTERFACE] >>> Attempting to import from inference_webui...")
        from inference_webui import ( #dont mind the errors here. Everything is working as intended
            change_gpt_weights as gpt_change_gpt,
            change_sovits_weights as gpt_change_sovits,
            get_tts_wav as gpt_get_tts
        )
        from tools.i18n.i18n import I18nAuto

        get_tts_wav = gpt_get_tts
        change_gpt_weights = gpt_change_gpt
        change_sovits_weights = gpt_change_sovits
        print("[TTS_INTERFACE] >>> SUCCESS: GPT-SoVITS modules imported.")
    except Exception as e:
        print(f"Error Details: {e}")
        traceback.print_exc()
else:
    print("!!! [TTS_INTERFACE] WARNING: GPT-SoVITS path not found. TTS will be disabled.")

def initialize_tts_models():
    """Initializes the TTS models using paths from config."""
    global TTS_MODELS_INITIALIZED, i18n_instance
    
    if TTS_MODELS_INITIALIZED:
        print(">>> EXIT: Models already initialized.")
        return True
        
    if not all([get_tts_wav, change_gpt_weights, change_sovits_weights]):
        print("!!! EXIT: FATAL - GPT-SoVITS library functions were not loaded. Cannot proceed.")
        return False

    try:
        i18n_instance = I18nAuto()
        print("--- i18n Initialized.")
        change_gpt_weights(gpt_path=GPT_MODEL_PATH)
        print("--- GPT weights loaded.")
        change_sovits_weights(sovits_path=SOVITS_MODEL_PATH)
        print("--- SoVITS weights loaded.")
        
        TTS_MODELS_INITIALIZED = True
        return True
        
    except Exception as e:
        print("\n" + "="*50)
        print("!!! [TTS_INTERFACE] FATAL INITIALIZATION ERROR !!!")
        print("An error occurred while loading the TTS models into memory.")
        print(f"Error Details: {e}")
        traceback.print_exc()
        print("="*50 + "\n")
        TTS_MODELS_INITIALIZED = False
        return False

def _blocking_tts_segment_producer(
    queue_to_async: asyncio.Queue,
    event_loop_for_queue: asyncio.AbstractEventLoop,
    text_to_synthesize: str
):
    """This is the blocking function that runs in a separate thread to generate audio."""
    global i18n_instance
    if not TTS_MODELS_INITIALIZED:
        print("TTS Producer Thread: Models not initialized. Aborting.")
        asyncio.run_coroutine_threadsafe(queue_to_async.put(None), event_loop_for_queue).result()
        return

    try:
        print(f"TTS Producer Thread: Starting generation for: \"{text_to_synthesize[:50]}...\"")
        
        tts_params = {
            "how_to_cut": i18n_instance("按标点符号切"),
            "sample_steps": sample_steps, "top_k": top_k, "top_p": top_p,
            "temperature": temperature, "speed": speed,
        }

        synthesis_generator = get_tts_wav(
            ref_wav_path=REF_AUDIO_PATH,
            prompt_text=REF_TEXT_CONTENT,
            prompt_language=i18n_instance(REF_LANG),
            text=text_to_synthesize,
            text_language=i18n_instance(TARGET_LANG),
            **tts_params
        )

        for sr, data_chunk in synthesis_generator:
            if data_chunk is not None and sr is not None:
                item_to_queue = (sr, data_chunk)
                asyncio.run_coroutine_threadsafe(queue_to_async.put(item_to_queue), event_loop_for_queue).result()

    except Exception as e:
        print(f"FATAL Error in TTS Producer Thread: {e}")
        traceback.print_exc()
        asyncio.run_coroutine_threadsafe(queue_to_async.put(e), event_loop_for_queue).result()
    finally:
        print("TTS Producer Thread: Sending completion sentinel.")
        asyncio.run_coroutine_threadsafe(queue_to_async.put(None), event_loop_for_queue).result()