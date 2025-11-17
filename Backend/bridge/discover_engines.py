import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.tts_manager import TTSManager
from src.core.llm_manager import LLMManager 
from src.log.custom_logger import logger

def discover():
    """Discovers and prints available TTS and LLM engines in a machine-readable format."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # --- Discover TTS Engines ---
        tts_manager = TTSManager(loop)
        tts_engines = tts_manager.get_available_engines()
        print(f"TTS_ENGINES:{','.join(tts_engines)}") 
        
        # --- Discover LLM Engines ---
        llm_manager = LLMManager(loop)
        llm_engines = llm_manager.get_available_engines()
        print(f"LLM_ENGINES:{','.join(llm_engines)}")
        
    except Exception as e:
        print(f"DISCOVERY_ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    discover()