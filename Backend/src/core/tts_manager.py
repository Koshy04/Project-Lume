import os
import sys
import importlib
import traceback
import asyncio
from typing import Optional, Dict, Any, Tuple

class TTSManager:
    """
    Dynamically discovers, loads, and manages different TTS engines.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.engines_dir = os.path.join(os.path.dirname(__file__), '..', 'tts', 'engines')
        self.available_engines = self._discover_engines()
        self.active_engine = None
        self.active_engine_name = None

    def _discover_engines(self) -> Dict[str, Any]:
        # ... (no changes in this method) ...
        engines = {}
        if not os.path.exists(self.engines_dir):
            return engines

        for engine_name in os.listdir(self.engines_dir):
            engine_path = os.path.join(self.engines_dir, engine_name)
            if os.path.isdir(engine_path) and '__init__.py' in os.listdir(engine_path):
                try:
                    module_path = f'src.tts.engines.{engine_name}.inference'
                    spec = importlib.util.spec_from_file_location(module_path, os.path.join(engine_path, 'inference.py'))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'TTSEngine'):
                        engines[engine_name] = module.TTSEngine
                        print(f"Discovered TTS Engine: {engine_name}")
                except Exception as e:
                    print(f"Failed to load TTS engine '{engine_name}': {e}")
                    traceback.print_exc()
        return engines

    def get_available_engines(self) -> list:
        return list(self.available_engines.keys())

    async def initialize_engine(self, engine_name: str) -> bool:
        if engine_name not in self.available_engines:
            print(f"Error: TTS engine '{engine_name}' not found.")
            return False

        if self.active_engine and self.active_engine_name == engine_name:
            print(f"Engine '{engine_name}' is already initialized.")
            return True

        try:
            print(f"--- Initializing TTS Engine: {engine_name} ---")
            EngineClass = self.available_engines[engine_name]
            self.active_engine = EngineClass()
            
            initialization_success = await self.loop.run_in_executor(None, self.active_engine.initialize)
            
            if initialization_success:
                self.active_engine_name = engine_name
                print(f"--- TTS Engine '{engine_name}' Initialized Successfully ---")
                return True
            else:
                print(f"!!! FATAL: Failed to initialize TTS engine '{engine_name}'.")
                self.active_engine = None
                return False

        except Exception as e:
            print(f"\n" + "="*50)
            print(f"!!! [TTS_MANAGER] FATAL INITIALIZATION ERROR for engine '{engine_name}' !!!")
            print(f"Error Details: {e}")
            traceback.print_exc()
            print("="*50 + "\n")
            self.active_engine = None
            return False

    def start_audio_generation(self, text: str) -> Tuple[asyncio.Task, asyncio.Queue]:
        """
        Starts the TTS generation in a background thread and returns the task and queue.
        This does NOT wait for the audio. It returns immediately.
        """
        if not self.active_engine:
            print("[Warning] TTS generation called but no engine is active.")
            async def dummy_task(): pass
            return asyncio.create_task(dummy_task()), asyncio.Queue()

        audio_data_queue = asyncio.Queue()
        
        producer_task = self.loop.run_in_executor(
            None, 
            self.active_engine.stream_generate, 
            text, 
            audio_data_queue, 
            self.loop
        )
        
        return producer_task, audio_data_queue