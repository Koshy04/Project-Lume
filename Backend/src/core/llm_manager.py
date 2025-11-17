import os
import importlib.util
import traceback
from typing import Dict, Any
from src.log.custom_logger import logger

class LLMManager:
    """Dynamically discovers, loads, and manages different LLM engines."""
    def __init__(self, loop):
        self.loop = loop
        self.engines_dir = os.path.join(os.path.dirname(__file__), '..', 'llm', 'engines')
        self.available_engines = self._discover_engines()
        self.active_engine = None
        self.active_engine_name = None

    def _discover_engines(self) -> Dict[str, Any]:
        """Discovers available LLM engines in the engines directory."""
        engines = {}
        if not os.path.exists(self.engines_dir):
            return engines

        for engine_name in os.listdir(self.engines_dir):
            engine_path = os.path.join(self.engines_dir, engine_name)

            if os.path.isdir(engine_path) and '__init__.py' in os.listdir(engine_path):
                inference_file_path = os.path.join(engine_path, 'inference.py')
                if not os.path.exists(inference_file_path):
                    continue
                
                try:
                    module_path = f'src.llm.engines.{engine_name}.inference'
                    
                    spec = importlib.util.spec_from_file_location(module_path, inference_file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, 'LLMEngine'):
                        engines[engine_name] = module.LLMEngine
                        logger.info(f"Discovered LLM Engine: {engine_name}")
                except Exception as e:
                    logger.error(f"Failed to load LLM engine '{engine_name}': {e}\n{traceback.format_exc()}")
        return engines

    def get_available_engines(self) -> list:
        return list(self.available_engines.keys())

    async def initialize_engine(self, engine_name: str) -> bool:
        if engine_name not in self.available_engines:
            logger.error(f"LLM engine '{engine_name}' not found.")
            return False

        if self.active_engine and self.active_engine_name == engine_name:
            logger.info(f"LLM Engine '{engine_name}' is already initialized.")
            return True

        try:
            logger.info(f"--- Initializing LLM Engine: {engine_name} ---")
            EngineClass = self.available_engines[engine_name]
            self.active_engine = EngineClass()

            initialization_success = self.active_engine.initialize()

            if initialization_success:
                self.active_engine_name = engine_name
                logger.info(f"--- LLM Engine '{engine_name}' Initialized Successfully ---")
                return True
            else:
                logger.critical(f"Failed to initialize LLM engine '{engine_name}'.")
                self.active_engine = None
                return False
        except Exception as e:
            logger.critical(f"FATAL error initializing LLM engine '{engine_name}': {e}\n{traceback.format_exc()}")
            self.active_engine = None
            return False

    async def generate(self, *args, **kwargs):
        if not self.active_engine:
            logger.warning("LLM generation called but no engine is active.")
            return "I am unable to think right now as no LLM engine is active."
        return await self.active_engine.generate(*args, **kwargs)
    
    async def classify_intent(self, transcription: str) -> str:
        """Delegates intent classification to the active engine."""
        if not self.active_engine:
            logger.warning("LLM intent classification called but no engine is active.")
            return ""
        
        return await self.active_engine.classify_intent(transcription)

    async def shutdown(self):
        """Unloads the active LLM engine to free up resources."""
        if self.active_engine and hasattr(self.active_engine, 'unload'):
            logger.info(f"Shutting down LLM engine: {self.active_engine_name}")
            try:
                await self.loop.run_in_executor(None, self.active_engine.unload)
                logger.info(f"LLM engine '{self.active_engine_name}' shut down successfully.")
            except Exception as e:
                logger.error(f"Error shutting down LLM engine '{self.active_engine_name}': {e}")
        self.active_engine = None
        self.active_engine_name = None