import httpx
from . import config as engine_config
from ..base_llm import BaseLLMEngine
from src.log.custom_logger import logger


class LLMEngine(BaseLLMEngine):
    """LLM Engine for Ollama."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.api_url = engine_config.API_URL
        self.model = engine_config.MODEL

    def initialize(self) -> bool:
        logger.info(f"Ollama Engine Initialized. Endpoint: {self.api_url}, Model: {self.model}")
        return True

    async def generate(self, prompt: str, system_prompt: str, temperature: float, stop_sequences: list) -> str:
        json_payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt,
            "options": {
                "temperature": temperature, "top_p": 1.5, "top_k": 40,
                "repeat_penalty": 1.15, "num_predict": 90, "num_ctx": 16384,
                "repeat_last_n": 32, "stop": stop_sequences
            },
            "stream": False
        }
        try:
            response = await self.client.post(f"{self.api_url}/api/generate", json=json_payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except httpx.RequestError as e:
            logger.error(f"Error connecting to Ollama API: {e}")
            return "Sorry, I'm having trouble connecting to my brain right now."
        except Exception as e:
            logger.error(f"Error in LLM inference: {e}")
            return "Sorry, something went wrong while I was thinking."

    async def classify_intent(self, transcription: str) -> str:
        """Uses Ollama for specialized intent classification."""
        system_prompt = """You are a 'Conversation Target' analysis expert. Your job is to analyze the user's text and identify who the user is speaking to.
You must respond with ONLY ONE of the following three words:
- 'Bot' if the user is addressing the AI bot.
- 'Other' if the user is addressing another person or group.
- 'General' if the user is making a general statement to no one in particular."""

        full_prompt = f"System: {system_prompt}\n\nUser: \"{transcription}\"\nSystem:"

        json_payload = {
            "model": self.model,
            "prompt": full_prompt,
            "options": {"temperature": 0.0, "top_p": 0.1, "num_predict": 5, "stop": ["\n", "."]},
            "stream": False
        }

        try:
            response = await self.client.post(f"{self.api_url}/api/generate", json=json_payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Error during Ollama intent classification: {e}")
            return "" # Return empty on failure