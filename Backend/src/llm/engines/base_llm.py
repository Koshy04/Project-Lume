from abc import ABC, abstractmethod

class BaseLLMEngine(ABC):
    """Abstract base class for all LLM inference engines."""

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initializes the engine, loading models or setting up clients.
        Returns True on success, False on failure.
        """
        pass

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str, temperature: float, stop_sequences: list) -> str:
        """
        Generates a response from the language model.

        Args:
            prompt (str): The user's input prompt.
            system_prompt (str): The system instruction or context.
            temperature (float): The generation temperature.
            stop_sequences (list): A list of strings that should stop generation.

        Returns:
            str: The generated text from the model.
        """
        pass

    @abstractmethod #currently this only used for multi mode
    async def classify_intent(self, transcription: str) -> str:
        """
        Performs a specialized, low-temperature generation to classify user intent.
        Should return a single, clean word or phrase as the classification.
        """
        pass