from abc import ABC, abstractmethod
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class BotInterface(ABC):
    """
    Defines the methods a plugin is allowed to call from the main bot.
    This acts as a controlled 'API' for plugins, preventing them from
    accessing complex or sensitive parts of the bot directly.
    """
    @abstractmethod
    def is_bot_busy(self) -> bool:
        """Checks if the bot is currently speaking, processing, or on cooldown."""
        pass

    @abstractmethod
    async def request_bot_speech(self, prompt: str, user_id: str, emotion: str):
        """A plugin calls this to make the bot generate and speak a response."""
        pass

class BasePlugin(ABC):
    """
    The abstract base class that all plugins must inherit from.
    """
    def __init__(self, bot_interface: BotInterface):
        self.bot_interface = bot_interface
        self.is_enabled = False

    @property
    @abstractmethod
    def name(self) -> str:
        """A unique, lowercase name for the plugin used in commands."""
        pass

    @abstractmethod
    async def poll(self):
        """
        The main loop for the plugin, called periodically by the PluginManager.
        This is where the plugin's core logic resides.
        """
        pass