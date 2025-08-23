import os
import importlib
import inspect
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from plugins.base_plugin import BasePlugin, BotInterface

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.join(SCRIPT_DIR, "plugins")
POLLING_JOB_ID = "plugin_poller"

class PluginManager:
    def __init__(self, bot_interface: BotInterface):
        self.bot_interface = bot_interface
        self.available_plugins: dict[str, type[BasePlugin]] = {}
        self.active_plugins: dict[str, BasePlugin] = {}
        self.scheduler = None

    def discover_plugins(self):
        """
        Finds all valid plugin classes in the plugins directory and stores
        them in 'available_plugins' without creating instances.
        """
        logging.info("--- Discovering available plugins ---")
        for filename in os.listdir(PLUGIN_DIR):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = f"plugins.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                            temp_instance = obj(self.bot_interface)
                            self.available_plugins[temp_instance.name] = obj
                            logging.info(f"   -> Discovered plugin: '{temp_instance.name}'")
                except Exception as e:
                    logging.error(f"Failed to discover plugin from {filename}: {e}")
        logging.info("--- Plugin discovery complete ---")

    def activate_plugin(self, name: str) -> BasePlugin | None:
        """
        Loads and instantiates a plugin by name if it's not already active.
        Returns the plugin instance.
        """
        if name in self.active_plugins:
            return self.active_plugins[name]

        if name in self.available_plugins:
            try:
                plugin_class = self.available_plugins[name]
                instance = plugin_class(self.bot_interface)
                self.active_plugins[name] = instance
                logging.info(f"Plugin '{name}' has been activated and loaded into memory.")
                return instance
            except Exception as e:
                logging.error(f"Failed to activate and instantiate plugin '{name}': {e}")
                return None
        
        return None # Plugin name is not known

    async def _poll_all_plugins(self):
        """The actual task that the scheduler runs. Iterates over active plugins."""
        for plugin in self.active_plugins.values():
            if plugin.is_enabled:
                await plugin.poll()
    
    def start_polling(self, scheduler: AsyncIOScheduler):
        """Adds the polling job to the bot's scheduler."""
        if not self.scheduler:
            self.scheduler = scheduler
        try:
            if not self.scheduler.get_job(POLLING_JOB_ID):
                self.scheduler.add_job(
                    self._poll_all_plugins, 'interval', seconds=2, id=POLLING_JOB_ID
                )
                logging.info("Plugin polling job started.")
        except Exception as e:
            logging.error(f"Could not start plugin polling: {e}")

    def stop_polling(self):
        """Removes the polling job from the scheduler."""
        if self.scheduler and self.scheduler.get_job(POLLING_JOB_ID):
            self.scheduler.remove_job(POLLING_JOB_ID)
            logging.info("Plugin polling job stopped.")