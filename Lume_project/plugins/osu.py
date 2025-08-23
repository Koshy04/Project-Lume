import requests
import json
import logging
from .base_plugin import BasePlugin, BotInterface

class OsuPlugin(BasePlugin):
    """
    Connects to StreamCompanion and uses the BotInterface to comment on gameplay.
    """
    def __init__(self, bot_interface: BotInterface):
        super().__init__(bot_interface)
        self.game_state = {}
        self.streamcompanion_url = "http://127.0.0.1:20727/json"
        self.is_enabled = True # Enabled by default, can be toggled.

    @property
    def name(self) -> str:
        return "osu"

    def _get_state_change_prompt(self, new_state):
        """Analyzes state change and returns a prompt and emotion if applicable."""
        prompt = None
        emotion = "neutral"
        
        previous_status = self.game_state.get('status', 0)
        current_status = new_state.get('status', 0)

        # Health bar dropped to zero while playing.
        previous_hp = self.game_state.get('playerHp', 1.0)
        current_hp = new_state.get('playerHp', 1.0)
        if current_status == 2 and previous_hp > 0 and current_hp == 0.0:
            logging.info("OsuPlugin Event: Map Failed (HP Drop).")
            prompt = "The user just failed the map. Roast them for failing so pathetically."
            emotion = "anger"
        
        # Status changed from Playing to something else.
        elif previous_status == 2 and current_status != 2:
            logging.info(f"OsuPlugin Event: Status Change from Playing -> {current_status}")
            grade_map = {9: 'SS', 8: 'S', 7: 'S', 6: 'A', 5: 'B', 4: 'C', 3: 'D'}
            
            if current_status == 7: # Results Screen
                grade = grade_map.get(new_state.get('grade', 0), 'an Unknown Grade')
                prompt = (f"The user finished. Stats: Grade: '{grade}', Acc: {new_state.get('acc', 0.0):.2f}%, "
                          f"Combo: {new_state.get('maxCombo', 0)}x, Misses: {new_state.get('miss', 0)}. React to this.")
                emotion = "joy" if grade in ['SS', 'S', 'A'] and new_state.get('miss', 0) == 0 else "neutral"
            elif current_status == 15: # Fail Screen
                 prompt = f"The user just completely failed the map with {new_state.get('miss', 0)} misses. Roast them."
                 emotion = "anger"
            elif current_status == 1: # Quit to Menu
                prompt = f"The user quit with a {self.game_state.get('combo', 0)}x combo. Make a snarky comment."
                emotion = "neutral"
        
        return prompt, emotion

    async def poll(self):
        if not self.is_enabled:
            return

        proxies = {"http": None, "https": None}
        try:
            response = requests.get(self.streamcompanion_url, timeout=1, proxies=proxies)
            response.raise_for_status()
            new_state = json.loads(response.content.decode('utf-8-sig'))
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return

        prompt, emotion = self._get_state_change_prompt(new_state)

        if prompt:
            if not self.bot_interface.is_bot_busy():
                logging.info(f"OsuPlugin: Decision: CLEAR to speak. Requesting comment.")
                await self.bot_interface.request_bot_speech(prompt, "osu_system", emotion)

        self.game_state.update(new_state)