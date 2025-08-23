import asyncio
import random
import time
from collections import deque
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.type import AuthScope
from twitchAPI.chat import Chat, EventData, ChatMessage
import config
from ai_core import analyze_emotions

class TwitchBot:
    """The class that connects to Twitch using the twitchAPI library."""
    def __init__(self, response_queue: deque):
        self.response_queue = response_queue
        self.chat_messages = deque(maxlen=15)
        self.target_channel = config.TWITCH_CHANNEL
        self.app = None
        self.chat = None
        self.is_running = False
        self.last_prompt_time = 0.0

    async def on_message(self, msg: ChatMessage):
        """Callback for when a message is received in chat."""
        ignored_users = ['streamelements', 'nightbot', config.TWITCH_NICK.lower()]
        if msg.user.name.lower() in ignored_users:
            return
        
        print(f"Twitch chat ({msg.room.name}): {msg.user.name}: {msg.text}")
        self.chat_messages.append(f"{msg.user.name}: {msg.text}")

    async def run(self):
        """Initializes the connection and starts the chat listener."""
        self.app = await Twitch(config.TWITCH_APP_ID, config.TWITCH_APP_SECRET)

        user_scope = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
        
        await self.app.authenticate_user(user_scope)
        
        self.chat = await Chat(self.app)
        self.chat.register_event(ChatMessage, self.on_message)
        self.chat.start()
        print(f"Twitch bot connected and listening to '{self.target_channel}'")
        self.is_running = True
        
        await self.chat.join_room(self.target_channel)
        print(f"Joined Twitch channel: {self.target_channel}")
        
    async def stop(self):
        """Stops the chat listener and disconnects."""
        if self.chat:
            await self.chat.leave_room(self.target_channel)
            self.chat.stop()
        if self.app:
            await self.app.close()
        self.is_running = False
        print("Twitch bot disconnected.")

    def get_random_chat_prompt(self) -> str | None:
        """Picks one random message from the recent chat history."""
        if self.chat_messages:
            return random.choice(self.chat_messages)
        return None

async def check_external_chats(twitch_bot: TwitchBot, response_queue: deque):
    """Periodically checks Twitch for messages, respecting the cooldown."""
    if not (twitch_bot and twitch_bot.is_running):
        return
    
    current_time = time.time()
    if current_time - twitch_bot.last_prompt_time < config.TWITCH_COOLDOWN_SECONDS:
        return
    
    prompt = twitch_bot.get_random_chat_prompt()
    if prompt:
        print(f"Injecting prompt from Twitch: '{prompt}'")
        response_queue.append(("twitch_user", prompt, analyze_emotions(prompt), "twitch_chat_history"))

        twitch_bot.last_prompt_time = time.time()