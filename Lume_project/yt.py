import asyncio
import random
import time
from collections import deque
import pytchat
import config

class YouTubeBot:
    """The class that connects to YouTube Live chat using pytchat """
    def __init__(self, response_queue: deque):
        self.response_queue = response_queue
        self.chat_messages = deque(maxlen=10)
        self.video_id = config.YOUTUBE_VIDEO_ID
        self.chat = None
        self.is_running = False
        self._stop_event = asyncio.Event()

    def _run_sync(self):
        """
        The synchronous chat fetching loop based on the proven working snippet.
        This is designed to be run in a separate thread.
        """
        print(f"Pytchat thread starting for video ID: {self.video_id}...")
        try:
            # Create the chat object, disabling interruptable to prevent crashes in threads
            self.chat = pytchat.create(video_id=self.video_id, interruptable=False)

            while self.chat.is_alive() and not self._stop_event.is_set():
                for c in self.chat.get().sync_items():
                    # Format the message exactly as your example shows
                    formatted_message = f"{c.author.name} just said '{c.message}' in chat."
                    print(f"YouTube chat received: {formatted_message}")
                    self.chat_messages.append(formatted_message)

                time.sleep(2)

        except Exception as e:
            print(f"ERROR in Pytchat thread: {e}")
        finally:
            if self.chat:
                self.chat.terminate()
            print("Pytchat thread finished.")

    async def run(self):
        """Starts the chat listener in a separate thread to avoid blocking asyncio."""
        if not self.video_id:
            print("YouTube bot cannot start: YOUTUBE_VIDEO_ID not set in config.")
            return
        if self.is_running:
            print("YouTube bot is already running.")
            return

        self._stop_event.clear()
        self.is_running = True
        loop = asyncio.get_event_loop()
        # Run the synchronous _run_sync method in a background thread
        await loop.run_in_executor(None, self._run_sync)
        self.is_running = False

    async def stop(self):
        """Stops the chat listener thread."""
        if self.is_running and not self._stop_event.is_set():
            print("Stopping YouTube bot...")
            self._stop_event.set()
            # Give the thread a moment to see the event and break its loop
            await asyncio.sleep(0.5)
        self.is_running = False
        print("YouTube bot stop signal sent.")

    def get_random_chat_prompt(self) -> str | None:
        """Picks one random message from the recent chat history and removes it."""
        if self.chat_messages:
            chosen_index = random.randrange(len(self.chat_messages))
            chosen_message = self.chat_messages[chosen_index]
            del self.chat_messages[chosen_index]
            return chosen_message
        return None