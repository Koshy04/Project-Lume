import asyncio
import torch
import traceback
import sounddevice as sd
import numpy as np
import tempfile
import wave
import os
from collections import deque, defaultdict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
import config
from src.core.ai_core import AICore
from src.core.tts_manager import TTSManager
from src.core.transcription import Transcription
from src.input.vision import VisionInput
from src.services.memory.summarizer import summarizer
from src.services.chat.yt import YouTubeBot
from src.services.movement.vts_interface import trigger_vts_animation, close_vts_connection

# --- Utility Functions ---
def find_audio_device_id(device_name_query: str) -> int | None:
    """Finds the first output audio device matching a query string."""
    if not device_name_query: return None
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name_query.lower() in device['name'].lower() and device['max_output_channels'] > 0:
                return i
    except Exception as e:
        print(f"Lipsync: Error querying audio devices: {e}")
    return None

def play_audio_on_device(data, samplerate, device_id):
    """Blocking function to play audio on a specific device."""
    try:
        sd.play(data, samplerate, device=device_id)
        sd.wait()
    except Exception as e:
        print(f"Lipsync: Error playing audio on device {device_id}: {e}")

# --- Local Audio Player (Acts as the "Sink" for Local Mode) ---
class LocalAudioPlayer:
    """A class to handle asynchronous playback of audio to a local device."""
    def __init__(self, bot: 'Bot'):
        self.bot = bot
        self.audio_playback_queue = asyncio.Queue()
        self.playback_task = None
        self._shutdown = False
        self.lipsync_device_id = find_audio_device_id(config.VIRTUAL_MIC_NAME)

    async def start(self):
        if self.playback_task is None or self.playback_task.done():
            self.playback_task = asyncio.create_task(self._dedicated_audio_player())

    async def stop(self):
        self._shutdown = True
        if self.playback_task and not self.playback_task.done():
            self.playback_task.cancel()
            await asyncio.sleep(0.1)

    async def enqueue_audio_segment(self, audio_item):
        if self._shutdown: return
        await self.audio_playback_queue.put(audio_item)

    async def _dedicated_audio_player(self):
        loop = asyncio.get_event_loop()
        main_output_device_id = config.LOCAL_OUTPUT_DEVICE
        
        def play_and_wait(data, sr, device_id):
            """Plays audio and blocks until it's finished."""
            sd.play(data, sr, device=device_id)
            sd.wait()

        while not self._shutdown:
            try:

                audio_item = await self.audio_playback_queue.get()
                if audio_item is None: break
                samplerate, audio_data = audio_item
                
                play_main_future = loop.run_in_executor(None, play_and_wait, audio_data, samplerate, main_output_device_id)
                
                lipsync_future = None
                if (self.bot.vts_enabled or self.bot.veadotube_enabled):
                    if self.lipsync_device_id is not None:
                        lipsync_future = loop.run_in_executor(None, play_and_wait, audio_data, samplerate, self.lipsync_device_id)
                
                tasks_to_run = [play_main_future]
                if lipsync_future:
                    tasks_to_run.append(lipsync_future)
                
                await asyncio.gather(*tasks_to_run)
                
                self.audio_playback_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                traceback.print_exc()

# --- The Main Bot Class ---
class Bot:
    """
    The central class that manages the bot's state, components, and core logic.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

        # --- Core State ---
        self.active_mode: str | None = None
        self.vts_enabled = False
        self.veadotube_enabled = False
        self.vision_mode_enabled = False
        self.youtube_enabled = False

        # --- Components ---
        self.ai = AICore()
        self.tts_manager = TTSManager(self.loop)
        self.transcription = Transcription()
        self.vision_system: VisionInput | None = None
        self.youtube_bot: YouTubeBot | None = None
        self.local_audio_player: LocalAudioPlayer | None = None
        self.scheduler: AsyncIOScheduler | None = None

        # --- Data Structures ---
        self.response_queue = asyncio.Queue()
        self.conversation_history_for_prompt = defaultdict(lambda: deque(maxlen=config.CONVERSATION_HISTORY_LIMIT))
        self.conversation_log_for_summary = defaultdict(list)

    async def setup(self, selected_tts_engine: str): # MODIFIED
        """Initializes all blocking (in executors) and async components."""
        print("--- Initializing Core Bot Systems ---")
        await self.loop.run_in_executor(None, self.transcription.initialize_model)
        # Initialize the selected TTS engine
        await self.tts_manager.initialize_engine(selected_tts_engine)
        if config.VISION_STARTUP and config.VISION_ACTION_WORDS:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.vision_system = VisionInput(device=device)
            print("Vision system initialized.")
        print("--- Core Bot Systems Ready ---")

    async def shutdown(self):
        """Gracefully shuts down all active services and connections."""
        print("Initiating bot shutdown sequence...")
        if self.youtube_bot and self.youtube_bot.is_running: await self.youtube_bot.stop()
        if self.local_audio_player: await self.local_audio_player.stop()
        await close_vts_connection()
        print("Bot shutdown complete.")

    async def generate_and_play_tts(self, text: str, sink):
        """
        Generates TTS audio and passes it to the sink. The sink is responsible
        for all playback, including lipsync.
        """
        if not sink:
            print("[Warning] generate_and_play_tts called with no active sink.")
            return

        # Start the TTS generation and get the queue for audio chunks
        producer_task, audio_data_queue = self.tts_manager.start_audio_generation(text)

        # This loop simply forwards every audio chunk to the sink
        while True:
            try:
                item = await asyncio.wait_for(audio_data_queue.get(), timeout=180.0)
                if item is None: break  # End of stream
                if isinstance(item, Exception):
                    print(f"[Error] TTS producer thread failed: {item}"); break
                
                await sink.enqueue_audio_segment(item)

            except asyncio.TimeoutError:
                print("[Error] TTS generation timed out."); break
            except Exception as e:
                print(f"[Error] TTS consumer loop failed: {e}"); break
        
        await producer_task

    async def handle_vision_request(self, user_id: str, sink, channel_id: str):
        """Processes a request to analyze the screen."""
        if not self.vision_system:
            await self.generate_and_play_tts("My vision system isn't enabled right now.", sink)
            return
        try:
            vision_result = await self.loop.run_in_executor(None, self.vision_system.process_screen, 1)
            if not vision_result or not vision_result.get('success'):
                await self.generate_and_play_tts("Sorry, I had a problem looking at the screen.", sink)
                return
            caption = vision_result.get('caption', 'I couldn\'t generate a caption.')
            ocr_text = self.vision_system.get_detected_text(ocr_results=vision_result.get('ocr_results', []))
            prompt = (f"You are looking at the user's screen. Your analysis is as follows:\n"
                      f"- **Overall Scene:** '{caption}'\n- **Text on Screen:** '{ocr_text}'\n\n"
                      f"Based on this information, formulate a helpful and context-aware response.")
            
            conversation_log = "\n".join(self.conversation_history_for_prompt[channel_id])
            ai_response = await self.ai.chat_with_ai(prompt, user_id, {"dominant_emotion": "neutral"}, conversation_log)
            if ai_response:
                self.conversation_history_for_prompt[channel_id].append(f"{config.BOT_NAME}: {ai_response}")
                await self.generate_and_play_tts(ai_response, sink)
        except Exception:
            print(f"Error during vision handling:\n{traceback.format_exc()}")
            await self.generate_and_play_tts("I ran into an unexpected error trying to process the screen.", sink)

    async def process_ai_response_queue(self, get_active_sink_callback):
        """
        The main consumer loop that processes tasks from the queue.
        Tasks can contain text from Discord or raw audio from local mode.
        """
        print("AI Response Queue Processor is running.")
        while True:
            try:
                user_id, task_data, emotion_data, channel_id = await self.response_queue.get()
                
                transcription = None
                
                if isinstance(task_data, np.ndarray): # Handle raw audio from local mode
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                        file_path = tmp_wav.name
                        with wave.open(file_path, 'wb') as wf:
                            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                            wf.writeframes(task_data.tobytes())
                    transcription = await self.loop.run_in_executor(None, self.transcription.transcribe, file_path)
                    os.remove(file_path)
                    
                    if not transcription or len(transcription.strip()) < 2:
                        self.response_queue.task_done(); continue
                    
                    print(f"You ({config.user_name}): {transcription}")
                    emotion_data = self.ai.analyze_emotions(transcription)

                elif isinstance(task_data, str): # Handle text from Discord
                    transcription = task_data
                
                if not transcription:
                    self.response_queue.task_done(); continue

                active_sink = get_active_sink_callback()
                if not active_sink:
                    self.response_queue.task_done(); continue
                
                user_name = config.USER_NAMES.get(user_id, config.user_name if user_id == "local_user" else f"User({user_id})")
                self.conversation_history_for_prompt[channel_id].append(f"{user_name}: {transcription}")
                self.conversation_log_for_summary[channel_id].append({"role": user_name, "content": transcription, "user_id": user_id})

                if self.vision_mode_enabled and self.ai.is_vision_request(transcription):
                    asyncio.create_task(self.handle_vision_request(user_id, active_sink, channel_id))
                else:
                    conversation_log = "\n".join(self.conversation_history_for_prompt[channel_id])
                    ai_response = await self.ai.chat_with_ai(transcription, user_id, emotion_data, conversation_log)
                    
                    print(f"{config.BOT_NAME} to {user_name}: {ai_response}")

                    if ai_response and "I'm having trouble responding right now." not in ai_response:
                        self.conversation_history_for_prompt[channel_id].append(f"{config.BOT_NAME}: {ai_response}")
                        self.conversation_log_for_summary[channel_id].append({"role": config.BOT_NAME, "content": ai_response, "user_id": "Bot"})
                        
                        if len(self.conversation_log_for_summary[channel_id]) >= 8:
                            # Pass the bot's AI instance (self.ai) to the summarizer.
                            asyncio.create_task(summarizer.consolidate_and_store(
                                list(self.conversation_log_for_summary[channel_id]),
                                self.ai
                            ))
                            self.conversation_log_for_summary[channel_id].clear()

                        ai_emotion_data = self.ai.analyze_emotions(ai_response)
                        if self.vts_enabled:
                            await trigger_vts_animation(ai_emotion_data.get("dominant_emotion", "neutral"))
 
                        tts_response = ai_response.replace('-', ' ')
                        await self.generate_and_play_tts(tts_response, active_sink)

                self.response_queue.task_done()
            except Exception:
                print(f"CRITICAL Error in response queue processor:\n{traceback.format_exc()}")