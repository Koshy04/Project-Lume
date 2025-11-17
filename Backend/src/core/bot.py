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
import io
import base64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import config
from src.log.custom_logger import logger
from src.core.ai_core import AICore
from src.core.tts_manager import TTSManager
from src.core.llm_manager import LLMManager
from src.core.transcription import Transcription
from src.input.vision import VisionInput
from src.services.memorys.summarizer import summarizer
from src.services.chat.yt import YouTubeBot
from src.services.movement.vts_animator import VTSAnimator

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
        logger.error(f"Lipsync: Error querying audio devices: {e}")
    return None

# --- Local Audio Player Class ---
class LocalAudioPlayer:
    """A class to handle asynchronous playback of audio files to a local device."""
    def __init__(self, bot: 'Bot'):
        self.bot = bot
        self.audio_file_queue = asyncio.Queue()
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

    async def enqueue_audio_file(self, audio_path: str):
        if self._shutdown: return
        await self.audio_file_queue.put(audio_path)

    async def _dedicated_audio_player(self):
        loop = asyncio.get_event_loop()
        main_output_device_id = config.LOCAL_OUTPUT_DEVICE

        def play_and_wait(path: str, device_id: int | None):
            try:
                with wave.open(path, 'rb') as wf:
                    samplerate = wf.getframerate()
                    data = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(data, dtype=np.int16)
                
                sd.play(audio_data, samplerate, device=device_id)
                sd.wait()
            except Exception as e:
                logger.error(f"Error playing audio file {path} on device {device_id}: {e}")

        while not self._shutdown:
            try:
                audio_path = await self.audio_file_queue.get()
                if audio_path is None: break
                play_main_future = loop.run_in_executor(None, play_and_wait, audio_path, main_output_device_id)
                await play_main_future
                self.audio_file_queue.task_done()
            except asyncio.CancelledError: break
            except Exception: logger.error(f"Error in dedicated audio player: {traceback.format_exc()}")
            
# --- The Main Bot Class ---
class Bot:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop; self.active_mode: str | None = None; self.vts_animation_enabled = False
        self.youtube_enabled = False; self.vision_mode_enabled = False; self.vts_animator: VTSAnimator | None = None
        self.llm_manager = LLMManager(self.loop); self.ai = AICore(self.llm_manager)
        self.tts_manager = TTSManager(self.loop); self.transcription = Transcription()
        self.vision_system: VisionInput | None = None; self.youtube_bot: YouTubeBot | None = None
        self.local_audio_player: LocalAudioPlayer | None = None; self.scheduler: AsyncIOScheduler | None = None
        self.vision_lock = asyncio.Lock(); self.response_queue = asyncio.Queue()
        self.vision_context = {"caption": "N/A", "ocr_text": "N/A"}
        self.conversation_history_for_prompt = defaultdict(lambda: deque(maxlen=config.CONVERSATION_HISTORY_LIMIT))
        self.conversation_log_for_summary = defaultdict(list); self.screenshot_region = None
        self.current_animation_task: asyncio.Task | None = None

    async def setup(self, selected_tts_engine: str, selected_llm_engine: str, vision_enabled: bool = False, vts_enabled: bool = False):
        logger.info("--- Initializing Core Bot Systems ---")
        await self.llm_manager.initialize_engine(selected_llm_engine)
        await self.loop.run_in_executor(None, self.transcription.initialize_model)
        await self.tts_manager.initialize_engine(selected_tts_engine)
        if vision_enabled:
            self.vision_mode_enabled = True; logger.info("Vision mode is ENABLED on startup.")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.vision_system = VisionInput(device=device); await self.start_vision_updates()
        else: logger.info("Vision mode is DISABLED on startup.")
        if vts_enabled:
            logger.info("VTube Studio Animation mode is ENABLED on startup.")
            await self.start_vts_animation_system()
        else: logger.info("VTube Studio Animation mode is DISABLED on startup.")
        logger.info("--- Core Bot Systems Ready ---")

    async def shutdown(self):
        logger.info("--- SHUTDOWN SEQUENCE ---")
        if self.scheduler and self.scheduler.running: self.scheduler.shutdown()
        if self.youtube_bot and self.youtube_bot.is_running: await self.youtube_bot.stop()
        if self.local_audio_player: await self.local_audio_player.stop()
        if self.vts_animator and self.vts_animator.is_running: await self.vts_animator.stop()
        if self.current_animation_task and not self.current_animation_task.done():
            self.current_animation_task.cancel()
        logger.info("Unloading AI models...")
        if self.vision_system: await self.loop.run_in_executor(None, self.vision_system.unload_model)
        if self.llm_manager: await self.llm_manager.shutdown()
        if self.tts_manager: await self.tts_manager.shutdown()
        logger.info("--- SHUTDOWN COMPLETE ---")

    async def play_animation_sequence(self, animation_names: list[str]):
        if not self.vts_animation_enabled or not self.vts_animator: return
        for anim_name in animation_names:
            try:
                duration = self.vts_animator.get_animation_duration(anim_name)
                if duration > 0: self.vts_animator.play_animation(anim_name); await asyncio.sleep(duration)
            except asyncio.CancelledError:
                logger.info(f"Animation sequence '{animation_names}' was cancelled.")
                break
            except Exception as e: logger.error(f"Error playing animation '{anim_name}' in sequence: {e}")

    async def start_vts_animation_system(self):
        if self.vts_animator and self.vts_animator.is_running: logger.info("VTS Animation system is already running."); return
        self.vts_animator = VTSAnimator()
        success = await self.vts_animator.start()
        if success: self.vts_animation_enabled = True
        else: logger.error("Failed to start VTS Animation system."); self.vts_animator = None; self.vts_animation_enabled = False

    async def stop_vts_animation_system(self):
        if self.vts_animator and self.vts_animator.is_running: await self.vts_animator.stop()
        self.vts_animator = None; self.vts_animation_enabled = False; logger.info("VTS Animation system stopped.")
    
    async def set_screenshot_region(self, region: dict | None):
        self.screenshot_region = region; logger.info(f"Screenshot region updated to: {self.screenshot_region}")
        
    async def toggle_vision_mode(self, keep_model_in_memory: bool = True):
        async with self.vision_lock:
            logger.info("--- Received request to toggle vision mode ---")
            previous_state = self.vision_mode_enabled; self.vision_mode_enabled = not self.vision_mode_enabled
            logger.info(f"Vision state changing from {previous_state} to {self.vision_mode_enabled}")
            if self.vision_mode_enabled:
                if self.vision_system is None:
                    logger.info("Vision system is not loaded. Initializing...")
                    try:
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'; self.vision_system = VisionInput(device=device)
                        logger.info("Vision system initialized successfully.")
                    except Exception as e:
                        logger.error(f"Failed to initialize vision system: {e}\n{traceback.format_exc()}"); self.vision_mode_enabled = False; return
                else: logger.info("Vision system is already loaded in memory.")
                await self.start_vision_updates()
            else:
                if self.scheduler and self.scheduler.running:
                    self.scheduler.shutdown(); self.scheduler = None; logger.info("Vision context updater has been stopped.")
                if not keep_model_in_memory and self.vision_system:
                    logger.info(f"Unloading vision model (keep_model_in_memory={keep_model_in_memory})...")
                    try:
                        await self.loop.run_in_executor(None, self.vision_system.unload_model); self.vision_system = None
                        logger.info("Vision model unloaded successfully.")
                    except Exception as e: logger.error(f"Failed to unload vision model: {e}\n{traceback.format_exc()}")
                    
    async def start_vision_updates(self):
        if self.vision_system and not (self.scheduler and self.scheduler.running):
            self.scheduler = AsyncIOScheduler()
            self.scheduler.add_job(self._update_vision_context, 'interval', seconds=config.VISION_UPDATE_INTERVAL_SECONDS, id='vision_context_updater', replace_existing=True)
            self.scheduler.start(); logger.info(f"Vision context updater scheduled every {config.VISION_UPDATE_INTERVAL_SECONDS}s.")
            
    async def _update_vision_context(self):
        async with self.vision_lock:
            if not self.vision_system: logger.debug("Vision update skipped: system is not active or unloaded."); return
            try:
                vision_result = await self.loop.run_in_executor(None, lambda: self.vision_system.process_screen(monitor_index=config.DEFAULT_MONITOR, region=self.screenshot_region))
                if vision_result and vision_result.get('success'):
                    self.vision_context['caption'] = vision_result.get('caption', 'N/A')
                    self.vision_context['ocr_text'] = self.vision_system.get_detected_text(ocr_results=vision_result.get('ocr_results', []))
                    logger.info(f"Vision context updated. Caption: {self.vision_context['caption'][:50]}...")
                    if vision_result['screenshot'] is not None:
                        try:
                            img = vision_result['screenshot']; img.thumbnail((640, 360)); buffered = io.BytesIO()
                            img.save(buffered, format="JPEG"); img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            print(f"VISION_FEED:{img_str}", flush=True)
                        except Exception as e: logger.error(f"Failed to process and send vision feed image: {e}")
            except asyncio.CancelledError: logger.info("Vision update task was canceled, likely during shutdown. This is normal.")
            except Exception: logger.error(f"Error during vision update: {traceback.format_exc()}")

    async def play_audio_to_speakers(self, audio_path: str, start_signal: asyncio.Event):
        if not audio_path:
            start_signal.set()
            return
        loop = asyncio.get_event_loop()
        device_id = config.LOCAL_OUTPUT_DEVICE
        def play_and_wait():
            try:
                with wave.open(audio_path, 'rb') as wf:
                    samplerate = wf.getframerate(); data = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(data, dtype=np.int16)
                loop.call_soon_threadsafe(start_signal.set)
                sd.play(audio_data, samplerate, device=device_id)
                sd.wait()
            except Exception as e:
                logger.error(f"Error playing audio file {audio_path}: {e}")
                if not start_signal.is_set():
                    loop.call_soon_threadsafe(start_signal.set)
        await loop.run_in_executor(None, play_and_wait)

    async def process_ai_response_queue(self, get_active_sink_callback):
        """The main consumer loop that processes tasks from the response queue."""
        logger.info("AI Response Queue Processor is running.")
        while True:
            audio_path_to_clean = None
            try:
                user_id, task_data, emotion_data, channel_id = await self.response_queue.get()
                transcription = None
                if isinstance(task_data, np.ndarray):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        with wave.open(f.name, 'wb') as wf:
                            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(task_data.tobytes())
                        transcription = await self.loop.run_in_executor(None, self.transcription.transcribe, f.name)
                    os.remove(f.name)
                    if not transcription or len(transcription.strip()) < 2: self.response_queue.task_done(); continue
                    logger.info(f"You ({config.user_name}): {transcription}")
                    emotion_data = self.ai.analyze_emotions(transcription)
                elif isinstance(task_data, str): transcription = task_data
                
                if not transcription:
                    self.response_queue.task_done(); continue
                
                user_name = config.USER_NAMES.get(user_id, config.user_name if user_id == "local_user" else f"User({user_id})")
                self.conversation_history_for_prompt[channel_id].append(f"{user_name}: {transcription}")
                self.conversation_log_for_summary[channel_id].append({"role": user_name, "content": transcription, "user_id": user_id})

                current_vision_context = self.vision_context if self.vision_system and self.vision_mode_enabled else None
                ai_response = await self.ai.chat_with_ai(
                    transcription, user_id, emotion_data, "\n".join(self.conversation_history_for_prompt[channel_id]),
                    vision_context=current_vision_context)
                
                logger.info(f"{config.BOT_NAME} to {user_name}: {ai_response}")

                if ai_response and "I'm having trouble responding right now." not in ai_response:
                    self.conversation_history_for_prompt[channel_id].append(f"{config.BOT_NAME}: {ai_response}")
                    self.conversation_log_for_summary[channel_id].append({"role": config.BOT_NAME, "content": ai_response, "user_id": "Bot"})
                    if len(self.conversation_log_for_summary[channel_id]) >= 8:
                        asyncio.create_task(summarizer.consolidate_and_store(list(self.conversation_log_for_summary[channel_id]), self.ai))
                        self.conversation_log_for_summary[channel_id].clear()

                    audio_path = await self.tts_manager.generate_tts_file(ai_response.replace('-', ' '))
                    if not audio_path:
                        logger.error("TTS failed to generate an audio file. Aborting response."); self.response_queue.task_done(); continue
                    
                    audio_path_to_clean = audio_path
                    
                    lipsync_cues = None
                    if self.vts_animation_enabled and self.vts_animator:
                        lipsync_cues = await self.vts_animator.get_lipsync_data(audio_path)

                    start_signal = asyncio.Event()

                    if self.current_animation_task and not self.current_animation_task.done():
                        self.current_animation_task.cancel()

                    all_tasks = set()
                    
                    audio_task = asyncio.create_task(self.play_audio_to_speakers(audio_path, start_signal))
                    all_tasks.add(audio_task)
                    
                    if self.vts_animation_enabled and self.vts_animator:
                        anim_list = self.vts_animator.get_available_animations()
                        if anim_list:
                            animation_sequence = await self.ai.determine_animation_sequence(ai_response, anim_list)
                            if animation_sequence:
                                self.current_animation_task = asyncio.create_task(self.play_animation_sequence(animation_sequence))
                                all_tasks.add(self.current_animation_task)

                    if self.vts_animation_enabled and self.vts_animator and lipsync_cues:
                        lipsync_task = self.vts_animator.start_lipsync_animation(lipsync_cues, start_signal)
                        if lipsync_task:
                            all_tasks.add(lipsync_task)

                    if all_tasks:
                        done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
                        
                        if audio_task in done:
                            for task in pending:
                                task.cancel()
                            if pending:
                                await asyncio.gather(*pending, return_exceptions=True)
                        else:
                            await audio_task
                            for task in all_tasks:
                                if not task.done():
                                    task.cancel()

                self.response_queue.task_done()
            except Exception:
                logger.critical(f"CRITICAL Error in response queue processor:\n{traceback.format_exc()}")
            finally:
                if audio_path_to_clean and os.path.exists(audio_path_to_clean):
                    try: os.remove(audio_path_to_clean)
                    except Exception as e: logger.error(f"Failed to clean up temp audio file {audio_path_to_clean}: {e}")