import discord
import asyncio
import datetime
import time
import re
import os
import io
import wave
import tempfile
import functools
import traceback
import logging
import torch
import subprocess
import sys
from collections import deque, defaultdict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from discord.ext import voice_recv
from fuzzywuzzy import fuzz, process
import sounddevice as sd
import soundfile as sf
import config
from ai_core import chat_with_ai, analyze_emotions, classify_speech_intent
from transcription import transcribe_audio
import tts_interface
from vts_interface import trigger_vts_animation
from vision import VisionInput
from summarizer import summarizer
from memory import memory_manager
from plugin_manager import PluginManager
from plugins.base_plugin import BotInterface
from twitch import TwitchBot
from yt import YouTubeBot

# --- Global Bot State ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)
active_mode = config.DEFAULT_BOT_MODE
vts_enabled = False
veadotube_enabled = False
vision_mode_enabled = False
twitch_enabled = False
youtube_enabled = False
twitch_bot: TwitchBot | None = None
youtube_bot: YouTubeBot | None = None
vision_system: VisionInput | None = None
active_voice_clients = {}
response_queue = deque()
api_process = None
plugin_manager: PluginManager | None = None # Will hold the plugin manager instance
external_chat_scheduler: AsyncIOScheduler | None = None

# --- Conversation History Management ---
conversation_history_for_prompt = defaultdict(lambda: deque(maxlen=config.CONVERSATION_HISTORY_LIMIT))
conversation_log_for_summary = defaultdict(list)

class BotInterfaceImpl(BotInterface):
    def is_bot_busy(self) -> bool:
        """Checks all conditions to see if the bot should speak."""
        vc_data = next(iter(active_voice_clients.values()), None)
        if not vc_data:
            return True

        if any(vc_data['sink'].is_processing_user_buffer.values()): return True
        if vc_data['vc'].is_playing(): return True
        if response_queue: return True

        return False

    async def request_bot_speech(self, prompt: str, user_id: str, emotion: str):
        """Handles a plugin's request to generate and speak a line."""
        print(f"[DEBUG] BotInterface received request to speak: '{prompt}'")

        vc_data = next(iter(active_voice_clients.values()), None)
        if not vc_data: return

        sink = vc_data['sink']
        channel_id = str(sink.text_channel.id)
        conversation_log = "\n".join(conversation_history_for_prompt[channel_id])
        ai_response = await chat_with_ai(prompt, user_id, {"dominant_emotion": emotion}, conversation_log)

        print(f"[DEBUG] AI Response: '{ai_response}'")

        if ai_response and "I can't speak now" not in ai_response:
            conversation_history_for_prompt[channel_id].append(f"{config.BOT_NAME}: {ai_response}")
            conversation_log_for_summary[channel_id].append({"role": config.BOT_NAME, "content": ai_response, "user_id": "Bot"})
            await generate_and_play_tts(ai_response, sink)

# --- Audio & Lipsync ---
def find_audio_device_id(device_name_query):
    if not device_name_query: return None
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name_query.lower() in device['name'].lower() and device['max_output_channels'] > 0:
                print(f"Lipsync: Found audio device '{device['name']}' with ID {i}")
                return i
    except Exception as e:
        print(f"Lipsync: Error querying audio devices: {e}")
    return None

def play_audio_on_device(data, samplerate, device_id):
    try:
        sd.play(data, samplerate, device=device_id)
        sd.wait()
    except Exception as e:
        print(f"Lipsync: Error playing audio on device {device_id}: {e}")

# --- Core Voice Handling Class ---
class BufferSink(voice_recv.AudioSink):
    def __init__(self, voice_client, text_channel):
        self.buf = defaultdict(bytearray)
        self.sample_width = 2
        self.sample_rate = 96000
        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()
        self._voice_client = voice_client
        self.text_channel = text_channel
        self.is_processing_user_buffer = defaultdict(bool)
        self.audio_playback_queue = asyncio.Queue()
        self.playback_task = None
        self._shutdown = False
        self.queue_processor_job = self.scheduler.add_job(
            process_ai_response_queue, 'interval', seconds=0.5, id='ai_queue_processor'
        )

    def wants_opus(self) -> bool: return False

    def write(self, user, data):
        if user is None or not data.pcm or self._shutdown: return
        user_id = str(user.id)
        self.buf[user_id] += data.pcm
        job_id = f'vc_buffer_timer_{user_id}'
        run_time = datetime.datetime.now() + datetime.timedelta(seconds=0.5)
        self.scheduler.add_job(vc_reply, 'date', run_date=run_time, args=[user_id], id=job_id, replace_existing=True)

    def freshen(self, user_id):
        if user_id in self.buf: self.buf[user_id] = bytearray()

    async def cleanup(self):
        print("BufferSink cleanup initiated.")
        self._shutdown = True
        if self.scheduler.running: self.scheduler.shutdown(wait=False)
        if self.playback_task and not self.playback_task.done(): self.playback_task.cancel()
        print("BufferSink cleanup finished.")

    async def enqueue_audio_segment(self, audio_item):
        if self._shutdown: return
        await self.audio_playback_queue.put(audio_item)
        if self.playback_task is None or self.playback_task.done():
            self.playback_task = asyncio.create_task(self._dedicated_audio_player())

    async def _dedicated_audio_player(self):
        loop = asyncio.get_event_loop()
        while not self._shutdown:
            try:
                audio_item = await asyncio.wait_for(self.audio_playback_queue.get(), timeout=60.0)
                if audio_item is None: break
                samplerate, audio_data = audio_item
                lipsync_future = None
                if vts_enabled or veadotube_enabled:
                    device_id = find_audio_device_id(config.VIRTUAL_MIC_NAME)
                    if device_id is not None:
                        lipsync_future = loop.run_in_executor(None, play_audio_on_device, audio_data, samplerate, device_id)

                in_memory_file = io.BytesIO()
                sf.write(in_memory_file, audio_data, samplerate, format='WAV', subtype='PCM_16')
                in_memory_file.seek(0)
                source = discord.FFmpegPCMAudio(in_memory_file, pipe=True)
                self._voice_client.play(source)

                while self._voice_client.is_playing(): await asyncio.sleep(0.05)
                if lipsync_future: await lipsync_future
                self.audio_playback_queue.task_done()
            except (asyncio.TimeoutError, asyncio.CancelledError): break
            except Exception: traceback.print_exc(); break
        self.playback_task = None

# --- Social Intelligence & Decision Making ---
def is_definitely_for_bot(transcription: str) -> bool:
    text = transcription.lower().strip()

    wake_words = '|'.join(re.escape(name.lower()) for name in config.BOT_WAKE_WORDS)
    wake_prefixes = '|'.join(re.escape(prefix.lower()) for prefix in config.WAKE_PREFIXES)
    # Check for wake prefix + bot name
    if re.search(rf'^({wake_prefixes})\s+({wake_words})\b', text):
        return True
    # Check for bot name at start
    if re.search(rf'^({wake_words})[\s,]', text):
        return True
    # Check for bot name at end
    if re.search(rf'[\s,]+({wake_words})\??\s*$', text):
        return True

    return False

async def is_speech_for_ai(transcription: str, user_id: str) -> bool:
    cleaned_transcription = transcription.lower().strip()
    if not cleaned_transcription: return False
    if is_definitely_for_bot(cleaned_transcription):
        print("Decision: Respond (Priority rule-based trigger detected)")
        return True
    if cleaned_transcription in config.IGNORE_EXPRESSIONS:
        print(f"Decision: Ignore (Phrase '{cleaned_transcription}' is on the ignore list)")
        return False
    if active_mode == config.BOT_MODES["SINGLE"]:
        print("Decision: Respond (Single Mode is active)")
        return True
    print("Deferring to AI for final analysis...")
    is_for_bot = await asyncio.to_thread(classify_speech_intent, cleaned_transcription)
    print(f"Decision: {'Respond' if is_for_bot else 'Ignore'} (AI classifier fallback)")
    return is_for_bot

def is_vision_request(transcription: str) -> bool:

    if not vision_mode_enabled: 
        return False
    
    text = transcription.lower().strip()
    # 1st method (trigger phases)
    for phrase in config.VISION_TRIGGER_PHRASES:
        if phrase in text:
            print(f"Vision request detected via trigger phrase: '{phrase}'")
            return True
        
    # 2nd method (context clues)
    for clue in config.VISION_CONTEXT_CLUES:
        if clue in text:
            print(f"Vision request detected via context clue: '{clue}'")
            return True

    # 3rd method (fuzzy matching)
    action_match = process.extractOne(text, config.VISION_ACTION_WORDS, scorer=fuzz.partial_token_sort_ratio)
    target_present = any(word in text for word in config.VISION_TARGET_WORDS)

    if action_match and action_match[1] > config.VISION_CONFIDENCE_THRESHOLD:
        if target_present:
            print(f"Vision request detected via fuzzy matching: Action '{action_match[0]}' with confidence {action_match[1]}")
            return True
        
    # 4th method (question detection)
        question_starter = ["what", "whats", "what's", "how", "where", "who", "can you"]
        if any(text.startswith(starter) for starter in question_starter):
             print("Vision request detected via question format: '{action_match[0]}'")
             return True
        
    #5th method (imperative commands)
    imperative_commands = ["see", "look", "show", "describe", "read", "check", "analyze"]
    if any(text.startswith(cmd) for cmd in imperative_commands):
        print(f"Vision request detected via imperative command: '{text}'")
        return True

    return False

async def vc_reply(user_id: str):
    vc_data = next((vc for vc in active_voice_clients.values() if vc['sink'].buf.get(user_id)), None)
    if not vc_data or vc_data['sink'].is_processing_user_buffer.get(user_id): return
    vc_data['sink'].is_processing_user_buffer[user_id] = True
    try:
        user_audio_data = bytes(vc_data['sink'].buf.get(user_id, b''))
        vc_data['sink'].freshen(user_id)
        if len(user_audio_data) < 20000: return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            file_path = tmp_wav.name
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(vc_data['sink'].sample_width)
                wf.setframerate(vc_data['sink'].sample_rate); wf.writeframes(user_audio_data)

        transcription = await asyncio.to_thread(transcribe_audio, file_path)
        os.remove(file_path)

        if transcription:
            user_name = config.USER_NAMES.get(user_id, f"User({user_id})")
            print(f"Transcription from {user_name}: {transcription}")
            if await is_speech_for_ai(transcription, user_id):
                channel_id = str(vc_data['sink'].text_channel.id)
                conversation_history_for_prompt[channel_id].append(f"{user_name}: {transcription}")
                conversation_log_for_summary[channel_id].append({"role": user_name, "content": transcription, "user_id": user_id})

                if is_vision_request(transcription):
                    asyncio.create_task(handle_vision_request(user_id, vc_data['sink']))
                else:
                    emotion_data = analyze_emotions(transcription)
                    response_queue.append((user_id, transcription, emotion_data, channel_id))
    except Exception as e: print(f"Error in vc_reply: {e}\n{traceback.format_exc()}")
    finally:
        if user_id in vc_data['sink'].is_processing_user_buffer:
            vc_data['sink'].is_processing_user_buffer[user_id] = False

async def handle_vision_request(user_id: str, sink: BufferSink):
    if not vision_system: return
    try:
        screenshot = await asyncio.to_thread(vision_system.capture_screenshot, 1)
        caption, ocr_text = await asyncio.gather(
            asyncio.to_thread(vision_system.generate_caption, screenshot),
            asyncio.to_thread(vision_system.get_detected_text, screenshot)
        )
        prompt = f"You looked at the user's screen. My analysis is:\nScene: '{caption}'\nText: '{ocr_text}'\n\nFormulate a response based on this information. Remember you are talking to {config.USER_NAMES.get(user_id, 'User')}"
        channel_id = str(sink.text_channel.id)
        conversation_log = "\n".join(conversation_history_for_prompt[channel_id])
        ai_response = await chat_with_ai(prompt, user_id, {"dominant_emotion": "neutral"}, conversation_log)
        if ai_response:
            conversation_history_for_prompt[channel_id].append(f"{config.BOT_NAME}: {ai_response}")
            await generate_and_play_tts(ai_response, sink)
    except Exception as e: print(f"Error during vision handling: {e}\n{traceback.format_exc()}")

async def process_ai_response_queue():
    if not response_queue: return
    user_id, transcription, emotion_data, channel_id = response_queue.popleft()
    vc_data = next((vc for vc in active_voice_clients.values() if vc['vc'].is_connected()), None)
    if not vc_data: return
    try:
        print(f"Processing queue for user {user_id}: '{transcription}'")
        conversation_log = "\n".join(conversation_history_for_prompt[channel_id])
        ai_response = await chat_with_ai(transcription, user_id, emotion_data, conversation_log)
        user_name = config.USER_NAMES.get(user_id, user_id)
        print(f"{config.BOT_NAME} to {user_name}: {ai_response}")
        if ai_response and "I'm having trouble responding right now." not in ai_response:
            conversation_history_for_prompt[channel_id].append(f"{config.BOT_NAME}: {ai_response}")
            conversation_log_for_summary[channel_id].append({"role": config.BOT_NAME, "content": ai_response, "user_id": "Bot"})
            if len(conversation_log_for_summary[channel_id]) >= 8:
                print("--- Triggering memory consolidation ---")
                history_to_summarize = list(conversation_log_for_summary[channel_id])
                conversation_log_for_summary[channel_id].clear()
                asyncio.create_task(summarizer.consolidate_and_store(history_to_summarize))
            ai_emotion_data = analyze_emotions(ai_response)
            if vts_enabled: await trigger_vts_animation(ai_emotion_data.get("dominant_emotion", "neutral"))
            await generate_and_play_tts(ai_response, vc_data['sink'])
    except Exception as e: print(f"Error processing response queue: {e}\n{traceback.format_exc()}")

async def generate_and_play_tts(text: str, sink: BufferSink):
    loop = asyncio.get_event_loop()
    audio_data_queue = asyncio.Queue()
    producer_task = loop.run_in_executor(None, functools.partial(tts_interface._blocking_tts_segment_producer, audio_data_queue, loop, text))
    segment_count = 0
    start_time = time.time()
    last_chunk_time = start_time
    while True:
        try:
            item = await asyncio.wait_for(audio_data_queue.get(), timeout=180.0)
            if item is None:
                break
            if isinstance(item, Exception):
                print(f"ERROR from TTS producer thread: {item}")
                break
            current_time = time.time()
            segment_count += 1
            if segment_count == 1:
                time_to_first_chunk = current_time - start_time
                print(f"✅ TTS: Time to first audio chunk: {time_to_first_chunk:.2f}s.")
            else:
                time_for_this_chunk = current_time - last_chunk_time
                print(f"TTS: Received audio chunk {segment_count} in {time_for_this_chunk:.2f}s.")
            last_chunk_time = current_time
            await sink.enqueue_audio_segment(item)
        except asyncio.TimeoutError:
            print("TTS generation timed out waiting for audio chunk.")
            break
        except Exception as e:
            print(f"ERROR in TTS consumer loop: {e}")
            break
    await producer_task
    total_time = time.time() - start_time
    if segment_count > 0:
        print(f"✅ TTS: Finished. Generated {segment_count} audio segments in a total of {total_time:.2f}s.")
    else:
        print("⚠️ WARNING: TTS producer finished but generated no audio segments.")

async def check_external_chats():
    """Periodically checks Twitch/YouTube for messages and injects them into the queue."""
    # Twitch Check
    if twitch_enabled and twitch_bot and twitch_bot.is_running:
        current_time = time.time()
        if current_time - twitch_bot.last_prompt_time >= config.CHAT_COOLDOWN_SECONDS:
            prompt = twitch_bot.get_random_chat_prompt()
            if prompt:
                print(f"Injecting prompt from Twitch: '{prompt}'")
                response_queue.append(("twitch_user", prompt, analyze_emotions(prompt), "twitch_chat_history"))
                twitch_bot.last_prompt_time = time.time()

    # YouTube Check
    if youtube_enabled and youtube_bot and youtube_bot.is_running:
        prompt = youtube_bot.get_random_chat_prompt()
        if prompt:
            print(f"Injecting prompt from YouTube: '{prompt}'")
            response_queue.append(("youtube_user", prompt, analyze_emotions(prompt), "youtube_chat_history"))

# --- Discord Event Handlers ---
@client.event
async def on_ready():
    global vision_system, plugin_manager, external_chat_scheduler
    logging.basicConfig(level=logging.INFO)
    print(f"Logged in as {client.user}")

    # --- INITIALIZE PLUGIN SYSTEM ---
    bot_interface = BotInterfaceImpl()
    plugin_manager = PluginManager(bot_interface)
    plugin_manager.discover_plugins()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config.VISION_ACTION_WORDS: vision_system = VisionInput(device=device)
    
    # --- Initialize External Chat Scheduler ---
    if config.TWITCH_TOKEN or config.YOUTUBE_VIDEO_ID:
        external_chat_scheduler = AsyncIOScheduler()
        external_chat_scheduler.add_job(
            check_external_chats, 
            'interval', 
            seconds=15, 
            id='external_chat_checker'
        )
        external_chat_scheduler.start()
        print("Scheduler for external chats (Twitch/YouTube) started.")
    else:
        print("Skipping external chat scheduler: No credentials provided in config.")

    print("online and ready.")


@client.event
async def on_message(message: discord.Message):
    global active_mode, vts_enabled, veadotube_enabled, vision_mode_enabled, api_process, twitch_enabled, twitch_bot, youtube_enabled, youtube_bot
    if message.author == client.user or not message.content.startswith('!'): return

    command, *args = message.content.lower().split(' ')

    if command == "!help":
        embed = discord.Embed(title="Commands", description=f"{config.BOT_NAME}'s main purpose is to listen and respond in voice channels.", color=discord.Color.purple())
        embed.add_field(name="!join / !leave", value="Joins or leaves your current voice channel.", inline=False)
        embed.add_field(name="!s / !m", value="Switch between **Single Mode** (respond to all) and **Multiple Mode** (respond to name).", inline=False)
        embed.add_field(name="Plugin Commands", value="---", inline=False)
        embed.add_field(name="!plugin list", value="Lists all background plugins and their current status.", inline=False)
        embed.add_field(name="!plugin <name> <on|off>", value="Loads/Enables or disables a specific plugin (e.g., `!plugin osu on`).", inline=False)
        embed.add_field(name="Other Toggles", value="---", inline=False)
        embed.add_field(name="!ss", value=f"Toggle screen vision. (Currently: **{'ON' if vision_mode_enabled else 'OFF'}**)", inline=False)
        embed.add_field(name="!vts / !png", value="Toggle VTube Studio or Veadotube lipsync integrations.", inline=False)
        embed.add_field(name="!twitch", value=f"Toggle Twitch chat integration. (Currently: **{'ON' if twitch_enabled else 'OFF'}**)", inline=False)
        embed.add_field(name="!yt", value=f"Toggle YouTube chat integration. (Currently: **{'ON' if youtube_enabled else 'OFF'}**)", inline=False)
        embed.add_field(name=f"!api <start|stop|status>", value="Manage the external API server for game engines.", inline=False)
        await message.channel.send(embed=embed)
        return

    if command == "!join":
        if not message.author.voice: await message.channel.send("You're not in a voice channel."); return
        channel = message.author.voice.channel
        if message.guild.voice_client: await message.channel.send("I'm already in a voice channel here."); return
        try:
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
            sink = BufferSink(vc, message.channel)
            vc.listen(sink)
            active_voice_clients[message.guild.id] = {'vc': vc, 'sink': sink}
            if plugin_manager:
                plugin_manager.start_polling(sink.scheduler)
            await message.channel.send(f"Fine, I've joined {channel.name}.")
        except Exception as e: await message.channel.send(f"Error joining voice channel: {e}")

    elif command == "!leave":
        if message.guild.id in active_voice_clients:
            if plugin_manager:
                plugin_manager.stop_polling()
            vc_data = active_voice_clients.pop(message.guild.id)
            await vc_data['sink'].cleanup()
            await vc_data['vc'].disconnect(force=False)
            await message.channel.send("Alright, I'm out.")
        else: await message.channel.send("I'm not in a voice channel.")

    elif command == "!plugin":
        if not plugin_manager: await message.channel.send("The plugin system isn't initialized."); return
        if not args or args[0] == 'list':
            if not plugin_manager.available_plugins:
                await message.channel.send("No plugins were discovered."); return
            embed = discord.Embed(title="Plugin Status", color=discord.Color.blue())
            for name in plugin_manager.available_plugins:
                if name in plugin_manager.active_plugins:
                    instance = plugin_manager.active_plugins[name]
                    status = "✅ ENABLED" if instance.is_enabled else "⚖️ LOADED (DISABLED)"
                else:
                    status = "💤 INACTIVE (NOT LOADED)"
                embed.add_field(name=f"**`{name}`**", value=status, inline=False)
            await message.channel.send(embed=embed)
            return

        if len(args) < 2: await message.channel.send(f"Usage: `!plugin <name> <on|off>`"); return
        plugin_name, action = args[0].lower(), args[1].lower()
        if plugin_name not in plugin_manager.available_plugins:
            await message.channel.send(f"Plugin '{plugin_name}' not found. Available: `{', '.join(plugin_manager.available_plugins.keys())}`"); return

        if action == "on":
            instance = plugin_manager.activate_plugin(plugin_name)
            if instance:
                instance.is_enabled = True
                await message.channel.send(f"Plugin **{plugin_name}** is now **LOADED and ENABLED**.")
            else:
                await message.channel.send(f"Failed to activate plugin '{plugin_name}'. Check logs.")
        elif action == "off":
            if plugin_name in plugin_manager.active_plugins:
                plugin_instance = plugin_manager.active_plugins[plugin_name]
                plugin_instance.is_enabled = False
                await message.channel.send(f"Plugin **{plugin_name}** is now **DISABLED** (but remains loaded in memory).")
            else:
                await message.channel.send(f"Plugin **{plugin_name}** is already inactive and not loaded.")
        else:
            await message.channel.send("Invalid action. Use `on` or `off`.")

    elif command == f"!api":
        if not message.author.guild_permissions.administrator:
            await message.channel.send("You need to be an administrator to use this command."); return
        sub_command = args[0] if args else "status"
        if sub_command == "start":
            if api_process and api_process.poll() is None:
                await message.channel.send(f"The API is already running.")
            else:
                try:
                    api_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")
                    api_process = subprocess.Popen([sys.executable, api_script_path])
                    await message.channel.send(f"The API is **starting...**")
                except Exception as e:
                    await message.channel.send(f"Failed to start the API server: `{e}`")
        elif sub_command == "stop":
            if not api_process or api_process.poll() is not None:
                await message.channel.send(f"The API is not currently running.")
            else:
                api_process.terminate(); api_process.wait(); api_process = None
                await message.channel.send(f"The API has been **stopped**.")
        elif sub_command == "status":
            if api_process and api_process.poll() is None:
                await message.channel.send(f"The API is **online**. PID: `{api_process.pid}`.")
            else:
                await message.channel.send(f"The API is **offline**.")
        else:
            await message.channel.send(f"Invalid command. Usage: `!api <start|stop|status>`")

    elif command == "!s": active_mode = config.BOT_MODES["SINGLE"]; await message.channel.send("Switched to **Single Mode**.")
    elif command == "!m": active_mode = config.BOT_MODES["MULTIPLE"]; await message.channel.send("Switched to **Multiple Mode**.")
    elif command == "!vts": vts_enabled = not vts_enabled; status = "ENABLED" if vts_enabled else "DISABLED"; await message.channel.send(f"VTube Studio integration is now **{status}**.")
    elif command == "!png": veadotube_enabled = not veadotube_enabled; status = "ENABLED" if veadotube_enabled else "DISABLED"; await message.channel.send(f"Veadotube lipsync is now **{status}**.")
    elif command == "!ss": vision_mode_enabled = not vision_mode_enabled; status = "ON" if vision_mode_enabled else "OFF"; await message.channel.send(f"{config.BOT_NAME}'s screen vision ability is now **{status}**.")

    elif command == "!twitch":
        if not all([config.TWITCH_APP_ID, config.TWITCH_APP_SECRET, config.TWITCH_CHANNEL]):
            await message.channel.send("Twitch integration is not fully configured. Please set `TWITCH_APP_ID`, `TWITCH_APP_SECRET`, and `TWITCH_CHANNEL` in the config.")
            return

        twitch_enabled = not twitch_enabled
        status = "ENABLED" if twitch_enabled else "DISABLED"
        await message.channel.send(f"Twitch integration is now **{status}**.")
        
        if twitch_enabled:
            if not twitch_bot:
                 twitch_bot = TwitchBot(response_queue)
            if not twitch_bot.is_running:
                await message.channel.send("Connecting to Twitch chat... (Check console for first-time auth link)")
                asyncio.create_task(twitch_bot.run())
        elif not twitch_enabled and twitch_bot:
            if twitch_bot.is_running:
                await message.channel.send("Disconnecting from Twitch chat...")
                await twitch_bot.stop()

    elif command == "!yt":
        if not config.YOUTUBE_VIDEO_ID:
            await message.channel.send("YouTube integration is not configured. Please set `YOUTUBE_VIDEO_ID` in the config.")
            return

        youtube_enabled = not youtube_enabled
        status = "ENABLED" if youtube_enabled else "DISABLED"
        await message.channel.send(f"YouTube integration is now **{status}**.")

        if youtube_enabled:
            if not youtube_bot:
                youtube_bot = YouTubeBot(response_queue)
            if not youtube_bot.is_running:
                await message.channel.send("Connecting to YouTube chat...")
                asyncio.create_task(youtube_bot.run())
        elif not youtube_enabled and youtube_bot:
            if youtube_bot.is_running:
                await message.channel.send("Disconnecting from YouTube chat...")
                await youtube_bot.stop()
