import discord
import asyncio
import datetime
import os
import io
import wave
import tempfile
import traceback
import subprocess
import sys
from collections import defaultdict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from discord.ext import voice_recv
import soundfile as sf
import config
from src.log.custom_logger import logger

from src.core.bot import Bot
from src.core.bot import find_audio_device_id

# --- Discord-specific Global State ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)

active_voice_clients = {}
api_process = None
external_chat_scheduler: AsyncIOScheduler | None = None

# --- Core Voice Handling Class (Discord Specific) ---
class BufferSink(voice_recv.AudioSink):
    def __init__(self, voice_client, text_channel, bot: Bot):
        self.bot = bot
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

    def wants_opus(self) -> bool: return False

    def write(self, user, data):
        if user is None or not data.pcm or self._shutdown: return
        user_id = str(user.id)
        self.buf[user_id] += data.pcm
        job_id = f'vc_buffer_timer_{user_id}'
        run_time = datetime.datetime.now() + datetime.timedelta(seconds=0.5)
        self.scheduler.add_job(vc_reply, 'date', run_date=run_time, args=[user_id, self.bot], id=job_id, replace_existing=True)

    def freshen(self, user_id):
        if user_id in self.buf: self.buf[user_id] = bytearray()

    async def cleanup(self):
        self._shutdown = True
        if self.scheduler.running: self.scheduler.shutdown(wait=False)
        if self.playback_task and not self.playback_task.done(): self.playback_task.cancel()

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

                discord_playback_finished = asyncio.Event()
                def after_discord_play(error):
                    if error: logger.error(f"Discord playback error: {error}")
                    loop.call_soon_threadsafe(discord_playback_finished.set)

                lipsync_future = None
                if (self.bot.vts_enabled or self.bot.veadotube_enabled):
                    device_id = find_audio_device_id(config.VIRTUAL_MIC_NAME)
                    if device_id is not None:
                        lipsync_future = loop.run_in_executor(None, audio_data, samplerate, device_id)
                
                in_memory_file = io.BytesIO()
                sf.write(in_memory_file, audio_data, samplerate, format='WAV', subtype='PCM_16')
                in_memory_file.seek(0)
                source = discord.FFmpegPCMAudio(in_memory_file, pipe=True)
                
                self._voice_client.play(source, after=after_discord_play)

                await discord_playback_finished.wait()
                if lipsync_future:
                    await lipsync_future

                self.audio_playback_queue.task_done()
            except (asyncio.TimeoutError, asyncio.CancelledError): break
            except Exception: 
                logger.error(f"Error in Discord dedicated audio player:\n{traceback.format_exc()}")
                break
        self.playback_task = None

async def vc_reply(user_id: str, bot: Bot):
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
                wf.setnchannels(1)
                wf.setsampwidth(vc_data['sink'].sample_width)
                wf.setframerate(vc_data['sink'].sample_rate)
                wf.writeframes(user_audio_data)
        
        transcription = await bot.loop.run_in_executor(None, bot.transcription.transcribe, file_path)
        os.remove(file_path)

        if transcription:
            user_name = config.USER_NAMES.get(user_id, f"User({user_id})")
            logger.info(f"Transcription from {user_name}: {transcription}")
            
            if await bot.ai.is_speech_for_ai(transcription, user_id, bot.active_mode):
                channel_id = str(vc_data['sink'].text_channel.id)
                
                bot.conversation_history_for_prompt[channel_id].append(f"{user_name}: {transcription}")
                bot.conversation_log_for_summary[channel_id].append({"role": user_name, "content": transcription, "user_id": user_id})
                
                emotion_data = bot.ai.analyze_emotions(transcription)
                
                await bot.response_queue.put((user_id, transcription, emotion_data, channel_id))

    except Exception as e: 
        logger.error(f"Error in vc_reply: {e}\n{traceback.format_exc()}")
    finally:
        if vc_data and user_id in vc_data['sink'].is_processing_user_buffer:
            vc_data['sink'].is_processing_user_buffer[user_id] = False

async def check_external_chats(bot: Bot):
    if bot.youtube_enabled and bot.youtube_bot and bot.youtube_bot.is_running:
        prompt = bot.youtube_bot.get_random_chat_prompt()
        if prompt:
            emotion_data = bot.ai.analyze_emotions(prompt)
            await bot.response_queue.put(("youtube_user", prompt, emotion_data, "youtube_chat_history"))

@client.event
async def on_ready():
    global external_chat_scheduler
    logger.info(f"Logged in as {client.user}")
    
    if config.YOUTUBE_VIDEO_ID:
        external_chat_scheduler = AsyncIOScheduler()
        external_chat_scheduler.add_job(check_external_chats, 'interval', seconds=15, id='external_chat_checker', args=[client.bot])
        external_chat_scheduler.start()
    
    logger.info("Discord bot is online and ready.")

@client.event
async def on_message(message: discord.Message):
    global api_process
    bot = client.bot

    if message.author == client.user or not message.content.startswith('!'): return
    command, *args = message.content.lower().split(' ')

    # --- Text-based Chat Command ---
    if command == "!h":
        if message.guild.voice_client and message.guild.voice_client.is_connected():
            await message.channel.send("I'm in a voice channel right now, commands are limited.")
            return

        prompt = ' '.join(args)
        if not prompt:
            await message.channel.send("You need to say something!")
            return

        user_id = str(message.author.id)
        channel_id = str(message.channel.id)
        
        emotion_data = bot.ai.analyze_emotions(prompt)
        conversation_log = "\n".join(bot.conversation_history_for_prompt[channel_id])
        
        async with message.channel.typing():
            ai_response = await bot.ai.chat_with_ai(prompt, user_id, emotion_data, conversation_log)
        
        if ai_response:
            bot.conversation_history_for_prompt[channel_id].append(f"{message.author.name}: {prompt}")
            bot.conversation_history_for_prompt[channel_id].append(f"{config.BOT_NAME}: {ai_response}")
            await message.channel.send(ai_response)
        return

    # --- Help Command ---
    if command == "!help":
        embed = discord.Embed(title="Commands", description=f"{config.BOT_NAME}'s main purpose is to listen and respond in voice channels.", color=discord.Color.purple())
        embed.add_field(name="!join / !leave", value="Joins or leaves your current voice channel.", inline=False)
        embed.add_field(name="!s / !m", value="Switch between **Single Mode** (respond to all) and **Multiple Mode** (respond to name).", inline=False)
        embed.add_field(name="Other Toggles", value="---", inline=False)
        ss_status = 'ON' if bot.vision_mode_enabled else 'OFF'
        yt_status = 'ON' if bot.youtube_enabled else 'OFF'
        embed.add_field(name="!ss", value=f"Toggle screen vision. (Currently: **{ss_status}**)", inline=False)
        embed.add_field(name="!vts / !png", value="Toggle VTube Studio or Veadotube lipsync integrations.", inline=False)
        embed.add_field(name="!yt", value=f"Toggle YouTube chat integration. (Currently: **{yt_status}**)", inline=False)
        embed.add_field(name="!api <start|stop|status>", value="Manage the external API server for game engines.", inline=False)
        await message.channel.send(embed=embed)
        return

    # --- Voice Channel Commands ---
    if command == "!join":
        if not message.author.voice: await message.channel.send("You're not in a voice channel."); return
        channel = message.author.voice.channel
        if message.guild.voice_client: await message.channel.send("I'm already in a voice channel here."); return
        try:
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
            sink = BufferSink(vc, message.channel, bot)
            vc.listen(sink)
            active_voice_clients[message.guild.id] = {'vc': vc, 'sink': sink}
            await message.channel.send(f"Fine, I've joined {channel.name}.")
        except Exception as e: await message.channel.send(f"Error joining voice channel: {e}")

    elif command == "!leave":
        if message.guild.id in active_voice_clients:
            vc_data = active_voice_clients.pop(message.guild.id)
            await vc_data['sink'].cleanup()
            await vc_data['vc'].disconnect(force=False)
            await message.channel.send("Alright, I'm out.")
        else: await message.channel.send("I'm not in a voice channel.")

    # --- API Server Command ---
    elif command == "!api":
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

    # --- Mode and Integration Toggles ---
    elif command == "!s": 
        config.DEFAULT_BOT_MODE = config.BOT_MODES["SINGLE"]
        await message.channel.send("Switched to **Single Mode** (will respond to all speech).")
    
    elif command == "!m": 
        config.DEFAULT_BOT_MODE = config.BOT_MODES["MULTIPLE"]
        await message.channel.send("Switched to **Multiple Mode** (will only respond when addressed).")

    elif command == "!vts": 
        bot.vts_enabled = not bot.vts_enabled
        status = "ENABLED" if bot.vts_enabled else "DISABLED"
        await message.channel.send(f"VTube Studio integration is now **{status}**.")
    
    elif command == "!png": 
        bot.veadotube_enabled = not bot.veadotube_enabled
        status = "ENABLED" if bot.veadotube_enabled else "DISABLED"
        await message.channel.send(f"Veadotube lipsync is now **{status}**.")
    
    elif command == "!ss": 
        if not bot.vision_system and not bot.vision_mode_enabled:
            await message.channel.send("First time enabling vision, this may take a moment...")
            try:
                import torch
                from src.input.vision import VisionInput
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                bot.vision_system = VisionInput(device=device)
                await message.channel.send("Vision system initialized.")
            except Exception as e:
                await message.channel.send(f"Failed to initialize vision system: {e}")
                return

        bot.vision_mode_enabled = not bot.vision_mode_enabled
        status = "ON" if bot.vision_mode_enabled else "OFF"
        await message.channel.send(f"{config.BOT_NAME}'s screen vision ability is now **{status}**.")

    elif command == "!yt":
        if not config.YOUTUBE_VIDEO_ID:
            await message.channel.send("YouTube integration is not configured.")
            return

        bot.youtube_enabled = not bot.youtube_enabled
        status = "ENABLED" if bot.youtube_enabled else "DISABLED"
        await message.channel.send(f"YouTube integration is now **{status}**.")

        if bot.youtube_enabled:
            if not bot.youtube_bot:
                from src.services.chat.yt import YouTubeBot
                bot.youtube_bot = YouTubeBot(bot.response_queue)
            if not bot.youtube_bot.is_running:
                await message.channel.send("Connecting to YouTube chat...")
                asyncio.create_task(bot.youtube_bot.run())
        elif not bot.youtube_enabled and bot.youtube_bot:
            if bot.youtube_bot.is_running:
                await message.channel.send("Disconnecting from YouTube chat...")
                await bot.youtube_bot.stop()