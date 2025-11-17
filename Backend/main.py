import asyncio
import sys
import traceback
import config
from src.log.custom_logger import logger
from src.core.bot import Bot, LocalAudioPlayer
import src.input.discord_bot as discord_bot_module
from src.input.local_audio import LocalAudioHandler

# --- Mode Execution Functions ---
async def run_local_mode_async(bot: Bot, audio_mode: str):
    """
    Initializes and runs the bot in local audio mode.
    """
    logger.info(f"--- Starting Local Mode ({audio_mode.upper()}) ---")
    bot.active_mode = "local"

    bot.local_audio_player = LocalAudioPlayer(bot)
    await bot.local_audio_player.start()

    def get_local_sink():
        return bot.local_audio_player

    asyncio.create_task(bot.process_ai_response_queue(get_local_sink))
    logger.info("Global AI response queue processor started for Local Mode.")

    handler = LocalAudioHandler(mode=audio_mode, loop=bot.loop, queue=bot.response_queue)
    await bot.loop.run_in_executor(None, handler.start)

    logger.info("--- Local Mode is fully running. Press Ctrl+C to stop. ---")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        logger.info("Stopping local audio handler...")
        handler.stop()


async def run_discord_mode_async(bot: Bot):
    """
    Initializes and runs the Discord bot client.
    """
    logger.info("\n--- STARTING IN DISCORD BOT MODE ---")
    bot.active_mode = "discord"
    
    discord_bot_module.client.bot = bot

    def get_discord_sink():
        vc_data = next((vc for vc in discord_bot_module.active_voice_clients.values() if vc['vc'].is_connected()), None)
        return vc_data['sink'] if vc_data else None

    asyncio.create_task(bot.process_ai_response_queue(get_discord_sink))
    logger.info("Global AI response queue processor started for Discord Mode.")

    await discord_bot_module.client.start(config.DISCORD_TOKEN)


async def async_main():
    """The main asynchronous entry point for the application."""
    logger.info("--- BOOT SEQUENCE ---")
    loop = asyncio.get_running_loop()
    bot = Bot(loop)

    # --- Engine Selections (TTS & LLM) ---
    available_tts_engines = bot.tts_manager.get_available_engines()
    selected_tts_engine = None
    if not available_tts_engines:
        logger.critical("FATAL: No TTS engines found.")
        return
    if sys.stdin.isatty():
        if len(available_tts_engines) == 1:
            selected_tts_engine = available_tts_engines[0]
        else:
            while True:
                print("Please choose a TTS engine:"); [print(f"  ({i}) {e}") for i, e in enumerate(available_tts_engines, 1)]
                try:
                    choice = int(input("Enter choice number: ").strip()) - 1
                    if 0 <= choice < len(available_tts_engines): selected_tts_engine = available_tts_engines[choice]; break
                    else: logger.warning("Invalid choice.")
                except (ValueError, IndexError): logger.warning("Invalid input.")
    else: selected_tts_engine = available_tts_engines[0]
    logger.info(f"Using TTS Engine: '{selected_tts_engine}'")

    available_llm_engines = bot.llm_manager.get_available_engines()
    selected_llm_engine = None
    if not available_llm_engines:
        logger.critical("FATAL: No LLM engines found.")
        return
    if sys.stdin.isatty():
        if len(available_llm_engines) == 1:
            selected_llm_engine = available_llm_engines[0]
        else:
            while True:
                print("\nPlease choose an LLM engine:"); [print(f"  ({i}) {e}") for i, e in enumerate(available_llm_engines, 1)]
                try:
                    choice = int(input("Enter choice number: ").strip()) - 1
                    if 0 <= choice < len(available_llm_engines): selected_llm_engine = available_llm_engines[choice]; break
                    else: logger.warning("Invalid choice.")
                except (ValueError, IndexError): logger.warning("Invalid input.")
    else: selected_llm_engine = available_llm_engines[0]
    logger.info(f"Using LLM Engine: '{selected_llm_engine}'")
    
    # --- NEW: Feature Selection (Vision & VTS) ---
    vision_enabled = config.VISION_STARTUP
    vts_enabled = config.VTS_ANIMATION_STARTUP

    if sys.stdin.isatty():
        # Vision Selection
        default_vision = 'Y' if config.VISION_STARTUP else 'N'
        choice = input(f"\nEnable Vision Mode? (Default: {default_vision}) [Y/n]: ").strip().lower()
        if choice == 'n': vision_enabled = False
        elif choice == 'y': vision_enabled = True
        
        # VTS Animation Selection
        default_vts = 'Y' if config.VTS_ANIMATION_STARTUP else 'N'
        choice = input(f"Enable VTube Studio Animation System? (Default: {default_vts}) [Y/n]: ").strip().lower()
        if choice == 'n': vts_enabled = False
        elif choice == 'y': vts_enabled = True

    if not selected_tts_engine or not selected_llm_engine:
        logger.critical("Could not determine which engines to use. Exiting.")
        return

    try:
        await bot.setup(
            selected_tts_engine, 
            selected_llm_engine,
            vision_enabled=vision_enabled,
            vts_enabled=vts_enabled
        )
        logger.info("--- CORE INITIALIZATION COMPLETE ---")

        # --- Mode Selection (Local vs. Discord) ---
        mode = '2'; audio_mode = 'ptt'
        if sys.stdin.isatty():
            while True:
                choice = input("\nChoose mode: (1) Local Audio | (2) Discord Bot\nEnter choice: ").strip()
                if choice in ['1', '2']: mode = choice; break
                logger.warning("Invalid choice. Please enter 1 or 2.")
            if mode == '1':
                while True:
                    choice = input("Choose input mode: (1) PTT | (2) VAD\nEnter choice: ").strip().lower()
                    if choice in ['1', 'ptt']: audio_mode = "ptt"; break
                    elif choice in ['2', 'vad']: audio_mode = "vad"; break
                    else: logger.warning("Invalid choice.")
        else: 
            try:
                line = sys.stdin.readline().strip().split(','); mode = line[0]
                if len(line) > 1: audio_mode = line[1]
            except (IndexError, ValueError): mode = '2'

        if mode == '1':
            await run_local_mode_async(bot, audio_mode)
        elif mode == '2':
            if not config.DISCORD_TOKEN:
                logger.critical("FATAL: DISCORD_TOKEN not found in config or .env file.")
                return
            await run_discord_mode_async(bot)

    except KeyboardInterrupt:
        logger.info("\nApplication terminated by user.")
    except Exception as e:
        logger.critical(f"\n--- CRITICAL ERROR ---\n{traceback.format_exc()}")
    finally:
        logger.info("\n--- SHUTDOWN SEQUENCE ---")
        await bot.shutdown()
        logger.info("Shutdown sequence finished. Goodbye.")


def main():
    """Synchronous entry point."""
    try:
        asyncio.run(async_main())
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise

if __name__ == "__main__":
    main()