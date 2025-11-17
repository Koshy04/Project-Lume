import asyncio
import sys
import traceback
import config
import os
from src.log.custom_logger import logger
from src.core.bot import Bot
import src.input.discord_bot as discord_bot_module
from src.input.local_audio import LocalAudioHandler

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def listen_for_stdin_thread(bot: Bot, loop: asyncio.AbstractEventLoop):
    """
    Synchronous function to run in a separate thread, listening for stdin commands.
    This is the robust way to handle stdin on Windows with asyncio.
    """
    logger.info("Starting stdin command listener thread.")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break 
            
            command = line.strip()
            if not command:
                continue

            logger.info(f"Received command from stdin: '{command}'")

            if command.startswith("TOGGLE_VISION:"):
                try:
                    keep_model_str = command.split(":")[1]
                    keep_model = keep_model_str.lower() == 'true'
                    loop.call_soon_threadsafe(asyncio.create_task, bot.toggle_vision_mode(keep_model_in_memory=keep_model))
                except IndexError:
                    logger.error("Invalid TOGGLE_VISION command format. Expected 'TOGGLE_VISION:true/false'")

            elif command == "SHUTDOWN":
                logger.info("Shutdown command received. Initiating graceful shutdown.")
                loop.call_soon_threadsafe(loop.stop)
                break # Exit the listener thread

        except Exception as e:
            logger.error(f"Error in stdin listener thread: {e}")
            break
    logger.info("Stdin command listener thread has stopped.")


async def run_local_mode_async(bot: Bot, audio_mode: str):
    """Initializes and runs the bot in local mode."""
    logger.info(f"--- Starting Local Mode ({audio_mode.upper()}) ---")
    from src.core.bot import LocalAudioPlayer # Local import
    bot.active_mode = "local"
    bot.vts_enabled = True 
    bot.local_audio_player = LocalAudioPlayer(bot)
    await bot.local_audio_player.start()
    
    asyncio.create_task(bot.process_ai_response_queue(lambda: bot.local_audio_player))
    
    handler = LocalAudioHandler(mode=audio_mode, loop=bot.loop, queue=bot.response_queue)
    await bot.loop.run_in_executor(None, handler.start)
    
    try:
        await asyncio.Event().wait() # Keep running indefinitely
    finally:
        handler.stop()


async def run_discord_mode_async(bot: Bot):
    """Initializes and runs the bot in Discord mode."""
    logger.info("--- Starting Discord Mode ---")
    bot.active_mode = "discord"
    discord_bot_module.client.bot = bot

    def get_discord_sink():
        vc_data = next((vc for vc in discord_bot_module.active_voice_clients.values() if vc['vc'].is_connected()), None)
        return vc_data['sink'] if vc_data else None

    asyncio.create_task(bot.process_ai_response_queue(get_discord_sink))
    await discord_bot_module.client.start(config.DISCORD_TOKEN)


async def async_main():
    """The main asynchronous entry point for the application."""
    logger.info("--- BOOT SEQUENCE ---")
    
    if len(sys.argv) < 6:
        logger.critical(f"FATAL: Invalid args. Expected 5, got {len(sys.argv) - 1}. Received: {sys.argv}"); return
    
    mode, audio_mode, tts_engine, llm_engine, vision_str = sys.argv[1:6]
    vision_on_startup = vision_str.lower() == 'true'
    
    logger.info(f"Args: mode={mode}, audio={audio_mode}, tts={tts_engine}, llm={llm_engine}, vision={vision_on_startup}")

    loop = asyncio.get_running_loop()
    bot = Bot(loop)
    
    loop.run_in_executor(None, listen_for_stdin_thread, bot, loop)

    try:
        await bot.setup(tts_engine, llm_engine, vision_enabled=vision_on_startup)
        logger.info("--- CORE INITIALIZATION COMPLETE ---")

        if mode == '1': await run_local_mode_async(bot, audio_mode)
        elif mode == '2':
            if not config.DISCORD_TOKEN:
                logger.critical("FATAL: DISCORD_TOKEN not found."); return
            await run_discord_mode_async(bot)
        else: logger.error(f"Invalid mode: '{mode}'. Exiting.")

    except (KeyboardInterrupt, EOFError): logger.info("\nProcess terminated by user.")
    except Exception: logger.critical(f"\n--- CRITICAL UNHANDLED ERROR ---:\n{traceback.format_exc()}")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(async_main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e) or "stopped before Future completed" in str(e):
            pass
        else:
            raise 