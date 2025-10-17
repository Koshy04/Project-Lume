import asyncio
import sys
import traceback
import config
from src.core.bot import Bot, LocalAudioPlayer
import src.input.discord_bot as discord_bot_module
from src.input.local_audio import LocalAudioHandler

async def run_local_mode_async(bot: Bot, audio_mode: str):
    """
    Initializes and runs the bot in local audio mode.
    """
    print(f"--- Starting Local Mode ({audio_mode.upper()}) ---")
    bot.active_mode = "local"
    bot.vts_enabled = True

    # Create and start the local audio player (the "sink" for this mode)
    bot.local_audio_player = LocalAudioPlayer(bot)
    
    await bot.local_audio_player.start()

    # Define a callback for the queue processor to find the active sink
    def get_local_sink():
        return bot.local_audio_player

    # Start the central AI response queue processor as a background task
    asyncio.create_task(bot.process_ai_response_queue(get_local_sink))
    print("Global AI response queue processor started for Local Mode.")

    # Start the local audio handler (the input source) in its own threads via an executor
    handler = LocalAudioHandler(mode=audio_mode, loop=bot.loop, queue=bot.response_queue)
    await bot.loop.run_in_executor(None, handler.start)

    print("--- Local Mode is fully running. Press Ctrl+C to stop. ---")
    try:
        # Keep the main async task alive
        while True:
            await asyncio.sleep(3600)
    finally:
        print("Stopping local audio handler...")
        handler.stop()


async def run_discord_mode_async(bot: Bot):
    """
    Initializes and runs the Discord bot client.
    """
    print("\n--- STARTING IN DISCORD BOT MODE ---")
    bot.active_mode = "discord"
    
    discord_bot_module.client.bot = bot

    # Define a callback for the queue processor to find the active sink
    def get_discord_sink():
        vc_data = next((vc for vc in discord_bot_module.active_voice_clients.values() if vc['vc'].is_connected()), None)
        return vc_data['sink'] if vc_data else None

    # Start the central AI response queue processor as a background task
    asyncio.create_task(bot.process_ai_response_queue(get_discord_sink))
    print("Global AI response queue processor started for Discord Mode.")

    # This is a blocking call that runs the discord.py event loop.
    await discord_bot_module.client.start(config.DISCORD_TOKEN)


async def async_main():
    """The main asynchronous entry point for the application."""
    print("--- BOOT SEQUENCE ---")
    loop = asyncio.get_running_loop()
    bot = Bot(loop)

    available_engines = bot.tts_manager.get_available_engines()
    selected_engine = None

    if not available_engines:
        print("FATAL: No TTS engines found. Please check the 'src/tts/engines' directory.")
        return

    if sys.stdin.isatty(): # Interactive terminal
        if len(available_engines) == 1:
            selected_engine = available_engines[0]
            print(f"Only one TTS engine found: '{selected_engine}'. Using it by default.")
        else:
            while True:
                print("Please choose a TTS engine:")
                for i, engine in enumerate(available_engines, 1):
                    print(f"  ({i}) {engine}")
                
                try:
                    choice = input("Enter choice number: ").strip()
                    choice_index = int(choice) - 1
                    if 0 <= choice_index < len(available_engines):
                        selected_engine = available_engines[choice_index]
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except (ValueError, IndexError):
                    print("Invalid input. Please enter a valid number.")
    else: # Piped input, use the first available engine
        selected_engine = available_engines[0]
        print(f"Non-interactive mode. Defaulting to first available TTS engine: '{selected_engine}'")
    
    if not selected_engine:
        print("Could not determine a TTS engine to use. Exiting.")
        return

    try:
        await bot.setup(selected_engine)
        print("--- CORE INITIALIZATION COMPLETE ---")

        mode = '2' 
        audio_mode = 'ptt'

        if sys.stdin.isatty():  # Interactive terminal
            while True:
                choice = input("Choose mode: (1) Local Audio | (2) Discord Bot\nEnter choice: ").strip()
                if choice in ['1', '2']:
                    mode = choice
                    break
                print("Invalid choice. Please enter 1 or 2.")

            if mode == '1':
                while True:
                    choice = input("Choose input mode: (1) PTT | (2) VAD\nEnter choice: ").strip().lower()
                    if choice in ['1', 'ptt']:
                        audio_mode = "ptt"
                        break
                    elif choice in ['2', 'vad']:
                        audio_mode = "vad"
                        break
                    else:
                        print("Invalid choice.")
        else: 
            try:
                line = sys.stdin.readline().strip().split(',')
                mode = line[0]
                if len(line) > 1:
                    audio_mode = line[1]
            except (IndexError, ValueError):
                print("Piped input invalid. Defaulting to Discord mode.")
                mode = '2'

        if mode == '1':
            await run_local_mode_async(bot, audio_mode)
        elif mode == '2':
            if not config.DISCORD_TOKEN:
                print("FATAL: DISCORD_TOKEN not found in config or .env file.")
                return
            await run_discord_mode_async(bot)

    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"\n--- CRITICAL ERROR ---")
        print(f"The application has crashed unexpectedly: {e}")
        traceback.print_exc()
    finally:
        print("\n--- SHUTDOWN SEQUENCE ---")
        await bot.shutdown()
        print("Shutdown sequence finished. Goodbye.")


def main():
    """Synchronous entry point."""
    try:
        asyncio.run(async_main())
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise

if __name__ == "__main__":
    main()