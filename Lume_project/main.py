import config
import memory
import asyncio
import traceback
import subprocess
import summarizer
from vts_interface import close_vts_connection
from tts_interface import initialize_tts_models
from transcription import initialize_transcription_model
from discord_bot import client, active_voice_clients, api_process, twitch_bot

async def full_cleanup():
    """A function to gracefully shut down all active connections."""
    print("Initiating full cleanup...")

    if api_process and api_process.poll() is None:
        print("Shutting down the Lume API server...")
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
            print("API server process terminated.")
        except subprocess.TimeoutExpired:
            print("API server did not terminate in time, killing.")
            api_process.kill()
    if twitch_bot and twitch_bot.is_running:
        print("Shutting down Twitch bot...")
        await twitch_bot.stop()
        print("Twitch bot disconnected.")


    cleanup_tasks = []

    for guild_id, vc_data in list(active_voice_clients.items()):
        print(f"Cleaning up voice client for guild {guild_id}...")
        if vc_data.get('sink'):
            cleanup_tasks.append(vc_data['sink'].cleanup())
        if vc_data.get('vc') and vc_data['vc'].is_connected():
            cleanup_tasks.append(vc_data['vc'].disconnect(force=True))

    cleanup_tasks.append(close_vts_connection())

    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    print("Full cleanup complete.")

def main():
    """Main entry point for the bot."""
    if not config.DISCORD_TOKEN:
        print("FATAL: DISCORD_TOKEN not found in config. Please set it in your .env file.")
        return

    print("--- BOOT SEQUENCE ---")

    if not initialize_transcription_model():
        print("Could not initialize transcription model. Performance may be affected.")

    if not initialize_tts_models():
        print("Could not initialize TTS models. Voice output will be disabled.")

    print("--- INITIALIZATION COMPLETE ---")

    try:
        client.run(config.DISCORD_TOKEN)
    except Exception as e:
        print(f"CRITICAL ERROR: The bot has crashed. Exception: {e}")
        traceback.print_exc()
    finally:
        print("\n--- SHUTDOWN SEQUENCE ---")
        loop = asyncio.get_event_loop()
        try:
            if loop.is_running():
                cleanup_task = loop.create_task(full_cleanup())
                loop.run_until_complete(cleanup_task)
            else:
                loop.run_until_complete(full_cleanup())
        except Exception as e:
            print(f"Error during shutdown cleanup: {e}")
        finally:
            print("shutdown sequence finished.")

if __name__ == "__main__":
    main()