import sys
import os
import re
import time
import queue
import threading
import subprocess
import gradio as gr
import json
import traceback
import ast # <-- IMPORT THE CORRECT TOOL

# --- Global State ---
bot_process = None
log_queue = queue.Queue()
log_reader_thread = None

def enqueue_output(pipe, q):
    """Reads lines from a subprocess pipe and puts them in a queue."""
    try:
        for line in iter(pipe.readline, ''):
            q.put(line)
    finally:
        pipe.close()

# --- Bot Control Functions ---
def start_bot(mode_choice, audio_mode_choice):
    """Starts the main.py script as a subprocess and pipes its output."""
    global bot_process, log_reader_thread

    if bot_process and bot_process.poll() is None:
        return "Bot is already running.", get_status()

    # Clear the log queue from any previous runs
    while not log_queue.empty():
        try: log_queue.get_nowait()
        except queue.Empty: continue

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_path = os.path.join(script_dir, "main.py")

    if not os.path.exists(main_script_path):
        return f"Error: main.py not found at {main_script_path}", get_status()

    if "Discord" in mode_choice:
        mode_input = "2\n"
    else: # Local Audio Mode
        audio_mode = "ptt" if "Push-to-Talk" in audio_mode_choice else "vad"
        mode_input = f"1,{audio_mode}\n"
    
    try:
        command = [sys.executable, "-u", main_script_path] # -u for unbuffered output

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=script_dir,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            encoding='utf-8'
        )
        
        log_reader_thread = threading.Thread(
            target=enqueue_output, 
            args=(process.stdout, log_queue),
            daemon=True
        )
        log_reader_thread.start()
        
        process.stdin.write(mode_input)
        process.stdin.flush()
        
        bot_process = process
        time.sleep(3) 
        return f"Bot started in {mode_choice} mode. PID: {bot_process.pid}", get_status()
    except Exception as e:
        return f"Failed to start bot: {e}", get_status()

def stop_bot():
    """Stops the bot subprocess."""
    global bot_process
    if bot_process and bot_process.poll() is None:
        try:
            bot_process.terminate()
            bot_process.wait(timeout=5)
            status_message = "Bot stopped successfully."
        except subprocess.TimeoutExpired:
            bot_process.kill()
            status_message = "Bot did not respond, it has been forcibly stopped."
        except Exception as e:
            status_message = f"Error stopping bot: {e}"
        finally:
            bot_process = None
        return status_message, get_status()
    else:
        return "Bot is not currently running.", get_status()

def get_status():
    """Checks and returns the current status of the bot process."""
    if bot_process and bot_process.poll() is None:
        return f"🟢 Running (PID: {bot_process.pid})"
    return "🔴 Stopped"

def stream_logs_realtime():
    """A generator that yields new log lines in the format Gradio expects."""
    log_history = []
    yield log_history

    while True:
        try:
            line = log_queue.get_nowait()
            if line:
                log_history.append({"role": "assistant", "content": line.strip()})
        except queue.Empty:
            time.sleep(0.1)
        
        yield log_history

# --- Settings Management Functions (Now More Robust) ---
def get_config_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")

# --- SAFE PARSING HELPERS ---
def _safe_search(pattern, content, default=""):
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else default

def _safe_parse_literal(pattern, content, default_value):
    """Safely parses a Python literal from a string, handling comments."""
    match_str = _safe_search(pattern, content)
    if match_str:
        try:
            return ast.literal_eval(match_str)
        except (ValueError, SyntaxError):
            return default_value
    return default_value

def _format_list_for_save(input_str):
    """Converts a comma-separated string to a Python list string."""
    items = [f'"{item.strip()}"' for item in input_str.split(',') if item.strip()]
    return f"[{', '.join(items)}]"

def _format_dict_for_save(input_str):
    """Converts a multi-line 'key: value' string to a Python dict string."""
    items = {}
    for line in input_str.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            items[key.strip().replace('"', '')] = val.strip()
    return json.dumps(items, indent=4)

def load_settings():
    """Loads all settings from config.py into the UI components."""
    try:
        with open(get_config_path(), "r", encoding="utf-8") as f:
            content = f.read()

        settings = {
            "YOUTUBE_VIDEO_ID": _safe_search(r"YOUTUBE_VIDEO_ID\s*=\s*['\"](.*?)['\"]", content),
            "OLLAMA_API_URL": _safe_search(r"OLLAMA_API_URL\s*=\s*['\"](.*?)['\"]", content, "http://localhost:11434"),
            "OLLAMA_MODEL": _safe_search(r"OLLAMA_MODEL\s*=\s*['\"](.*?)['\"]", content),
            "BOT_NAME": _safe_search(r"BOT_NAME\s*=\s*['\"](.*?)['\"]", content, "Lume"),
            "VIRTUAL_MIC_NAME": _safe_search(r"VIRTUAL_MIC_NAME\s*=\s*['\"](.*?)['\"]", content),
            "GPT_SOVITS_BASE_PATH": _safe_search(r"GPT_SOVITS_BASE_PATH\s*=\s*['\"](.*?)['\"]", content),
            "REF_TEXT_CONTENT": _safe_search(r"REF_TEXT_CONTENT\s*=\s*['\"](.*?)['\"]", content),
            "REF_LANG": _safe_search(r"REF_LANG\s*=\s*['\"](.*?)['\"]", content, "英文"),
            "TARGET_LANG": _safe_search(r"TARGET_LANG\s*=\s*['\"](.*?)['\"]", content, "英文"),
            "top_p": float(_safe_search(r"top_p\s*=\s*([\d\.]+)", content, "0.5")),
            "top_k": int(float(_safe_search(r"top_k\s*=\s*([\d\.]+)", content, "50"))),
            "temperature": float(_safe_search(r"temperature\s*=\s*([\d\.]+)", content, "0.5")),
            "STT_MODEL": _safe_search(r"STT_MODEL\s*=\s*['\"](.*?)['\"]", content, "small"),
            "STT_COMPUTE_TYPE": _safe_search(r"STT_COMPUTE_TYPE\s*=\s*['\"](.*?)['\"]", content, "int8"),
            "STT_BEAM_SIZE": int(float(_safe_search(r"STT_BEAM_SIZE\s*=\s*([\d\.]+)", content, "5"))),
            "DEFAULT_BOT_MODE": _safe_search(r'DEFAULT_BOT_MODE\s*=\s*BOT_MODES\["(.*?)"\]', content, "SINGLE"),
            "CONVERSATION_HISTORY_LIMIT": int(float(_safe_search(r"CONVERSATION_HISTORY_LIMIT\s*=\s*([\d\.]+)", content, "20"))),
            "BASED_PERSONALITY": _safe_search(r'BASED_PERSONALITY\s*=\s*"""(.*?)"""', content),
            # --- FIX: Use ast.literal_eval for robust parsing ---
            "BOT_WAKE_WORDS": ", ".join(_safe_parse_literal(r"BOT_WAKE_WORDS\s*=\s*(\[.*?\])", content, [])),
            "WAKE_PREFIXES": ", ".join(_safe_parse_literal(r"WAKE_PREFIXES\s*=\s*(\[.*?\])", content, [])),
            "IGNORE_EXPRESSIONS": "\n".join(_safe_parse_literal(r"IGNORE_EXPRESSIONS\s*=\s*(\[.*?\])", content, [])),
            "USER_NAMES": "\n".join([f"{k}: {v}" for k, v in _safe_parse_literal(r"USER_NAMES\s*=\s*(\{.*?\})", content, {}).items()]),
            "VISION_CONFIDENCE_THRESHOLD": int(float(_safe_search(r"VISION_CONFIDENCE_THRESHOLD\s*=\s*([\d\.]+)", content, "70"))),
            "VISION_TRIGGER_PHRASES": "\n".join(_safe_parse_literal(r"VISION_TRIGGER_PHRASES\s*=\s*(\[.*?\])", content, []))
        }
        return tuple(settings.values())

    except Exception:
        print(f"--- ERROR LOADING SETTINGS ---")
        traceback.print_exc()
        return ("",) * len(settings_components)

def save_settings_and_restart(*args):
    """Saves all settings to config.py and then restarts the bot if it's running."""
    keys = [
        "YOUTUBE_VIDEO_ID", "OLLAMA_API_URL", "OLLAMA_MODEL", "BOT_NAME", "VIRTUAL_MIC_NAME",
        "GPT_SOVITS_BASE_PATH", "REF_TEXT_CONTENT", "REF_LANG", "TARGET_LANG", "top_p", "top_k", "temperature",
        "STT_MODEL", "STT_COMPUTE_TYPE", "STT_BEAM_SIZE", "DEFAULT_BOT_MODE", "CONVERSATION_HISTORY_LIMIT",
        "BASED_PERSONALITY", "BOT_WAKE_WORDS", "WAKE_PREFIXES", "IGNORE_EXPRESSIONS", "USER_NAMES",
        "VISION_CONFIDENCE_THRESHOLD", "VISION_TRIGGER_PHRASES", "mode_choice", "audio_mode_choice"
    ]
    settings = dict(zip(keys, args))

    try:
        config_path = get_config_path()
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        def replace_line(key, value, is_string=True):
            nonlocal content
            if is_string:
                content = re.sub(fr"^{key}\s*=\s*.*$", f'{key} = "{value}"', content, flags=re.MULTILINE)
            else:
                 content = re.sub(fr"^{key}\s*=\s*.*$", f'{key} = {value}', content, flags=re.MULTILINE)

        def replace_block(key, value):
            nonlocal content
            escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
            content = re.sub(fr'{key}\s*=\s*"""(.*?)"""', fr'{key} = """{escaped_value}"""', content, flags=re.DOTALL)

        def replace_list(key, value_str):
            nonlocal content
            content = re.sub(fr"^{key}\s*=\s*.*$", f"{key} = {_format_list_for_save(value_str)}", content, flags=re.MULTILINE)

        def replace_list_multiline(key, value_str):
            nonlocal content
            items = [f'    "{line.strip()}"' for line in value_str.split('\n') if line.strip()]
            formatted_list = "[\n" + ",\n".join(items) + "\n]"
            content = re.sub(fr"{key}\s*=\s*\[.*?\]", f"{key} = {formatted_list}", content, flags=re.DOTALL)

        def replace_dict(key, value_str):
            nonlocal content
            content = re.sub(fr"{key}\s*=\s*{{.*?}}", f"{key} = {_format_dict_for_save(value_str)}", content, flags=re.DOTALL)

        # Save each setting
        replace_line("YOUTUBE_VIDEO_ID", settings["YOUTUBE_VIDEO_ID"])
        replace_line("OLLAMA_API_URL", settings["OLLAMA_API_URL"])
        replace_line("OLLAMA_MODEL", settings["OLLAMA_MODEL"])
        replace_line("BOT_NAME", settings["BOT_NAME"])
        replace_line("VIRTUAL_MIC_NAME", settings["VIRTUAL_MIC_NAME"])
        replace_line("GPT_SOVITS_BASE_PATH", settings["GPT_SOVITS_BASE_PATH"])
        replace_line("REF_TEXT_CONTENT", settings["REF_TEXT_CONTENT"])
        replace_line("REF_LANG", settings["REF_LANG"])
        replace_line("TARGET_LANG", settings["TARGET_LANG"])
        replace_line("STT_MODEL", settings["STT_MODEL"])
        replace_line("STT_COMPUTE_TYPE", settings["STT_COMPUTE_TYPE"])
        replace_line("top_p", settings["top_p"], is_string=False)
        replace_line("top_k", int(settings["top_k"]), is_string=False)
        replace_line("temperature", settings["temperature"], is_string=False)
        replace_line("STT_BEAM_SIZE", int(settings["STT_BEAM_SIZE"]), is_string=False)
        content = re.sub(r"^DEFAULT_BOT_MODE\s*=\s*.*$", f'DEFAULT_BOT_MODE = BOT_MODES["{settings["DEFAULT_BOT_MODE"]}"]', content, flags=re.MULTILINE)
        replace_line("CONVERSATION_HISTORY_LIMIT", int(settings["CONVERSATION_HISTORY_LIMIT"]), is_string=False)
        replace_block("BASED_PERSONALITY", settings["BASED_PERSONALITY"])
        replace_list("BOT_WAKE_WORDS", settings["BOT_WAKE_WORDS"])
        replace_list("WAKE_PREFIXES", settings["WAKE_PREFIXES"])
        replace_list_multiline("IGNORE_EXPRESSIONS", settings["IGNORE_EXPRESSIONS"])
        replace_dict("USER_NAMES", settings["USER_NAMES"])
        replace_line("VISION_CONFIDENCE_THRESHOLD", int(settings["VISION_CONFIDENCE_THRESHOLD"]), is_string=False)
        replace_list_multiline("VISION_TRIGGER_PHRASES", settings["VISION_TRIGGER_PHRASES"])
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        save_message = "Settings saved successfully!"

    except Exception:
        traceback.print_exc()
        return f"Error saving settings: {traceback.format_exc()}", get_status(), f"Error saving settings"

    # --- Restart the bot ---
    action_log_message = save_message
    current_status = get_status()
    if "Running" in current_status:
        action_log_message += "\nRestarting bot to apply changes..."
        stop_message, _ = stop_bot()
        time.sleep(2)
        start_message, new_status = start_bot(settings["mode_choice"], settings["audio_mode_choice"])
        action_log_message += f"\n{stop_message}\n{start_message}"
        current_status = new_status
    else:
        action_log_message += "\nBot is not running. Start it to apply new settings."

    return "Settings saved. Bot may be restarting.", current_status, action_log_message

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="Lume Control Panel") as demo:
    gr.Markdown("# 🤖 Lume Control Panel")
    
    with gr.Tabs():
        with gr.TabItem("Control & Logs"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Control")
                    status_output = gr.Textbox(value=get_status, label="Current Status", interactive=False, every=5)
                    mode_dropdown = gr.Dropdown(["Local Audio Mode", "Discord Bot Mode"], label="Select Mode", value="Discord Bot Mode")
                    audio_mode_dropdown = gr.Dropdown(["Push-to-Talk", "Voice Activity Detection (VAD)"], label="Audio Input Mode", value="Push-to-Talk", info="Only for Local Audio Mode.")
                    start_button = gr.Button("🚀 Start Bot", variant="primary")
                    stop_button = gr.Button("🛑 Stop Bot", variant="stop")
                    control_output = gr.Textbox(label="Action Log", interactive=False, lines=4)
                
                with gr.Column(scale=3):
                    gr.Markdown("### 📜 Real-Time Logs")
                    log_chatbot = gr.Chatbot(label="Live Log Output", height=500, show_label=False, type="messages")
                    demo.load(stream_logs_realtime, None, log_chatbot)

        with gr.TabItem("Settings"):
            gr.Markdown("⚠️ **Important:** API tokens (like Discord Token) are managed in your `.env` file and will not appear here. When you save, the bot will **automatically restart** if it is running.")
            
            with gr.Tabs():
                with gr.TabItem("General & AI"):
                    gr.Markdown("### Core Bot Identity")
                    s_youtube_video_id = gr.Textbox(label="YouTube Live Video ID", info="For connecting to a live chat.")
                    with gr.Row():
                        s_ollama_api_url = gr.Textbox(label="Ollama API URL", value="http://localhost:11434")
                        s_ollama_model = gr.Textbox(label="Ollama Model Name", info="The name of the model Ollama should use.")
                    with gr.Row():
                        s_bot_name = gr.Textbox(label="Bot Name")
                        s_virtual_mic = gr.Textbox(label="Virtual Mic Name for Lipsync", info="e.g., CABLE Input (VB-Audio Virtual Cable)")
                
                with gr.TabItem("TTS & STT"):
                    gr.Markdown("### Speech-to-Text (STT) & Text-to-Speech (TTS) Configuration")
                    with gr.Accordion("GPT-SoVITS TTS Settings", open=False):
                        s_gpt_sovits_path = gr.Textbox(label="GPT-SoVITS Base Path", info="Absolute path to your GPT-SoVITS installation.")
                        s_ref_text = gr.Textbox(label="Reference Audio Text", lines=2, info="The exact text spoken in your reference audio file.")
                        with gr.Row():
                            s_ref_lang = gr.Dropdown(["英文", "中文", "中英混合", "日英混合", "多语种混合"], label="Reference Language")
                            s_target_lang = gr.Dropdown(["英文", "中文", "中英混合", "日英混合", "多语种混合"], label="Target Language")
                        with gr.Row():
                            s_top_p = gr.Slider(0, 1, value=0.5, step=0.05, label="Top P")
                            s_top_k = gr.Slider(1, 100, step=1, value=50, label="Top K")
                            s_temp = gr.Slider(0, 2, value=0.5, step=0.05, label="Temperature")
                    with gr.Accordion("Faster-Whisper STT Settings", open=True):
                        with gr.Row():
                            s_stt_model = gr.Dropdown(["tiny", "base", "small", "medium", "large-v2", "large-v3"], label="STT Model")
                            s_stt_compute = gr.Dropdown(["int8", "fp16", "fp32"], label="STT Compute Type", info="int8 is fastest.")
                            s_stt_beam = gr.Slider(1, 10, step=1, label="STT Beam Size", info="Higher is more accurate but slower.")
                
                with gr.TabItem("Social & Personality"):
                     gr.Markdown("### Conversation Logic & Bot Character")
                     with gr.Row():
                        s_bot_mode = gr.Dropdown(["SINGLE", "MULTIPLE"], label="Default Bot Mode")
                        s_history = gr.Slider(1, 100, step=1, label="Conversation History Limit")
                     s_personality = gr.Textbox(label="Based Personality Prompt", lines=10, info="The core system prompt defining the bot's personality.")
                     with gr.Row():
                        s_wake_words = gr.Textbox(label="Bot Wake Words (comma-separated)", info="For Multiple mode.")
                        s_wake_prefixes = gr.Textbox(label="Wake Prefixes (comma-separated)", info="e.g., hey, yo, ok")
                     with gr.Row():
                        s_ignore_phrases = gr.Textbox(label="Ignore Expressions (one per line)", lines=5)
                        s_user_names = gr.Textbox(label="User ID to Name Mapping (ID: Name, one per line)", lines=5, info="Maps Discord IDs to names.")

                with gr.TabItem("Vision"):
                    gr.Markdown("### Screen Reading & Vision Capabilities")
                    s_vision_confidence = gr.Slider(0, 100, step=1, label="Vision Confidence Threshold", info="Minimum confidence for detecting objects/text.")
                    s_vision_triggers = gr.Textbox(label="Vision Trigger Phrases (one per line)", lines=7)
            
            with gr.Row():
                load_settings_button = gr.Button("🔄 Load All Settings from config.py")
                save_settings_button = gr.Button("💾 Save All Settings & Restart Bot", variant="primary")
            settings_feedback = gr.Textbox(label="Status", interactive=False)

    # --- Component Lists ---
    settings_components = [
        s_youtube_video_id, s_ollama_api_url, s_ollama_model, s_bot_name, s_virtual_mic,
        s_gpt_sovits_path, s_ref_text, s_ref_lang, s_target_lang, s_top_p, s_top_k, s_temp,
        s_stt_model, s_stt_compute, s_stt_beam, s_bot_mode, s_history, s_personality,
        s_wake_words, s_wake_prefixes, s_ignore_phrases, s_user_names, s_vision_confidence, s_vision_triggers
    ]

    # --- Event Listeners ---
    start_button.click(start_bot, inputs=[mode_dropdown, audio_mode_dropdown], outputs=[control_output, status_output])
    stop_button.click(stop_bot, inputs=None, outputs=[control_output, status_output])
    
    load_settings_button.click(load_settings, inputs=None, outputs=settings_components)
    
    save_settings_button.click(
        save_settings_and_restart, 
        inputs=settings_components + [mode_dropdown, audio_mode_dropdown], 
        outputs=[settings_feedback, status_output, control_output]
    )

if __name__ == "__main__":
    demo.launch()