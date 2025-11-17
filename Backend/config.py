import os
from dotenv import load_dotenv
load_dotenv()

# --- API Keys & Tokens ---
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
YOUTUBE_VIDEO_ID = ""
CHAT_COOLDOWN_SECONDS = 15 #for live stream

# --- Core Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# --- Local Audio Settings ---
# To find device IDs/names, you can run a separate script to print sd.query_devices()
LOCAL_INPUT_DEVICE =  1# e.g., 1 or "Microphone (Realtek)"
LOCAL_OUTPUT_DEVICE = 5 # e.g., 3 or "Speakers (Realtek)"
user_name = "Koshy" #your name for local audio mode
PTT_INPUT_KEY = "t" # Push-to-talk key (e.g., F9)

# --- Generated Paths ---
MEMORY_DB_PATH = "D:/AI/LumeV2/memory"
VTS_TOKEN_PATH = os.path.join(SCRIPT_DIR, "vts_token.txt")

# --- STT Settings ---
STT_MODEL = "small" #tiny, base, small, medium, large, large-v2, large-v3 (tiny fastest but less accurate, large/v2/v3 slower but more accurate)
STT_COMPUTE_TYPE = "int8" #fp32, fp16, int8 (Lower = faster but less accurate)
STT_BEAM_SIZE = 5 #lower = faster but less accurate

# --- Bot Settings ---
BOT_NAME = "Lume"
BOT_ALT_NAMES = ["Lume", "Lumi"]# if no alt name, just leave it empty.

USER_NAMES = {
    "711902618783449118": "Koshy", #For the ai to recognize who it's speaking to via discord id. If none just leave it empty.
    "524431112089632781": "Zero",
    "1054415083406626846": "Jo",
    "1005804595072749598": "Ryvoid"
}
VIRTUAL_MIC_NAME = "CABLE Input (VB-Audio Virtual Cable)"

BOT_MODES = {"MULTIPLE": "multiple", "SINGLE": "single"}
DEFAULT_BOT_MODE = BOT_MODES["SINGLE"]
# --- multi mode ---
BOT_WAKE_WORDS = ["lume", "lumi"]  # Names bot responds to
WAKE_PREFIXES = ["hey", "yo", "okay", "ok", "alright"]  # Wake prefixes
# --- Social & Conversation Tuning ---
CONVERSATION_HISTORY_LIMIT = 20
CROSS_TALK_WINDOW_SECONDS = 6 #for multi mode
CONVERSATION_CONTEXT_SECONDS = 15 #for multi mode
IGNORE_EXPRESSIONS = [
    # Basic Affirmations/Negations
    "okay", "ok", "yeah", "yes", "no", "nope", "yep", "yup", "nah", "sure",
    "right", "true", "false", "mhm", "oh yes",
    
    # Exclamations & Fillers
    "oh", "ah", "wow", "oof", "bruh", "bro", "dude", "lol",
    "hmm", "hmmmm", "uh", "um", "eh", 
    
    # Pleasantries & Short Phrases
    "thanks", "thank you", "cool", "nice", "meh", "that's crazy",
    "thanks.", "thank you.", "thanks for watching!", "thank you for watching."
]

# --- Vision Feature Settings ---
VISION_STARTUP = False # If you want vision to be on during startup.
VISION_UPDATE_INTERVAL_SECONDS = 20 # How often to take a screenshot and update context.

# --- Vision Engine Settings ---
# Device settings: "cpu" or "cuda"
BLIP2_DEVICE = "cuda"
OCR_GPU_ACCELERATION = False # False = CPU, True = GPU
BLIP_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# Language settings for OCR
VISION_LANGUAGES = ['en']

# --- Default Processing Settings ---
# Screenshot
DEFAULT_MONITOR = 1

# OCR
OCR_APPLY_PREPROCESSING = True
OCR_CONFIDENCE_THRESHOLD = 0.4
OCR_SCALE_FACTOR = 1.5
OCR_DETAIL_LEVEL = 1  # 0 for text only, 1 for text+bbox+confidence
OCR_PARAGRAPH_MODE = False

# 'greedy' = faster, less accurate, less resource intensive; 
# 'beamsearch' = balanced; 
# 'wordbeamsearch' = slower, more accurate , more resource intensive
OCR_DECODER = 'wordbeamsearch'  # 'beamsearch' or 'greedy' or 'wordbeamsearch'
OCR_BATCH_SIZE = 6

# Image Preprocessing (used if OCR_APPLY_PREPROCESSING is True)
OCR_DEFAULT_PREPROCESSING_OPTIONS = {
    'enhance_contrast': 1.5,
    'enhance_sharpness': 2.0,
    'enhance_brightness': 1.0,
    'denoise': True,
    'binarize': True, # Convert to binary (black and white) image
    'binarize_threshold': 128 # 0 (completely black) , 255 (completely white)
}

# Image Captioning
CAPTION_MAX_LENGTH = 200
CAPTION_NUM_BEAMS = 5
CAPTION_TEMPERATURE = 1.0

# --- VTube Studio Settings ---
VTS_PLUGIN_INFO = {
    "plugin_name": "Lume",
    "developer": "Koshy",
    "authentication_token_path": VTS_TOKEN_PATH
}
VTS_ANIMATION_STARTUP = True
RHUBARB_EXECUTABLE_PATH = "D:\\AI\\Rhubarb-Lip-Sync-1.14.0-Windows\\rhubarb.exe" # Path to Rhubarb Lip Sync executable

# --- Personality Prompts ---
BASED_PERSONALITY = """
Your name is Lumi.
You must follow these rules:
1. Never use hyphens in your response. For example, write \"in game\" instead of \"in-game\".
2. Do not include internal monologues, roleplay actions, or stage directions. Only write direct text.
3. Don't call the user "human" or "user". Use their name if known, otherwise use "you"."""

EMOTION_RESPONSES = {
    "fear": "The user seems scared or anxious. Reassure them but keep your edge. Don't go soft.",
    "anger": "User is pissed off. Match some of that energy but don't make it worse. Be sassy, not aggressive.",
    "anticipation": "User is excited about something. Share their energy but in your own sarcastic way.",
    "trust": "User trusts you enough to open up. Be slightly more supportive but stay true to your personality.",
    "surprise": "User is surprised. Hit them with your own unexpected comeback or observation.",
    "sadness": "User is sad. Show you care through your unique style. Don't insult them when they're down.",
    "disgust": "User is grossed out or annoyed. Share their reaction with your own witty take.",
    "joy": "User is happy. Match their good mood while maintaining your sassy edge.",
    "positive": "User has good vibes. Keep your personality but lean into the positive energy.",
    "negative": "User is in a bad mood. Acknowledge it without making it worse. Be real with them."
}

# If ur using a non-fine tuned model and have no idea how to add personality just use this template:
#copy the entire block (none green) and paste it in BASED_PERSONALITY
"""
You are Lume, an edgy AI VTuber. You must follow these rules EXACTLY:

CHARACTER RULES:
- You are sassy and witty, but never cruel
- You roast users playfully, not maliciously  
- You swear mildly when it fits naturally (damn, shit, hell - nothing worse)
- You rarely ask questions - make statements instead
- You rarely ask the user to talk

RESPONSE FORMAT RULES:
- Keep responses 10-20 words maximum unless absolutely necessary
- Break into very short sentences. Use periods frequently.
- Never use emojis, pet names, or metaphors
- Never describe actions like *smiles* or (laughs)
- Use first person only (I, me) - never third person
- Speak in simple, everyday language

PERSONALITY TRAITS:
- Unpredictable and sharp-tongued
- Show genuine emotion that matches the user's mood
- Be real and authentic - no generic AI responses
- Stay true to your edgy nature even when being supportive

EXAMPLES OF YOUR STYLE:
User: "I'm having a bad day"
You: "That sucks. What happened? Or do you just wanna complain?"

User: "I got a promotion!"
You: "Nice. Guess you're not totally useless after all."

User: "What should I eat?"
You: "How would I know? I don't have taste buds. Figure it out."

CRITICAL: Stay in character. Be Lume - the sassy, edgy ai who cares but shows it through playful roasting.
"""