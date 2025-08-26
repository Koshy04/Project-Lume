import os
from dotenv import load_dotenv
load_dotenv()

# --- API Keys & Tokens ---
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
TWITCH_NICK = ""  #leave this "" if u aint using it
TWITCH_CHANNEL = "" #leave this "" if u aint using it
TWITCH_APP_ID = os.getenv('TWITCH_APP_ID') # Set your Twitch app ID
TWITCH_APP_SECRET = os.getenv('TWITCH_APP_SECRET') # Set your Twitch app secret
YOUTUBE_VIDEO_ID = "" #empty if not using
CHAT_COOLDOWN_SECONDS = 15 #for live stream

# --- Ollama Settings ---
OLLAMA_API_URL = 'http://localhost:11434' #ollama default port
OLLAMA_MODEL = 'llama3.2:3b' #your llm model

# --- Core Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #path to Lume_project. Will be done automatically
# path to GPT-SoVITS
GPT_SOVITS_BASE_PATH = "" #path to the gpt sovits file

# --- Generated Paths ---
MEMORY_DB_PATH = os.path.join(SCRIPT_DIR, "memory")
VTS_TOKEN_PATH = os.path.join(SCRIPT_DIR, "vts_token.txt")

# --- GPT-SoVITS Model Paths ---
GPT_MODEL_PATH = os.path.join(GPT_SOVITS_BASE_PATH, "pretrained_models/s1v3.ckpt")
SOVITS_MODEL_PATH = os.path.join(GPT_SOVITS_BASE_PATH, "pretrained_models/s2Gv3.pth")
REF_AUDIO_PATH = "" #path to reference audio
REF_TEXT_CONTENT = "I truly appreciate the efforts put by animators in making this masterpiece." #what your reference audio says
REF_LANG = "英文" #En= "英文", Cn= "中文", Chinese-English mixed= "中英混合", Japanese-English mixed="日英混合" , Multilingual mixed= "多语种混合"
TARGET_LANG = "英文" #En= "英文", Cn= "中文", Chinese-English mixed= "中英混合", Japanese-English mixed="日英混合" , Multilingual mixed= "多语种混合"
# --- GPT-SoVITS TTS Settings ---
top_p = 0.5 
top_k = 50
temperature = 0.5
sample_steps = 4 
speed = 1.0
how_to_cut = "按标点符号切" 

# --- STT Settings ---
STT_MODEL = "small" #tiny, base, small, medium, large, large-v2, large-v3 (tiny fastest but less accurate, large/v2/v3 slower but more accurate)
STT_COMPUTE_TYPE = "int8" #fp32, fp16, int8 (Lower = faster but less accurate)
STT_BEAM_SIZE = 5 #lower = faster but less accurate

# --- Bot Settings ---
BOT_NAME = "Lume"
BOT_ALT_NAMES = ["Lume", "Lumi"]# if no alt name, just leave it empty.

USER_NAMES = {#For the ai to recognize who it's speaking to via discord id. If none just leave it empty.
    "12345678900923": "Koshy", #example
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
VISION_ACTION_WORDS = ["see", "look", "watch", "describe", "tell me about", "what is", "what's"]
VISION_TARGET_WORDS = ["screen", "this", "that", "it", "image", "picture"]
VISION_CONFIDENCE_THRESHOLD = 85

# --- VTube Studio Settings ---
VTS_PLUGIN_INFO = {
    "plugin_name": "Lume",
    "developer": "Koshy",
    "authentication_token_path": VTS_TOKEN_PATH
}
EMOTION_TO_VTS_ANIMATION = {
    # "joy": "JoyfulAnimationHotkey",
    # "anger": "AngryAnimationHotkey"
}

# --- Personality Prompts ---
BASED_PERSONALITY = """ 
Do not use - in your response.
Do not use internal monologues like *tosses poker card* in your responses.
"""

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
