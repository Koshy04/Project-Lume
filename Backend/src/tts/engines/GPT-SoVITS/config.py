import os

BASE_PATH = os.path.abspath(os.path.dirname(__file__))

# --- GPT-SoVITS Model Paths (now relative to this engine's folder) ---
GPT_MODEL_PATH = os.path.join(BASE_PATH, "pretrained_models/s1v3.ckpt")
SOVITS_MODEL_PATH = os.path.join(BASE_PATH, "pretrained_models/s2Gv3.pth")
REF_AUDIO_PATH = os.path.join(BASE_PATH, "voice/Lume2.wav") #path to reference audio

# --- GPT-SoVITS Reference Settings ---
REF_TEXT_CONTENT = "I truly appreciate the efforts put by animators in making this masterpiece." #what your reference audio says
REF_LANG = "英文" #En= "英文", Cn= "中文", Chinese-English mixed= "中英混合", Japanese-English mixed="日英混合" , Multilingual mixed= "多语种混合"
TARGET_LANG = "英文" #En= "英文", Cn= "中文", Chinese-English mixed= "中英混合", Japanese-English mixed="日英混合" , Multilingual mixed= "多语种混合"

# --- GPT-SoVITS Generation Settings ---
top_p = 0.5 
top_k = 50
temperature = 0.5
sample_steps = 4 
speed = 1.0
how_to_cut = "按标点符号切" 