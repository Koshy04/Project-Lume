# Project Lume

This project is an AI Vtuber/Companion that I originally made for my own personal usage but now I plan to make it open source and as user friendly as possible. Do be patient as most of my codes are refactor from google colabs. Beware of messy code.

This software uses libraries from the FFmpeg project under the LGPLv2.1

# *This project does not come with any vtuber models, ai models and ai voice models. You'll have to get those yourself. This project only gives the template to making an ai companion/vtuber.*
# Features

## Core (Main features)
Notes: This is all working around discord.
1) STT with faster whisper
2) T2T with ollama
3) TTS with GPT-SoVITS for easy voice customization 

## Side (addon feature to core)
1) Vtube studio auto toggle and lipsync
2) Vaedotube lipsync (pngtuber)
3) Emotion detection. AI will respond accordingly to how the user feel.
4) long and short term memory
5) ignored expressions for simple fliter
6) base plugins to make the ai be able to interact with you while you play games
7) user detection with discord id
## Experimental (working in progress/ early testing)
1) Vision
2) an api (currently have POST /chat and GET /status only)
3) Multi mode to have more then 2 ppl in the same call. 
4) Yt and twitch chat reading


# System Requirements
### It is recommanded to have a **NVIDIA** GPU with at least 12 gb of vram

# Install from scratch
notes: It's highly recommanded to use venv

## Installing dependencies
### 1) Install python 3.12 :
https://www.python.org/downloads/release/python-3120/
### 2) Install ffmpeg and add it into ur windows path :
https://ffmpeg.org/index.html?pubDate=20250822
### 3) Install cuda 12.8 :
https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64
### 4) Install pytorch with cuda 12.8 in the terminal
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
### 5) Clone the repo
```bash
git clone https://github.com/Koshy04/Project-Lume.git
```
### 6) head on to Lume project dic
```bash
cd Lume-project
```
### 7) install dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
you need to then 
```bash
pip install jieba_fast
```
idk why u cant do it automatically in requirements.txt but this is to avoid errors
### 8) Head to https://huggingface.co/koshy04/Project_Lume/tree/main
Download : Every content in pretrained_models and place it in pretrained_models in GPT-SoVITS
### 9) Download virtual cable:
https://vb-audio.com/Cable/
### 10) create a .env file 
copy the .env.example file and paste it in .env. Next add your discord bot token.

# Customizing
After you have done installing and setting up the files. You need to add GPT-SoVITS path in config.py before you can use it. (labled under core path)
## voice 
You'll need a clean 3-10 seconds clean single speaker voice clip and reference text of what the voice clip.
Add it the voice clip path to REF_AUDIO_PATH in config.py and reference text in config.py too.

**Note**: Emotions and accent of the ai voice is based on the voice clips provided.
## Personality
You'll need to install ollama from :
https://ollama.com/download

once downloded you'll need to download a model from ollama website:
https://ollama.com/search

follow the instruction in ollama on how to install a model.

After that, in config.py add your model's name that you just downloded from ollama in OLLAMA_MODEL.

Next, head down to BASED_PERSONALITY and describe how you want your ai to act (or just follow the template). You can do the same for EMOTION_RESPONSES.

# Config file
### Everything in the config file can be changed to your liking.

*Note*: This is my first time making a repo readme. Excuse the terrible explaination.

# Usage
Head to the main.py in Lume_project and run it. Once the bot is running. Head to discord and type !help.


# Personal Notes
If you have any problems or want to suggest something, join the discord server. I'll mostly be there.

Discord :  https://discord.gg/HYHe8tCSrZ

Yt : https://www.youtube.com/@Koshy-04


# Resources
TTS = https://github.com/RVC-Boss/GPT-SoVITS

STT = https://github.com/SYSTRAN/faster-whisper

