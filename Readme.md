<h1 align="center">Project Lume</h1>

<h4 align="center">A template on making ur OWN AI companion/Vtuber</h4>

<p align="center" >
  <a href="#about-this-project">About This Project</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#customizing ">Customizing </a> •
  <a href="#applications">Applications</a> •
  <a href="#how-to-use">How To Use</a> •
</p>

## About This Project
This project is a template that allows u to make ur very own ai companion/vtuber with tons of customization for streaming or personal use.


This project was built with customizable, low resource usage and running fully local in mind.

This software uses libraries from the FFmpeg project under the LGPLv2.1

## Key Features
- Realtime promptable AI personality with text and speech input
- Interchangeable TTS model and LLM inference 
- Useable with a webui or via the terminal
- Runs fully locally
- Long term and short term memory
- Vision that auto updates
- Supports reading YT chats (Twitch soon)
- A custom script that triggers Live2d model movement
- Useable via discord and local

## Installation
 **It's highly recommended to uses a virtual env** 
### 1) Install python 3.12 :
https://www.python.org/downloads/release/python-3100/
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
### 6) install dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
you need to then 
```bash
pip install jieba_fast
```
idk why u cant do it automatically in requirements.txt but this is to avoid errors
### 7) Download virtual cable:
https://vb-audio.com/Cable/
### 8) create a .env file 
copy the .env.example file and paste it in .env. Follow the .example file


## Customizing
### Configs
The main config is found on *Backend*, I've labaled most of the settings found on it

The other config is found on the TTS and LLM folder with it's own config file. *Note*: Config found on *Backend* controls the entire AI while config found on TTS or LLM only controls it's own file.

### Vtube Studio
This template only uses Live2D for now. It has a few build in movement script that can be played by the AI automatically but the default script may not work with ur model.

Head to Backend/services/movement/config/presets.json to add ur own movement. 

For hotkeys triggering and idle settings, head to each of its json file. Hotkeys is mapped to a presets so when a preset  is trigger only then the hotkeys will be triggered.

Lipsync uses rhubarb  for  mouth  movement so u  need to install  it ([rhubarb](https://github.com/DanielSWolf/rhubarb-lip-sync)), once installed, head to Backend/config and paste the .exe path. U can then customize the mouth movement in Backend/services/movement/config/visemes.json. If u have no idea what ur doing, just use the default. 

**Note**: u need to use virtual cable as input on Vtube Studio for lipsync

## Applications
This project supports interchangeable tts engine and llm inference engine.
Those "pugins" can be found on my profile ([Koshy04](https://github.com/Koshy04))

In short, clone the plugin project, placed it on its respected folder and install the dependecies.


## How To Use
This project is useable with a webui or via the terminal.

### Terminal
Find main.py in Backend/ and just run that

### Webui
Head to Frontend

```bash
cd Frontend
```
and install the application

```bash
npm install
```

and run the app

```bash
npm start
```