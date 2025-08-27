import requests
import json
import datetime
import re
from nrclex import NRCLex
import asyncio
from memory import memory_manager
import config

def analyze_emotions(text: str) -> dict:
    """Analyzes text for emotions and returns a dominant emotion and scores."""
    if not text or not text.strip():
        return {"dominant_emotion": "neutral", "emotions": {}}
    try:
        emotion_analyzer = NRCLex(text)
        emotions = emotion_analyzer.affect_frequencies
        core_emotions = {k: v for k, v in emotions.items() if k not in ['positive', 'negative'] and v > 0}
        if not core_emotions:
            dominant = 'positive' if emotions.get('positive', 0) > emotions.get('negative', 0) else ('negative' if emotions.get('negative', 0) > 0 else 'neutral')
        else:
            dominant = max(core_emotions, key=core_emotions.get)
        dominant = 'anticipation' if dominant == 'anticip' else dominant
        return {"dominant_emotion": dominant, "emotions": emotions}
    except Exception as e:
        print(f"Error analyzing emotions for text '{text[:30]}...': {e}")
        return {"dominant_emotion": "neutral", "emotions": {}}
    
#multi mode speech detection
def classify_speech_intent(transcription: str, bot_name: str = None) -> bool:
    """Uses "Target Analysis" to determine if speech is for the bot."""
    if bot_name is None:
        bot_name = getattr(config, 'BOT_NAME', 'Assistant') # Default to 'Assistant' if BOT_NAME not set in config

    # Get alternative names from config or use default variations
    alt_names = getattr(config, 'BOT_ALT_NAMES', [bot_name.lower(), bot_name])
    alt_names_str = "', '".join(set(alt_names))
    
    system_prompt = f"""You are a 'Conversation Target' analysis expert. Your job is to analyze the user's text and identify who the user is speaking to.
The AI bot's name is '{bot_name}' and may also be called '{alt_names_str}'.
You must respond with ONLY ONE of the following three words:
- '{bot_name}' if the user is addressing the bot.
- 'Other' if the user is addressing another person or group.
- 'General' if the user is making a general statement to no one in particular.

# --- EXAMPLES ---
User: "hey {bot_name.lower()} what's up"
System: {bot_name}
User: "yo {alt_names[0] if alt_names else bot_name.lower()} can you hear me"
System: {bot_name}
User: "what do you guys think?"
System: Other
User: "i think he went left"
System: Other
User: "that's crazy"
System: General
User: "I think {bot_name} is pretty cool."
System: General
User: "What about you, {bot_name}?"
System: {bot_name}
User: "Tell me something interesting, {bot_name}."
System: {bot_name}
User: "wow that's amazing"
System: General
User: "Her name is {bot_name}, right?"
System: Other
"""
    try:
        response = requests.post(
            f"{config.OLLAMA_API_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": f"System: {system_prompt}\n\nUser: \"{transcription}\"\nSystem:",
                "options": {
                    "temperature": 0.1, "top_p": 0.5, "top_k": 20,
                    "repeat_penalty": 1.2, "num_predict": 5,
                    "stop": ["\n", "."],
                },
                "stream": False
            },
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        generated_text = result.get("response", "").strip().lower()
        print(f"Target Analysis for '{transcription}': AI returned -> '{generated_text}'")
        
        # Check if any of the bot names appear in the response
        bot_names_to_check = [bot_name.lower()] + [name.lower() for name in alt_names]
        return any(name in generated_text for name in bot_names_to_check)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama for intent classification: {e}")
        return False
    except Exception as e:
        print(f"Error during intent classification: {e}")
        return False
    
#Main T2T
async def llm_inference(prompt: str, system_prompt: str, temperature: float = 0.8, stop_sequences: list = None) -> str:
    """Generic async function to call the Ollama API."""
    try:
        response = await asyncio.to_thread(
            requests.post,
            f"{config.OLLAMA_API_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "system": system_prompt,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9, 
                    "top_k": 40,
                    "repeat_penalty": 1.15,
                    "num_predict": 90, #max token to generate
                    "num_ctx": 16384, #max memory (context size)
                    "repeat_last_n": 32,     
                    "stop": stop_sequences or ["\nUser:", "\nHuman:", "\nSystem:"]
                },
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        return f'Sorry, something went wrong. (Connection error: {type(e).__name__})'
    except Exception as e:
        print(f"Error in llm_inference: {e}")
        return f'Sorry, something went wrong. (Error: {type(e).__name__})'

async def chat_with_ai(prompt: str, user_id: str, emotion_data: dict, conversation_history: str) -> str:
    """Generates a response from the AI using conversation history and vector memory."""
    user_name = config.USER_NAMES.get(str(user_id), f"User({user_id})")
    dominant_emotion = emotion_data.get("dominant_emotion", "neutral")

    # Get semantically relevant memories for this user and prompt
    relevant_memory_context = memory_manager.search_memories(prompt, user_id)

    # Build the system prompt with all context
    system_instruction = (
        f"{getattr(config, 'BASED_PERSONALITY', 'You are a helpful AI assistant.')}\n\n" #Try to use BASED_PERSONALITY but if fail, will use 'You are a helpful AI assistant.'
        f"RECENT CONVERSATION:\n{conversation_history}\n\n"
        f"{relevant_memory_context}"
    )

    if dominant_emotion in getattr(config, 'EMOTION_RESPONSES', {}):
        system_instruction += f"\n\nEMOTION CONTEXT: {config.EMOTION_RESPONSES[dominant_emotion]}"

    # Single temperature based on emotion
    temp = 0.8
    if dominant_emotion == "joy": temp = 0.9
    elif dominant_emotion == "sadness": temp = 0.7
    elif dominant_emotion == "surprise": temp = 1.0
    elif dominant_emotion == "anger": temp = 0.75

    #print(system_instruction)     #Remove # infront of this print to see what prompts + memories are given to the ai 

    generated_text = await llm_inference(
        prompt=prompt,
        system_prompt=system_instruction,
        temperature=temp,
        stop_sequences=["\nUser:", "\nHuman:", f"\n{user_name}:", f"\n{user_name.lower()}:"]
    )
   
    if "I'm having trouble responding right now." in generated_text: 
        return generated_text

    # Save the raw turn to the vector memory
    if prompt.strip() and generated_text:
        memory_manager.add_raw_turn(user_name, prompt, generated_text, str(user_id))

    return generated_text or "I don't know what to say right now."
