import httpx
import re
from nrclex import NRCLex
from fuzzywuzzy import fuzz, process
import config
from src.services.memory.memory import memory_manager

class AICore:
    """
    Manages all AI-related tasks including language model inference,
    emotion analysis, and intent classification using a non-blocking HTTP client.
    """
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)

        wake_words_pattern = '|'.join(re.escape(name.lower()) for name in config.BOT_WAKE_WORDS)
        wake_prefixes_pattern = '|'.join(re.escape(prefix.lower()) for prefix in config.WAKE_PREFIXES)
        
        self.wake_word_patterns = {
            "prefix": re.compile(rf'^({wake_prefixes_pattern})\s+({wake_words_pattern})\b'),
            "start": re.compile(rf'^({wake_words_pattern})[\s,]'),
            "end": re.compile(rf'[\s,]+({wake_words_pattern})\??\s*$')
        }
        print("AI Core initialized with Ollama model:", config.OLLAMA_MODEL)

    # --- Public Methods ---

    async def chat_with_ai(self, prompt: str, user_id: str, emotion_data: dict, conversation_history: str) -> str:
        """
        Generates a contextual AI response using conversation history and vector memory.
        This is the primary method for getting a text response from the bot.
        """
        user_name = config.USER_NAMES.get(str(user_id), f"User({user_id})")
        dominant_emotion = emotion_data.get("dominant_emotion", "neutral")

        # Get semantically relevant memories for this user and prompt
        relevant_memory_context = memory_manager.search_memories(prompt, user_id)

        system_instruction = (
            f"{getattr(config, 'BASED_PERSONALITY', 'You are a helpful AI assistant.')}\n\n"
            f"--- RECENT CONVERSATION ---:\n{conversation_history}\n\n"
            f"--- RELEVANT MEMORIES ---:\n{relevant_memory_context}\n\n"
            f"--- You are talking to {user_name} ---\n"
        )

        if dominant_emotion in getattr(config, 'EMOTION_RESPONSES', {}):
            system_instruction += f"\n\nEMOTION CONTEXT: {config.EMOTION_RESPONSES[dominant_emotion]}"

        temp = self._get_temperature_for_emotion(dominant_emotion)

        generated_text = await self._llm_inference(
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

    # --- Analysis & Classification Methods ---

    def analyze_emotions(self, text: str) -> dict:
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

    async def is_speech_for_ai(self, transcription: str, user_id: str, active_mode: str) -> bool:
        """Determines if a given transcription is intended for the AI."""
        cleaned_transcription = transcription.lower().strip()
        if not cleaned_transcription:
            return False
            
        if self.is_definitely_for_bot(cleaned_transcription):
            return True
            
        if cleaned_transcription in config.IGNORE_EXPRESSIONS:
            return False
        
        # In local mode or single-user mode, always assume it's for the bot.
        if active_mode == "local" or config.DEFAULT_BOT_MODE == config.BOT_MODES["SINGLE"]:
            return True
            
        return await self._classify_speech_intent_with_ai(cleaned_transcription)

    def is_definitely_for_bot(self, transcription: str) -> bool:
        """Checks for explicit wake words or direct address patterns."""
        text = transcription.lower().strip()
        if self.wake_word_patterns["prefix"].search(text): return True
        if self.wake_word_patterns["start"].search(text): return True
        if self.wake_word_patterns["end"].search(text): return True
        return False

    def is_vision_request(self, transcription: str) -> bool:
        """Checks if a transcription is likely a request for the vision system."""
        text = transcription.lower().strip()
        
        if any(phrase in text for phrase in config.VISION_TRIGGER_PHRASES):
            return True
            
        if any(clue in text for clue in config.VISION_CONTEXT_CLUES):
            return True
        
        action_match = process.extractOne(text, config.VISION_ACTION_WORDS, scorer=fuzz.partial_token_sort_ratio)
        target_present = any(word in text for word in config.VISION_TARGET_WORDS)
        
        if action_match and action_match[1] > config.VISION_CONFIDENCE_THRESHOLD and target_present:
            return True
        
        imperative_commands = ["see", "look", "show", "describe", "read", "check", "analyze"]
        if any(text.startswith(cmd) for cmd in imperative_commands):
            return True
            
        return False

    # --- Internal Helper Methods ---

    def _get_temperature_for_emotion(self, dominant_emotion: str) -> float:
        """Returns a fine-tuned temperature setting based on the user's emotion."""
        emotion_temps = {
            "joy": 0.9,
            "sadness": 0.7,
            "surprise": 1.0,
            "anger": 0.75
        }
        return emotion_temps.get(dominant_emotion, 0.8)

    async def _llm_inference(self, prompt: str, system_prompt: str, temperature: float, stop_sequences: list) -> str:
        """Internal async method to call the Ollama API with httpx."""
        json_payload = {
            "model": config.OLLAMA_MODEL,
            "system": system_prompt,
            "prompt": prompt,
            "options": {
                "temperature": temperature, "top_p": 1.5, "top_k": 40,
                "repeat_penalty": 1.15, "num_predict": 90, "num_ctx": 16384,
                "repeat_last_n": 32, "stop": stop_sequences
            },
            "stream": False
        }
        try:
            response = await self.client.post(f"{config.OLLAMA_API_URL}/api/generate", json=json_payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except httpx.RequestError as e:
            print(f"Error connecting to Ollama API: {e}")
            return "Sorry, I'm having trouble connecting to my brain right now."
        except Exception as e:
            print(f"Error in LLM inference: {e}")
            return "Sorry, something went wrong while I was thinking."
            
    async def _classify_speech_intent_with_ai(self, transcription: str) -> bool:
        """Uses a specialized LLM call to determine if speech is for the bot."""
        bot_name = getattr(config, 'BOT_NAME', 'Assistant')
        alt_names = getattr(config, 'BOT_ALT_NAMES', [bot_name.lower(), bot_name])
        alt_names_str = "', '".join(set(alt_names))
        

        system_prompt = f"""You are a 'Conversation Target' analysis expert. Your job is to analyze the user's text and identify who the user is speaking to.
The AI bot's name is '{bot_name}' and may also be called '{alt_names_str}'.
You must respond with ONLY ONE of the following three words:
- '{bot_name}' if the user is addressing the bot.
- 'Other' if the user is addressing another person or group.
- 'General' if the user is making a general statement to no one in particular."""
        
        full_prompt = f"System: {system_prompt}\n\nUser: \"{transcription}\"\nSystem:"
        
        json_payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": full_prompt,
            "options": {"temperature": 0.0, "top_p": 0.1, "num_predict": 5, "stop": ["\n", "."]},
            "stream": False
        }
        
        try:
            response = await self.client.post(f"{config.OLLAMA_API_URL}/api/generate", json=json_payload)
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("response", "").strip().lower()
            print(f"Target Analysis for '{transcription}': AI returned -> '{generated_text}'")
            
            bot_names_to_check = [bot_name.lower()] + [name.lower() for name in alt_names]
            return any(name in generated_text for name in bot_names_to_check)
        except httpx.RequestError as e:
            print(f"Error connecting to Ollama for intent classification: {e}")
            return False
        except Exception as e:
            print(f"Error during intent classification: {e}")
            return False