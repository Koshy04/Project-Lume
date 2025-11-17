import httpx
import re
from nrclex import NRCLex
from fuzzywuzzy import fuzz, process
import config
from src.services.memorys.memory import memory_manager
from src.log.custom_logger import logger

class AICore:
    """
    Manages all AI-related tasks including language model inference,
    emotion analysis, and intent classification.
    """
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.client = httpx.AsyncClient(timeout=60.0)

        wake_words_pattern = '|'.join(re.escape(name.lower()) for name in config.BOT_WAKE_WORDS)
        wake_prefixes_pattern = '|'.join(re.escape(prefix.lower()) for prefix in config.WAKE_PREFIXES)
        
        self.wake_word_patterns = {
            "prefix": re.compile(rf'^({wake_prefixes_pattern})\s+({wake_words_pattern})\b'),
            "start": re.compile(rf'^({wake_words_pattern})[\s,]'),
            "end": re.compile(rf'[\s,]+({wake_words_pattern})\??\s*$')
        }

        logger.info("AI Core initialized.")

    async def chat_with_ai(self, prompt: str, user_id: str, emotion_data: dict, conversation_history: str, vision_context: dict | None = None) -> str:
        """
        Generates a contextual AI response using conversation history and vector memory.
        """
        user_name = config.USER_NAMES.get(str(user_id), f"User({user_id})")
        dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
        relevant_memory_context = memory_manager.search_memories(prompt, user_id)
        system_instruction = (
            f"{getattr(config, 'BASED_PERSONALITY', 'You are a helpful AI assistant.')}\n\n"
            f"--- RECENT CONVERSATION ---:\n{conversation_history}\n\n"
            f"--- RELEVANT MEMORIES ---:\n{relevant_memory_context}\n\n"
            f"--- You are talking to {user_name} ---\n"
        )
        
        if vision_context:
            caption = vision_context.get('caption', 'N/A')
            ocr_text = vision_context.get('ocr_text', 'N/A')
            system_instruction += (
                f"\n--- CURRENT SCREEN CONTEXT (What you can see right now) ---\n"
                f"- Overall Scene: '{caption}'\n- Text on Screen: '{ocr_text}'\n"
            )

        if dominant_emotion in getattr(config, 'EMOTION_RESPONSES', {}):
            system_instruction += f"\n\nEMOTION CONTEXT: {config.EMOTION_RESPONSES[dominant_emotion]}"

        temp = self._get_temperature_for_emotion(dominant_emotion)
        generated_text = await self.llm_manager.generate(
            prompt=prompt,
            system_prompt=system_instruction,
            temperature=temp,
            stop_sequences=["\nUser:", "\nHuman:", f"\n{user_name}:", f"\n{user_name.lower()}:"]
        )
   
        if "I'm having trouble responding right now." in generated_text: 
            return generated_text

        if prompt.strip() and generated_text:
            memory_manager.add_raw_turn(user_name, prompt, generated_text, str(user_id))

        return generated_text or "I don't know what to say right now."

    async def determine_animation_sequence(self, text: str, available_animations: list) -> list[str]:
        """
        Uses the LLM to choose a sequence of animations based on dialogue length and energy.
        """
        if not available_animations:
            return []
            
        anim_list_str = ", ".join(available_animations)
        
        system_prompt = (
            "You are an expert Animation Director for a VTuber. Your task is to create a short sequence of animations "
            "that matches the bot's dialogue. Your response will be a comma-separated list of animation names."
        )
        
        prompt = (
            f"RULES:\n"
            f"1. For short, simple dialogue (under 10 words), respond with ONE animation.\n"
            f"2. For longer or more expressive dialogue, respond with 2-3 animations.\n"
            f"3. If the dialogue is very energetic or emotional, you can REPEAT an animation (e.g., HappyBounce, HappyBounce).\n"
            f"4. Choose ONLY from the provided list. Your response must be a comma-separated list and nothing else.\n"
            f"5. If no animation fits, respond with 'Idle'.\n\n"
            f"--- EXAMPLE ---\n"
            f"Dialogue: \"Oh wow, that is absolutely insane! I love it!\"\n"
            f"Available Animations: [{anim_list_str}]\n"
            f"Response: Surprise, HappyBounce, HappyBounce\n\n"
            f"--- TASK ---\n"
            f"Dialogue: \"{text}\"\n"
            f"Available Animations: [{anim_list_str}]\n"
            f"Response:"
        )
        
        try:
            llm_response = await self.llm_manager.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                stop_sequences=["\n"]
            )
            
            # Sanitize and split the response into a list of potential animation names
            raw_choices = [choice.strip().strip('.?!,"\'') for choice in llm_response.split(',')]
            
            final_sequence = []
            for choice in raw_choices:
                if not choice or choice.lower() == 'idle':
                    continue
                
                # Fuzzy match each choice to ensure it's a valid animation
                match = process.extractOne(choice, available_animations, score_cutoff=85)
                if match:
                    final_sequence.append(match[0])
            
            logger.info(f"Animation Director raw: '{llm_response}', Final sequence: {final_sequence}")
            return final_sequence

        except Exception as e:
            logger.error(f"Error in animation director: {e}")
            return []

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
            logger.error(f"Error analyzing emotions for text '{text[:30]}...': {e}")
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

    def _get_temperature_for_emotion(self, dominant_emotion: str) -> float:
        """Returns a fine-tuned temperature setting based on the user's emotion."""
        emotion_temps = { "joy": 0.9, "sadness": 0.7, "surprise": 1.0, "anger": 0.75 }
        return emotion_temps.get(dominant_emotion, 0.8)

    async def _classify_speech_intent_with_ai(self, transcription: str) -> bool:
        """Uses the active LLM engine to determine if speech is for the bot."""
        try:
            generated_text = await self.llm_manager.classify_intent(transcription)
            generated_text = generated_text.strip().lower()

            logger.info(f"Target Analysis for '{transcription}': AI returned -> '{generated_text}'")
            
            bot_names_to_check = ['bot'] + [name.lower() for name in config.BOT_WAKE_WORDS] + [config.BOT_NAME.lower()]
            return any(name in generated_text for name in bot_names_to_check)

        except Exception as e:
            logger.error(f"Error during intent classification via LLMManager: {e}")
            return False