import asyncio
import config
import ai_core as ai_core
from memory import memory_manager

class SummarizationManager:
    def __init__(self):
        print("-> Initializing Summarization Manager...")
        pass

    def _get_summarization_prompt(self, transcript: str, user_name: str) -> str:
        """Creates the prompt for the LLM to generate a summary."""
        return (f"You are a 'Memory Consolidation' AI. Below is a conversation transcript involving a user named '{user_name}'. "
                f"Your task is to extract the single most important, lasting piece of information about '{user_name}' from this exchange. "
                f"The memory should be a concise, third-person statement about their preferences, decisions, or core personality traits. "
                f"**Be strictly factual based ONLY on the provided transcript. Do not add your own commentary.** "
                f"Examples: 'Koshy's favorite character in a game is Karlach.' or 'Zero feels that story is more important than gameplay.' or 'Jo is learning to code.'\n\n"
                f"If no significant new memory was formed, respond ONLY with the word 'NO_MEMORY'.\n\n"
                f"Conversation Transcript:\n---\n{transcript}\n---\n\nSingle most important memory about {user_name}:")

    async def consolidate_and_store(self, conversation_history: list):
        """
        Processes a conversation history, generates a summary, and stores it as a key memory.
        """
        if not conversation_history: return

        # Identify the primary user in this conversation segment
        user_turns = [turn for turn in conversation_history if turn['role'].lower() != config.BOT_NAME.lower()]
        if not user_turns:
            print("Summarizer: No user turns in history to summarize.")
            return

        # Find the most frequent user in this segment
        try:
            user_ids = [turn['user_id'] for turn in user_turns]
            primary_user_id = max(set(user_ids), key=user_ids.count)
            primary_user_name = config.USER_NAMES.get(primary_user_id, f"User({primary_user_id})")
        except (ValueError, IndexError):
            print("Summarizer: Could not determine a primary user for this segment.")
            return

        transcript = "\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation_history])
        prompt = self._get_summarization_prompt(transcript, primary_user_name)
        
        # Use a neutral, factual system prompt for summarization
        system_prompt = "You are a helpful AI assistant that is excellent at extracting factual information from text."
        
        # We call the core llm_inference function directly for this system task
        summary = await ai_core.llm_inference(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0, # Be very factual
            stop_sequences=["\n"]
        )

        if summary and "NO_MEMORY" not in summary:
            # Clean any potential notes or extra text from the summary
            summary = summary.split('(Note:')[0].strip().replace('"', '')
            # Store the summary with the user ID it pertains to
            memory_manager.add_summarized_memory(summary, user_id=primary_user_id)
        else:
            print("   Summarizer: No significant memory to consolidate from segment.")

# --- Singleton Instance ---
summarizer = SummarizationManager()