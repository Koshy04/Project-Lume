import chromadb
import os
import time
import uuid
import config
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class VectorMemoryManager:
    def __init__(self, collection_name="conversation_memory"):
        print("-> Initializing Vector Memory Manager...")
        if not os.path.exists(config.MEMORY_DB_PATH):
            os.makedirs(config.MEMORY_DB_PATH)
            
        self.client = chromadb.PersistentClient(
            path=config.MEMORY_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Using a CPU-based model for broader compatibility
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
        print("Vector Memory Manager initialized.")

    def add_raw_turn(self, user_name: str, user_input: str, ai_response: str, user_id: str):
        """Adds a new raw conversation turn to the memory."""
        try:
            # We create an embedding based on the user's input to find it later
            embedding_text = f"{user_name}: {user_input}"
            
            # The document we store contains the full context of the turn
            document_text = f"[{user_name}]: \"{user_input}\" -> [{config.BOT_NAME}]: \"{ai_response}\""

            self.collection.add(
                embeddings=[self.embedding_model.encode(embedding_text).tolist()],
                documents=[document_text],
                metadatas=[{"user_id": str(user_id), "timestamp": time.time(), "type": "turn"}],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            print(f"ERROR: Failed to add raw memory turn: {e}")

    def add_summarized_memory(self, summary_text: str, user_id: str):
        """Adds a new high-level, summarized memory to the database."""
        try:
            self.collection.add(
                embeddings=[self.embedding_model.encode(summary_text).tolist()],
                documents=[summary_text],
                metadatas=[{"user_id": str(user_id), "timestamp": time.time(), "type": "summary"}],
                ids=[str(uuid.uuid4())]
            )
            print(f"✅ Consolidated Memory Added: '{summary_text}' for user {user_id}")
        except Exception as e:
            print(f"ERROR: Failed to add summarized memory: {e}")

    def search_memories(self, query_text: str, user_id: str, n_results: int = 5) -> str:
        """Searches for memories semantically similar to the query text."""
        if self.collection.count() == 0:
            return "" # Return empty string if no memories
            
        try:
            query_embedding = self.embedding_model.encode(query_text).tolist()

            # Query for memories related to the specific user OR general summaries that belong to them
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                where={"user_id": str(user_id)},
                include=['documents', 'metadatas']
            )
            
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    if meta.get('type') == 'summary':
                        formatted_results.append(f"[Key Memory]: {doc}")
                    else:
                        formatted_results.append(f"[Past Conversation]: {doc}")
            
            if not formatted_results:
                return "" # Return empty string if no relevant memories

            return "RELEVANT MEMORIES:\n- " + "\n- ".join(formatted_results)

        except Exception as e:
            print(f"   ERROR: Failed to search memories: {e}")
            return "I had a problem searching my memory."

# This ensures we use the same memory manager across the application.
memory_manager = VectorMemoryManager()