import asyncio
from flask import Flask, request, jsonify
from waitress import serve
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ai_core import chat_with_ai, analyze_emotions, add_to_recent_responses, is_too_similar_to_recent
    from config import USER_NAMES
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"API ERROR: Could not import Lume modules: {e}. The API will run but return error messages.")
    IMPORTS_SUCCESSFUL = False

# --- Initialize Flask App ---
app = Flask(__name__)

# --- API Endpoints ---

@app.route('/status', methods=['GET'])
def get_status():
    """A simple endpoint to confirm the API is running."""
    return jsonify({"status": "Lume API is online", "ai_core_loaded": IMPORTS_SUCCESSFUL}), 200

@app.route('/chat', methods=['POST'])
async def handle_chat():
    """
    Main endpoint to interact with the bot.
    Expects JSON: { "user_id": "some_id", "message": "hello " }
    """
    if not IMPORTS_SUCCESSFUL:
        return jsonify({"error": "AI Core modules are not loaded. API cannot process requests."}), 503

    data = request.get_json()
    if not data or 'user_id' not in data or 'message' not in data:
        return jsonify({"error": "Invalid request. 'user_id' and 'message' are required."}), 400

    user_id = str(data['user_id'])
    user_message = data['message']
    user_name = USER_NAMES.get(user_id, f"User({user_id})")

    try:
        conversation_history = f"User: {user_message}"
        emotion_data = analyze_emotions(user_message)

        ai_response_text = await chat_with_ai(user_message, user_id, emotion_data, conversation_history)

        if "Meow?" in ai_response_text or is_too_similar_to_recent(ai_response_text):
            final_response = "I'm not quite sure how to respond to that."
            final_emotion = "neutral"
        else:
            add_to_recent_responses(ai_response_text)
            final_response = ai_response_text
            response_emotion_data = analyze_emotions(final_response)
            final_emotion = response_emotion_data.get("dominant_emotion", "neutral")

        current_timestamp = await asyncio.to_thread(lambda: __import__('datetime').datetime.now().isoformat())

        response_payload = {
            "user_id": user_id,
            "user_name": user_name,
            "ai_response": final_response,
            "emotion": final_emotion,
            "timestamp": current_timestamp
        }
        return jsonify(response_payload), 200

    except Exception as e:
        print(f"ERROR in /chat endpoint: {e}\n{__import__('traceback').format_exc()}")
        return jsonify({"error": "An internal error occurred while processing the AI response."}), 500

def main():
    """Main entry point for running the API server."""
    print("--- API Server ---")
    print("Starting Waitress server on http://0.0.0.0:8080")
    print("The API is now available for game plugins.")
    serve(app, host='0.0.0.0', port=8080)

if __name__ == '__main__':
    main()