import os
from datetime import datetime
import json
from config import INGEST_PATH

class ChatHistoryManager:
    def __init__(self):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ingest_path = os.path.join(script_dir, INGEST_PATH.lstrip("../"))
        self.chat_dir = os.path.join(ingest_path, "chat_history")
        self.current_file = None
        self._ensure_chat_directory()
        self._set_current_file()

    def _ensure_chat_directory(self):
        """Ensure the chat history directory exists"""
        if not os.path.exists(self.chat_dir):
            os.makedirs(self.chat_dir)

    def _set_current_file(self):
        """Set the current chat file name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = os.path.join(self.chat_dir, f"chat-{timestamp}.json")

    def save_history(self, history):
        """Save chat history to the current file only if there are messages"""
        if history:  # Only save if there are messages
            formatted_history = []
            for user_msg, assistant_msg in history:
                # Only save pairs where both messages exist
                if user_msg and assistant_msg:
                    formatted_pair = {
                        "user": user_msg,
                        "assistant": assistant_msg
                    }
                    formatted_history.append(formatted_pair)
            
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_history, f, ensure_ascii=False, indent=2)

    def load_history(self):
        """Load chat history from the current file"""
        if os.path.exists(self.current_file):
            with open(self.current_file, 'r', encoding='utf-8') as f:
                formatted_history = json.load(f)
                history = []
                for msg_pair in formatted_history:
                    user_msg = msg_pair["user"].replace("user: ", "", 1) if msg_pair["user"] else ""
                    assistant_msg = msg_pair["assistant"].replace("assistant: ", "", 1) if msg_pair["assistant"] else ""
                    history.append((user_msg, assistant_msg))
                return history
        return []

    def get_new_messages(self, last_processed_index=0):
        """Get messages that haven't been processed for embeddings yet"""
        history = self.load_history()
        new_messages = history[last_processed_index:]
        return new_messages, len(history)

    def format_for_embedding(self, messages):
        """Format messages for embedding, excluding references"""
        formatted_texts = []
        for user_msg, assistant_msg in messages:
            if user_msg:
                formatted_texts.append(f"User: {user_msg}")
            if assistant_msg:
                # Remove reference sections if they exist
                response = assistant_msg.split("References:", 1)[0].strip()
                formatted_texts.append(f"Assistant: {response}")
        return "\n".join(formatted_texts)
