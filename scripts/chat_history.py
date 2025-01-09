# scripts/chat_history.py

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
                # Extract timestamp from messages if present
                user_timestamp = None
                if '[' in user_msg and ']' in user_msg:
                    timestamp_line = user_msg.split('\n')[0]  # Get first line containing timestamp
                    user_timestamp = timestamp_line[timestamp_line.find('[')+1:timestamp_line.find(']')]
                
                # Only save pairs where both messages exist
                if user_msg and assistant_msg:
                    formatted_pair = {
                        "timestamp": user_timestamp,
                        "user": user_msg,
                        "assistant": assistant_msg
                    }
                    formatted_history.append(formatted_pair)
            
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_history, f, ensure_ascii=False, indent=2)

    def format_for_embedding(self, messages):
        """Format messages for embedding, including timestamps"""
        formatted_texts = []
        for user_msg, assistant_msg in messages:
            if user_msg:
                formatted_texts.append(user_msg)  # Already includes timestamp and newline
            if assistant_msg:
                # Remove reference sections if they exist, but keep timestamp
                response = assistant_msg.split("References:", 1)[0].strip()
                formatted_texts.append(response)
        return "\n".join(formatted_texts)
