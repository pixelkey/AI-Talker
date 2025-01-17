import os
import json
from datetime import datetime
import pytz

class SelfReflectionHistoryManager:
    def __init__(self, base_dir="/home/andrew/projects/app/python/talker/ingest/self_reflection_history"):
        """Initialize the self reflection history manager"""
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.current_file = None
        self.current_reflections = []

    def start_new_session(self):
        """Start a new reflection session with a new file"""
        # Generate filename with timestamp
        tz = pytz.timezone('Australia/Adelaide')
        timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        filename = f"reflection-{timestamp}.json"
        self.current_file = os.path.join(self.base_dir, filename)
        self.current_reflections = []
        self._save_reflections()

    def add_reflection(self, reflection, context=None):
        """
        Add a new reflection to the current session
        
        Args:
            reflection (str): The reflection text
            context (dict, optional): Any additional context to store
        """
        if not self.current_file:
            self.start_new_session()

        # Convert any datetime objects in context to ISO format strings
        if context:
            context = self._serialize_datetime_values(context)

        entry = {
            "timestamp": datetime.now(pytz.timezone('Australia/Adelaide')).strftime('%Y-%m-%d %H:%M:%S%z'),
            "reflection": reflection,
            "context": context or {}
        }
        
        self.current_reflections.append(entry)
        self._save_reflections()
        
    def get_current_reflections(self):
        """Get all reflections from the current session"""
        return self.current_reflections

    def _save_reflections(self):
        """Save reflections to the current file"""
        if self.current_file:
            with open(self.current_file, 'w') as f:
                json.dump({
                    "reflections": self.current_reflections
                }, f, indent=2)

    def _serialize_datetime_values(self, obj):
        """Recursively convert datetime objects to ISO format strings"""
        if isinstance(obj, dict):
            return {key: self._serialize_datetime_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime_values(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S%z')
        return obj
