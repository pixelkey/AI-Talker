import os
import json
from datetime import datetime
import pytz

class SelfReflectionHistoryManager:
    def __init__(self, base_dir="/home/andrew/projects/app/python/talker/ingest/self_reflection_history"):
        """Initialize the self reflection history manager"""
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def add_reflection(self, memory_content, context=None):
        """
        Save a reflection to a file in the same format as embeddings
        
        Args:
            memory_content (str): The memory content
            context (dict, optional): Metadata about the memory
        """
        # Generate filename with timestamp
        tz = pytz.timezone('Australia/Adelaide')
        timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        filename = f"reflection-{timestamp}.txt"
        filepath = os.path.join(self.base_dir, filename)
        
        # Format metadata as a header
        metadata = {
            'source': 'memory',
            'type': context.get('memory_type') if context else 'unknown',
            'expiry_date': context.get('expiry_date') if context else None,
            'surprise_score': context.get('surprise_score') if context else 0.0,
            'timestamp': context.get('timestamp') if context else datetime.now(tz).isoformat(),
            'content_type': 'conversation_memory'
        }
        
        # Write memory content with metadata header
        with open(filepath, 'w') as f:
            f.write("=== Metadata ===\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n=== Content ===\n")
            f.write(memory_content)
