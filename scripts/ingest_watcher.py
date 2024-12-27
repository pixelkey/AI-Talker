import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import INGEST_PATH
import logging
from pathlib import Path
import threading
from typing import Optional, Callable

class IngestHandler(FileSystemEventHandler):
    def __init__(self, on_change_callback: Callable):
        self.on_change_callback = on_change_callback
        self._last_processed_time = time.time()
        self._debounce_delay = 1.0  # Delay in seconds to debounce multiple events
        self._lock = threading.Lock()
        self._pending_changes = set()  # Track pending file changes

    def _should_process_event(self, event) -> bool:
        """Check if the event should be processed"""
        # Process all directory events
        if event.is_directory:
            return True
            
        file_path = Path(event.src_path)
        
        # Ignore temporary files, hidden files, and system files
        if file_path.name.startswith(('.', '~', '$')):
            return False
            
        # Process all other files
        return True

    def _handle_event(self, event):
        """Handle file system event with debouncing"""
        if not self._should_process_event(event):
            return
            
        current_time = time.time()
        with self._lock:
            # Add the changed file to pending changes
            if not event.is_directory:
                self._pending_changes.add(event.src_path)
                logging.info(f"Added {event.src_path} to pending changes")
            
            if current_time - self._last_processed_time >= self._debounce_delay:
                self._last_processed_time = current_time
                # Process all pending changes
                if self._pending_changes:
                    logging.info(f"Processing {len(self._pending_changes)} changed files: {list(self._pending_changes)}")
                    self.on_change_callback(list(self._pending_changes))
                    self._pending_changes.clear()
                else:
                    # Directory event - process all files
                    self.on_change_callback(None)

    def on_modified(self, event):
        self._handle_event(event)

    def on_created(self, event):
        self._handle_event(event)
        
    def on_deleted(self, event):
        self._handle_event(event)
        
    def on_moved(self, event):
        self._handle_event(event)

class IngestWatcher:
    def __init__(self, on_change_callback: Callable):
        """
        Initialize the watcher for the ingest directory
        Args:
            on_change_callback: Callback function to be called when files change
        """
        self.observer: Optional[Observer] = None
        self.handler = IngestHandler(on_change_callback)
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.watch_path = os.path.join(script_dir, INGEST_PATH.lstrip("../"))
        self._ensure_watch_path()

    def _ensure_watch_path(self):
        """Ensure the watch path exists"""
        if not os.path.exists(self.watch_path):
            os.makedirs(self.watch_path)
            logging.info(f"Created ingest directory at {self.watch_path}")

    def start(self):
        """Start watching the ingest directory"""
        if self.observer is not None:
            logging.warning("Watcher is already running")
            return

        self.observer = Observer()
        self.observer.schedule(self.handler, self.watch_path, recursive=True)
        self.observer.start()
        logging.info(f"Started watching directory: {self.watch_path}")

    def stop(self):
        """Stop watching the ingest directory"""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logging.info("Stopped watching ingest directory")
