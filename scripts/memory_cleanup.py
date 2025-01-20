import threading
import time
import logging
import re
from datetime import datetime, timedelta
import pytz
from typing import Optional
import traceback
import os
from pathlib import Path
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class MemoryCleanupManager:
    def __init__(self, context, cleanup_interval: int = 3600):  
        """Initialize the memory cleanup manager.
        
        Args:
            context: Application context containing vector store and other components
            cleanup_interval: Time between cleanup runs in seconds
        """
        self.context = context
        self.cleanup_interval = cleanup_interval
        self.cleanup_thread = None
        self.stop_cleanup = threading.Event()
        self.is_cleaning = False
        self.last_cleanup_time = None
        
        # Setup logging directory
        self.log_dir = Path(self.context.get('base_dir', '.')) / 'logs' / 'memory_cleanup'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure file handler for cleanup logs
        cleanup_log_file = self.log_dir / 'memory_cleanup.log'
        file_handler = logging.FileHandler(cleanup_log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Prompt for evaluating memory usefulness
        self.usefulness_prompt = """Evaluate if this memory contains useful information worth retaining.

        A memory is NOT useful if it:
        1. Contains error messages or failed attempts (e.g., "unable to find...", "failed to...")
        2. Is purely procedural without content (e.g., "processing request...", "searching...")
        3. Contains no actual information (e.g., "no relevant information found")
        4. Is redundant or trivial
        5. Is a temporary status update
        6. Contains no meaningful context or insights

        A memory IS useful if it:
        1. Contains specific facts, knowledge, or insights
        2. Includes personal preferences or important details
        3. Has context that helps understand the user or conversation
        4. Contains meaningful search results or findings
        5. Includes decisions or conclusions
        6. Has time-sensitive but important information

        Memory to evaluate:
        {memory_content}

        Respond in this format only:
        KEEP: [YES/NO]
        REASON: [brief explanation]"""

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp with flexible format handling"""
        try:
            # Try different timestamp formats
            formats = [
                '%Y-%m-%dT%H:%M:%S%z',  # ISO format with timezone
                '%Y-%m-%d %H:%M:%S%z',  # Standard format with timezone
                '%Y-%m-%d %H:%M:%S',     # Without timezone
            ]
            
            # Clean up the timestamp string
            timestamp_str = timestamp_str.strip()
            
            # Handle timezone separately if it exists
            if '+' in timestamp_str:
                main_part, tz_part = timestamp_str.rsplit('+', 1)
                if ':' in tz_part:  # Handle +HH:MM format
                    tz_part = tz_part.replace(':', '')
                timestamp_str = f"{main_part}+{tz_part}"
            
            # Try each format
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # If none of the formats work, try parsing with dateutil
            from dateutil import parser
            return parser.parse(timestamp_str)
            
        except Exception as e:
            logger.error(f"Error parsing timestamp {timestamp_str}: {str(e)}")
            return None

    def _log_cleanup_details(self, removed_docs, evaluation_results):
        """Log detailed cleanup information to a JSON file"""
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f'cleanup_details_{timestamp}.json'
            
            cleanup_data = {
                'timestamp': timestamp,
                'total_removed': len(removed_docs),
                'removed_documents': removed_docs,
                'evaluation_results': evaluation_results
            }
            
            with open(log_file, 'w') as f:
                json.dump(cleanup_data, f, indent=2)
            
            logger.info(f"Detailed cleanup log saved to {log_file}")
            
            # Also maintain a summary file
            summary_file = self.log_dir / 'cleanup_summary.json'
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                summary = {'cleanups': []}
            
            summary['cleanups'].append({
                'timestamp': timestamp,
                'docs_removed': len(removed_docs),
                'log_file': str(log_file.name)
            })
            
            # Keep only last 100 entries
            summary['cleanups'] = summary['cleanups'][-100:]
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging cleanup details: {str(e)}")

    def _evaluate_memory_usefulness(self, memory_content: str) -> tuple[bool, str]:
        """Use LLM to evaluate if a memory is worth keeping"""
        try:
            # Create a simplified context for LLM
            temp_context = {
                'client': self.context['client'],
                'MODEL_SOURCE': self.context.get('MODEL_SOURCE', 'local'),
                'skip_web_search': True,
                'skip_emotion': True,
                'LLM_MODEL': self.context.get('LLM_MODEL', 'mistral')
            }
            
            # Format prompt with memory content
            prompt = self.usefulness_prompt.format(memory_content=memory_content)
            
            # Get evaluation from LLM
            response = temp_context['client'].chat(
                model=temp_context['LLM_MODEL'],
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': memory_content}
                ]
            )
            
            result = response['message']['content'].strip().upper()
            keep_memory = 'KEEP: YES' in result
            
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', result)
            reason = reason_match.group(1) if reason_match else "No reason provided"
            logger.debug(f"Memory evaluation - Keep: {keep_memory}, Reason: {reason}")
            
            return keep_memory, reason
            
        except Exception as e:
            logger.error(f"Error evaluating memory usefulness: {str(e)}")
            return True, f"Error during evaluation: {str(e)}"

    def _cleanup_expired_memories(self) -> None:
        """Remove expired memories from the vector store and save changes"""
        try:
            vector_store = self.context.get('vector_store')
            if not vector_store:
                logger.warning("No vector store found in context")
                return

            current_time = self.context.get('current_time')
            if isinstance(current_time, str):
                current_time = self._parse_timestamp(current_time)
            if not current_time:
                logger.error("Could not get current time")
                return

            logger.info(f"Starting memory cleanup at {current_time}")
            
            # Get all documents from vector store
            docstore = vector_store.docstore
            all_docs = docstore._dict if hasattr(docstore, '_dict') else {}
            docs_to_remove = []
            indices_to_remove = []  # Track FAISS indices to remove
            evaluation_results = {}  # Track evaluation results for logging

            # Check each document for expiry and usefulness
            for doc_id, doc in all_docs.items():
                try:
                    should_remove = False
                    removal_reason = None
                    expiry_date = doc.metadata.get('expiry_date')
                    
                    # Check if expired
                    if expiry_date:
                        expiry_datetime = self._parse_timestamp(expiry_date)
                        if expiry_datetime and expiry_datetime <= current_time:
                            should_remove = True
                            removal_reason = f"Expired (expiry: {expiry_date})"
                            
                    # If not expired or no expiry, check usefulness if it's not a long-term memory
                    if not should_remove and doc.metadata.get('type') != 'long_term':
                        keep_memory, reason = self._evaluate_memory_usefulness(doc.page_content)
                        if not keep_memory:
                            should_remove = True
                            removal_reason = f"Low usefulness: {reason}"
                    
                    if should_remove:
                        docs_to_remove.append(doc_id)
                        if hasattr(vector_store, 'index_to_docstore_id'):
                            for i, stored_id in enumerate(vector_store.index_to_docstore_id):
                                if stored_id == doc_id:
                                    indices_to_remove.append(i)
                        
                        # Store evaluation results for logging
                        evaluation_results[doc_id] = {
                            'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                            'metadata': doc.metadata,
                            'removal_reason': removal_reason
                        }
                        
                        logger.info(f"Marking document for removal: {doc_id} - {removal_reason}")
                        
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {str(e)}")
                    continue

            # Remove expired documents if any found
            if docs_to_remove:
                logger.info(f"Removing {len(docs_to_remove)} documents")
                try:
                    # Remove from docstore
                    for doc_id in docs_to_remove:
                        if doc_id in docstore._dict:
                            del docstore._dict[doc_id]
                    
                    # Remove from FAISS index if indices found
                    if indices_to_remove and hasattr(vector_store, 'index'):
                        indices_to_remove.sort(reverse=True)
                        for idx in indices_to_remove:
                            vector_store.index = vector_store._remove_vectors([idx], vector_store.index)
                            if hasattr(vector_store, 'index_to_docstore_id'):
                                del vector_store.index_to_docstore_id[idx]
                    
                    # Save changes to disk
                    from faiss_utils import save_faiss_index_metadata_and_docstore
                    save_faiss_index_metadata_and_docstore(
                        vector_store.index,
                        vector_store.docstore,
                        vector_store.index_to_docstore_id,
                        self.context.get('embeddings_dir', 'embeddings')
                    )
                    
                    # Log cleanup details
                    self._log_cleanup_details(evaluation_results, {
                        'total_docs': len(all_docs),
                        'docs_removed': len(docs_to_remove),
                        'cleanup_time': current_time.isoformat()
                    })
                    
                    logger.info("Memory cleanup completed and saved to disk")
                except Exception as e:
                    logger.error(f"Error during document removal: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.info("No documents to remove")

            self.last_cleanup_time = current_time

        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            logger.error(traceback.format_exc())

    def _get_last_cleanup_from_logs(self) -> Optional[datetime]:
        """Get the last cleanup time from logs"""
        try:
            summary_file = self.log_dir / 'cleanup_summary.json'
            if not summary_file.exists():
                return None
                
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                
            if not summary.get('cleanups'):
                return None
                
            # Get the most recent cleanup timestamp
            last_cleanup = summary['cleanups'][-1]['timestamp']
            return datetime.strptime(last_cleanup, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
            
        except Exception as e:
            logger.error(f"Error reading last cleanup time from logs: {str(e)}")
            return None

    def should_run_cleanup(self) -> bool:
        """Check if cleanup should run based on last cleanup time"""
        try:
            current_time = self.context.get('current_time')
            if isinstance(current_time, str):
                current_time = self._parse_timestamp(current_time)
            if not current_time:
                logger.error("Could not get current time")
                return False

            # Get last cleanup time from logs or memory
            last_cleanup = self._get_last_cleanup_from_logs()
            if not last_cleanup:
                # If no log found, use in-memory time or return True for first run
                last_cleanup = self.last_cleanup_time
                if not last_cleanup:
                    logger.info("No previous cleanup found, should run first cleanup")
                    return True

            # Check if an hour has passed
            time_since_cleanup = current_time - last_cleanup
            should_run = time_since_cleanup.total_seconds() >= 3600  # 1 hour

            if should_run:
                logger.info(f"Last cleanup was {time_since_cleanup.total_seconds()/3600:.1f} hours ago, should run cleanup")
            else:
                logger.debug(f"Only {time_since_cleanup.total_seconds()/3600:.1f} hours since last cleanup, skipping")

            return should_run

        except Exception as e:
            logger.error(f"Error checking if should run cleanup: {str(e)}")
            return False

    def _cleanup_loop(self) -> None:
        """Main cleanup loop that runs periodically"""
        while not self.stop_cleanup.is_set():
            try:
                # Only run cleanup if needed and no other processing is happening
                if self.should_run_cleanup() and not self.context.get('is_processing', False):
                    self.is_cleaning = True
                    self._cleanup_expired_memories()
                    self.is_cleaning = False
                
                # Check every minute instead of every hour
                for _ in range(60):  # 1 minute sleep broken into 1-second chunks
                    if self.stop_cleanup.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                logger.error(traceback.format_exc())
                self.is_cleaning = False
                time.sleep(60)  # Wait a bit before retrying after error

    def get_next_cleanup_time(self) -> Optional[datetime]:
        """Get the time of the next scheduled cleanup"""
        if not self.last_cleanup_time:
            return datetime.now(timezone.utc)  # First cleanup will happen soon
        
        next_cleanup = self.last_cleanup_time + timedelta(seconds=self.cleanup_interval)
        return next_cleanup

    def start_cleanup_thread(self) -> None:
        """Start the background cleanup thread"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            logger.warning("Cleanup thread is already running")
            return

        logger.info("Starting memory cleanup thread")
        self.stop_cleanup.clear()
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def stop_cleanup_thread(self) -> None:
        """Stop the background cleanup thread"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            logger.info("Stopping memory cleanup thread")
            self.stop_cleanup.set()
            self.cleanup_thread.join(timeout=5)
            if self.cleanup_thread.is_alive():
                logger.warning("Cleanup thread did not stop gracefully")
        self.cleanup_thread = None
