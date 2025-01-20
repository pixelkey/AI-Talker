import threading
import time
import logging
import re
from datetime import datetime
import pytz
from typing import Optional
import traceback

logger = logging.getLogger(__name__)

class MemoryCleanupManager:
    def __init__(self, context, cleanup_interval: int = 3600):  # Default interval: 1 hour
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

    def _evaluate_memory_usefulness(self, memory_content: str) -> bool:
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
            
            # Log the decision and reason
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', result)
            reason = reason_match.group(1) if reason_match else "No reason provided"
            logger.debug(f"Memory evaluation - Keep: {keep_memory}, Reason: {reason}")
            
            return keep_memory
            
        except Exception as e:
            logger.error(f"Error evaluating memory usefulness: {str(e)}")
            return True  # Keep memory if evaluation fails

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

            # Get all documents from vector store
            all_docs = vector_store.docstore.docs
            docs_to_remove = []
            indices_to_remove = []  # Track FAISS indices to remove

            # Check each document for expiry and usefulness
            for doc_id, doc in all_docs.items():
                try:
                    should_remove = False
                    expiry_date = doc.metadata.get('expiry_date')
                    
                    # Check if expired
                    if expiry_date:
                        expiry_datetime = self._parse_timestamp(expiry_date)
                        if expiry_datetime and expiry_datetime <= current_time:
                            should_remove = True
                            
                    # If not expired or no expiry, check usefulness if it's not a long-term memory
                    if not should_remove and doc.metadata.get('type') != 'long_term':
                        if not self._evaluate_memory_usefulness(doc.page_content):
                            should_remove = True
                            logger.info(f"Marking document for removal due to low usefulness: {doc_id}")
                    
                    if should_remove:
                        docs_to_remove.append(doc_id)
                        # Get the index in the FAISS store
                        if hasattr(vector_store, 'index_to_docstore_id'):
                            for i, stored_id in enumerate(vector_store.index_to_docstore_id):
                                if stored_id == doc_id:
                                    indices_to_remove.append(i)
                        logger.info(f"Marking document for removal: {doc_id}")
                        
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {str(e)}")
                    continue

            # Remove expired documents if any found
            if docs_to_remove:
                logger.info(f"Removing {len(docs_to_remove)} expired documents")
                try:
                    # Remove from docstore
                    for doc_id in docs_to_remove:
                        if doc_id in vector_store.docstore.docs:
                            del vector_store.docstore.docs[doc_id]
                    
                    # Remove from FAISS index if indices found
                    if indices_to_remove and hasattr(vector_store, 'index'):
                        # Sort in descending order to remove from end first
                        indices_to_remove.sort(reverse=True)
                        for idx in indices_to_remove:
                            vector_store.index = vector_store._remove_vectors([idx], vector_store.index)
                            # Update the index mapping
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
                    
                    logger.info("Memory cleanup completed and saved to disk")
                except Exception as e:
                    logger.error(f"Error during document removal: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.info("No expired memories found")

            self.last_cleanup_time = current_time

        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            logger.error(traceback.format_exc())

    def _cleanup_loop(self) -> None:
        """Main cleanup loop that runs periodically"""
        while not self.stop_cleanup.is_set():
            try:
                # Only run cleanup if no other important processing is happening
                if not self.context.get('is_processing', False):
                    self.is_cleaning = True
                    self._cleanup_expired_memories()
                    self.is_cleaning = False
                
                # Sleep for the cleanup interval, but check stop flag every second
                for _ in range(self.cleanup_interval):
                    if self.stop_cleanup.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                logger.error(traceback.format_exc())
                self.is_cleaning = False
                time.sleep(60)  # Wait a bit before retrying after error

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
