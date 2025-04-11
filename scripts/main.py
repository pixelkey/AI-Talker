# scripts/main.py

import torch
import gc
import logging
import time
from datetime import datetime
import threading
from initialize import initialize_model_and_retrieval
from continuous_listener import ContinuousListener
from tts_utils import TTSManager
from chatbot_functions import chatbot_response, retrieve_and_format_references
from vector_store_client import VectorStoreClient
from document_processing import normalize_text
from embedding_updater import EmbeddingUpdater
from self_reflection import SelfReflection
from memory_cleanup import MemoryCleanupManager
from ingest_watcher import IngestWatcher
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear CUDA memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

class VoiceChatbot:
    """Backend-only voice chatbot that uses activation/deactivation words"""
    
    def __init__(self, context):
        """Initialize the voice chatbot"""
        logger.info("Initializing voice chatbot...")
        
        # Store context
        self.context = context
        
        # Initialize components that were in the Gradio interface
        self._initialize_components()
        
        # Initialize history
        self.history = []
        
        # Initialize continuous listener
        activation_word = self.context.get('ACTIVATION_WORD', 'activate')
        deactivation_word = self.context.get('DEACTIVATION_WORD', 'stop')
        
        logger.info(f"Setting up continuous listener with activation word: '{activation_word}' and deactivation word: '{deactivation_word}'")
        self.continuous_listener = ContinuousListener(
            activation_word=activation_word,
            deactivation_word=deactivation_word,
            callback=self.handle_speech_input
        )
        self.context['continuous_listener'] = self.continuous_listener
        
        # Flag to track if we're running
        self.running = False
        
    def _initialize_components(self):
        """Initialize all the components that were in the Gradio interface"""
        # Initialize TTS at startup
        tts_manager = TTSManager(self.context)
        self.context['tts_manager'] = tts_manager  # Add TTS manager to context
        self.tts_manager = tts_manager
        
        # Initialize vector store client
        vector_store_client = VectorStoreClient(
            self.context['vector_store'],
            self.context['embeddings'],
            normalize_text
        )
        self.context['vector_store_client'] = vector_store_client
        
        # Initialize Ollama client for local LLM if needed
        if self.context.get('MODEL_SOURCE') == 'local':
            self.context['client'] = ollama
            
        # Initialize embedding updater
        embedding_updater = EmbeddingUpdater(self.context)
        self.context['embedding_updater'] = embedding_updater
        
        # Initialize self reflection
        self_reflection = SelfReflection(self.context)
        self.context['self_reflection'] = self_reflection
        
        # Don't start reflection thread here, we'll process reflections manually
        # self_reflection.start_reflection_thread()
        
        # Initialize memory cleanup manager
        memory_cleanup = MemoryCleanupManager(self.context)
        self.context['memory_cleanup'] = memory_cleanup
        memory_cleanup.start_cleanup_thread()
        
        # Create and set up ingest watcher
        watcher = IngestWatcher(embedding_updater.update_embeddings)
        self.context['watcher'] = watcher
        
    def parse_timestamp(self, timestamp):
        """Parse timestamp string to datetime object"""
        if isinstance(timestamp, datetime):
            return timestamp
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")
        
    def handle_speech_input(self, text):
        """Process speech input and generate response"""
        try:
            logger.info(f"Processing speech input: '{text}'")
            
            # Notify self-reflection about user input
            if 'self_reflection' in self.context:
                self.context['self_reflection'].notify_user_input()
            
            # Set current time
            current_time = datetime.now().isoformat()
            self.context['current_time'] = current_time
            
            # Get references
            refs, filtered_docs, context_documents = retrieve_and_format_references(text, self.context)
            
            # Generate response
            _, response, refs, _ = chatbot_response(text, context_documents, self.context, self.history)
            
            # Format messages
            dt = datetime.now()
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            user_msg = f"[{formatted_time}] User: {text}"
            bot_msg = f"[{formatted_time}] Bot: {response}"
            
            # Update history
            new_history = self.history.copy()
            new_history.append((user_msg, bot_msg))
            self.history = new_history
            
            # Log the interaction
            logger.info(f"User: {text}")
            logger.info(f"Bot: {response}")
            
            if refs:
                logger.info("References:")
                logger.info(refs)
            
            # Generate speech for the response
            logger.info("Generating TTS response...")
            tts_text = self.context.get('tts_response', response)
            logger.info(f"DEBUG - Original response: {response[:100]}...")
            logger.info(f"DEBUG - TTS text with emotion: {tts_text[:100]}...")
            audio_path = self.tts_manager.text_to_speech(tts_text)
            logger.info(f"TTS generation complete: {audio_path}")
            
            # Wait for TTS to fully complete before continuing
            while self.tts_manager.is_processing:
                time.sleep(0.1)
            
            # Process reflection asynchronously after TTS is done
            # Use a separate thread to avoid blocking the main flow
            if 'self_reflection' in self.context:
                # Create a copy of history to prevent modifications affecting the conversation
                history_copy = list(self.history)
                threading.Thread(
                    target=self._process_reflection_safely,
                    args=(history_copy,),
                    daemon=True
                ).start()
            
            # Signal continuous listener that response is complete
            self.continuous_listener.signal_response_complete()
            
        except Exception as e:
            logger.error(f"Error processing speech input: {e}", exc_info=True)
            self.continuous_listener.signal_response_complete()

    def _process_reflection_safely(self, history_copy):
        """Process reflection in a safe manner that doesn't affect the main conversation flow"""
        try:
            # Use the start_reflection method with enhanced logging, passing the latest exchange
            logger.info("Starting self-reflection processing...")
            self.context['self_reflection'].start_reflection(history_copy)
            logger.info("Self-reflection triggered successfully")
        except Exception as e:
            logger.error(f"Error in reflection processing: {e}", exc_info=True)
    
    def start(self):
        """Start the voice chatbot"""
        if self.running:
            logger.info("Voice chatbot is already running")
            return
        
        try:
            logger.info("Starting voice chatbot...")
            self.running = True
            
            # Start continuous listener
            self.continuous_listener.start()
            
            # Keep main thread alive
            logger.info("Voice chatbot started and ready")
            logger.info(f"Say '{self.context.get('ACTIVATION_WORD', 'activate')}' to begin a conversation")
            logger.info(f"Say '{self.context.get('DEACTIVATION_WORD', 'stop')}' to end a conversation")
            
            # Keep application running until keyboard interrupt
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, shutting down...")
                self.stop()
                
        except Exception as e:
            logger.error(f"Error starting voice chatbot: {e}", exc_info=True)
            self.stop()
    
    def stop(self):
        """Stop the voice chatbot"""
        logger.info("Stopping voice chatbot...")
        self.running = False
        
        # Stop continuous listener
        if self.continuous_listener:
            self.continuous_listener.stop()
            
        # Stop other background threads
        if 'self_reflection' in self.context:
            self.context['self_reflection'].stop_reflection.set()
            
        logger.info("Voice chatbot stopped")

def main():
    # Clear GPU memory at startup
    clear_gpu_memory()
    
    # Initialize the model, embeddings, and retrieval components
    context = initialize_model_and_retrieval()

    # Initialize and start voice chatbot
    voice_chatbot = VoiceChatbot(context)
    voice_chatbot.start()

if __name__ == "__main__":
    main()
