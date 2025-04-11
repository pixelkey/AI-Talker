# scripts/voice_chatbot.py

import os
import logging
import time
from datetime import datetime
import threading
from continuous_listener import ContinuousListener
from tts_utils import TTSManager
from chatbot_functions import chatbot_response, retrieve_and_format_references
from initialize import initialize_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceChatbot:
    """Backend-only voice chatbot that uses activation/deactivation words"""
    
    def __init__(self):
        """Initialize the voice chatbot"""
        logger.info("Initializing voice chatbot...")
        
        # Initialize context and components
        self.context = initialize_context()
        
        # Initialize TTS
        self.tts_manager = TTSManager(self.context)
        self.context['tts_manager'] = self.tts_manager
        
        # Initialize history
        self.history = []
        
        # Initialize continuous listener
        activation_word = self.context.get('ACTIVATION_WORD', 'activate')
        deactivation_word = self.context.get('DEACTIVATION_WORD', 'stop')
        
        logger.info(f"Setting up continuous listener with activation word: '{activation_word}' and deactivation word: '{deactivation_word}'")
        self.continuous_listener = ContinuousListener(
            activation_word=activation_word,
            deactivation_word=deactivation_word,
            on_speech_recognized=self.handle_speech_input
        )
        self.context['continuous_listener'] = self.continuous_listener
        
        # Connect TTS to continuous listener
        self.tts_manager.continuous_listener = self.continuous_listener
        self.context['processing_text_prompt'] = False
        
        # Flag to track if we're running
        self.running = False
        
    def handle_speech_input(self, text):
        """Process speech input and generate response"""
        try:
            logger.info(f"Processing speech input: '{text}'")
            
            # Set current time
            current_time = datetime.now().isoformat()
            self.context['current_time'] = current_time
            
            # Get references
            refs, filtered_docs, context_documents = retrieve_and_format_references(text, self.context)
            
            # Generate response
            _, response, refs, _ = chatbot_response(text, context_documents, self.context, self.history)
            
            # Format messages
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_msg = f"[{formatted_time}] User: {text}"
            bot_msg = f"[{formatted_time}] Bot: {response}"
            
            # Update history
            self.history.append((user_msg, bot_msg))
            
            # Log the interaction
            logger.info(f"User: {text}")
            logger.info(f"Bot: {response}")
            
            if refs:
                logger.info("References:")
                logger.info(refs)
            
            # Generate TTS
            logger.info("Generating TTS response...")
            
            # Debug logs for better tracking
            logger.info(f"DEBUG - Original response: {response[:100]}...")
            
            # Add emotion to the text if detected
            tts_text = response
            if 'emotion' in self.context:
                tts_text = f"[{self.context['emotion']}] {response}"
                logger.info(f"DEBUG - TTS text with emotion: {tts_text[:100]}...")
            
            # Generate TTS
            self.tts_manager.text_to_speech(tts_text)
            
            # Signal to the listener that response processing is complete
            self.continuous_listener.signal_response_complete()
            
        except Exception as e:
            logger.error(f"Error processing speech input: {e}")
            # Still signal completion even on error
            try:
                self.continuous_listener.signal_response_complete()
            except:
                pass
    
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
        
        logger.info("Voice chatbot stopped")

def main():
    """Main function to run the voice chatbot"""
    voice_chatbot = VoiceChatbot()
    voice_chatbot.start()

if __name__ == "__main__":
    main()
