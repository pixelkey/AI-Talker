# scripts/continuous_listener.py

import speech_recognition as sr
import threading
import time
import logging
import queue
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class ContinuousListener:
    """
    A class to continuously listen for speech input with activation word detection.
    """
    def __init__(self, 
                 activation_word: str = "activate", 
                 deactivation_word: str = "stop",
                 callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the continuous listener.
        
        Args:
            activation_word: Word to activate listening mode (default: "activate")
            deactivation_word: Word to deactivate listening mode (default: "stop")
            callback: Function to call with recognized text when recognized
        """
        self.recognizer = sr.Recognizer()
        # Configure pause threshold for longer pauses between words/phrases
        self.recognizer.pause_threshold = 2.0  # Default is 0.8
        self.recognizer.phrase_threshold = 0.3  # Default is 0.3
        self.recognizer.non_speaking_duration = 1.0  # Default is 0.5
        
        self.activation_word = activation_word.lower()
        self.deactivation_word = deactivation_word.lower()
        self.callback = callback
        self.is_listening = False
        self.is_active = False
        self.stop_requested = False
        self.thread = None
        self.response_queue = queue.Queue()
        self.active_thread = None
        
    def _adjust_for_ambient_noise(self, source):
        """Adjust recognizer sensitivity for ambient noise level."""
        logger.info("Adjusting for ambient noise...")
        self.recognizer.adjust_for_ambient_noise(source, duration=1)
        logger.info("Ambient noise adjustment complete")
        
    def listen_for_activation(self):
        """Background thread that listens for the activation word."""
        try:
            with sr.Microphone() as source:
                self._adjust_for_ambient_noise(source)
                
                logger.info(f"Waiting for activation word: '{self.activation_word}'")
                while not self.stop_requested:
                    try:
                        # Short timeout for checking stop_requested frequently
                        # Using only timeout and phrase_time_limit parameters
                        audio = self.recognizer.listen(
                            source, 
                            timeout=2, 
                            phrase_time_limit=3
                        )
                        
                        try:
                            text = self.recognizer.recognize_google(audio).lower()
                            logger.debug(f"Heard: {text}")
                            
                            # Check if activation word is detected
                            if self.activation_word in text:
                                logger.info(f"Activation word detected: '{text}'")
                                self.start_active_listening()
                                break
                                
                        except sr.UnknownValueError:
                            # Speech not recognized - normal, continue listening
                            pass
                        except sr.RequestError as e:
                            logger.error(f"Google Speech Recognition service error: {e}")
                            time.sleep(2)  # Wait before retrying
                            
                    except sr.WaitTimeoutError:
                        # Timeout is expected - allows checking stop_requested periodically
                        pass
                        
        except Exception as e:
            logger.error(f"Error in activation listener: {e}")
            self.stop_requested = True
            
    def start_active_listening(self):
        """Start the active listening mode after activation word is detected."""
        if self.active_thread and self.active_thread.is_alive():
            logger.info("Active listening is already running")
            return
        
        self.is_active = True
        self.active_thread = threading.Thread(target=self._active_listening_loop)
        self.active_thread.daemon = True
        self.active_thread.start()
        
    def _active_listening_loop(self):
        """Active listening loop that captures user speech until deactivation word."""
        try:
            with sr.Microphone() as source:
                self._adjust_for_ambient_noise(source)
                
                logger.info("Active listening mode started. Say something...")
                while self.is_active and not self.stop_requested:
                    self.is_listening = True
                    logger.info("Listening for input... (say 'stop' to end conversation)")
                    
                    try:
                        # Using only timeout and phrase_time_limit parameters
                        # Increased phrase_time_limit to 30 seconds for longer sentences
                        audio = self.recognizer.listen(
                            source, 
                            timeout=5, 
                            phrase_time_limit=30
                        )
                        
                        try:
                            text = self.recognizer.recognize_google(audio).lower()
                            logger.info(f"Recognized: {text}")
                            
                            # Check for deactivation word
                            if self.deactivation_word in text:
                                logger.info("Deactivation word detected, ending active listening")
                                self.is_active = False
                                self.is_listening = False
                                # Restart activation word listener
                                self.restart_activation_listener()
                                break
                                
                            # Process the recognized text if not the deactivation word
                            if self.callback:
                                logger.info(f"Submitting recognized text: '{text}'")
                                self.callback(text)
                                
                                # Wait for response before continuing
                                logger.info("Waiting for response to complete...")
                                try:
                                    # Wait with timeout for response signal
                                    self.response_queue.get(timeout=30)
                                    logger.info("Response completed, ready for next input")
                                except queue.Empty:
                                    logger.warning("Timed out waiting for response, continuing anyway")
                            
                        except sr.UnknownValueError:
                            logger.info("Could not understand audio")
                        except sr.RequestError as e:
                            logger.error(f"Google Speech Recognition service error: {e}")
                            
                    except sr.WaitTimeoutError:
                        # No speech detected in timeout period
                        logger.info("No speech detected, continuing to listen...")
                    
                    self.is_listening = False
                    
        except Exception as e:
            logger.error(f"Error in active listening loop: {e}")
            self.is_active = False
            self.is_listening = False
            self.restart_activation_listener()
            
    def restart_activation_listener(self):
        """Restart the activation word listener after deactivation."""
        if not self.stop_requested:
            self.thread = threading.Thread(target=self.listen_for_activation)
            self.thread.daemon = True
            self.thread.start()
            
    def signal_response_complete(self):
        """Signal that response processing is complete and ready for next input."""
        try:
            self.response_queue.put(True, block=False)
        except queue.Full:
            # Queue already has an item, which is fine
            pass
            
    def start(self):
        """Start the continuous listener in a background thread."""
        if self.thread and self.thread.is_alive():
            logger.info("Listener is already running")
            return
            
        self.stop_requested = False
        self.thread = threading.Thread(target=self.listen_for_activation)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Continuous listener started")
        
    def stop(self):
        """Stop the continuous listener."""
        logger.info("Stopping continuous listener...")
        self.stop_requested = True
        self.is_active = False
        self.is_listening = False
        # Wait for threads to end naturally
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.active_thread and self.active_thread.is_alive():
            self.active_thread.join(timeout=2)
        logger.info("Continuous listener stopped")
