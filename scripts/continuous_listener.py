# scripts/continuous_listener.py

import speech_recognition as sr
import threading
import time
import queue
from typing import Optional, Callable
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class ContinuousListener:
    """A class to continuously listen for speech input with activation word detection."""
    
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
        
        # TTS state tracking
        self.is_tts_playing = False        # Flag for when TTS is actively playing
        self.last_tts_end_time = 0         # When TTS last ended
        self.tts_cooldown_period = 0.3     # Very short cooldown after TTS ends
    
    def _adjust_for_ambient_noise(self, source):
        """Adjust recognizer sensitivity for ambient noise level."""
        logger.info("Adjusting for ambient noise...")
        try:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Ambient noise adjustment complete")
        except Exception as e:
            logger.error(f"Error adjusting for ambient noise: {e}")
            
    def notify_tts_started(self):
        """Notify that TTS playback has started."""
        logger.info("TTS playback started - pausing recognition")
        self.is_tts_playing = True
        
    def notify_tts_finished(self):
        """Notify that TTS playback has finished."""
        logger.info("TTS playback finished - resuming recognition after short delay")
        self.is_tts_playing = False
        self.last_tts_end_time = time.time()
        
    def listen_for_activation(self):
        """Background thread that listens for the activation word."""
        try:
            with sr.Microphone() as source:
                self._adjust_for_ambient_noise(source)
                
                logger.info(f"Waiting for activation word: '{self.activation_word}'")
                while not self.stop_requested:
                    # Add a short delay to prevent CPU overuse
                    time.sleep(0.05)
                    
                    # Skip listening if TTS is playing
                    if self.is_tts_playing:
                        continue
                    
                    # Also skip if we just finished TTS playback (slight cooldown)
                    time_since_tts_end = time.time() - self.last_tts_end_time
                    if time_since_tts_end < self.tts_cooldown_period:
                        continue
                    
                    try:
                        audio = self.recognizer.listen(source, phrase_time_limit=5, timeout=1)
                        try:
                            text = self.recognizer.recognize_google(audio).lower()
                            logger.debug(f"Heard: {text}")
                            
                            if self.activation_word in text:
                                logger.info(f"Activation word detected: '{text}'")
                                self._start_active_listening()
                                break  # Exit loop once activated
                                
                        except sr.UnknownValueError:
                            # Speech was unintelligible
                            pass
                        except sr.RequestError as e:
                            logger.error(f"Google Speech Recognition service error: {e}")
                            
                    except sr.WaitTimeoutError:
                        # No speech detected within timeout
                        pass
        except Exception as e:
            logger.error(f"Error in listen_for_activation: {e}")
        
    def _start_active_listening(self):
        """Start the active listening mode after activation word is detected."""
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
                    
                    # Skip listening if TTS is playing
                    if self.is_tts_playing:
                        logger.debug("TTS is playing - skipping recognition")
                        time.sleep(0.1)
                        continue
                        
                    # Also skip if we just finished TTS playback (brief cooldown)
                    time_since_tts_end = time.time() - self.last_tts_end_time
                    if time_since_tts_end < self.tts_cooldown_period:
                        logger.debug(f"In TTS cooldown ({time_since_tts_end:.2f}s < {self.tts_cooldown_period}s)")
                        time.sleep(0.1)
                        continue
                    
                    try:
                        # Listen for user speech with reasonable timeouts
                        audio = self.recognizer.listen(
                            source, 
                            timeout=5, 
                            phrase_time_limit=30
                        )
                        
                        try:
                            text = self.recognizer.recognize_google(audio).lower()
                            logger.info(f"Recognized: {text}")
                            
                            # Check for deactivation word to end conversation
                            if text == self.deactivation_word:
                                logger.info("Deactivation word detected, ending conversation")
                                self.is_active = False
                                self.is_listening = False
                                self._start_listen_for_activation_thread()
                                break
                            
                            # Send recognized text to callback
                            if self.callback:
                                logger.info(f"Submitting recognized text: '{text}'")
                                self.callback(text)
                                
                                # Wait for response to complete before listening again
                                logger.info("Waiting for response to complete...")
                                self._wait_for_response_complete()
                                logger.info("Response completed, ready for next input")
                            
                        except sr.UnknownValueError:
                            logger.info("Could not understand audio")
                        except sr.RequestError as e:
                            logger.error(f"Google Speech Recognition service error: {e}")
                    
                    except sr.WaitTimeoutError:
                        logger.info("No speech detected, continuing to listen...")
                    
                    except Exception as e:
                        logger.error(f"Error in active listening loop: {e}")
                        
        except Exception as e:
            logger.error(f"Error in active listening loop: {e}")
            
    def _wait_for_response_complete(self):
        """Wait for a response to be placed in the queue."""
        try:
            self.response_queue.get(timeout=60)  # Wait up to 60 seconds for response
        except queue.Empty:
            logger.warning("Response timeout - continuing without waiting")
            
    def notify_response_complete(self):
        """Mark that a response is complete and we can listen again."""
        self.response_queue.put(True)
        
    def _start_listen_for_activation_thread(self):
        """Start the thread that listens for the activation word."""
        self.thread = threading.Thread(target=self.listen_for_activation)
        self.thread.daemon = True
        self.thread.start()
        
    def start(self):
        """Start the continuous listener thread."""
        self._start_listen_for_activation_thread()
        
    def stop(self):
        """Stop the continuous listener."""
        logger.info("Stopping continuous listener...")
        self.stop_requested = True
        
        # Wait for any active threads to finish
        if self.active_thread and self.active_thread.is_alive():
            try:
                self.active_thread.join(timeout=2)
            except Exception as e:
                logger.error(f"Error joining active thread: {e}")
                
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=2)
            except Exception as e:
                logger.error(f"Error joining listener thread: {e}")
                
        self.is_active = False
        self.is_listening = False
