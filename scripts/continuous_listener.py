# scripts/continuous_listener.py

import speech_recognition as sr
import threading
import time
import queue
from typing import Optional, Callable
import logging
import numpy as np
import os
import sounddevice as sd
import soundfile as sf

# Configure logging
logger = logging.getLogger(__name__)

class ContinuousListener:
    """A class to continuously listen for speech input with activation word detection."""
    
    def __init__(self, 
                 activation_word: str = "activate", 
                 deactivation_word: str = "stop",
                 callback: Optional[Callable[[str], None]] = None,
                 on_speech_recognized: Optional[Callable[[str], None]] = None):
        """
        Initialize the continuous listener.
        
        Args:
            activation_word: Word to activate listening mode (default: "activate")
            deactivation_word: Word to deactivate listening mode (default: "stop")
            callback: Function to call with recognized text when recognized
            on_speech_recognized: Function to call with recognized text when recognized (new callback)
        """
        self.recognizer = sr.Recognizer()
        # Configure pause threshold for longer pauses between words/phrases
        self.recognizer.pause_threshold = 2  # Default is 0.8
        self.recognizer.phrase_threshold = 0.3  # Default is 0.3
        self.recognizer.non_speaking_duration = 0.5  # Default is 0.5
        
        self.activation_word = activation_word.lower()
        self.deactivation_word = deactivation_word.lower()
        self.callback = callback
        self.on_speech_recognized = on_speech_recognized
        self.is_listening = False
        self.is_active = False
        self.stop_requested = False
        self.thread = None
        self.response_queue = queue.Queue()
        self.active_thread = None
        
        # TTS state tracking
        self.is_tts_playing = False        # Flag for when TTS is actively playing
        self.last_tts_end_time = 0         # When TTS last ended
        self.tts_cooldown_period = 1.0     # Longer cooldown after TTS ends (was 0.3)
        
        # Keep track of recent TTS output to detect feedback loops
        self.recent_tts_phrases = []       # Store phrases recently spoken by TTS
        self.max_recent_phrases = 5        # Maximum number of phrases to remember
        
        # Response tracking
        self.last_response_time = 0        # Timestamp for the last response
        
        # Sound effect paths
        self.sound_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sounds")
        self.activation_sound = os.path.join(self.sound_dir, "activation.wav")
        self.deactivation_sound = os.path.join(self.sound_dir, "deactivation.wav")
        self.recognition_sound = os.path.join(self.sound_dir, "recognition.wav")
        self.searching_sound = os.path.join(self.sound_dir, "searching.wav")
        self.self_reflection_sound = os.path.join(self.sound_dir, "self_reflection.wav")
        
        # Create sound directory if it doesn't exist
        os.makedirs(self.sound_dir, exist_ok=True)
        
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
        
    def notify_tts_content(self, text):
        """
        Store the content of TTS to avoid recognizing it as user speech.
        
        Args:
            text: The text being converted to speech
        """
        try:
            # Log the notification
            logger.info(f"TTS content notification received: '{text[:50]}...'")
            
            # Store in recent TTS phrases queue
            self.recent_tts_phrases.append(text.lower())
            
            # Trim the queue if it gets too long
            if len(self.recent_tts_phrases) > self.max_recent_phrases:
                self.recent_tts_phrases.pop(0)
                
            # Set TTS as playing
            self.is_tts_playing = True
            self.last_tts_content = text.lower()
            
            logger.info(f"Stored TTS content ({len(self.recent_tts_phrases)} phrases in memory)")
        except Exception as e:
            logger.error(f"Error in notify_tts_content: {e}")
            
    def set_tts_playing(self, is_playing):
        """
        Directly set the TTS playing state.
        
        Args:
            is_playing: Boolean indicating if TTS is currently playing
        """
        try:
            self.is_tts_playing = is_playing
            
            # If TTS just ended, update the last end time
            if not is_playing:
                self.last_tts_end_time = time.time()
                logger.info(f"TTS playback ended, setting cooldown until {self.last_tts_end_time + self.tts_cooldown_period:.2f}s")
            else:
                logger.info("TTS playback started")
                
        except Exception as e:
            logger.error(f"Error in set_tts_playing: {e}")
            
    def signal_response_complete(self):
        """
        Signal that a response has been completed.
        This helps with tracking conversation state and avoiding feedback loops.
        """
        logger.info("Response processing completed")
        # Set a timestamp for the last response
        self.last_response_time = time.time()
        # Ensure we're not in TTS mode
        self.is_tts_playing = False
        # Reset last TTS end time to avoid unnecessary cooldown
        self.last_tts_end_time = time.time()
            
    def should_ignore_audio(self):
        """
        Determine if audio should be ignored (during TTS playback or cooldown period).
        
        Returns:
            bool: True if audio should be ignored, False otherwise
        """
        try:
            # If TTS is currently playing, ignore audio
            if self.is_tts_playing:
                logger.debug("Ignoring audio: TTS is currently playing")
                return True
                
            # If we're still in the cooldown period after TTS, ignore audio
            time_since_tts = time.time() - self.last_tts_end_time
            if time_since_tts < self.tts_cooldown_period:
                logger.debug(f"Ignoring audio: Within TTS cooldown period ({time_since_tts:.2f}/{self.tts_cooldown_period:.2f}s)")
                return True
                
            # No reason to ignore, process audio normally
            return False
            
        except Exception as e:
            logger.error(f"Error in should_ignore_audio: {e}")
            # Default to not ignoring audio on error
            return False
            
    def _is_likely_self_speech(self, text):
        """
        Check if the recognized text is likely the system's own speech being picked up.
        
        Args:
            text: The recognized text to check
            
        Returns:
            bool: True if the text is likely from TTS output, False otherwise
        """
        # If we have no recent TTS phrases, it can't be self-speech
        if not self.recent_tts_phrases:
            return False
            
        # Normalize text for comparison (lowercase, strip punctuation)
        import string
        normalized_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        
        # Check if the recognized text is a substring of any recent TTS output
        # or if any recent TTS output is a substring of the recognized text
        for phrase in self.recent_tts_phrases:
            norm_phrase = phrase.lower().translate(str.maketrans('', '', string.punctuation))
            
            # Check for significant overlap between the texts
            # Either one contains the other, or they share many words
            if norm_phrase in normalized_text or normalized_text in norm_phrase:
                return True
                
            # Check for word overlap
            text_words = set(normalized_text.split())
            phrase_words = set(norm_phrase.split())
            common_words = text_words.intersection(phrase_words)
            
            # If more than 75% of words match, likely self-speech
            if len(common_words) >= len(text_words) * 0.75:
                return True
                
        return False
        
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
                                self.play_activation_sound()
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
        """Actively listen and respond to user speech."""
        logger.info("Starting active listening loop")
        
        with sr.Microphone() as source:
            # Adjust for ambient noise
            self._adjust_for_ambient_noise(source)
            
            while self.is_active and not self.stop_requested:
                try:
                    # Check if we should be ignoring audio input right now
                    if self.should_ignore_audio():
                        # Short sleep to avoid busy waiting, then check again
                        time.sleep(0.1)
                        continue
                        
                    logger.info("Listening for speech...")
                    try:
                        # Listen with timeout to avoid blocking indefinitely
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=30)
                    except sr.WaitTimeoutError:
                        logger.debug("No speech detected, continuing")
                        continue
                    
                    # Wait before processing to ensure self-speech detection is accurate
                    time.sleep(0.1)
                    
                    # Check again if we should ignore audio (TTS might have started during listen)
                    if self.should_ignore_audio():
                        logger.debug("TTS started during listen, discarding audio")
                        continue
                    
                    try:
                        # Process the audio
                        text = self.recognizer.recognize_google(audio).lower()
                        
                        # Skip empty or extremely short text 
                        if not text or len(text.strip()) <= 1:
                            logger.debug(f"Skipping empty or very short text: '{text}'")
                            continue
                            
                        # If this looks like the TTS system's own speech, ignore it
                        if self._is_likely_self_speech(text):
                            logger.info(f"Ignoring likely self-speech: '{text}'")
                            continue
                            
                        logger.info(f"Recognized: {text}")
                        
                        # Play a beep to indicate speech recognition
                        self.play_recognition_sound()
                        
                        # Process the recognized text
                        logger.info(f"Submitting recognized text: '{text}'")
                        if self.on_speech_recognized:
                            self.on_speech_recognized(text)
                        elif self.callback:
                            self.callback(text)
                            
                        # Check for deactivation word to end conversation
                        if self.deactivation_word and text.strip() == self.deactivation_word:
                            logger.info("Deactivation word detected, ending conversation")
                            self.play_deactivation_sound()
                            self.is_active = False
                            self._start_listen_for_activation_thread()
                            break
                                
                    except sr.UnknownValueError:
                        logger.debug("Speech not recognized")
                    except sr.RequestError as e:
                        logger.error(f"Could not request results: {e}")
                    
                except Exception as e:
                    logger.error(f"Error in active listening loop: {e}")
                    time.sleep(1)  # Sleep on error to avoid tight loop
                    
        logger.info("Exiting active listening loop")
        
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
        
    def play_activation_sound(self):
        """Play a sound effect when voice activation occurs"""
        try:
            if os.path.exists(self.activation_sound):
                logger.info("Playing activation sound")
                # Add the activation message to recent_tts_phrases to prevent recognition
                activation_message = "Hello, I'm listening. How can I help you?"
                self.recent_tts_phrases.append(activation_message.lower())
                if len(self.recent_tts_phrases) > self.max_recent_phrases:
                    self.recent_tts_phrases.pop(0)
                
                # Play the sound file
                data, samplerate = sf.read(self.activation_sound)
                sd.play(data, samplerate)
            else:
                # Generate a simple beep if sound file doesn't exist
                logger.info("Activation sound file not found, generating beep")
                samplerate = 44100
                t = np.linspace(0, 0.3, int(0.3 * samplerate), False)
                tone = 0.3 * np.sin(2 * np.pi * 880 * t)  # A5 tone
                sd.play(tone, samplerate)
        except Exception as e:
            logger.error(f"Error playing activation sound: {e}")
    
    def play_deactivation_sound(self):
        """Play a sound effect when voice deactivation occurs"""
        try:
            if os.path.exists(self.deactivation_sound):
                logger.info("Playing deactivation sound")
                # Add the deactivation message to recent_tts_phrases to prevent recognition
                deactivation_message = "I'll stop listening now."
                self.recent_tts_phrases.append(deactivation_message.lower())
                if len(self.recent_tts_phrases) > self.max_recent_phrases:
                    self.recent_tts_phrases.pop(0)
                
                # Play the sound file
                data, samplerate = sf.read(self.deactivation_sound)
                sd.play(data, samplerate)
            else:
                # Generate a simple beep if sound file doesn't exist
                logger.info("Deactivation sound file not found, generating beep")
                samplerate = 44100
                t = np.linspace(0, 0.3, int(0.3 * samplerate), False)
                tone = 0.2 * np.sin(2 * np.pi * 440 * t)  # A4 tone
                sd.play(tone, samplerate)
        except Exception as e:
            logger.error(f"Error playing deactivation sound: {e}")
            
    def play_recognition_sound(self):
        """Play a sound effect when speech is recognized"""
        try:
            if os.path.exists(self.recognition_sound):
                logger.info("Playing recognition sound")
                # Add the recognition message to recent_tts_phrases to prevent recognition
                recognition_message = "hmm"
                self.recent_tts_phrases.append(recognition_message.lower())
                if len(self.recent_tts_phrases) > self.max_recent_phrases:
                    self.recent_tts_phrases.pop(0)
                
                # Play the sound file
                data, samplerate = sf.read(self.recognition_sound)
                sd.play(data, samplerate)
            else:
                # Generate a simple beep if sound file doesn't exist
                logger.info("Recognition sound file not found, generating beep")
                samplerate = 44100
                t = np.linspace(0, 0.2, int(0.2 * samplerate), False)
                tone = 0.2 * np.sin(2 * np.pi * 660 * t)  # E5 tone
                sd.play(tone, samplerate)
        except Exception as e:
            logger.error(f"Error playing recognition sound: {e}")
            
    def play_searching_sound(self):
        """Play a sound effect when web search is initiated"""
        try:
            if os.path.exists(self.searching_sound):
                logger.info("Playing searching sound")
                # Add the searching message to recent_tts_phrases to prevent recognition
                searching_message = "Ok, just wait a few moments while I search the web."
                self.recent_tts_phrases.append(searching_message.lower())
                if len(self.recent_tts_phrases) > self.max_recent_phrases:
                    self.recent_tts_phrases.pop(0)
                
                # Play the sound file
                data, samplerate = sf.read(self.searching_sound)
                sd.play(data, samplerate)
            else:
                # Generate a simple beep if sound file doesn't exist
                logger.info("Searching sound file not found, generating beep")
                samplerate = 44100
                t = np.linspace(0, 0.3, int(0.3 * samplerate), False)
                # Create a slightly rising tone for searching
                freqs = np.linspace(440, 587, int(0.3 * samplerate))  # A4 to D5
                tone = 0.3 * np.sin(2 * np.pi * np.cumsum(freqs) / samplerate)
                sd.play(tone, samplerate)
        except Exception as e:
            logger.error(f"Error playing searching sound: {e}")
            
    def play_self_reflection_sound(self):
        """Play a sound effect when self-reflection is initiated"""
        try:
            if os.path.exists(self.self_reflection_sound):
                logger.info("Playing self-reflection sound")
                # Add the self-reflection message to recent_tts_phrases to prevent recognition
                self_reflection_message = "hmmm interesting."
                self.recent_tts_phrases.append(self_reflection_message.lower())
                if len(self.recent_tts_phrases) > self.max_recent_phrases:
                    self.recent_tts_phrases.pop(0)
                
                # Play the sound file
                data, samplerate = sf.read(self.self_reflection_sound)
                sd.play(data, samplerate)
            else:
                # Generate a simple beep if sound file doesn't exist
                logger.info("Self-reflection sound file not found, generating beep")
                samplerate = 44100
                t = np.linspace(0, 0.4, int(0.4 * samplerate), False)
                # Create a complex tone for self-reflection - a chord with a bit of wavering
                base_freq = 330  # E4
                # Create a chord (E minor)
                tone1 = 0.15 * np.sin(2 * np.pi * base_freq * t)  # E4
                tone2 = 0.15 * np.sin(2 * np.pi * (base_freq * 1.2) * t)  # G4
                tone3 = 0.15 * np.sin(2 * np.pi * (base_freq * 1.5) * (t + 0.05 * np.sin(2 * t)))  # B4 with slight wavering
                
                # Combine the tones
                tone = tone1 + tone2 + tone3
                sd.play(tone, samplerate)
        except Exception as e:
            logger.error(f"Error playing self-reflection sound: {e}")
        
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
