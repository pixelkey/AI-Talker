import sys
sys.path.append('./csm')
import torch
import torchaudio
import os
import tempfile
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import logging
from typing import Tuple
import simpleaudio as sa
from generator import load_csm_1b

# Configure logging
logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self, context):
        """Initialize TTS Manager with context"""
        self.context = context
        self.generator = None
        self.device = self.get_device()
        self.speaker = 0  # Default speaker ID; can be made configurable
        self.sample_rate = 24000  # Default sample rate for CSM
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.playback_thread.start()
        self.initialize_tts()

    def get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def initialize_tts(self):
        print("\n=== Initializing Sesame CSM TTS ===")
        try:
            self.generator = load_csm_1b(device=self.device)
            self.sample_rate = self.generator.sample_rate
            print(f"Sesame CSM TTS initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing Sesame CSM TTS: {e}")
            import traceback
            traceback.print_exc()

    def text_to_speech(self, text):
        """Convert text to speech using Sesame CSM TTS."""
        logger = logging.getLogger(__name__)
        logger.info("\n=== Starting text_to_speech (CSM) ===")
        self.is_processing = True
        self.context['is_processing'] = True
        self.notify_continuous_listener_start(text)
        try:
            if self.generator is None:
                self.initialize_tts()
            if self.generator is None:
                raise RuntimeError("Failed to initialize Sesame CSM TTS system")
            # You may want to parse for style cues here and map to speaker if needed
            # For now, just use self.speaker and no context
            audio = self.generator.generate(
                text=text,
                speaker=self.speaker,
                context=[],
                max_audio_length_ms=10000,
            )
            # Save audio to temp file and queue for playback
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                temp_path = fp.name
                torchaudio.save(temp_path, audio.unsqueeze(0).cpu(), self.sample_rate)
                print(f"Audio saved to {temp_path}")
            # Play asynchronously
            audio_data = audio.cpu().numpy()
            self.play_audio_async(audio_data)
            return temp_path
        except Exception as e:
            print(f"Error in Sesame CSM text-to-speech: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.is_processing = False
            self.context['is_processing'] = False
            self.notify_continuous_listener_end()

    def _process_audio_queue(self):
        """Process audio data from the queue and play it."""
        play_obj = None
        
        while True:
            # Get audio data from the queue
            audio_data = self.audio_queue.get()
            if audio_data is None:  # Signal to stop
                break
                
            try:
                # Notify listener that TTS playback is starting
                self._notify_playback_start()
                
                # Convert to int16 and play the audio
                audio_int16 = (audio_data * 32767).astype(np.int16)  
                play_obj = sa.play_buffer(
                    audio_int16,
                    num_channels=1,
                    bytes_per_sample=2,
                    sample_rate=self.sample_rate
                )
                
                # Wait until audio playback is finished
                play_obj.wait_done()
                
                # Add a small buffer period after playback
                time.sleep(0.1)
            
                # Notify listener that TTS playback is finished  
                self._notify_playback_finish()
                
                logger.debug("Audio playback completed")
                
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
                # Still notify that playback is finished even if there was an error
                self._notify_playback_finish()
                
            finally:
                self.audio_queue.task_done()
            
    def _notify_playback_start(self):
        """Notify the continuous listener that TTS playback has started."""
        try:
            continuous_listener = self.context.get('continuous_listener')
            if continuous_listener and hasattr(continuous_listener, 'notify_tts_started'):
                continuous_listener.notify_tts_started()
                logger.info("Notified listener of TTS playback start")
        except Exception as e:
            logger.error(f"Error notifying TTS start: {e}")
            
    def _notify_playback_finish(self):
        """Notify the continuous listener that TTS playback has finished."""
        try:
            continuous_listener = self.context.get('continuous_listener')
            if continuous_listener and hasattr(continuous_listener, 'notify_tts_finished'):
                continuous_listener.notify_tts_finished()
                logger.info("Notified listener of TTS playback finish")
        except Exception as e:
            logger.error(f"Error notifying TTS finish: {e}")

    def play_audio_async(self, audio_data):
        """Queue audio data for playback without blocking the main process."""
        self.audio_queue.put(audio_data)

    def notify_continuous_listener_start(self, text):
        """
        Notify the continuous listener that TTS is starting and what content is being spoken.
        This helps prevent the system from responding to its own TTS output.
        
        Args:
            text: The text being converted to speech
        """
        try:
            # Get continuous listener from context
            continuous_listener = self.context.get('continuous_listener')
            
            if continuous_listener:
                # Notify about the TTS content
                if hasattr(continuous_listener, 'notify_tts_content'):
                    continuous_listener.notify_tts_content(text)
                    logger.info("Notified continuous listener of TTS content")
                    
                # Mark TTS as playing
                if hasattr(continuous_listener, 'set_tts_playing'):
                    continuous_listener.set_tts_playing(True)
                    logger.info("Notified continuous listener that TTS is starting")
        except Exception as e:
            logger.error(f"Error notifying continuous listener of TTS start: {e}")
            
    def notify_continuous_listener_end(self):
        """
        Notify the continuous listener that TTS has finished.
        This helps prevent the system from responding to its own TTS output.
        """
        try:
            # Get continuous listener from context
            continuous_listener = self.context.get('continuous_listener')
            
            if continuous_listener:
                # Mark TTS as no longer playing
                if hasattr(continuous_listener, 'set_tts_playing'):
                    continuous_listener.set_tts_playing(False)
                    logger.info("Notified continuous listener that TTS has finished")
        except Exception as e:
            logger.error(f"Error notifying continuous listener of TTS end: {e}")
            
    def __del__(self):
        """Clean up resources when the object is garbage collected."""
        # Signal the playback thread to stop
        if hasattr(self, 'audio_queue'):
            self.audio_queue.put(None)
