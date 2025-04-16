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
        self.voice_name = os.getenv("TTS_VOICE", "Alex_0")
        self.voice_id = None  # Will be extracted from voice_name
        self.first_audio = None
        self.first_text = None
        
        # Extract numeric voice ID from name if present (e.g., Alex_0 -> 0)
        if '_' in self.voice_name and self.voice_name.split('_')[-1].isdigit():
            self.voice_id = int(self.voice_name.split('_')[-1])
            logger.info(f"Extracted voice ID {self.voice_id} from voice name {self.voice_name}")
        
        self.voice_path = self._find_voice_checkpoint(self.voice_name)
        # This is crucial for voice consistency - store a voice context
        self.voice_context = []
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
            # Set speaker ID to match voice checkpoint if available
            if self.voice_id is not None:
                self.speaker = self.voice_id
                logger.info(f"Using voice ID {self.speaker} to match voice checkpoint {self.voice_name}")
            else:
                logger.info(f"No voice ID found in {self.voice_name}, using default speaker ID {self.speaker}")
                
            # Load the selected voice checkpoint - handle both tensor and state_dict formats
            if self.voice_path:
                try:
                    checkpoint = torch.load(self.voice_path, map_location=self.device)
                    
                    # Check if the checkpoint is a tensor (voice embedding) or state dict
                    if isinstance(checkpoint, torch.Tensor):
                        logger.info("Loaded voice checkpoint as tensor (voice embedding)")
                        # Store the voice embedding tensor for future use
                        self.voice_embedding = checkpoint
                    else:
                        # It's a state dict, try different ways to load it
                        logger.info("Loaded voice checkpoint as state dict")
                        # First check model._model for CSM generator
                        if hasattr(self.generator, '_model') and hasattr(self.generator._model, 'load_state_dict'):
                            logger.info("Loading voice checkpoint into generator._model")
                            self.generator._model.load_state_dict(checkpoint)
                        # Next try direct load_state_dict
                        elif hasattr(self.generator, 'load_state_dict'):
                            logger.info("Loading voice checkpoint directly into generator")
                            self.generator.load_state_dict(checkpoint)
                        # Finally try model attribute
                        elif hasattr(self.generator, 'model') and hasattr(self.generator.model, 'load_state_dict'):
                            logger.info("Loading voice checkpoint into generator.model")
                            self.generator.model.load_state_dict(checkpoint)
                        else:
                            logger.error("Unable to find appropriate target for loading voice checkpoint")
                            print("WARNING: Unable to load voice checkpoint into generator. Please check integration.")
                    
                    print(f"Loaded voice: {self.voice_name} from {self.voice_path}")
                except Exception as e:
                    logger.error(f"Error loading voice checkpoint: {e}")
                    print(f"Error loading voice checkpoint: {e}")
            
            self.sample_rate = self.generator.sample_rate
            print(f"Sesame CSM TTS initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing Sesame CSM TTS: {e}")
            import traceback
            traceback.print_exc()

    def text_to_speech(self, text):
        """Convert text to speech using Sesame CSM TTS.
        
        This method now uses text_to_speech_sentences internally to process
        text one sentence at a time for improved reliability.
        """
        logger = logging.getLogger(__name__)
        logger.info("\n=== Starting text_to_speech (CSM) ===")
        logger.info(f"Using robust sentence-by-sentence processing for reliability")
        
        try:
            # Call the enhanced sentence-by-sentence method for better reliability
            temp_paths = self.text_to_speech_sentences(text)
            
            # Return the path to the first audio file for backward compatibility
            return temp_paths[0] if temp_paths else None
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")
            import traceback
            traceback.print_exc()
            return None

    def split_text_to_sentences(self, text):
        """Split text into sentences using a simple regex."""
        import re
        logger.info(f"Splitting text into sentences: {text}")
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        logger.info(f"Split into {len(sentences)} sentences: {sentences}")
        return [s for s in sentences if s]

    def text_to_speech_sentences(self, text):
        """Convert text to speech one sentence at a time, queueing each for playback."""
        logger = logging.getLogger(__name__)
        logger.info(f"\n=== Starting text_to_speech_sentences (CSM) ===")
        logger.info(f"Using voice: {self.voice_name} (checkpoint: {self.voice_path})")
        self.is_processing = True
        self.context['is_processing'] = True
        self.notify_continuous_listener_start(text)
        try:
            if self.generator is None:
                self.initialize_tts()
            if self.generator is None:
                raise RuntimeError("Failed to initialize Sesame CSM TTS system")
            sentences = self.split_text_to_sentences(text)
            temp_paths = []
            
            # Define a simple local Segment class since import from csm.generator may not be available
            class Segment:
                def __init__(self, speaker, text, audio):
                    self.speaker = speaker
                    self.text = text
                    self.audio = audio
            
            # If we don't have a voice context set up yet, generate a silent one-time context
            # This ensures all future sentences use the same voice characteristics
            voice_context = []
            if not hasattr(self, 'first_audio') or self.first_audio is None:
                logger.info("[TTS] Creating initial voice context using a silent generation")
                try:
                    # Generate a short, silent segment to create a consistent voice context
                    silent_text = "."  # Just a period, which will be nearly silent
                    silent_audio = self.generator.generate(
                        text=silent_text,
                        speaker=self.speaker,
                        context=[],  # Empty for first generation
                        max_audio_length_ms=1000,  # Very short
                    )
                    # Save this for future use
                    self.first_audio = silent_audio
                    self.first_text = silent_text
                    logger.info("[TTS] Created initial voice context")
                except Exception as e:
                    logger.error(f"[TTS] Error creating initial voice context: {e}")
                    # Continue without voice context if it fails
            
            # Create context if we have an initial audio sample
            if hasattr(self, 'first_audio') and self.first_audio is not None:
                voice_context = [Segment(
                    speaker=self.speaker,
                    text=self.first_text,
                    audio=self.first_audio
                )]
                logger.info("[TTS] Using voice context for consistent voice")
            
            for idx, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                clean_sentence = self.clean_text_for_tts(sentence.strip())
                logger.info(f"[TTS] Processing sentence {idx+1}/{len(sentences)}: '{clean_sentence}'")
                try:
                    # Always use the same speaker ID and voice context for all sentences
                    audio = self.generator.generate(
                        text=clean_sentence,
                        speaker=self.speaker,  # Same speaker ID for all sentences
                        context=voice_context,  # Important: Use the same voice context for all sentences
                        max_audio_length_ms=10000,
                    )
                    
                    # Update the first audio if this is the first sentence and we don't have one yet
                    if idx == 0 and (not hasattr(self, 'first_audio') or self.first_audio is None):
                        self.first_audio = audio
                        self.first_text = clean_sentence
                        logger.info("[TTS] Stored first sentence audio for future voice consistency")
                    
                    # Save audio to temp file and queue for playback
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                        temp_path = fp.name
                        torchaudio.save(temp_path, audio.unsqueeze(0).cpu(), self.sample_rate)
                        logger.info(f"[TTS] Audio saved to {temp_path} (sentence {idx+1})")
                    # Play asynchronously
                    audio_data = audio.cpu().numpy()
                    self.play_audio_async(audio_data)
                    temp_paths.append(temp_path)
                    logger.info(f"[TTS] Sentence {idx+1} processed and queued for playback.")
                except Exception as e:
                    logger.error(f"[TTS] Error processing sentence {idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
            logger.info(f"[TTS] Finished processing all sentences. {len(temp_paths)} audio files created.")
            return temp_paths
        except Exception as e:
            logger.error(f"Error in Sesame CSM text-to-speech (sentences): {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            self.is_processing = False
            self.context['is_processing'] = False
            self.notify_continuous_listener_end()

    def clean_text_for_tts(self, text):
        """
        Clean text to make it suitable for TTS processing.
        Removes problematic symbols while preserving emotion markers.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text ready for TTS processing
        """
        import re
        
        # Skip cleaning if text is empty
        if not text:
            return text
            
        # Preserve emotion markers at the start of text like [happy], [excited], etc.
        emotion_marker = None
        emotion_match = re.match(r'^\s*\[(.*?)\]', text)
        if emotion_match:
            emotion_marker = emotion_match.group(0)
            text = text[len(emotion_marker):].strip()
            logger.info(f"Preserved emotion marker: {emotion_marker}")
            
        # Replace problematic symbols with their spoken form or spaces
        replacements = {
            ':': ' ', 
            ';': ',',
            '&': ' and ',
            '+': ' plus ',
            '=': ' equals ',
            '@': ' at ',
            '#': ' number ',
            '%': ' percent ',
            '|': ' or ',
            '$': ' dollars ',
            '•': ', ',
            '·': ', ',
            '…': '...',
            '*': ' star ',
            '^': '',
            '~': '',
            '`': '',
            '<': '',
            '>': '',
            '"': "'",
            '"': "'",
            '"': "'",
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '©': '',
            '®': '',
            '™': '',
        }
        
        # Apply replacements
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, replacement)
            
        # Fix common abbreviations
        abbreviations = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'etcetera',
            'vs.': 'versus',
            'approx.': 'approximately',
            'hr.': 'hour',
            'hr ': 'hour ',
            'hrs.': 'hours',
            'hrs ': 'hours ',
            'min.': 'minute',
            'min ': 'minute ',
            'mins.': 'minutes',
            'mins ': 'minutes ',
            'sec.': 'second',
            'sec ': 'second ',
            'secs.': 'seconds',
            'secs ': 'seconds ',
        }
        
        # Apply abbreviation replacements - must match whole word with word boundaries
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
        
        # Fix some specific patterns that cause issues
        # URLs and email addresses - replace with "link" or "email"
        text = re.sub(r'https?://\S+', 'link', text)
        text = re.sub(r'\S+@\S+\.\S+', 'email address', text)
        
        # Handle repeated spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Re-add emotion marker if it was present
        if emotion_marker:
            text = emotion_marker + ' ' + text
            
        # Ensure there's a period at the end for better TTS phrasing if not already present
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
            
        return text.strip()

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

    def _find_voice_checkpoint(self, voice_name):
        """Find the .pt checkpoint for the given voice name in assets/voices."""
        voices_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "voices")
        # Accept both exact and partial (case-insensitive) matches
        for fname in os.listdir(voices_dir):
            if fname.lower().endswith(".pt") and voice_name.lower() in fname.lower():
                return os.path.join(voices_dir, fname)
        raise FileNotFoundError(f"Voice checkpoint for '{voice_name}' not found in {voices_dir}")

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
