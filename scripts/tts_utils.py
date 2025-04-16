import sys
import os

# Add paths to find modules in both execution contexts
sys.path.append('./csm')  # For when run from scripts directory 
sys.path.append('../csm')  # For when run from scripts directory directly
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csm_dir = os.path.join(base_dir, 'csm')
sys.path.append(csm_dir)  # Absolute path to csm directory

import torch
import torchaudio
import tempfile
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import logging
from typing import Tuple
import simpleaudio as sa

# Try different import approaches for generator module
try:
    from generator import load_csm_1b
except ImportError:
    try:
        from csm.generator import load_csm_1b
    except ImportError:
        # Try absolute import as last resort
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generator", os.path.join(csm_dir, "generator.py")
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        load_csm_1b = generator.load_csm_1b

# Configure logging
logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self, context):
        """Initialize TTS Manager with context"""
        self.context = context
        self.generator = None
        self.device = self.get_device()
        self.speaker = 0  # Default speaker ID
        self.sample_rate = 24000  # Default sample rate for CSM
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.playback_thread.start()
        self.voice_name = os.getenv("TTS_VOICE", "Alex_0")
        self.voice_id = None
        self.first_audio = None
        self.first_text = None
        
        # Extract numeric voice ID from name if present (e.g., Alex_0 -> 0)
        if '_' in self.voice_name and self.voice_name.split('_')[-1].isdigit():
            self.voice_id = int(self.voice_name.split('_')[-1])
            logger.info(f"Extracted voice ID {self.voice_id} from voice name {self.voice_name}")
        
        self.voice_path = self._find_voice_checkpoint(self.voice_name)
        self.voice_context = []  # Store voice context for consistency
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
            # Initialize the CSM generator
            self.generator = load_csm_1b(device=self.device)
            
            # CRITICAL FIX: The key is using the correct numeric speaker ID
            # In the CSM model, speaker ID 0 = Alex, 1 = Carter, etc.
            if self.voice_id is not None:
                self.speaker = self.voice_id
                print(f"Using speaker ID {self.speaker} for voice {self.voice_name}")
            else:
                print(f"No voice ID found in {self.voice_name}, using default speaker ID {self.speaker}")
            
            # Load the voice embedding from the voice file
            self.voice_embedding = self._load_voice_embedding()
            
            # Set sample rate from generator
            self.sample_rate = self.generator.sample_rate
            
            # Pre-create the voice context to ensure all sentences use the same voice
            self._create_voice_context()
            
            print(f"TTS initialized with voice {self.voice_name} (speaker ID: {self.speaker})")
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_voice_embedding(self):
        """Load the voice embedding from the voice file."""
        if not self.voice_path:
            print("No voice file available. Using default voice.")
            return None
            
        try:
            print(f"Loading voice embedding from: {self.voice_path}")
            voice_data = torch.load(self.voice_path, map_location=self.device)
            
            if isinstance(voice_data, torch.Tensor):
                print(f"Loaded voice embedding tensor with shape: {voice_data.shape}")
                print(f"Voice tensor stats: min={voice_data.min().item():.4f}, max={voice_data.max().item():.4f}")
                return voice_data
            elif isinstance(voice_data, dict) and 'embedding' in voice_data:
                print(f"Loaded voice embedding from dictionary")
                return voice_data['embedding']
            else:
                print(f"Voice file has unexpected format: {type(voice_data)}")
                return None
        except Exception as e:
            print(f"Error loading voice embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_voice_context(self):
        """Create a voice context for consistent voice across generations"""
        print("Creating voice context for consistent speech characteristics...")
        try:
            # Define a Segment class for the voice context
            class Segment:
                def __init__(self, speaker, text, audio):
                    self.speaker = speaker
                    self.text = text
                    self.audio = audio
            
            # If we already have a pre-computed voice context from a previous run,
            # try to load it instead of creating a new one
            voice_context_path = self._get_voice_context_path()
            if os.path.exists(voice_context_path):
                try:
                    print(f"Loading pre-computed voice context from: {voice_context_path}")
                    saved_context = torch.load(voice_context_path, map_location=self.device)
                    if isinstance(saved_context, dict) and 'audio' in saved_context and 'text' in saved_context:
                        self.first_audio = saved_context['audio']
                        self.first_text = saved_context['text']
                        print(f"Loaded voice context with shape: {self.first_audio.shape}")
                        
                        # Create the voice context object
                        self.voice_context = [Segment(
                            speaker=self.speaker,
                            text=self.first_text,
                            audio=self.first_audio
                        )]
                        print("Successfully loaded pre-computed voice context")
                        return
                except Exception as e:
                    print(f"Error loading pre-computed voice context: {e}, generating new one")
            
            # Generate a short sentence to establish voice characteristics
            test_text = "This is a voice test."
            print(f"Generating reference audio with speaker ID {self.speaker}")
            
            # Create a special generation context for the first generation
            # If we have a voice embedding, try to use it
            if self.voice_embedding is not None:
                print("Using voice embedding for context generation")
                # Try to patch the model's speaker embedding for this generation
                if hasattr(self.generator, 'model') and hasattr(self.generator.model, 'speaker_embedding'):
                    if isinstance(self.generator.model.speaker_embedding, torch.nn.Embedding):
                        print("Patching model's speaker embedding")
                        # Save original embedding
                        original_weight = self.generator.model.speaker_embedding.weight[self.speaker].clone()
                        
                        # Reshape voice embedding if needed
                        if self.voice_embedding.shape != original_weight.shape:
                            if self.voice_embedding.numel() == original_weight.numel():
                                voice_embedding = self.voice_embedding.reshape(original_weight.shape)
                                print(f"Reshaped voice embedding from {self.voice_embedding.shape} to {voice_embedding.shape}")
                            else:
                                print(f"Cannot reshape voice embedding: {self.voice_embedding.shape} to {original_weight.shape}")
                                voice_embedding = self.voice_embedding
                        else:
                            voice_embedding = self.voice_embedding
                        
                        try:
                            # Apply the voice embedding
                            self.generator.model.speaker_embedding.weight[self.speaker] = voice_embedding
                            print("Applied voice embedding to model")
                            
                            # Generate with the voice embedding
                            test_audio = self.generator.generate(
                                text=test_text,
                                speaker=self.speaker,
                                context=[],  # Empty for first generation
                                max_audio_length_ms=3000,
                            )
                            
                            # Restore original embedding
                            self.generator.model.speaker_embedding.weight[self.speaker] = original_weight
                        except Exception as e:
                            print(f"Error applying voice embedding: {e}")
                            # Restore original embedding on error
                            self.generator.model.speaker_embedding.weight[self.speaker] = original_weight
                            # Fall back to regular generation
                            test_audio = self.generator.generate(
                                text=test_text,
                                speaker=self.speaker,
                                context=[],
                                max_audio_length_ms=3000,
                            )
                    else:
                        print("Speaker embedding is not an embedding layer, using default generation")
                        test_audio = self.generator.generate(
                            text=test_text,
                            speaker=self.speaker,
                            context=[],
                            max_audio_length_ms=3000,
                        )
                else:
                    # No model.speaker_embedding, try custom injection
                    try:
                        # Create a dictionary of model parameters to pass along
                        gen_kwargs = {
                            'text': test_text,
                            'speaker': self.speaker,
                            'context': [],
                            'max_audio_length_ms': 3000,
                        }
                        
                        # Try to inject the voice embedding if the function supports it
                        import inspect
                        gen_sig = inspect.signature(self.generator.generate)
                        if 'voice_embedding' in gen_sig.parameters:
                            print("Generator supports voice_embedding parameter")
                            gen_kwargs['voice_embedding'] = self.voice_embedding
                            
                        # Generate with possibly injected voice embedding
                        test_audio = self.generator.generate(**gen_kwargs)
                    except Exception as e:
                        print(f"Error with custom voice injection: {e}")
                        # Fall back to normal generation
                        test_audio = self.generator.generate(
                            text=test_text,
                            speaker=self.speaker,
                            context=[],
                            max_audio_length_ms=3000,
                        )
            else:
                # No voice embedding, use standard generation
                test_audio = self.generator.generate(
                    text=test_text,
                    speaker=self.speaker,
                    context=[],
                    max_audio_length_ms=3000,
                )
            
            print(f"Created voice context with shape {test_audio.shape}")
            print(f"Audio stats: min={test_audio.min().item():.4f}, max={test_audio.max().item():.4f}")
            
            # Store for future use
            self.first_audio = test_audio
            self.first_text = test_text
            
            # Create the voice context
            self.voice_context = [Segment(
                speaker=self.speaker,
                text=self.first_text,
                audio=self.first_audio
            )]
            
            # Save the voice context for future runs
            try:
                context_save_path = self._get_voice_context_path()
                save_dir = os.path.dirname(context_save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                torch.save({
                    'audio': self.first_audio,
                    'text': self.first_text,
                    'speaker': self.speaker,
                    'voice_name': self.voice_name,
                }, context_save_path)
                print(f"Saved voice context to: {context_save_path}")
            except Exception as e:
                print(f"Error saving voice context: {e}")
            
            print(f"Voice context created successfully for speaker ID {self.speaker}")
        except Exception as e:
            print(f"Error creating voice context: {e}")
            import traceback
            traceback.print_exc()
            
    def _get_voice_context_path(self):
        """Get the path to save/load the voice context."""
        # Create a directory for voice contexts if it doesn't exist
        voice_context_dir = os.path.join(base_dir, 'assets', 'voice_contexts')
        return os.path.join(voice_context_dir, f"{self.voice_name}_context.pt")

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
            print(f"Error in text_to_speech: {e}")
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
        logger.info(f"Using voice: {self.voice_name} (speaker ID: {self.speaker})")
        self.is_processing = True
        self.context['is_processing'] = True
        self.notify_continuous_listener_start(text)
        try:
            if self.generator is None:
                self.initialize_tts()
            if self.generator is None:
                raise RuntimeError("Failed to initialize Sesame CSM TTS system")
            
            sentences = self.split_text_to_sentences(text)
            print(f"Processing {len(sentences)} sentences with voice {self.voice_name} (ID: {self.speaker})")
            temp_paths = []
            
            # Define Segment class for voice context
            class Segment:
                def __init__(self, speaker, text, audio):
                    self.speaker = speaker
                    self.text = text
                    self.audio = audio
            
            # CRITICAL: Ensure we have a voice context object
            if not self.voice_context:
                print("No voice context found. Creating one now...")
                self._create_voice_context()
            
            if not self.voice_context:
                print("WARNING: Failed to create voice context!")
            else:
                print(f"Using voice context with {len(self.voice_context)} segments")
                if hasattr(self, 'first_audio') and self.first_audio is not None:
                    print(f"Voice context audio shape: {self.first_audio.shape}")
            
            for idx, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                clean_sentence = self.clean_text_for_tts(sentence.strip())
                print(f"Processing sentence {idx+1}/{len(sentences)}: '{clean_sentence}'")
                try:
                    # CRITICAL FIX: Always use the same speaker ID and voice context
                    print(f"Generating with speaker ID {self.speaker} and context: {bool(self.voice_context)}")
                    
                    # Generate audio with consistent speaker ID and voice context
                    audio = self.generator.generate(
                        text=clean_sentence,
                        speaker=self.speaker,
                        context=self.voice_context,
                        max_audio_length_ms=10000,
                    )
                    
                    print(f"Generated audio with shape: {audio.shape}")
                    
                    # Save audio to temp file and queue for playback
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                        temp_path = fp.name
                        torchaudio.save(temp_path, audio.unsqueeze(0).cpu(), self.sample_rate)
                        print(f"Audio saved to {temp_path}")
                    
                    # Play asynchronously
                    audio_data = audio.cpu().numpy()
                    self.play_audio_async(audio_data)
                    temp_paths.append(temp_path)
                except Exception as e:
                    logger.error(f"Error processing sentence {idx+1}: {e}")
                    print(f"Error processing sentence {idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"Finished processing all sentences. {len(temp_paths)} audio files created.")
            return temp_paths
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
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
        """
        # Strip any leading/trailing whitespace
        text = text.strip()
        
        # Add a period if the text doesn't end with a punctuation
        if text and text[-1] not in ['.', '!', '?', ';', ':', ',']:
            text += '.'
            
        return text

    def play_audio_async(self, audio_data):
        """Queue audio data for asynchronous playback."""
        self.audio_queue.put(audio_data)

    def _process_audio_queue(self):
        """Process audio queue in background thread."""
        while True:
            try:
                audio_data = self.audio_queue.get()
                if audio_data is None:
                    # None is a signal to stop
                    break
                    
                # Sound device playback
                samples = (audio_data * 32767).astype(np.int16)
                play_obj = sa.play_buffer(samples, 1, 2, self.sample_rate)
                play_obj.wait_done()
                
                self.audio_queue.task_done()
            except Exception as e:
                print(f"Error in audio playback: {e}")
                import traceback
                traceback.print_exc()
                continue

    def notify_continuous_listener_start(self, text):
        """Notify that continuous listener should be stopped."""
        # This is a hook for the UI to stop the continuous listener
        if hasattr(self.context, 'continuous_listener_running') and self.context.get('continuous_listener_running', False):
            print("Stopping continuous listener for TTS playback")
            try:
                self.context['stop_continuous_listener']()
            except Exception as e:
                print(f"Error stopping continuous listener: {e}")

    def notify_continuous_listener_end(self):
        """Notify that continuous listener can be restarted."""
        pass

    def _find_voice_checkpoint(self, voice_name):
        """Find the voice checkpoint file."""
        # Try to find the voice file
        voice_dir = os.path.join(base_dir, 'assets', 'voices')
        voice_path = os.path.join(voice_dir, f"{voice_name}.pt")
        if os.path.exists(voice_path):
            return voice_path
            
        # If full path not found, try just the name part
        if '_' in voice_name:
            name_part = voice_name.split('_')[0]
            voice_path = os.path.join(voice_dir, f"{name_part}.pt")
            if os.path.exists(voice_path):
                return voice_path
                
        # Look for any file with a similar name
        if os.path.exists(voice_dir):
            for file in os.listdir(voice_dir):
                if file.endswith('.pt') and voice_name.lower() in file.lower():
                    return os.path.join(voice_dir, file)
                    
        # No voice file found
        print(f"No voice file found for {voice_name}")
        return None
