import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import os
import tempfile
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import torchaudio
import logging
import gc
from typing import Tuple
import simpleaudio as sa
import re

# Configure logging
logger = logging.getLogger(__name__)


class TTSManager:
    def __init__(self, context):
        """Initialize TTS Manager with context"""
        self.context = context
        self.tts = None
        self.voice_samples = None
        self.conditioning_latents = None
        self.is_processing = False  # Track TTS processing status
        self.gen_config = None
        self._gpu_memory = self.get_gpu_memory()
        self._use_deepspeed = self._gpu_memory >= 4
        self.last_generated_audio_path = None  # Track last generated audio path
        print(
            f"Initialized TTS Manager - GPU Memory: {self._gpu_memory:.1f}GB, DeepSpeed: {'Enabled' if self._use_deepspeed else 'Disabled'}"
        )

        # Setup audio queue system
        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(
            target=self._process_audio_queue, daemon=True
        )
        self.playback_thread.start()

        # Voice fingerprint tracking
        self.voice_fingerprint_samples = []
        self.max_fingerprint_samples = 5  # Number of samples to keep for fingerprinting

        self.initialize_tts()

    def clear_gpu_memory(self, reinitialize=False):
        """Clear GPU memory cache and run garbage collection"""
        if torch.cuda.is_available():
            # Empty CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()

            # Only clear models if explicitly requested
            if reinitialize:
                # Clear any existing models from memory
                if hasattr(self, "tts") and self.tts is not None:
                    del self.tts
                if (
                    hasattr(self, "conditioning_latents")
                    and self.conditioning_latents is not None
                ):
                    del self.conditioning_latents
                self.tts = None
                self.conditioning_latents = None
                # Run garbage collection
                gc.collect()
                torch.cuda.empty_cache()

                # Only reinitialize if explicitly requested
                if reinitialize:
                    self._initialize_tts_internal()

    def get_gpu_memory(self):
        """Get the total GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )  # Convert to GB
        return 0

    def can_use_deepspeed(self):
        """Check if DeepSpeed can be used on this system"""
        return self._use_deepspeed

    def get_optimal_tts_config(self):
        """Get optimal TTS configuration based on GPU memory"""
        gpu_memory = self._gpu_memory
        print(f"\nDetected GPU memory: {gpu_memory:.2f} GB")

        # Check DeepSpeed compatibility
        can_use_deepspeed = self.can_use_deepspeed()
        print(f"DeepSpeed status: {'Enabled' if can_use_deepspeed else 'Disabled'}")

        # Base configuration for TTS initialization
        init_config = {
            "kv_cache": True,
            "half": False,  # Default to full precision
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "autoregressive_batch_size": 1,
            "use_deepspeed": can_use_deepspeed,
        }

        # Generation configuration that will be used in tts_with_preset
        gen_config = {}

        # Optimize settings for different GPU memory sizes
        if gpu_memory >= 35:  # For very high-end GPUs (A5000, A6000, etc.)
            init_config["autoregressive_batch_size"] = 4  # More conservative batch size
            gen_config.update(
                {
                    "diffusion_iterations": 60,
                    "num_autoregressive_samples": 4,
                    "length_penalty": 0.8,  # Slightly reduce length penalty
                    "repetition_penalty": 2.5,  # Increase repetition penalty
                    "top_k": 50,  # Add top_k sampling
                    "top_p": 0.85,  # Slightly increase top_p
                }
            )
        elif gpu_memory >= 24:  # For high-end GPUs (24GB+)
            init_config["autoregressive_batch_size"] = 4
            gen_config.update(
                {"diffusion_iterations": 60, "num_autoregressive_samples": 4}
            )
        elif gpu_memory >= 16:  # For GPUs with 16-24GB
            init_config["autoregressive_batch_size"] = 3
            gen_config.update(
                {"diffusion_iterations": 50, "num_autoregressive_samples": 3}
            )
        elif gpu_memory >= 11.5:  # For GPUs with 12-16GB
            init_config["autoregressive_batch_size"] = 2
            gen_config.update(
                {
                    "diffusion_iterations": 25,
                    "num_autoregressive_samples": 2,
                    "half": True,
                }
            )
        else:  # For GPUs with less than 12GB
            init_config.update(
                {
                    "autoregressive_batch_size": 1,  # Increased from 1 to 2 since we're using half precision
                    "half": True,  # Enable half-precision for low memory GPUs
                }
            )
            gen_config.update(
                {"diffusion_iterations": 20, "num_autoregressive_samples": 1}
            )

        return init_config, gen_config

    def _initialize_tts_internal(self):
        """Internal method for TTS initialization without recursive cleanup"""
        # Set PyTorch memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Get optimal configuration based on GPU memory
        init_config, self.gen_config = self.get_optimal_tts_config()
        print(f"Initializing TTS with config: {init_config}")
        print(f"Will use generation config: {self.gen_config}")
        self.tts = TextToSpeech(**init_config)
        print("TTS object created successfully")

        # Set fixed seed for consistent voice
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Get voice from config
        voice_name = os.getenv("TTS_VOICE", "emma")
        print(f"Loading voice samples for {voice_name}...")
        self.voice_samples = load_voice(voice_name, extra_voice_dirs=[])[0]
        print(f"Voice samples loaded: {len(self.voice_samples)} samples")

        print("Computing conditioning latents...")
        self.conditioning_latents = self.tts.get_conditioning_latents(
            self.voice_samples
        )
        print("Conditioning latents generated")

        # Store in context
        self.context.update(
            {
                "tts": self.tts,
                "voice_samples": self.voice_samples,
                "conditioning_latents": self.conditioning_latents,
            }
        )

    def initialize_tts(self):
        """Initialize TTS and return the initialized objects"""
        print("\n=== Initializing TTS at startup ===")
        try:
            # Clear GPU memory before initialization
            self.clear_gpu_memory(reinitialize=False)
            self._initialize_tts_internal()

            # Generate system sounds immediately after successful TTS initialization
            print("\n=== Generating system notification sounds ===")
            self.generate_system_notification_sounds()

        except Exception as e:
            print(f"Error initializing TTS: {e}")
            import traceback

            traceback.print_exc()

    def ensure_tts_initialized(self):
        """Ensure TTS is initialized before use"""
        if self.tts is None:
            self.initialize_tts()
            # Generate system notification sounds after TTS is initialized
            # self.generate_system_notification_sounds()

    def determine_temperature(self, emotion_cue, is_retry=False):
        # Define temperature variations based on emotion with more dramatic ranges
        temperature_map = {
            # High energy emotions - more variation
            "excited": 1.8,
            "happy": 1.5,
            "enthusiastic": 1.6,
            "angry": 1.7,
            "energetic": 1.6,
            # Medium energy emotions - balanced variation
            "neutral": 1.0,
            "confident": 1.2,
            "professional": 0.9,
            "formal": 0.8,
            # Low energy emotions - more controlled
            "sad": 0.6,
            "gentle": 0.5,
            "calm": 0.7,
            "soft": 0.5,
            "tender": 0.6,
            "melancholic": 0.7,
        }

        # Get the base temperature
        base_temperature = temperature_map.get("neutral", 1.0)

        # Adjust temperature based on emotion cue
        if emotion_cue:
            emotion_parts = emotion_cue.lower().split(" and ")
            temperatures = []

            # Collect temperatures for all mentioned emotions
            for part in emotion_parts:
                part = part.strip()
                if part in temperature_map:
                    temperatures.append(temperature_map[part])

            # If we found any matching emotions, use their average
            if temperatures:
                temperature = sum(temperatures) / len(temperatures)
            else:
                temperature = base_temperature

            # Adjust for retry if needed
            if is_retry:
                temperature = max(
                    0.5, temperature * 0.8
                )  # Reduce by 20% but not below 0.5

            # Ensure we stay within safe bounds
            temperature = min(2.0, max(0.5, temperature))
        else:
            temperature = base_temperature

        return temperature

    def process_style_cue(self, style_cue: str) -> tuple[str, float]:
        """
        Process style and emotion cues to generate an enhanced prompt and temperature.

        Format examples:
        [happy and excited, fast]
        [sad, slow and gentle]
        [professional, clear and confident]
        [whispered, mysterious]
        [shouting, angry and energetic]

        These cues will influence the speech generation but won't be spoken.
        """
        logger = logging.getLogger(__name__)

        # Style modifiers affect both prompt and temperature
        style_modifiers = {
            # Speed modifiers
            "fast": {"prefix": "speaking quickly and energetically", "temp_mod": 1.2},
            "slow": {"prefix": "speaking slowly and deliberately", "temp_mod": 0.8},
            # Volume/Intensity modifiers
            "loud": {"prefix": "speaking loudly and clearly", "temp_mod": 1.3},
            "soft": {"prefix": "speaking softly and gently", "temp_mod": 0.7},
            "whispered": {"prefix": "whispering intimately", "temp_mod": 0.6},
            "shouting": {"prefix": "shouting energetically", "temp_mod": 1.4},
            # Style modifiers
            "clear": {"prefix": "speaking very clearly and precisely", "temp_mod": 0.9},
            "mysterious": {"prefix": "speaking mysteriously", "temp_mod": 1.1},
            "dramatic": {"prefix": "speaking dramatically", "temp_mod": 1.4},
            "playful": {"prefix": "speaking playfully", "temp_mod": 1.3},
            "formal": {"prefix": "speaking formally", "temp_mod": 0.8},
            "casual": {"prefix": "speaking casually", "temp_mod": 1.1},
        }

        if not style_cue:
            return "", self.determine_temperature("")

        # Split into parts and clean
        parts = [p.strip().lower() for p in style_cue.split(",")]

        # Process each part for emotions and styles
        emotion_temp = self.determine_temperature(
            parts[0]
        )  # First part is always emotion

        # Start building the enhanced prompt
        prompt_parts = []
        temp_modifiers = []

        # Add base emotion
        prompt_parts.append(parts[0])

        # Process additional style modifiers
        for part in parts[1:] if len(parts) > 1 else []:
            words = part.split()
            for word in words:
                if word in style_modifiers:
                    prompt_parts.append(style_modifiers[word]["prefix"])
                    temp_modifiers.append(style_modifiers[word]["temp_mod"])

        # Combine everything into a final prompt
        final_prompt = f"{', '.join(prompt_parts)}"

        # Calculate final temperature with modifiers
        final_temp = emotion_temp
        for modifier in temp_modifiers:
            final_temp *= modifier

        # Ensure temperature stays within bounds
        final_temp = min(2.0, max(0.5, final_temp))

        logger.info(
            f"Processed style cue '{style_cue}' into prompt '{final_prompt}' with temperature {final_temp}"
        )
        return final_prompt, final_temp

    def safe_tts_with_preset(self, text, **kwargs):
        """Safely attempt TTS generation with fallback options"""
        try:
            return self.tts.tts_with_preset(text, **kwargs)
        except RuntimeError as e:
            if "expected a non-empty list of Tensors" in str(e):
                # Try with more conservative settings
                conservative_kwargs = kwargs.copy()
                conservative_kwargs.update(
                    {
                        "num_autoregressive_samples": max(
                            1, kwargs.get("num_autoregressive_samples", 2) - 1
                        ),
                        "temperature": min(1.2, kwargs.get("temperature", 1.0) + 0.2),
                        "top_p": min(0.95, kwargs.get("top_p", 0.8) + 0.1),
                        "repetition_penalty": max(
                            1.5, kwargs.get("repetition_penalty", 2.0) - 0.2
                        ),
                    }
                )
                print("Retrying with more conservative settings:", conservative_kwargs)
                return self.tts.tts_with_preset(text, **conservative_kwargs)
            raise

    def validate_brackets(self, text: str) -> Tuple[bool, str]:
        """
        Validate that all emotion/style brackets are properly paired.
        Returns (is_valid, error_message)
        """
        # If no opening bracket, text is valid
        if "[" not in text:
            return True, ""

        # Text must start with [ if it contains one
        if not text.startswith("["):
            return False, "Emotion/style markers must be at the start of text"

        # Find matching closing bracket
        end_bracket = text.find("]")
        if end_bracket == -1:
            return False, "Missing closing bracket ']' for emotion/style marker"

        # Check for any additional brackets
        remaining_text = text[end_bracket + 1 :]
        if "[" in remaining_text or "]" in remaining_text:
            return False, "Only one emotion/style marker allowed at the start of text"

        return True, ""

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
                    audio_int16, num_channels=1, bytes_per_sample=2, sample_rate=24000
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
            continuous_listener = self.context.get("continuous_listener")
            if continuous_listener and hasattr(
                continuous_listener, "notify_tts_started"
            ):
                continuous_listener.notify_tts_started()
                logger.info("Notified listener of TTS playback start")
        except Exception as e:
            logger.error(f"Error notifying TTS start: {e}")

    def _notify_playback_finish(self):
        """Notify the continuous listener that TTS playback has finished."""
        try:
            continuous_listener = self.context.get("continuous_listener")
            if continuous_listener and hasattr(
                continuous_listener, "notify_tts_finished"
            ):
                continuous_listener.notify_tts_finished()
                logger.info("Notified listener of TTS playback finish")
        except Exception as e:
            logger.error(f"Error notifying TTS finish: {e}")

    def play_audio_async(self, audio_data):
        """Queue audio data for playback without blocking the main process."""
        self.audio_queue.put(audio_data)

    def text_to_speech(self, text, silent=False):
        """Convert text to speech using Tortoise TTS with emotional prompting support.

        The text can include emotional and style cues in brackets at the start, like:
        "[happy and excited, fast] Hello there!"
        "[sad, slow and gentle] I miss you"
        "[professional, clear and confident] Let me explain"
        "[whispered, mysterious] I have a secret"
        "[shouting, angry and energetic] I'm so angry!"

        These cues will influence the speech generation but won't be spoken.

        Args:
            text (str): The text to convert to speech, optionally with emotional cues
            silent (bool): If True, will generate the audio file but not play it
        """
        logger = logging.getLogger(__name__)
        logger.info("\n=== Starting text_to_speech ===")
        logger.info(f"Original text: {text[:50]}...")
        
        # Skip if TTS is not initialized
        if not self.is_initialized:
            logger.warning("TTS is not initialized. Call initialize_tts() first.")
            return False

        try:
            # Check if we're already processing a request
            if self.is_processing:
                logger.warning("Already processing a TTS request. Ignoring this one.")
                return False

            # Signal that we're starting to process
            self.is_processing = True
            
            # Extract any emotion/style cues from the text
            text, emotion_cue = self._extract_emotion_cue(text)
            logger.info(f"Text after removing emotion cue: {text[:50]}...")
            logger.info(f"Emotion cue: {emotion_cue}")
            
            # Create a timestamp for this request
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Setup output directory for generated audio
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(script_dir, "output", "audio")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"tts_{timestamp}.wav")
            
            # Keep track of the output path
            self.last_generated_audio_path = output_path
            
            # Start the TTS processing in a background thread
            thread = threading.Thread(
                target=self._process_tts_request,
                args=(text, output_path, silent)
            )
            thread.daemon = True
            thread.start()
            
            return True
        
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")
            self.is_processing = False
            return False

    def _process_tts_request(self, text, output_path, silent=False):
        """Process a TTS request in a background thread."""
        try:
            start_time = time.time()
            
            print(f"Starting TTS for: '{text}'")
            
            # If recognizer is provided and active, notify it (but only if not in silent mode)
            if not silent and self.recognizer_callback:
                self.recognizer_callback(text)
            
            # Process text in chunks for long text
            chunks = []
            if len(text) > self.MAX_TEXT_LENGTH:
                # Split the text into sentence-based chunks
                sentences = re.split(r'(?<=[.!?])\s+', text)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < self.MAX_TEXT_LENGTH:
                        if current_chunk:
                            current_chunk += " "
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk)
            else:
                chunks = [text]
                
            print(f"Processing {len(chunks)} chunks")
            
            # Process each text chunk
            all_audio = []
            
            # Notify listener that playback is starting (if not in silent mode)
            if not silent and self.listener_notify_start:
                self.listener_notify_start()
                
            for i, chunk in enumerate(chunks):
                print(f"Generating TTS for chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
                
                gen = self.tts.tts_with_preset(
                    chunk,
                    voice_samples=self.voice_samples,
                    conditioning_latents=self.conditioning_latents,
                    preset=self.preset
                )
                
                if gen is not None:
                    print(f"Audio shape for chunk {i}: {gen.shape}")

                    # Play the audio chunk asynchronously (if not in silent mode)
                    if not silent:
                        try:
                            audio_data = gen.squeeze(0).cpu().numpy()
                            print(f"Queueing audio chunk with shape: {audio_data.shape}")
                            self.play_audio_async(audio_data)
                        except Exception as e:
                            print(f"Error queueing audio for playback: {e}")

                    all_audio.append(gen)
            
            # Notify listener that playback is finished (if not in silent mode)
            if not silent and self.listener_notify_end:
                self.listener_notify_end()
            
            # Save the audio to the output path
            if all_audio:
                try:
                    # Check for tensor dimension mismatch
                    shapes = [a.shape for a in all_audio]
                    if len(set(s[1] for s in shapes)) > 1:
                        # Tensors have different dimensions, need to pad
                        max_len = max(s[1] for s in shapes)
                        print(
                            f"Audio chunks have different lengths, padding to {max_len}"
                        )
                        padded_audio = []
                        for audio in all_audio:
                            if audio.shape[1] < max_len:
                                # Create padding
                                padding = torch.zeros(
                                    1, max_len - audio.shape[1], device=audio.device
                                )
                                # Concatenate along dimension 1
                                padded = torch.cat([audio, padding], dim=1)
                                padded_audio.append(padded)
                            else:
                                padded_audio.append(audio)
                        final_audio = torch.cat(padded_audio, dim=0).mean(
                            dim=0, keepdim=True
                        )
                    else:
                        # All tensors have the same dimension, can concatenate directly
                        final_audio = torch.cat(all_audio, dim=1)
                except RuntimeError as e:
                    print(f"Error combining audio: {e}")
                    # Fallback: use the last generated audio
                    final_audio = all_audio[-1]
            else:
                raise RuntimeError("No audio was generated")

            # Save the audio to the output path
            torchaudio.save(output_path, final_audio.cpu(), 24000)
            print(f"Audio saved to {output_path}")

        except Exception as e:
            print(f"Error processing TTS request: {e}")
            import traceback

            traceback.print_exc()

    def generate_system_notification_sounds(self):
        """Generate system notification sound files for the application.

        This method creates TTS-based sound files for system notifications
        like activation, deactivation, etc.
        """
        logger = logging.getLogger(__name__)

        # Get the absolute path to the project directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        # Create sounds directory if it doesn't exist
        sound_dir = os.path.join(project_root, "sounds")
        os.makedirs(sound_dir, exist_ok=True)

        logger.info(f"Sound files will be saved to: {sound_dir}")
        print(f"Sound files will be saved to: {sound_dir}")

        logger.info("Generating system notification sounds...")

        # Define system messages with their style cues
        system_messages = {
            "activation": "[calm and welcoming] Hello, I'm listening. How can I help you?",
            "deactivation": "[gentle] I'll stop listening now.",
            "recognition": "[friendly and brief] hmm",
            "searching": "[helpful] Ok, just wait a few moments while I search the web.",
            "self_reflection": "[thoughtful] hmmm interesting."
        }

        # Save current state
        original_play_audio = self.play_audio

        # Temporarily disable audio playback
        self.play_audio = False

        # Generate each sound file
        for sound_name, message in system_messages.items():
            output_file = os.path.join(sound_dir, f"{sound_name}.wav")

            # Always generate new files, removing any existing ones
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                    logger.info(f"Removed existing sound file: {output_file}")
                except Exception as e:
                    logger.error(f"Error removing existing sound file: {e}")

            logger.info(f"Generating sound file for {sound_name}: {message}")

            try:
                # Use TTS to generate audio (without playing it)
                self.text_to_speech(message, silent=True)

                # Allow time for the audio to be generated and saved
                logger.info(f"Waiting for TTS to complete for {sound_name}")
                while self.is_processing:
                    time.sleep(0.1)

                # Get the latest generated audio file
                latest_audio = self.last_generated_audio_path

                if latest_audio and os.path.exists(latest_audio):
                    # Copy the audio file to the sound directory with the appropriate name
                    import shutil

                    shutil.copy(latest_audio, output_file)
                    logger.info(f"Created sound file: {output_file}")
                else:
                    logger.error(f"Failed to generate audio for {sound_name}")

            except Exception as e:
                logger.error(f"Error generating sound for {sound_name}: {e}")

        # Restore the original playback setting
        self.play_audio = original_play_audio
        logger.info("System notification sound generation complete!")

    def __del__(self):
        """Clean up resources when the object is garbage collected."""
        # Signal the playback thread to stop
        if hasattr(self, "audio_queue"):
            self.audio_queue.put(None)

        # Clear any CUDA tensors
        self.clear_gpu_memory()

    def notify_continuous_listener_start(self, text):
        """
        Notify the continuous listener that TTS is starting and what content is being spoken.
        This helps prevent the system from responding to its own TTS output.

        Args:
            text: The text being converted to speech
        """
        try:
            # Get continuous listener from context
            continuous_listener = self.context.get("continuous_listener")

            if continuous_listener:
                # Notify about the TTS content
                if hasattr(continuous_listener, "notify_tts_content"):
                    continuous_listener.notify_tts_content(text)
                    logger.info("Notified continuous listener of TTS content")

                # Mark TTS as playing
                if hasattr(continuous_listener, "set_tts_playing"):
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
            continuous_listener = self.context.get("continuous_listener")

            if continuous_listener:
                # Mark TTS as no longer playing
                if hasattr(continuous_listener, "set_tts_playing"):
                    continuous_listener.set_tts_playing(False)
                    logger.info("Notified continuous listener that TTS has finished")
        except Exception as e:
            logger.error(f"Error notifying continuous listener of TTS end: {e}")
