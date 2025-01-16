import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import os
import logging
import tempfile
import torchaudio
import gc
import json

class TTSManager:
    def __init__(self, context):
        """Initialize TTS Manager with context"""
        self.context = context
        self.tts = None
        self.voice_samples = None
        self.conditioning_latents = None
        self.is_processing = False  # Track TTS processing status
        self.initialize_tts()

    def clear_gpu_memory(self, reinitialize=False):
        """Clear GPU memory cache and run garbage collection"""
        if torch.cuda.is_available():
            # Empty CUDA cache
            torch.cuda.empty_cache()
            # Clear any existing models from memory
            if hasattr(self, 'tts') and self.tts is not None:
                del self.tts
            if hasattr(self, 'conditioning_latents') and self.conditioning_latents is not None:
                del self.conditioning_latents
            self.tts = None
            self.conditioning_latents = None
            # Run garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            # Only reinitialize if explicitly requested
            if reinitialize:
                self._initialize_tts_internal()
                
    def _initialize_tts_internal(self):
        """Internal method for TTS initialization without recursive cleanup"""
        # Configure DeepSpeed and PyTorch memory settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['DEEPSPEED_ZERO_STAGE'] = '2'  # Enable ZeRO stage 2
        os.environ['DEEPSPEED_ZERO_OPTIMIZATION'] = '1'  # Enable ZeRO optimization
        
        # Check CUDA capabilities
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = "cuda"
            # Check if GPU supports float16
            gpu_name = torch.cuda.get_device_name()
            compute_capability = torch.cuda.get_device_capability()
            supports_float16 = compute_capability[0] >= 7  # Volta or newer architecture
            logging.info(f"GPU: {gpu_name}, Compute Capability: {compute_capability}")
            
            # Configure DeepSpeed
            try:
                import deepspeed
                logging.info("DeepSpeed is installed")
                ds_version = deepspeed.__version__
                logging.info(f"DeepSpeed version: {ds_version}")
                
                # Initialize DeepSpeed config
                ds_config = {
                    "train_micro_batch_size_per_gpu": 1,
                    "gradient_accumulation_steps": 1,
                    "fp16": {
                        "enabled": supports_float16,
                        "loss_scale": 0,
                        "initial_scale_power": 16,
                        "loss_scale_window": 1000
                    },
                    "zero_optimization": {
                        "stage": 2,
                        "allgather_partitions": True,
                        "reduce_scatter": True,
                        "overlap_comm": True,
                        "contiguous_gradients": True
                    },
                    "zero_allow_untested_optimizer": True
                }
                
                # Set DeepSpeed config
                os.environ['DEEPSPEED_CONFIG'] = json.dumps(ds_config)
                use_deepspeed = True
                logging.info("DeepSpeed configured successfully")
            except ImportError:
                logging.warning("DeepSpeed is not installed")
                use_deepspeed = False
            except Exception as e:
                logging.warning(f"DeepSpeed error: {str(e)}")
                use_deepspeed = False
        else:
            device = "cpu"
            supports_float16 = False
            use_deepspeed = False
            logging.info("Running on CPU")
        
        # Initialize TTS with optimal configuration
        tts_config = {
            "kv_cache": True,
            "half": supports_float16,
            "device": device,
            "autoregressive_batch_size": 1,
            "use_deepspeed": use_deepspeed,
            "enable_redaction": False  # Disable redaction to save memory
        }
        logging.info(f"Initializing TTS with config: {tts_config}")
        self.tts = TextToSpeech(**tts_config)
        logging.info("TTS object created successfully")

        # Set fixed seed for consistent voice
        torch.manual_seed(42)
        if use_cuda:
            torch.cuda.manual_seed(42)

        # Get voice from config
        voice_name = os.getenv('TTS_VOICE', 'emma')
        logging.info(f"Loading voice samples for {voice_name}...")
        self.voice_samples = load_voice(voice_name, extra_voice_dirs=[])[0]
        logging.info(f"Voice samples loaded: {len(self.voice_samples)} samples")

        logging.info("Computing conditioning latents...")
        self.conditioning_latents = self.tts.get_conditioning_latents(self.voice_samples)
        logging.info("Conditioning latents generated")

        # Store in context
        self.context.update({
            'tts': self.tts,
            'voice_samples': self.voice_samples,
            'conditioning_latents': self.conditioning_latents
        })
            
    def initialize_tts(self):
        """Initialize TTS and return the initialized objects"""
        logging.info("\n=== Initializing TTS at startup ===")
        try:
            # Clear GPU memory before initialization
            self.clear_gpu_memory(reinitialize=False)
            self._initialize_tts_internal()
        except Exception as e:
            logging.info(f"Error initializing TTS: {e}")
            import traceback
            traceback.print_exc()

    def ensure_tts_initialized(self):
        """Ensure TTS is initialized before use"""
        if self.tts is None:
            self.initialize_tts()

    def determine_temperature(self, emotion_cue, is_retry=False):
        # Define temperature variations based on emotion with more dramatic ranges
        temperature_map = {
            # High energy emotions - more variation
            'excited': 1.8,
            'happy': 1.5,
            'enthusiastic': 1.6,
            'angry': 1.7,
            'energetic': 1.6,
            
            # Medium energy emotions - balanced variation
            'neutral': 1.0,
            'confident': 1.2,
            'professional': 0.9,
            'formal': 0.8,
            
            # Low energy emotions - more controlled
            'sad': 0.6,
            'gentle': 0.5,
            'calm': 0.7,
            'soft': 0.5,
            'tender': 0.6,
            'melancholic': 0.7
        }
        
        # Get the base temperature
        base_temperature = temperature_map.get('neutral', 1.0)
        
        # Adjust temperature based on emotion cue
        if emotion_cue:
            emotion_parts = emotion_cue.lower().split(' and ')
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
                temperature = max(0.5, temperature * 0.8)  # Reduce by 20% but not below 0.5
                
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
            'fast': {'prefix': 'speaking quickly and energetically', 'temp_mod': 1.2},
            'slow': {'prefix': 'speaking slowly and deliberately', 'temp_mod': 0.8},
            
            # Volume/Intensity modifiers
            'loud': {'prefix': 'speaking loudly and clearly', 'temp_mod': 1.3},
            'soft': {'prefix': 'speaking softly and gently', 'temp_mod': 0.7},
            'whispered': {'prefix': 'whispering intimately', 'temp_mod': 0.6},
            'shouting': {'prefix': 'shouting energetically', 'temp_mod': 1.4},
            
            # Style modifiers
            'clear': {'prefix': 'speaking very clearly and precisely', 'temp_mod': 0.9},
            'mysterious': {'prefix': 'speaking mysteriously', 'temp_mod': 1.1},
            'dramatic': {'prefix': 'speaking dramatically', 'temp_mod': 1.4},
            'playful': {'prefix': 'speaking playfully', 'temp_mod': 1.3},
            'formal': {'prefix': 'speaking formally', 'temp_mod': 0.8},
            'casual': {'prefix': 'speaking casually', 'temp_mod': 1.1}
        }
        
        if not style_cue:
            return "", self.determine_temperature("")
            
        # Split into parts and clean
        parts = [p.strip().lower() for p in style_cue.split(',')]
        
        # Process each part for emotions and styles
        emotion_temp = self.determine_temperature(parts[0])  # First part is always emotion
        
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
                    prompt_parts.append(style_modifiers[word]['prefix'])
                    temp_modifiers.append(style_modifiers[word]['temp_mod'])
        
        # Combine everything into a final prompt
        final_prompt = f"{', '.join(prompt_parts)}"
        
        # Calculate final temperature with modifiers
        final_temp = emotion_temp
        for modifier in temp_modifiers:
            final_temp *= modifier
            
        # Ensure temperature stays within bounds
        final_temp = min(2.0, max(0.5, final_temp))
        
        logger.info(f"Processed style cue '{style_cue}' into prompt '{final_prompt}' with temperature {final_temp}")
        return final_prompt, final_temp

    def text_to_speech(self, text):
        """Convert text to speech using Tortoise TTS with emotional prompting support.
        
        The text can include emotional and style cues in brackets at the start, like:
        "[happy and excited, fast] Hello there!"
        "[sad, slow and gentle] I miss you"
        "[professional, clear and confident] Let me explain"
        "[whispered, mysterious] I have a secret"
        "[shouting, angry and energetic] I'm so angry!"
        
        These cues will influence the speech generation but won't be spoken.
        """
        logger = logging.getLogger(__name__)
        logger.info("\n=== Starting text_to_speech ===")
        
        self.is_processing = True  # Set processing flag
        try:
            # Ensure TTS is initialized
            self.ensure_tts_initialized()
            
            if self.tts is None:
                raise RuntimeError("Failed to initialize TTS system")
                
            # Extract style cue if present
            style_cue = ""
            text_to_speak = text
            
            # Look for style cue in brackets at the start
            if text.startswith("["):
                end_bracket = text.find("]")
                if end_bracket != -1:
                    style_cue = text[1:end_bracket].strip()
                    text_to_speak = text[end_bracket + 1:].strip()
                    logger.info(f"Detected style cue: '{style_cue}'")
                    logger.info(f"Text to speak: '{text_to_speak[:100]}...'")
            else:
                logger.info("No style cue detected in text")
            
            # Process the style cue into an enhanced prompt and temperature
            emotion_prompt, temperature = self.process_style_cue(style_cue)
            
            print("Processing text chunks...")
            # Split long text into smaller chunks at sentence boundaries
            sentences = text_to_speak.split('.')
            max_chunk_length = 100  # Maximum characters per chunk
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chunk_length:
                    current_chunk += sentence + "."
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "."
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            print(f"Created {len(chunks)} chunks: {chunks}")
            
            # Process each chunk
            all_audio = []
            for i, chunk in enumerate(chunks, 1):
                print(f"\nProcessing chunk {i}/{len(chunks)}: '{chunk}'")
                if not chunk.strip():
                    print(f"Skipping empty chunk {i}")
                    continue
                
                print("Generating autoregressive samples...")
                try:
                    # Check GPU memory before processing
                    if torch.cuda.is_available():
                        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        if free_memory < 2 * 1024 * 1024 * 1024:  # Less than 2GB free
                            logger.warning(f"Low GPU memory ({free_memory/1024**3:.2f}GB free).")
                    
                    # Add enhanced emotion prompt back to each chunk if it exists
                    chunk_with_emotion = f"[{emotion_prompt}] {chunk}" if emotion_prompt else chunk
                    logger.info(f"Processing chunk with emotion prompt: '{chunk_with_emotion}'")
                    
                    # Get temperature based on emotion
                    logger.info(f"Using temperature {temperature} for emotional expression")
                    
                    gen = self.tts.tts_with_preset(
                        chunk_with_emotion,
                        voice_samples=self.voice_samples,
                        conditioning_latents=self.conditioning_latents,
                        preset='ultra_fast',  
                        use_deterministic_seed=True,
                        num_autoregressive_samples=2,  
                        diffusion_iterations=30,
                        cond_free=True,
                        cond_free_k=5.0,
                        temperature=temperature,  
                        length_penalty=1.0,
                        repetition_penalty=2.0,
                        top_p=0.8
                    )
                    print(f"Generated audio for chunk {i}")
                except RuntimeError as e:
                    print(f"Error generating audio for chunk {i}: {e}")
                    if "expected a non-empty list of Tensors" in str(e) or "out of memory" in str(e):
                        print("Retrying with different configuration...")
                        chunk_with_emotion = f"[{emotion_prompt}] {chunk}" if emotion_prompt else chunk
                        retry_temperature = self.determine_temperature("", is_retry=True)
                        logger.info(f"Retry attempt using temperature: {retry_temperature}")
                        
                        gen = self.tts.tts_with_preset(
                            chunk_with_emotion,
                            voice_samples=self.voice_samples,
                            conditioning_latents=self.conditioning_latents,
                            preset='ultra_fast',
                            use_deterministic_seed=True,
                            num_autoregressive_samples=1,  
                            diffusion_iterations=15,  
                            cond_free=False,  
                            temperature=retry_temperature,
                            length_penalty=1.0,
                            repetition_penalty=2.0,
                            top_p=0.8
                        )
                        print("Retry successful")
                    else:
                        raise
                
                if isinstance(gen, tuple):
                    gen = gen[0]
                if len(gen.shape) == 3:
                    gen = gen.squeeze(0)
                
                print(f"Audio shape for chunk {i}: {gen.shape}")
                all_audio.append(gen)

            # Combine all audio chunks
            print("\nCombining audio chunks...")
            if all_audio:
                combined_audio = torch.cat(all_audio, dim=1)
                print(f"Audio chunks combined successfully, final shape: {combined_audio.shape}")
            else:
                print("No audio generated")
                return None

            # Create a temporary file
            print("Saving audio to file...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                temp_path = fp.name
                torchaudio.save(temp_path, combined_audio.cpu(), 24000)
                print(f"Audio saved to {temp_path}")
                
            return temp_path

        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.is_processing = False  # Clear processing flag
