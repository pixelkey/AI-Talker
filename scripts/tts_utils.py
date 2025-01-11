import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import os
import logging
import tempfile
import torchaudio
import gc

class TTSManager:
    def __init__(self, context):
        """Initialize TTS Manager with context"""
        self.context = context
        self.tts = None
        self.voice_samples = None
        self.conditioning_latents = None
        self.initialize_tts()

    def clear_gpu_memory(self):
        """Clear GPU memory cache and run garbage collection"""
        if torch.cuda.is_available():
            # Empty CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
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

    def initialize_tts(self):
        """Initialize TTS and return the initialized objects"""
        print("\n=== Initializing TTS at startup ===")
        try:
            # Clear GPU memory before initialization
            self.clear_gpu_memory()

            # Initialize TTS with optimal configuration
            tts_config = {
                "kv_cache": True,
                "half": True,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "autoregressive_batch_size": 1,  # larger GPU memory usage if set more than 1
                "use_deepspeed": True
            }
            print(f"Initializing TTS with config: {tts_config}")
            self.tts = TextToSpeech(**tts_config)
            print("TTS object created successfully")

            # Set fixed seed for consistent voice
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)

            # Get voice from config
            voice_name = os.getenv('TTS_VOICE', 'emma')
            print(f"Loading voice samples for {voice_name}...")
            self.voice_samples = load_voice(voice_name, extra_voice_dirs=[])[0]
            print(f"Voice samples loaded: {len(self.voice_samples)} samples")

            print("Computing conditioning latents...")
            self.conditioning_latents = self.tts.get_conditioning_latents(self.voice_samples)
            print("Conditioning latents generated")

            # Store in context
            self.context.update({
                'tts': self.tts,
                'voice_samples': self.voice_samples,
                'conditioning_latents': self.conditioning_latents
            })

        except Exception as e:
            print(f"Error initializing TTS: {e}")
            import traceback
            traceback.print_exc()

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

    def text_to_speech(self, text):
        """Convert text to speech using Tortoise TTS with emotional prompting support.
        
        The text can include emotional cues in brackets at the start, like:
        "[happy and excited] Hello there!" or "[sad and slow] I miss you"
        These cues will influence the speech generation but won't be spoken.
        """
        logger = logging.getLogger(__name__)
        logger.info("\n=== Starting text_to_speech ===")
        
        if not text:
            logger.warning("No text provided, returning None")
            return None

        if not all([self.tts, self.voice_samples, self.conditioning_latents]):
            logger.error("TTS not properly initialized")
            return None
        
        try:
            # Extract emotion cue if present
            emotion_cue = ""
            text_to_speak = text
            
            # Look for emotion cue in brackets at the start
            if text.startswith("["):
                end_bracket = text.find("]")
                if end_bracket != -1:
                    emotion_cue = text[1:end_bracket].strip()
                    text_to_speak = text[end_bracket + 1:].strip()
                    logger.info(f"Detected emotion cue: '{emotion_cue}'")
                    logger.info(f"Text to speak: '{text_to_speak[:100]}...'")
            else:
                logger.info("No emotion cue detected in text")
            
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
                    # Add emotion cue back to each chunk if it exists
                    chunk_with_emotion = f"[{emotion_cue}] {chunk}" if emotion_cue else chunk
                    logger.info(f"Processing chunk with emotion: '{chunk_with_emotion}'")
                    
                    # Get temperature based on emotion
                    temperature = self.determine_temperature(emotion_cue)
                    logger.info(f"Using temperature {temperature} for emotional expression")
                    
                    gen = self.tts.tts_with_preset(
                        chunk_with_emotion,
                        voice_samples=self.voice_samples,
                        conditioning_latents=self.conditioning_latents,
                        preset='fast',  # Changed from standard to ultra_fast for memory efficiency
                        use_deterministic_seed=True,
                        num_autoregressive_samples=2,  # Keep at 1 for memory efficiency
                        diffusion_iterations=30,
                        cond_free=True,
                        cond_free_k=5.0,
                        temperature=temperature,  # Use emotion-based temperature
                        length_penalty=1.0,
                        repetition_penalty=2.0,
                        top_p=0.8
                    )
                    print(f"Generated audio for chunk {i}")
                except RuntimeError as e:
                    print(f"Error generating audio for chunk {i}: {e}")
                    if "expected a non-empty list of Tensors" in str(e):
                        print("Retrying with different configuration...")
                        # Try again with modified settings and emotion
                        chunk_with_emotion = f"[{emotion_cue}] {chunk}" if emotion_cue else chunk
                        
                        # Use a slightly lower temperature for retry
                        retry_temperature = self.determine_temperature(emotion_cue, is_retry=True)
                        logger.info(f"Retry attempt using temperature: {retry_temperature}")
                        
                        gen = self.tts.tts_with_preset(
                            chunk_with_emotion,
                            voice_samples=self.voice_samples,
                            conditioning_latents=self.conditioning_latents,
                            preset='ultra_fast',  # Changed from standard to ultra_fast for retry as well
                            use_deterministic_seed=True,
                            num_autoregressive_samples=1,
                            diffusion_iterations=20,
                            cond_free=True,
                            cond_free_k=4.0,
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
