import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import os
import logging
import tempfile
import torchaudio

class TTSManager:
    def __init__(self, context):
        """Initialize TTS Manager with context"""
        self.context = context
        self.tts = None
        self.voice_samples = None
        self.conditioning_latents = None
        self.initialize_tts()

    def initialize_tts(self):
        """Initialize TTS and return the initialized objects"""
        print("\n=== Initializing TTS at startup ===")
        try:
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

    def text_to_speech(self, text):
        """Convert text to speech using Tortoise TTS"""
        print("\n=== Starting text_to_speech ===")
        
        if not text:
            print("No text provided, returning None")
            return None

        if not all([self.tts, self.voice_samples, self.conditioning_latents]):
            print("TTS not properly initialized")
            return None
        
        try:
            print("Processing text chunks...")
            # Split long text into smaller chunks at sentence boundaries
            sentences = text.split('.')
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
                    gen = self.tts.tts_with_preset(
                        chunk,
                        voice_samples=self.voice_samples,
                        conditioning_latents=self.conditioning_latents,
                        preset='ultra_fast',
                        use_deterministic_seed=True,
                        num_autoregressive_samples=1,
                        diffusion_iterations=10,
                        cond_free=True,
                        cond_free_k=2.0,
                        temperature=0.8
                    )
                    print(f"Generated audio for chunk {i}")
                except RuntimeError as e:
                    print(f"Error generating audio for chunk {i}: {e}")
                    if "expected a non-empty list of Tensors" in str(e):
                        print("Retrying with different configuration...")
                        # Try again with modified settings
                        gen = self.tts.tts_with_preset(
                            chunk,
                            voice_samples=self.voice_samples,
                            conditioning_latents=self.conditioning_latents,
                            preset='ultra_fast',
                            use_deterministic_seed=True,
                            num_autoregressive_samples=2,
                            diffusion_iterations=10,
                            cond_free=False,
                            temperature=0.8
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
