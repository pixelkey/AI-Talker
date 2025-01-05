import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import os
import logging

def initialize_tts():
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
        tts = TextToSpeech(**tts_config)
        print("TTS object created successfully")

        # Set fixed seed for consistent voice
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Get voice from config
        voice_name = os.getenv('TTS_VOICE', 'emma')
        print(f"Loading voice samples for {voice_name}...")
        voice_samples = load_voice(voice_name, extra_voice_dirs=[])[0]
        print(f"Voice samples loaded: {len(voice_samples)} samples")

        print("Computing conditioning latents...")
        gen_conditioning_latents = tts.get_conditioning_latents(voice_samples)
        print("Conditioning latents generated")

        return {
            'tts': tts,
            'voice_samples': voice_samples,
            'conditioning_latents': gen_conditioning_latents
        }
    except Exception as e:
        print(f"Error initializing TTS: {e}")
        import traceback
        traceback.print_exc()
        return None
