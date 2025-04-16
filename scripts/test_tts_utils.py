import os
import sys
import argparse
import glob
import gc
import torch
# Add the parent directory to sys.path to find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import logging
from tts_utils import TTSManager

# Configure more detailed logging for the test
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_available_voices():
    """Get a list of available voice files in the assets/voices directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voices_dir = os.path.join(base_dir, 'assets', 'voices')
    voice_files = glob.glob(os.path.join(voices_dir, '*.pt'))
    
    # Extract voice names from filenames
    voice_names = []
    for voice_file in voice_files:
        voice_name = os.path.splitext(os.path.basename(voice_file))[0]
        # Skip context files
        if not voice_name.endswith('_context'):
            voice_names.append(voice_name)
    
    return sorted(voice_names)

def clear_gpu_memory():
    """Clear CUDA memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    print("Memory cleared")

def test_voice_consistency(voice_name=None, short_test=False):
    """Test specifically for voice consistency between sentences."""
    # Set environment variables
    if voice_name:
        os.environ["TTS_VOICE"] = voice_name
        print(f"\n==== TESTING VOICE CONSISTENCY FOR {voice_name} ====")
    else:
        print("\n==== TESTING VOICE CONSISTENCY ====")
    
    # Set up a dummy context
    context = {'is_processing': False}
    # Initialize TTS with watermarking disabled to save memory
    tts = TTSManager(context, voice_name=voice_name, disable_watermark=True)

    # Use an appropriate test length
    if short_test:
        test_text = "This is a short test."
        print("Using short test to conserve memory")
    else:
        test_text = "This is sentence one. This is sentence two. This is sentence three."
    
    print(f"Processing text with {len(tts.split_text_to_sentences(test_text))} sentences.")
    
    try:
        temp_paths = tts.text_to_speech_sentences(test_text)
        print(f"Generated {len(temp_paths)} audio files:")
        for i, path in enumerate(temp_paths):
            filesize = os.path.getsize(path)
            print(f"  - Audio {i+1}: {path} ({filesize} bytes)")

        # Check that all files were created
        sentences = tts.split_text_to_sentences(test_text)
        assert len(temp_paths) == len(sentences), f"Expected {len(sentences)} files, got {len(temp_paths)}"
        
        print("\nAll audio files created successfully. Waiting for audio playback to finish...")
        # Wait for audio queue to finish (give time for playback)
        tts.audio_queue.join()
        print("Test complete. Cleaning up temp files...")
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
        print("Cleanup complete.")
    except Exception as e:
        print(f"Error testing voice: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up memory
        del tts
        clear_gpu_memory()

def list_voices():
    """List all available voices and their IDs."""
    voices = get_available_voices()
    print("\n==== AVAILABLE VOICES ====")
    if not voices:
        print("No voice files found in assets/voices directory")
        return
    
    print(f"Found {len(voices)} voice files:")
    for voice in voices:
        # Try to extract speaker ID from voice name (format: Name_ID)
        speaker_id = "Unknown"
        if '_' in voice and voice.split('_')[-1].isdigit():
            speaker_id = voice.split('_')[-1]
        print(f"  - {voice} (Speaker ID: {speaker_id})")
    
    print("\nTo test a specific voice, run:")
    print("  python scripts/test_tts_utils.py --voice Voice_Name")
    print("\nFor memory-constrained systems, try:")
    print("  python scripts/test_tts_utils.py --voice Voice_Name --short")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TTS voice consistency')
    parser.add_argument('--voice', '-v', type=str, help='Voice name to test (e.g., Alex_0, Carter_1)')
    parser.add_argument('--list', '-l', action='store_true', help='List available voices')
    parser.add_argument('--short', '-s', action='store_true', help='Use shorter test to conserve memory')
    
    args = parser.parse_args()
    
    if args.list:
        list_voices()
    elif args.voice:
        test_voice_consistency(args.voice, args.short)
    else:
        # Default behavior: list voices and then test default voice
        list_voices()
        print("\nTesting with default voice:")
        test_voice_consistency(short_test=args.short)
