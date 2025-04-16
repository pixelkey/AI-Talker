import os
import sys
# Add the parent directory to sys.path to find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import logging
os.environ["TTS_VOICE"] = "Alex_0"  # Ensure the correct voice is used for testing
from tts_utils import TTSManager

# Configure more detailed logging for the test
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_voice_consistency():
    """Test specifically for voice consistency between sentences."""
    print("\n==== TESTING VOICE CONSISTENCY ====")
    # Set up a dummy context
    context = {'is_processing': False}
    tts = TTSManager(context)

    # Use a short test with just 3 sentences
    test_text = "This is sentence one. This is sentence two. This is sentence three."
    print(f"Processing text with {len(tts.split_text_to_sentences(test_text))} sentences.")
    
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

if __name__ == "__main__":
    test_voice_consistency()
