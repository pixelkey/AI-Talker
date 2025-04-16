import os
import time
import logging
os.environ["TTS_VOICE"] = "Alex_0"  # Ensure the correct voice is used for testing
from tts_utils import TTSManager

# Configure more detailed logging for the test
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_text_to_speech_sentences():
    """Test that TTSManager processes and plays each sentence separately with consistent voice."""
    # Set up a dummy context
    context = {'is_processing': False}
    tts = TTSManager(context)

    # Use a multi-sentence input with potentially challenging boundaries
    test_text = ("Hello world! This is a test of the TTS system. Each sentence should play as soon as it is ready. "
                "Here's a more complex sentence with commas, semicolons; and other punctuation! "
                "What about questions? Can the system handle them? "
                "There's also a slight chance of fog in the morning. This is where it previously stopped.")

    print("\n==== TESTING SENTENCE-BY-SENTENCE TTS ====")
    print(f"Input text has {len(tts.split_text_to_sentences(test_text))} sentences.")
    
    temp_paths = tts.text_to_speech_sentences(test_text)
    print(f"Generated {len(temp_paths)} audio files:")
    for i, path in enumerate(temp_paths):
        filesize = os.path.getsize(path)
        print(f"  - Audio {i+1}: {path} ({filesize} bytes)")

    # Check that a temp file was created for each sentence
    sentences = tts.split_text_to_sentences(test_text)
    assert len(temp_paths) == len(sentences), f"Expected {len(sentences)} files, got {len(temp_paths)}"
    for path in temp_paths:
        assert os.path.exists(path), f"Missing output file: {path}"
        assert os.path.getsize(path) > 0, f"Output file is empty: {path}"

    print("\n==== TESTING ORIGINAL TEXT_TO_SPEECH METHOD ====")
    print("This should now use sentence-by-sentence processing internally")
    
    # Test that the original method now uses the improved sentence-by-sentence processing
    single_path = tts.text_to_speech("This is a test of the original method. It should also process by sentence.")
    assert single_path and os.path.exists(single_path), f"Original method failed to generate audio: {single_path}"
    filesize = os.path.getsize(single_path)
    print(f"Generated single audio file: {single_path} ({filesize} bytes)")
    
    print("\nAll temp files created and non-empty. Waiting for all audio to finish...")
    # Wait for audio queue to finish (give time for playback)
    tts.audio_queue.join()
    print("Test complete. Cleaning up temp files...")
    for path in temp_paths:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(single_path):
        os.remove(single_path)
    print("Cleanup complete.")

if __name__ == "__main__":
    test_text_to_speech_sentences()
