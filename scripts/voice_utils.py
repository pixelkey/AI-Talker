import speech_recognition as sr
import os
from gtts import gTTS
import tempfile
from playsound import playsound

def speech_to_text():
    """
    Convert speech from microphone to text.
    Returns:
        str: Transcribed text from speech, or empty string if transcription fails.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            print("Processing speech...")
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""
    except Exception as e:
        print(f"Error occurred: {e}")
        return ""

def text_to_speech(text):
    """
    Convert text to speech and play it.
    Args:
        text (str): Text to convert to speech
    """
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
            
        # Generate speech
        tts = gTTS(text=text, lang='en')
        tts.save(temp_filename)
        
        # Play the audio
        playsound(temp_filename)
        
        # Clean up the temporary file
        os.unlink(temp_filename)
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
