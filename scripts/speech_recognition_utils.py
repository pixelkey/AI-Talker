import speech_recognition as sr
import numpy as np
import io
import wave
import tempfile

class SpeechRecognizer:
    def __init__(self):
        """Initialize speech recognizer"""
        self.recognizer = sr.Recognizer()

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Either a string path to an audio file, or a tuple of (sample_rate, numpy_array)
            
        Returns:
            tuple: (transcribed_text, transcribed_text, status_message)
        """
        if audio_data is None:
            return "", "", None
        
        try:
            if isinstance(audio_data, tuple):
                # Handle numpy array data
                sample_rate, audio_array = audio_data
                
                # Create a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                    # Save the numpy array as a WAV file
                    with wave.open(temp_file.name, 'wb') as wav_file:
                        # Prepare the audio data
                        audio_array = np.clip(audio_array, -1.0, 1.0)
                        audio_int16 = (audio_array * 32767).astype(np.int16)
                        
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    # Use the temporary file for recognition
                    with sr.AudioFile(temp_file.name) as source:
                        audio = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio)
                        print(f"Successfully transcribed audio to: {text}")
                        return text, text, f"Transcribed: {text}"
            else:
                # Handle file path
                with sr.AudioFile(audio_data) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                    return text, text, f"Transcribed: {text}"
                    
        except Exception as e:
            error_msg = f"Error transcribing audio: {str(e)}"
            print(error_msg)
            return "", "", error_msg
