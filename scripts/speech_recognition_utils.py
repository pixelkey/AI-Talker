import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self):
        """Initialize speech recognizer"""
        self.recognizer = sr.Recognizer()

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text.
        
        Args:
            audio_path (str): Path to the audio file to transcribe
            
        Returns:
            tuple: (transcribed_text, transcribed_text, status_message)
        """
        if audio_path is None:
            return "", "", None
        
        try:
            # Use the audio file directly
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            
            # Perform the recognition
            text = self.recognizer.recognize_google(audio)
            return text, text, f"Transcribed: {text}"
        except Exception as e:
            return "", "", f"Error transcribing audio: {str(e)}"
