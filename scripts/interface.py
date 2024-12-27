# scripts/interface.py

import gradio as gr
from chatbot_functions import chatbot_response, clear_history, retrieve_and_format_references
import speech_recognition as sr
import numpy as np
import io
import soundfile as sf
from gtts import gTTS
import tempfile
import os

def setup_gradio_interface(context):
    """
    Sets up the Gradio interface.
    Args:
        context (dict): Context containing client, memory, and other settings.
    Returns:
        gr.Blocks: Gradio interface object.
    """
    with gr.Blocks(css=".separator { margin: 8px 0; border-bottom: 1px solid #ddd; }") as app:
        # Output fields
        chat_history = gr.Chatbot(label="Chat History", height=400)
        references = gr.Textbox(label="References", lines=2, interactive=False)
        audio_output = gr.Audio(
            label="Response Audio", 
            visible=True, 
            autoplay=True,
            show_download_button=False
        )
        debug_output = gr.Textbox(label="Debug Output", lines=2, interactive=False)

        # Input fields
        with gr.Row():
            input_text = gr.Textbox(label="Input Text", scale=4)
            audio_input = gr.Audio(
                label="Voice Input", 
                sources=["microphone"],
                type="filepath"
            )

        with gr.Row():
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear")

        # Initialize session state separately for each user
        session_state = gr.State(value=[])

        def text_to_speech(text):
            """Convert text to speech using gTTS"""
            if not text:
                return None
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                    temp_path = fp.name
                
                # Generate speech
                tts = gTTS(text=text, lang='en')
                tts.save(temp_path)
                
                return temp_path
            except Exception as e:
                print(f"Error in text-to-speech: {str(e)}")
                return None

        # Define a function to handle both reference retrieval and LLM response generation
        def handle_user_input(input_text, history):
            if not input_text.strip():
                return history, references.value, "", history, None, "No input provided"
                
            references, filtered_docs, context_documents = retrieve_and_format_references(input_text, context)
            
            # Initialize history if needed
            if history is None:
                history = []
            
            # Add user message to history
            history.append((input_text, None))
            yield history, references, "", history, None, ""

            # Generate the LLM response if references were found
            if filtered_docs:
                _, response, _ = chatbot_response(input_text, context_documents, context, history)
                # Add assistant response to history
                history[-1] = (input_text, response)
                # Generate speech for the response
                audio_path = text_to_speech(response)
                yield history, references, "", history, audio_path, ""
            else:
                response = "I don't have any relevant information to help with that query."
                history[-1] = (input_text, response)
                audio_path = text_to_speech(response)
                yield history, references, "", history, audio_path, ""

        def transcribe_audio(audio_path):
            if audio_path is None:
                return "", "No audio input received"
            
            try:
                # Initialize recognizer
                recognizer = sr.Recognizer()
                
                # Use the audio file directly
                with sr.AudioFile(audio_path) as source:
                    print("Recording from audio file...")
                    audio_data = recognizer.record(source)
                
                # Perform the recognition
                print("Starting speech recognition...")
                text = recognizer.recognize_google(audio_data)
                print(f"Recognized text: {text}")
                return text, f"Successfully transcribed: {text}"
            except Exception as e:
                error_msg = f"Error in speech recognition: {str(e)}"
                print(error_msg)
                return "", error_msg

        def clear_interface(history):
            cleared_history, cleared_refs, cleared_input = clear_history(context, history)
            return [], cleared_refs, cleared_input, [], None, ""

        # Setup event handlers with explicit state management
        submit_button.click(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output, debug_output],
        )
        
        clear_button.click(
            clear_interface,
            inputs=[session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output, debug_output],
        )

        # Add audio input handler
        audio_input.change(
            transcribe_audio,
            inputs=[audio_input],
            outputs=[input_text, debug_output]
        )

        # Add text input submission via Enter key
        input_text.submit(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output, debug_output],
        )

    return app
