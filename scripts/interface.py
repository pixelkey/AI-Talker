# scripts/interface.py

import gradio as gr
from chatbot_functions import chatbot_response, clear_history, retrieve_and_format_references
from chat_history import ChatHistoryManager
from ingest_watcher import IngestWatcher
from faiss_utils import save_faiss_index_metadata_and_docstore
import os
import logging
from vector_store_client import VectorStoreClient
from document_processing import normalize_text
import ollama
import time
from tts_utils import TTSManager
from embedding_updater import EmbeddingUpdater
from speech_recognition_utils import SpeechRecognizer
from self_reflection import SelfReflection
from queue import Queue
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    conversation: list = field(default_factory=list)

def setup_gradio_interface(context):
    """
    Sets up the Gradio interface.
    Args:
        context (dict): Context containing client, memory, and other settings.
    Returns:
        gr.Blocks: Gradio interface object.
    """
    print("\n=== Setting up Gradio interface ===")
    
    # Initialize TTS at startup
    tts_manager = TTSManager(context)
    context['tts_manager'] = tts_manager  # Add TTS manager to context
    if not context.get('tts'):
        print("Failed to initialize TTS")
        return None
    
    # Initialize speech recognizer
    speech_recognizer = SpeechRecognizer()
    
    # Initialize state
    state = {"last_processed_index": 0}

    # Initialize chat manager once at startup
    chat_manager = ChatHistoryManager()
    context['chat_manager'] = chat_manager

    # Initialize vector store client and LLM client
    vector_store_client = VectorStoreClient(
        context['vector_store'],
        context['embeddings'],
        normalize_text
    )
    context['vector_store_client'] = vector_store_client
    
    # Initialize Ollama client for local LLM
    if context.get('MODEL_SOURCE') == 'local':
        context['client'] = ollama

    # Initialize embedding updater
    embedding_updater = EmbeddingUpdater(context)
    context['embedding_updater'] = embedding_updater
    
    # Initialize self reflection
    self_reflection = SelfReflection(context)
    
    # Create the watcher but don't start it yet - we'll manually trigger updates
    watcher = IngestWatcher(embedding_updater.update_embeddings)
    context['watcher'] = watcher

    def parse_timestamp(timestamp):
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")

    def process_audio(audio, state):
        """Process incoming audio stream"""
        try:
            if audio is None:
                print("No audio data received")
                return None, state
            
            # audio is a tuple of (sample_rate, audio_data)
            print(f"Received audio chunk: sample_rate={audio[0]}, shape={audio[1].shape if audio[1] is not None else None}")
            
            if state.stream is None:
                state.stream = audio[1]
                state.sampling_rate = audio[0]
                print(f"Started new audio stream with sampling rate {state.sampling_rate}")
            else:
                state.stream = np.concatenate((state.stream, audio[1]))
                print(f"Updated stream length: {len(state.stream)}")
            
            return None, state
        except Exception as e:
            print(f"Error in process_audio: {str(e)}")
            return None, state

    def handle_stop(audio_data, state):
        """Handle when recording is stopped"""
        try:
            print(f"Stop handler called with audio_data={type(audio_data) if audio_data is not None else None}")
            print(f"State stream: shape={state.stream.shape if state.stream is not None else None}, rate={state.sampling_rate}")
            
            if state.stream is None:
                print("No audio data in state")
                return "", None
            
            print("Recording stopped, processing audio...")
            # Convert audio data to the format expected by the speech recognizer
            try:
                # Ensure audio data is float32 and between -1 and 1
                audio_float32 = state.stream.astype(np.float32)
                if audio_float32.max() > 1.0 or audio_float32.min() < -1.0:
                    audio_float32 = np.clip(audio_float32 / 32768.0, -1.0, 1.0)
                
                print(f"Audio stats - min: {audio_float32.min()}, max: {audio_float32.max()}, mean: {audio_float32.mean()}")
                result = speech_recognizer.transcribe_audio((state.sampling_rate, audio_float32))
                text = result[0] if result[0] else ""
                print(f"Transcribed text: {text}")
            except Exception as e:
                print(f"Error in transcription: {str(e)}")
                text = ""
            
            # Reset the stream
            state.stream = None
            state.sampling_rate = 0
            
            return text, gr.update(value=text, interactive=True)
        except Exception as e:
            print(f"Error in handle_stop: {str(e)}")
            return "", None

    def handle_transcription(state):
        """Handle the transcribed text"""
        try:
            if hasattr(state, 'value') and state.value:
                text = state.value
                print(f"Processing transcribed text: {text}")
                return text, gr.update(value=text, interactive=True)
            return "", None
        except Exception as e:
            print(f"Error in handle_transcription: {str(e)}")
            return "", None

    def handle_audio_input(audio_data, state):
        """Process completed audio recording"""
        try:
            if audio_data is None:
                return "", None, state
            
            result = speech_recognizer.transcribe_audio(audio_data)
            text = result[0] if result[0] else ""
            return text, gr.update(value=text), state
        except Exception as e:
            print(f"Error in handle_audio_input: {str(e)}")
            return "", None, state

    def handle_user_input(input_text, history, state):
        """Handle user input and generate response with proper state management."""
        try:
            # Notify self-reflection about user input
            self_reflection.notify_user_input()
            
            # Validate input
            if not input_text or not input_text.strip():
                return history, "", "", history, None
                
            # Get chat manager from context
            chat_manager = context['chat_manager']

            # Set current time in context
            context['current_time'] = '2025-01-08T15:10:11+10:30'  # Using the provided time
            
            # Get references and generate response
            refs, filtered_docs, context_documents = retrieve_and_format_references(input_text, context)
            
            # Generate the LLM response and get updated references
            _, response, refs, _ = chatbot_response(input_text, context_documents, context, history)
            
            # Format messages with timestamp
            dt = parse_timestamp(context['current_time'])
            formatted_time = dt.strftime("%A, %Y-%m-%d %H:%M:%S %z")
            user_msg = {"role": "user", "content": f"[{formatted_time}]\nUser: {input_text}"}
            bot_msg = {"role": "assistant", "content": f"[{formatted_time}]\nBot: {response}"}
            
            # Update history with the new user and bot messages
            new_history = history + [user_msg, bot_msg]
            chat_manager.save_history(new_history)
            
            # Generate speech for the response
            print("\nGenerating TTS response...")
            # Use the emotion-enhanced response if available
            tts_text = context.get('tts_response', response)
            print(f"\nDEBUG - Original response: {response[:100]}...")
            print(f"DEBUG - TTS text with emotion: {tts_text[:100]}...")
            audio_path = tts_manager.text_to_speech(tts_text)
            print("TTS generation complete")
            
            # Update embeddings in background
            context['embedding_updater'].update_chat_embeddings_async(history, state)
            
            # Wait for TTS to fully complete before starting reflection
            while tts_manager.is_processing:
                time.sleep(0.1)  # Short sleep to prevent busy waiting
            
            # Start reflection after TTS is done
            self_reflection.start_reflection(new_history, lambda x: None)
            
            return (
                new_history,  # chatbot
                refs,        # references
                "",         # input_text (clear it)
                new_history, # gr.State
                audio_path   # audio_output
            )
            
        except Exception as e:
            print(f"Error in handle_user_input: {str(e)}")
            import traceback
            traceback.print_exc()
            return history, "", "", history, None

    def clear_interface(history):
        # Stop any ongoing reflection by setting the event
        if self_reflection and hasattr(self_reflection, 'stop_reflection'):
            self_reflection.stop_reflection.set()
            
        cleared_history, cleared_refs, cleared_input = clear_history(context, history)
        return [], cleared_refs, cleared_input, [], None

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column(scale=1):
                references = gr.Markdown("References:")
            
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    show_label=False,
                    layout="bubble",
                    type="messages"  # Specify the type as messages to use OpenAI-style format
                )
                audio_output = gr.Audio(
                    label="Response",
                    show_label=True,
                    autoplay=True,
                    elem_id="audio-output"
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=6):
                        input_text = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter, or use the microphone...",
                            lines=1,
                            max_lines=1,
                            container=False
                        )
                    with gr.Column(scale=4, min_width=0):
                        with gr.Group():
                            audio_input = gr.Audio(
                                sources=["microphone"],
                                type="numpy",
                                streaming=True,
                                label="Speak",
                                show_label=True
                            )

                with gr.Row():
                    submit_text = gr.Button("Send", elem_id="submit-btn")
                    clear_btn = gr.Button("Clear", elem_id="clear-btn")

                state = gr.State(value=AppState())

                # Set up audio streaming and stopping
                audio_stream = audio_input.stream(
                    process_audio,
                    [audio_input, state],
                    [audio_input, state],
                    stream_every=0.5,
                    time_limit=30
                )

                # Handle stop recording
                audio_input.stop_recording(
                    handle_stop,
                    [audio_input, state],
                    [input_text, submit_text]
                ).success(
                    handle_user_input,
                    inputs=[input_text, chatbot, state],
                    outputs=[
                        chatbot,
                        references,
                        input_text,
                        gr.State(value=[]),
                        audio_output
                    ],
                    queue=True
                )

                # Start recording again after audio output finishes
                def restart_recording(state):
                    """Restart recording after audio output finishes"""
                    state.stream = None
                    state.sampling_rate = 0
                    # Return both audio input update and start recording
                    return gr.update(value=None, interactive=True, recording=True)

                audio_output.stop(
                    restart_recording,
                    [state],
                    [audio_input]
                )

                # Text input handlers
                submit_text.click(
                    handle_user_input,
                    inputs=[input_text, chatbot, state],
                    outputs=[
                        chatbot,
                        references,
                        input_text,
                        gr.State(value=[]),
                        audio_output
                    ],
                    queue=True
                )

                clear_btn.click(
                    clear_interface,
                    inputs=[chatbot],
                    outputs=[chatbot, references, input_text]
                )

        return interface

def clear_interface(chat_history):
    """Clears the interface"""
    return [], "", ""
